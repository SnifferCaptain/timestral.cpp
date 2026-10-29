# Inference-Side Design

[Model architecture](<architecture.md>) answers "what does one forward pass compute"; this document answers "how does one request become one or several forward passes, and which computations can be saved". The content follows the mechanisms that actually exist in the code: scheduling, chunked autoregression and post-processing, paged KV cache, multivariate and conditional completion, and the frozen-prefix sliding-window experiment that is only provided in the examples. Numerical evidence uniformly cites the local acceptance and progress records (not checked into the repo); no re-measurement was done for this document.

## From Sample Points to Tensors

The public interface only deals with sample points. `Request::variables` is a `vector<vector<Sample>>`, the outer level being channels, the inner level being time-increasing `Sample{time, value}`; NaN denotes a missing measurement; `horizonSeconds` gives the number of seconds to forecast from the latest valid observation, and when `sampleInterval` is 0 it is inferred from the timestamps. All channels share one time axis; which segment is conditioned and which segment is to be forecast is determined only by each channel's own observation positions — no role table is needed. The full declaration is in [timestral.hpp](<../../timestral/timestral.hpp>).

The naive approach is to re-encode the entire history on every request, then forecast block by block. It is simple and guaranteed correct; the problem is that the typical usage of a sensor stream is "forecast every time a small new chunk of data arrives", and the histories of two adjacent requests are mostly identical, so repeated encoding takes the main cost. Most of the inference-side design revolves around two things: reusing historical computation while guaranteeing consistency with a full recomputation, and merging multiple concurrent requests into fewer forward passes.

## Submission and Scheduling

![A request from submission to result](<../../assets/infer-flow.svg>)

`submit` completes all validation and grid computation on the calling thread ([Service::prepare](<../../timestral/utils/service.cpp#L218-L285>)); the worker thread only does numerical computation and page-cache maintenance. This way an illegal request throws an exception immediately at submission and does not occupy the queue; the worker thread also does not have to handle format errors. The service has a single worker thread, model weights are read-only after construction, and multiple threads can submit concurrently.

The semantics of the three strategies are:

- Fifo: take and execute one request from the front of the queue each time.
- Mailbox: at submission, if the queue already contains a not-yet-started request with the same `streamId`, the older request's future ends with `RequestSuperseded` and is dequeued, keeping only the latest one; switching to Mailbox also deduplicates the existing queue by source. Anonymous requests (empty `streamId`) each get a unique identity and are never deduplicated against each other. Suited for sensor streams that "only care about the latest data".
- Continuous: starting from the front of the queue, take batchable requests continuously in FCFS order, up to `maxBatch`; stop at the first non-batchable request — no skipping, no preemption. A round of requests is executed together and releases its slots together; there is no per-token intermediate state.

The conditions under which two requests can be batched are in [Service::batchable](<../../timestral/utils/service.cpp#L368-L408>): the number of channels, history length, steps, interval, whether the paged grid is used, and both post-processing switches all agree; condition-completion requests also require the observation layout to be pointwise identical. `abort(streamId)` invalidates all not-yet-started requests of that source and returns its references in the page cache.

After a round is taken, [executeRound](<../../timestral/utils/service.cpp#L410-L521>) splits into three paths: with only one request, execute them one by one; among multiple requests, those hitting identical page tables form a group sharing decoding; the remaining non-hitting requests are cold-batched into one `(B,V,T)` forward pass.

## Grid and Context

The standard path requires all channels to have identical timestamps and valid values at the tail. At submission, the most recent samples are kept according to `maxContext` (default 0, meaning the model's `context_length`); a history shorter than 2 patches is rejected outright. After that there are two grids:

- The full path left-pads the kept history to a multiple of the patch size; padded positions are marked as unobserved.
- The paged path left-aligns the history to patch boundaries; whole-patch prefixes enter pages, and the trailing part shorter than one patch does not enter pages but is encoded together with the horizon during decoding. Left alignment keeps the boundaries of earlier patches unchanged when the same source appends data, so page keys can hit.

Whether the paged grid is used is decided at submission: the backend supports the incremental interface; in continuous RoPE mode the history must be a whole multiple of the patch; after padding it must not exceed `maxContext`; the number of whole patches must not exceed `cachePages`. The default 2048 pages cover 65536 sample points, which is more than the 5M's 32768-point context, so under the default settings single-channel history all goes through the paged path; the decision is in [the decision code](<../../timestral/utils/service.cpp#L361-L366>). The mirror pass of symmetric averaging takes the prefix negative and builds a separate page chain, sharing the page pool with the positive pages; when capacity is insufficient, only the mirror pass falls back to a full recomputation.

## Chunked Autoregression and Refine

The model decodes at most `block_size` patches at once (32 for 5M, i.e. 1024 steps). `Settings::patchBatch` defaults to 16 and is clamped to `block_size` at construction. When the horizon is longer, [rawForecast](<../../timestral/utils/engine.cpp#L180-L426>) loops over blocks:

1. Forward one block with the current context; call `decodePages` when paged, otherwise `predictBlock`.
2. For blocks with more than 1 patch, do an advancing refine: the output frame of each forecast patch advances together with the generated content; the first patch reuses the frame of the last context patch, and subsequent patches fold the median according to the state after a Welford prefix merge and inverse-normalize the remaining quantiles. The prefix merge uses the Hillis-Steele scheme, preserving the pairwise merge association order of upstream.
3. Concatenate the target row's median to the end of the context as the input of the next block, marked as observed; a median whose absolute value is at least 1e30 or is non-finite is fed back as 0. Known covariate rows are fed back with their true future values, past covariate rows with blanks.
4. When the context exceeds `maxContext` the window shifts left; at this point the page-table grid is invalidated, the page table is released, and the remaining blocks are fully recomputed.

Chunked autoregression means the error of a long horizon accumulates. For the real 5M case of 1030 steps in the records, the maximum absolute error against upstream is 0.310, and the reference-output RMS is about 854, far larger than for short cases; one cannot claim from this that long and short horizons have comparable accuracy.

## Quantile Post-Processing

After the block loop finishes, [Engine::computePrediction](<../../timestral/utils/engine.cpp#L497-L628>) does three steps in order; the algorithms are in [postprocess.cpp](<../../timestral/utils/postprocess.cpp>):

- Range clipping: all outputs of each target trajectory are clamped into the range "the minimum and maximum of the initial context observations, each expanded outward by 1e4 × the last scale". The threshold 1e4 and the feedback threshold 1e30 are fixed post-processing semantics, not exposed as options.
- Symmetric averaging: when `symmetricAveraging` is true, negate the input and do a complete prediction once more; the mirror result's quantiles are flipped and negated, then averaged with the original result, i.e. `q[i] = 0.5 × (q[i] − m[Q−1−i])`. The mirror forward uses an empty page pool and does not go through the cache, so when this is enabled, half of the computation of one request cannot be saved by prefix caching.
- Non-negativity clipping: when `clipNonnegative` is true and all context observations of that channel are non-negative, the output is clipped to be at or above 0.

Any NaN output is reported as an exception; no fake prediction is returned. `Result` stores the 21 quantiles by (channel, step, level); `quantile(step, level)` interpolates linearly between adjacent quantile points and takes the endpoint when out of the grid; `lower` and `upper` give the two ends of the central interval, e.g. level=0.9 takes the 0.05 and 0.95 quantiles; `isObserved` distinguishes original observations from generated values.

## Paged KV Cache

First confirm what can be cached. On the standard path of all-target with no covariates, three conditions hold simultaneously: normalization is causal, the time-axis encoding mask is causal, and the forecast queries do not enter the encoding stream. Therefore the K/V of patch p at every time sublayer depends only on the inputs of the first p patches, and later data cannot change it. Variate attention only crosses channels within the same slot and does not need K/V across patches, but it makes a change in one channel affect the higher-level K/V of other channels in the same group, so one page must store the whole group of channels and cannot be hit independently per channel.

![Content-addressed paged prefix cache](<../../assets/infer-pages.svg>)

The definition of a page is in [cache.hpp](<../../timestral/utils/cache.hpp>): a page is the K/V at all time sublayers of a group of V rows for one patch, plus the per-row mask bits, RoPE positions, and the normalization accumulators at the patch tail. By the 5M geometry, the per-row K/V per page is 8 sublayers × 2 × 192 FP32, i.e. 12KiB, not including metadata; the default cap of 2048 pages for a single-channel page pool is at most about 24MiB; pages are allocated on demand, and actual usage grows with the number of pages used. These are numbers computed from the geometry; no process memory was actually measured.

The page pool is built separately for the actual number of channels V, with the capacity cap `cachePages`; a page's K/V buffer is allocated on first use and reused in place after reclamation. A page's identity is determined by its content, following vLLM's prefix caching:

$$
\text{key}_i=\text{FNV-1a}\left(\text{seed},\ \text{key}_{i-1},\ \text{observation bits},\ \text{value bit patterns},\ \text{mask bits},\ \text{position},\ \text{configuration fingerprint}\right)
$$

Values are hashed by their IEEE 754 bit patterns; NaN payloads and signed zero both affect the page key. A separate `inputFingerprint` covering only this page's input is computed with a different seed; lookups compare the page key, parent page key, and input fingerprint together, lowering the risk of a hash collision treating different content as the same page. The configuration fingerprint only covers the fields that determine K/V; quantile post-processing parameters do not participate; see [modelFingerprint](<../../timestral/utils/cache.cpp>).

A direct consequence of content addressing is that the cache is not isolated by source: two sources whose prefix content is identical hit the same physical pages, and the reference count increments by one. Each `streamId` holds only one view of page numbers. When the page pool is full, only the least-recently-used page with reference count 0 is evicted; when there is no page to evict, this request falls back to a full recomputation and reports 0-page statistics, without preempting a request in execution. An anonymous request's view is returned immediately after execution; the page keys are left for LRU reclamation. Any change to the normalization statistics, mask, position, or input values changes the page key; old pages naturally stop hitting and no explicit invalidation notification is needed.

## Frozen Prefix and Decode

The paged path splits one prediction into two stages, corresponding to prefill/decode of a language model:

- prefill: [appendPages](<../../timestral/utils/transformer.cpp#L997-L1236>) encodes the whole patches after the page table that are not yet in pages into the page pool; a trailing run of consecutive misses is encoded in one shot as a whole-segment prefill and lands in pages, while a hit prefix is reused directly. The BLAS backend enables `optimizedPrefill`: a pure prefill ends right after the K/V of the last time sublayer is written to the page, without computing the subsequent MLP and quantile head.
- decode: `decodePages` continues normalization of the not-yet-paged tail from the accumulators of the page table's last page, reads the prefix pages' K/V as the keys of time attention, and computes hidden states only for the tail and the query slots; the output frame takes the accumulators at the end of the last patch.

The correctness criterion is consistency with a full recomputation under the same mask and normalization semantics, with the tolerance max(1e-6, 1e-5 × output RMS). The progress records contain a set of historical measurements (default build, real 5M, interval 0.01 s): alignment and missing-data case max_abs=0; appending 1000 points step by step, max_abs=0, hit rate rising from 0.5 to about 0.90; a second request with an identical prefix computed 0 new pages, total time 25.52 ms, versus 67.57 ms for the first. The streamTest.cpp that produced these numbers is no longer in the current tests directory, and the numbers were not re-measured on the current code.

Multiple requests in the same round with exactly identical page tables go through `decodePagesShared`: the group shares one V-row page table, each request reads its corresponding channel's prefix K/V by `row % V`, the tails are encoded independently per request, and the shared path is read-only and does not append pages. The model layer validates that the prefix observations and values of all requests are pointwise identical, otherwise it rejects.

## Multivariate and Covariates

The channels of the same request form one joint-modeling group; within the group, information is exchanged via variate attention; different requests are isolated by `groupIds`, and batching never mixes channels of different devices into one group. `maxVariables` defaults to 8, a service protection value; the records note that the upstream configuration has no such hard limit. The output is still the marginal quantiles per channel per time step, not a joint distribution.

Internally the engine follows the upstream row roles: Target=0, PastCov=1, KnownCov=2; one role per row rather than per-point roles. Past covariates have history only; known covariates provide values for context plus horizon, are normalized by whole-window statistics, and are bidirectionally visible in time during the encoding. The public entry point does not generate past rows; the engine's past/known rows are mainly used to reproduce the upstream covariate comparisons.

In the public interface, when the observation tails of the channels are not aligned, it automatically segments and completes by observation availability; the implementation is in [condition.cpp](<../../timestral/utils/condition.cpp>). For example, three channels with history lengths 3, 7, 4, forecast to 10: first treat only A as target to forecast position 4; then treat A and C as targets and B as known to forecast 5 through 7; finally forecast 8 through 10 jointly for all three. Each segment calls the native short-horizon forecast; the context of the next segment is made of real readings and already-generated medians; real observations are never overwritten by model outputs. Segmentation changes the patch left-padding and the normalization start; it is an explicit inference strategy, not equivalent to the model natively supporting per-point roles.

The conditional path does not reuse historical KV across requests. The reason was already mentioned in [model architecture](<architecture.md>): known rows are bidirectionally visible and use whole-window statistics, and the target reads known across channels at the same slot, so the historical K/V of later layers would depend on future known values. All channels having observations is only a candidate condition, not enough to prove the hidden states are reusable.

## Experiment: Frozen Prefix and FIFO Sliding Window

![Frozen prefix and FIFO sliding window](<../../assets/infer-swa.svg>)

The [streaming example implementation](<../../example/tui/streaming.cpp>) provides an approximate path enabled only at the example layer, to answer "after the window is full, can we advance indefinitely with bounded KV storage". Whole patches of the common history are written into persistent pages along the all-target causal path; when the page table reaches capacity, the oldest page is released before appending new pages; `positionOffset` lets the retained pages keep their original absolute positions, not renumbering old KV; the normalization accumulators keep accumulating from the stream start and do not roll back when old pages are evicted. At forecast time, `predictFrozen` reads these frozen pages and only recomputes the auxiliary channels and the forecast tail.

It is not equivalent to a model that "recomputes the most recent samples in the window each round": the retained pages carry the context information from when they were built and are not recomputed after the window slides. It is not forcing a bidirectional encoder into a decoder: the native known-future path has bidirectional encoding and whole-window statistics, and freezing its historical dependencies is a deliberate approximation. The default public service does not go through this path. For the interface, configuration keys, and per-round output fields, see [the streaming UI](<../../example/tui/streamingTui.cpp>) and [the example description](<../../example/README.md>).

The scanning tool (a local tool `example-tools/swaBench.cpp`, not checked in) advanced the common history to 131072 points across six windows of 128, 512, 1k, 2k, 4k, and 8k, recomputing the 512-step forecast of the first channel once at each context of about a 1.2× step size, and compared it against a native full recomputation of the same window. Measurement conditions: real 5M, two correlated Fourier channels, 100 Hz, amplitude noise 0.03, horizon=512 (16 patches), 1 patch of feedback per position, symmetric averaging and non-negativity clipping disabled. K is the frozen common-history KV with FIFO eviction of the oldest pages; N is the native recomputation segmented by observation layout; RMSE is computed only over unobserved forecast positions, compared against the noise-free reference.

| Window | K RMSE @128k | N RMSE @128k | native full frame ms | cached full frame ms | factor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.484568 | 0.290793 | 7.88 | 6.09 | 1.3 |
| 256 | 0.314858 | 0.433044 | 7.56 | 6.24 | 1.2 |
| 512 | 0.321666 | 0.152261 | 10.35 | 6.44 | 1.6 |
| 1k | 0.194051 | 0.077387 | 14.39 | 6.94 | 2.1 |
| 2k | 0.112031 | 0.055461 | 25.11 | 7.70 | 3.3 |
| 4k | 0.083611 | 0.067024 | 49.64 | 9.45 | 5.3 |
| 8k | 0.069494 | 0.058639 | 123.95 | 13.15 | 9.4 |

![SWA window scan](<../../assets/swa-rmse.png>)

The RMSE columns and curves come from an existing Eigen FP32 four-thread scan, two-channel signal, 1 patch of feedback per position. The latency columns are a separate re-measurement under normal production conditions, not part of that scan: single variable, BLAS AVX2 single-thread, horizon=512 (16 patches, each patch forecast 32 points), one frame completing the whole 512-step forecast. The native full frame is `predictBlock`, no cache, re-encoding the whole window every frame; the cached full frame is `decodePages`, the window prefix is already resident in the paged KV, and only this frame's decoding is timed; the prefix `appendPages` is not inside the timing. Under the same accounting, a single complete forward and a prefill of the same length are of the same order: 8k window native full frame (8192 context plus 512 forecast) 123.95 ms, versus 104.15 ms for an 8k single-variable prefill on the same machine state, only about 19% more. The two paths are numerically consistent at the same frame granularity (8k window RMSE 0.0696039 vs 0.0696051). The median of 3 runs is taken, with one warm-up before each timing run, excluding model loading. Hardware is AMD Ryzen 7 7735H, MSVC Release; re-measurement is needed after changing machines or thread counts, and the power-supply state affects absolute values (with the same 8k two-channel prefill accounting, one collection in benchmark.md was 106.83 ms, and this re-test was 233.4 ms); across collection batches only relative factors and orders of magnitude are comparable. The raw data (RMSE table, full-frame latency table, no-sliding control) is kept in the local workspace.

The no-sliding control line (an 8192 window advanced only to that point, K=N):

| context | 128 | 256 | 512 | 1k | 2k | 4k | 8k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.9459 | 0.2154 | 0.0913 | 0.0733 | 0.0538 | 0.0494 | 0.0358 |

Observations:

- The gray control line is the no-sliding segment of an 8192 window advanced only to 8192; K and N agree within floating-point reduction error. It shows the pure capacity effect: about 0.95 at 128 points, about 0.21 at 256 points, about 0.094 at 320 points, a rise to about 0.11 near 672 points, then dropping again to about 0.036 at 8192 points — not a monotone square-root curve. The control line explicitly contains the window starts of 512, 1024, 2048, 4096, and 8192; each colored K curve joins this line from its own start before entering the eviction region.
- Under the 512 window, K has RMSE of about 0.09 at context=512, rises to 0.14–0.25 right after eviction starts, then fluctuates between 0.25 and 0.33 until 128k; the N of the same window fluctuates between 0.04 and 0.21.
- On this signal the error does not explode linearly with the context: after sliding, K enters a plateau, and at 128k it is still of the same order as the plateau at 2k–8k contexts. The plateau must not be read as "error does not accumulate"; it depends on signal stationarity and on the old pages still carrying useful information; after overlaying a 5×sin(0.03t) low-frequency drift on the 512-window signal, K rises from 0.046 to 0.367, enough to rule out any universal guarantee.
- The larger the window, the closer K is to N, but cached decoding is also slower: the cached full frame rises slowly from 6 ms at the 128 window to 13 ms at the 8k window, while the native full frame rises from 8 ms to 124 ms, about a 9.4× factor at the 8k window; the 512 window is 6.4 ms vs 10.4 ms. What is saved is mainly the repeated prefix encoding; the cost of reading attention within the window is still paid, and the gap between the two is small for small windows.
- The model's context_length is 32768 sample points; that does not mean the stream can only go to 32k. Here 128k is the stream length; each forward still only sees the window. A feasible long stream does not mean the model gains an infinite receptive field, nor that the information in the old pages is still correct.

Conclusion: SWA suits an explicitly chosen quality/latency trade-off; it cannot replace exact recomputation by default. Use the native path when default accuracy is needed; when a bounded budget is needed, one can use BLAS single-thread or limit Eigen threads with `threads`, but the backend and thread count do not change the approximation error above. Keeping the first token's KV as an attention sink has been implemented as an experimental interface and scanned (see the local sliding-window records): on stationary small windows it can partially recover distant phase anchors, with no measurable extra decoding cost, but it cannot stop the degradation of the frozen cache under statistical drift; it is not merged into the default path, and the interface remains experimental.

## Experiment: Continuous-Time RoPE

Off by default; set it before constructing the instance:

~~~cpp
settings.continuousRope = true;
settings.ropeReferenceInterval = 0.01; // reference sampling interval, in seconds
~~~

When `ropeReferenceInterval` is 0, the sampling interval resolved for this request is used; fixing it to a positive value can keep the same physical time scale across different sampling rhythms, and the reference should be changed in sync when changing time units.

Each patch uses the average of its sampling times as the time center; the coordinate is `(center_p−center_0)/(patchSize×referenceInterval)`; the model then anchors on the first observed patch of the same group; Unix seconds must not be used directly as the rotation position. Left padding and future sampling times are extrapolated by the request interval; the continuous coordinates are passed into encoding, decoding, autoregressive feedback, and shared-page validation; the KV fingerprint includes the position pattern, reference interval, and real-valued positions, so old-position K/V is not misused when time changes.

On a regular grid, when the reference interval equals the sampling interval, it degenerates exactly to discrete positions; the related tests require the forecast to match bit-for-bit. Non-uniform times only prove that the computation is entered; no accuracy improvement is promised on that basis. In a sample signal of the real 5M generated by real time, the MAE of the original discrete input is 0.139345, continuous RoPE is 0.158215, and after explicit linear resampling it is 0.0924159; continuous RoPE is worse on this sample; the experiment does not use observations after the forecast start for resampling; see the local acceptance records for the reproduction method and RMSE.

Experimental limitations:

- only the positional encoding between patches changes; a patch still has only one position, and the fixed input projection cannot represent all the different sampling intervals inside it
- the output still uses sampleInterval to build a regular grid; queries at arbitrary time points are not provided, and no resampling is done automatically
- multiple channels still share sampling positions within one standard forward; the interleaved-observation path first aligns to a common grid, which is uniform, so on the condition-completion path continuous mode is bit-for-bit identical to discrete
- continuous mode enables paging only for whole-patch history; non-whole-patch history takes the full forward; the default discrete cache behavior is unchanged
- when timestamp precision is not enough to represent the reference interval, it is explicitly rejected; one can shift the time origin first; unresolvable time differences are not silently accepted
- the existing weights were trained on discrete patches; this is an experimental inference path and does not claim the model natively supports general irregular time series

Related experiments on removing empty time columns, normalizing the time channel, and whole-frame decoding were not kept as available options; none showed a usable quality improvement.

## Trade-offs and Lessons

The correctness boundary of the cache is harder to pin down than the cache itself. The page key hashes in the input values, mask, position, and configuration; it can only guarantee "the same input gives the same page"; the point that "historical pages are unaffected by later inputs" must be guaranteed by the model's dependency structure, which a hash cannot prove. The standard path satisfies this condition, the conditional path does not, so the former is paged and the latter is fully recomputed; the dividing line is set by the mask and normalization, not by engineering convenience.

The second trade-off is to fall back to the full recomputation rather than preempt. When the page pool is full, the grid conditions are not met, or shared decoding fails, the request always falls back to per-request full recomputation; the result is still correct, just slower. The scheduler can therefore stay simple: a single worker thread, FCFS, whole-round execution, with no per-token intermediate state that needs to be migrated.

Finally, the sliding-window experiment shows that "the train-serve inconsistency costs very little" needs to be measured signal by signal; being close to the baseline on a stationary periodic signal does not imply the same on a drifting signal. Every approximate path is kept as an explicit experiment and reports both quality and time.
