# Historical Long Forecast and RoPE Parallel Experiments

The following preserves diagnostics superseded by the short-forecast protocol; they are not current default usage or performance recommendations. Cross-future-block RoPE Parallel is deprecated; current 1k backend comparison and 0-to-2k block AR measurements are in the [current benchmark](<benchmark.md>). Old data and failure records are not deleted, and this round's short-forecast comparison passing is not extrapolated as long-forecast being fixed.

## Long Forecast Quality Limitations

This single-variable signal is the first channel, same as the first channel in the [streaming config](<../../example/tui/streaming.json>): sin(4.908738521t) + 0.3sin(9.817477042t + 0.4), input overlaid with noise of standard deviation 0.03. True history is indices 0 to 16383, prediction and scoring at indices 16384 to 32767, sampling interval 0.01 seconds.

2026-10-26 re-predicted using current BLAS single-thread, native 32-patch block, symmetric averaging and nonnegative clipping disabled on both sides. Taking prefixes of the same 16k output, AR and RoPE Parallel both have RMSE 0.073598 for the first 512 points, 0.593199 for the first 1024 points, and 0.759291 and 0.936087 for the full span respectively. A separate 512-point native AR request is bit-identical to the first 512 points of the long forecast. Prefix statistics are not independent predictions requested at different lengths.

The RMSE of always predicting 0 for this future window is 0.738241, so neither 16k forecast's absolute quality is better than the zero baseline; the 23.28% difference between the two cannot be taken as evidence of usable quality. AR full-span output max absolute value 1.2453, Parallel 1.3218, no amplitude explosion. AR's offset segment 512-to-1023 RMSE has already reached 0.835675, still within the first native 1024-point block, no inter-block feedback has occurred yet, so the full degradation cannot be attributed to AR repeated feedback. The specific model-internal cause has not been located. The fixed-version official Python re-run this round also reproduces the phenomenon: raw network first 1024-point RMSE is 0.593843, after refine 0.593202; disabling attention split and using the full dual-stream window both give the same raw output, forcing math SDPA also does not eliminate the degradation. There is no evidence to label this as a found C++ implementation error.

Further multi-start and precision diagnostics (local records) found no confirmable C++ porting error: FP64 comparison did not restore first-block quality. The overlapping strategy that still generates 32 patches per round but only submits the first 512 points dropped to 0.3112 at one start point, but worsened from 0.7576 to 0.8119 at another, so it was not merged as a fix. An independent baseline estimating period from history alone reaches about 0.0026, showing the signal contains exploitable regularity, but that baseline is not Timestral output. The native 5M long-forecast quality issue remains unfixed.

Keeping the same official reference output and per-element error budget, numeric corrections are also insufficient to close the quality gap. Diagnostic data (local records) give an optimistic mathematical lower bound computed with future ground truth, not a usable forecast: within the interval where each element deviates from the reference by at most 1e-4 + 1e-3 x |reference|, the first 1024-point RMSE is still at least 0.592962, and the full 16k at least 0.759024. This bound applies only to the fixed reference output and does not exclude different weights or upstream semantic corrections with justification.

The main README's sliding-window chart is another old experiment: each new true observation triggers a 512-point prediction, jointly dual-channel but only scoring the first channel, and the [streaming forecast implementation](<../../example/tui/streaming.cpp>) still explicitly sets patchBatch=1. Its results are not a quality comparison against the current native 32-patch block, and 128k cumulative observation positions cannot substitute for prediction length. The original chart is retained as an old experiment record; the native block sliding-window comparison has not been re-measured.

## Official Decode Flow

Upstream, one decode block is 32 patches. Timestral-5M's patch_size is 32, so one block is 1024 sampling points. Long forecasting is block-autoregressive: predict a block, extend context with that block's median, predict the next block; when exceeding max_context, truncate the oldest part. The 32 patches within a block are one joint forward pass — this is the source of "parallelism"; between blocks it is still serial and must feed back.

The official forecast also enables symmetric averaging and nonnegative clipping by default. This repository's comparison disables both switches on the AR and RoPE Parallel sides to isolate the "whether to feed back" variable, so this comparison is not the official full default. Upstream full default and native block AR with these two disabled have their RMSE under same input and weights recorded in the verification file (locally retained).

## Multi-Function Long Forecast Curves

The [historical curves](<../../assets/decode-quality-20261026.png>) use six function configurations (locally retained), model still local Timestral-5M. Each function is inferred independently, not a six-channel joint input. Raw per-start cumulative metrics with model, config, program hashes and numeric comparison records are retained in the local workspace.

The six functions are: period-128 single sine, original-benchmark dual harmonic, period-512 slow sine, period-128-and-176 dual frequency, 5-odd-harmonic square wave approximation, and carrier-period-128/envelope-period-2048 amplitude-modulated sine. With sampling point n, phase angle is 2pi*n/P; dual harmonic is sin(2pi*n/128)+0.3sin(4pi*n/128+0.4), square wave approximation is Sum[k in {1,3,5,7,9}] sin(2pi*k*n/256)/k, AM is (1+0.5sin(2pi*n/2048))sin(2pi*n/128). Actual floating-point coefficients follow the JSON.

Sampling interval 0.01 seconds, input noise standard deviation 0.03, no sampling jitter or phase noise. Seed fixed at 42 + function index x 10007 per generator. Start points are 16384, 16416, 16448, 16480; each start reads only the prior 16384 true observations; future ground truth is scored only, not fed into context, mask, or feedback. Each full prediction is 16384 points, then scored cumulatively at the first 1024, 2048, ..., 16384 points, without mixing re-requested predictions at each length on one line.

The AR curve directly uses the fixed upstream Python's TimestralForecaster.forecast. The Parallel curve uses measured values from the C++ cached implementation, with an independent comparison constructed using the same fixed upstream's network, masks, scaling, and refine primitives: 16 groups of full 32-patch queries share a frozen history, only translating future query positions, no cross-group feedback. Both sides disable symmetric_averaging and clip_nonnegative, consistent with the existing speed comparison, not the official full default settings. No modifications to the inference library or weights.

For function s and cumulative length H, the curve is sqrt(Sum[4 starts, first H points](predicted median - noise-free truth)^2 / (4H)), not the average of four RMSEs. Shading is the min/max of each start's own cumulative RMSE. The blue dashed line is sqrt(Sum truth^2 / (4H)), i.e., the baseline of always predicting 0 in the same window; individual functions' baselines shift slightly with cumulative length, so a single constant for the full 16k cannot substitute all lengths. Being above that line only means not better than zero prediction, not a universal confidence bound.

New function C++/Python point-by-point comparisons have samples exceeding atol 1e-4 / rtol 1e-3, recorded in metadata; original budgets unchanged, failed samples not deleted. Numeric acceptance still fails by default; only explicit --allow-recorded-deviations permits exporting a quality chart with warnings. AR uses the official output rather than assuming C++ is identical; Parallel keeps actual C++ output, not replaced by comparison results. First-block official AR and independent Parallel reference are identical, C++ two paths' first blocks are also identical, but this does not constitute an accuracy guarantee, nor does it guarantee monotonic worsening at later lengths.

Reproduction entry points are the collection program, official comparison and cumulative check, and plotting scripts (local tools, not committed). Local point-by-point observation and prediction files are development review inputs; the published CSV retains all 6x4x16=384 cumulative metrics, no cross-function aggregate RMSE.

## Single-Patch AR Supplemental Experiment

At user request, explicitly set 1 patch per forward pass as a comparison with 32-patch block AR, without modifying defaults or the AR/Parallel charts above. Still using the same 5M weights, six functions, and 16k history, start points 16384 and 16448. Actually ran 1k and 16k requests separately; for each function, compute SSE at each start first, then merge the two starts to get RMSE, no cross-function quality metric synthesis.

The 16k request and per-length metrics, actual 1k request and program/data hashes, KV counts, and official re-verification are retained in the local workspace separately. The following comparison disables symmetric averaging and nonnegative clipping, consistent with the main curve protocol.

Also provided [full 1k-to-16k log curves](<../../assets/patch-ar-curves-20261026.png>) and near-range 1k-to-4k linear zoom (locally retained); purple is 1-patch, orange is 32-patch, blue dashed is the same-window zero-prediction baseline, shading is the two starts' min/max RMSE. Panel ranges differ, so direct visual slope comparison across functions is not appropriate.

| Function | 32-patch first 1k RMSE | 1-patch first 1k RMSE | 32-patch full 16k RMSE | 1-patch full 16k RMSE |
| --- | ---: | ---: | ---: | ---: |
| Single sine | 0.6159 | 0.1128 | 1.0119 | 1.2311 |
| Dual harmonic | 0.6267 | 0.0999 | 0.7507 | 5066.7015 |
| Slow sine | 0.5935 | 0.2506 | 1.2376 | 5529.5057 |
| Dual-frequency mix | 0.6004 | 0.0955 | 0.9729 | 2572.2193 |
| Square wave approx | 0.5441 | 0.0741 | 1.1209 | 618.7293 |
| AM sine | 0.7512 | 0.2321 | 0.9834 | 2.2968 |

The two-start merged first-1k error decreases for all six functions, but not every start is better: slow sine at start 16448 has 32-patch at 0.1075, 1-patch actually worse at 0.2037. Full 16k is worse for all six functions. The actual 1k request is bit-identical to the first 1k of its own 16k output; this prefix consistency cannot be interpreted as the two step sizes being equivalent.

Local BLAS FP32 single-thread, light daily load, each recipe warmed up once then timed 3 times taking the median, alternating order; official Python verification does not run concurrently with timing. Excludes initial observation prefill, includes private page pool construction/copy, feedback encoding, refine, and final post-processing. Aggregating the two starts' timing medians per function: actual 1k request, 1-patch about 220 to 236 ms, 32-patch about 19.8 to 20.6 ms; actual 16k request, 1-patch about 5.34 to 5.61 s, 32-patch about 0.98 to 1.05 s. New timings do not replace the original speed charts.

KV is measured through a read-only forwarding wrapper counting actual appendPages/decodePages calls, page table length, and page pool identity; model computation unchanged. Per-page K/V is 12288 bytes; the following counts only CPU-side K/V float payload, including the original 512-page true prefix and request-private pool, excluding page metadata, temporary query/activations, or process RSS.

| Actual prediction length | 1-patch decode/feedback append count | 32-patch decode/feedback append count | 1-patch KV pre-alloc payload | 32-patch KV pre-alloc payload |
| --- | ---: | ---: | ---: | ---: |
| 1k | 32/31 | 1/0 | 12.375 MiB | 6 MiB |
| 16k | 512/511 | 16/15 | 18 MiB | 18 MiB |

In the 16k request, both private pools pre-allocate 1024 pages; 1-patch peak effective pages 1023, 32-patch 992. In the 1k request, 32-patch decodes only one block, no private feedback pool needed; 1-patch allocates a 544-page private pool and copies the history prefix, so the current implementation actually uses more KV. Both step sizes leave the original observation KV fingerprint and page table unchanged after each prediction. This memory result is a measurement of the current paged implementation, not a theoretical requirement of 1-patch, and certainly not a GPU memory measurement; cumulative page table reference count also does not equal hardware DRAM traffic.

Official Python re-ran the single sine and dual harmonic at one start each with per_shot_horizon_patches=1/32 for the full 16k; short-term improvement and long-range instability are both reproducible, but not all point-by-point values pass the original budget. Single sine 1-patch comparison has 0 points over budget; dual harmonic 1-patch long trajectory has 2480 points over budget, max difference 14776.93, already showing sign disagreement near the clamp — cannot claim point-by-point agreement just because both sides' RMSE are close. Existing 32-patch over-budget points are retained, tolerances unchanged.

Additionally, only for the dual harmonic start 16384, official default post-processing is retained (symmetric_averaging=True, clip_nonnegative=True), changing only patches per round: 1-patch first 1k / full 16k RMSE is 0.044139 / 2615.403147, 32-patch is 0.604940 / 0.818449. This is an extra precision check; its results are not mixed into the above timing and KV tables with post-processing disabled, nor extrapolated as a guarantee for other functions.

This experiment shows that a smaller feedback step can improve near-range prediction on these signals, at a clear latency cost, and does not reduce current KV pre-allocation; it should not be directly used for 16k long extrapolation. 1-patch is an explicit non-default parameter allowed by the official interface; it cannot be deemed illegal just because the training block is 32 patches.

## Decode Protocol

16k history predicting the following 16k, single variable, symmetric averaging and nonnegative clipping disabled, native refine and quantile range processing retained. Initial prefill counted separately, both paths warmed up once, timed three times, alternating order, reporting the median time for the full 16k prediction.

- AR decodes one full 32-patch block per round, 16 rounds total, within-block forward with native causal mask and RoPE positions, feedback is the current round's predicted median. The generation branch uses an independent 1024-page trajectory, not overwriting the 512-page true observation prefix.
- RoPE Parallel treats 16 full blocks as independent query groups, one batched forward. Each group retains the intra-block 32-patch causal structure, native RoPE positions, and refine, only removing inter-block feedback, so it is still whole-block prediction, not per-patch prediction.

Each CSV row records frame_patches and frames corresponding to patch_size; the converter rejects measurements that degenerate to single-patch based on this. A previous version split the parallel path into 512 single-patch queries; that comparison is voided and not mixed with current results.

## Numeric Acceptance

The new long-frame path first passed an independent reference comparison: parallel output compared element-wise with the reference backend's same path, single-frame case compared with normal AR blocks, page table fingerprint unchanged before and after. Frozen tolerances were not relaxed; page read-only checked separately.

The current CPU ordered path and GPU compensated FP32 path have passed the frozen budget for this round's single-variable data. GPU extra 8-variable query, ytensor extra 2-variable query, and device 1 long query still have over-budget points; the old causal long-frame stress is also separately retained. See local current acceptance records for specific scope.
