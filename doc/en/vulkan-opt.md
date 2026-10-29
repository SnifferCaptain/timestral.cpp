# Deep Vulkan Backend Optimization Tree

## Current Model Accuracy Path

The model's common shapes use ordered attention, the RMS reduction, and compensated FP32 nonlinearities, and the shared prefix is no longer copied per query. The G1 projection acceleration is kept; A2 and the other online-reduction kernels remain in use only in standalone benchmarks and out-of-range fallbacks. For the deployment and numeric limits of the current 52-module package see [portability design](<vulkan-portability.md>); for the performance graphs and the long-decode table protocol see [benchmark notes](<benchmark.md>). The historical results in the optimization tree below do not represent the current default dispatch.

The records, logs, and dumps cited in this document all live in the local workspace and are not published with the repository.

## Latest Direction: Intel Portability and GPU Budget (2026-10-25)

The goal has been updated: target 13th- to 15th-generation, smaller-scale Intel iGPUs; no special-tuning for 680M. Decode is no longer accepted against a 1% MFU criterion; instead look at same-task latency, copy, and bandwidth. The earlier 300/30 numbers are kept only as a historical development record, not as an Intel deployment threshold.

The local ggml critical path provided by the user has been read; the remote commit is unknown, and the scope of review and SHA256 for the partial review are in [portability design](<vulkan-portability.md>). AI Max395/RDNA material has been searched, but the external text is still blocked by a non-public-IP check, so no full code review is claimed. No subagent was used and no RDNA-specific kernels were copied.

Timestral::Settings::gpuThreads was added: 0 means unlimited; a non-zero value must be a multiple of 256 and within the uint32 range. It constrains the logical invocations of each serial dispatch window, not a physical EU quota. The fixed local_size, legality, and variant sizes are checked on the host; when the budget is below 512 it may back off to a 512-thread pipeline; when the device reports 8/16 subgroups, no 32/64-lane-only fast kernels are created. The current complete package is 52 modules; the original 21 source/binding/push entries are unchanged.

The original M≤2048 upper bound on residual fusion was a 680M experimental guardrail and has been withdrawn from the default rationale. It is now off by default, and the explicit opt-in experiment no longer retains that large-M upper bound. No low-SLM attention, new GEMM panels, or UMA KV mirror are claimed as implemented. For the full interface, safety boundaries, verification, and uncovered Intel hardware see [new design](<vulkan-portability.md>).

## History: 300/30GFLOP/s and Multi-variable Results (early 2026-10-25)

This section is the adoption status and measurements from before this round's goal adjustment; the current default policy follows the portability design at the top. The 680M prefill 300GF/s and H1 decode 30GF/s targets proposed by the user were not reached, so phase six is not marked complete. No subagent; no changes to the public entry points, weight/Tensor layout; arena/offset push was not restored.

### Measured Results for the Final Default Policy

`Last Complete Matrix` corresponds to the current source; there is also a `structured summary`. 32768 samples per channel, different data within the same group, not independent batches; targets are jointly predicted across all channels. Cold prefill median of three, H1 decode median of 25 after 5 warmups, all tasks serial and not overlapping with the build. What is recorded is the median of this round, not the minimum across all rounds; real-time frequency/power was not measured, and cross-round variation cannot be attributed to the driver.

| Channels in group | 680M prefill ms | 680M H1 ms | prefill GF/s | 680M nominal MFU | 4060 prefill ms | 4060 H1 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.47 | 8.48 | 256.8 | 8.56% | 21.68 | 5.11 |
| 2 | 76.31 | 17.67 | 279.1 | 9.30% | 42.20 | 11.60 |
| 4 | 165.99 | 28.23 | 256.7 | 8.56% | 98.37 | 22.07 |
| 8 | 328.71 | 55.03 | 259.3 | 8.64% | 189.49 | 48.93 |

680M decode is only about 1.84 to 2.37GF/s, i.e. 0.061% to 0.079% nominal MFU, far below 1%. Single-variable prefill is 256.8GF/s, 8.56%; about a 16.8% throughput increase is still needed, i.e. bringing 41.47ms down to 35.50ms. The 2/4/8-channel prefill also did not reach 300GF/s. The 4060 does not estimate its own MFU from the 680M 3TFLOPS denominator.

The earlier `CPU reference window` on the same machine this round shows BLAS encode at 170.85/336.81/683.07/1424.89ms and decode at 4.21/7.85/14.90/36.04ms; this is not a per-item CPU re-measurement paired with the last GPU round. GPU prefill is clearly faster than that reference; decode is still slower. The 8-channel total input is 262144 samples, so the elapsed-time growth cannot be interpreted as a same-workload regression, and there is no total throughput that grows in proportion to the channel count.

### Known Future Covariates

One Target plus the remaining KnownCov, providing real future observations, with total in-group channels of 2/4/8. The current paged append input does not include roles/future; this mode is a fully conditioned forward, median of 5 after 5 warmups, and must not be labeled cached decode. The following takes steady-state timings from the log with metric=full_conditioned_forward; in this mode the decode_ms field of the compatible output is only a legacy field name and does not mean incremental decoding.

| Condition | Total channels | 680M full forward ms |
| --- | ---: | ---: |
| 1 Target + 1 KnownCov | 2 | 294.44 |
| 1 Target + 3 KnownCov | 4 | 644.15 |
| 1 Target + 7 KnownCov | 8 | 1344.05 |

These forwards return normalized quantiles, so the first output 0.50 cannot be directly compared in accuracy against the paged de-normalized 2.38. The necessary FLOPs of the full dual-stream case have not been derived separately, so no MFU is reported for that scenario, and no claim is made that covariates necessarily improve forecast quality.

### Adopted, Rolled-back, and No-Gain Leaves

- G1 fills in the real embedding projections (K,N)=(64,768)/(768,192)/(64,192), still per-K precise, bit-identical to v1 across 12 shapes × bias
- A3 uses an independent causal kernel to trim encoded future key blocks that can be proven entirely empty, without changing the real K/V strides; the KnownCov bidirectional and dual-stream suffix keep the original predicates
- A5 subgroup is enabled when compute Basic/Vote/Arithmetic and the 39300B shared capability are all satisfied. The shader verifies that each subgroup contains only the same query; only size32/64 take the fast path, otherwise the whole workgroup falls back; 64-lane direct reduction, 32-lane shared merge, no hard-coded warp32. The full conditioned forward was also checked against a long-context model
- P1 KV-only prefill stops after the K/V of the last time is ready; the useless trailing attention/gate/O, subsequent variate/MLP, out_norm, and hidden readback are not executed; full forward, trace, and decode are not truncated. The last QKV projection still computes a surplus Q, and that useless Q is not counted in the task output
- G2 is currently a 64-thread, 16-columns/group, merged-weight-load per-K sequential GEMV, used for the full native grid from M1 to 8. The 32-lane parallel-K candidate failed on the M1/N672/K768 operator and on the shared-cache/full-forward budget, was removed, and the budget was not relaxed
- A4, at count1, time/headDim64, with at least 512 keys, splits into 128-key chunks; each slice outputs max/denominator/unnormalized PV, then a stable merge; empty slices do not produce NaN. Passed tail-slice, missing, role, multi-row, and H1 long-prefix model checks, but no claim is made that the whole decode has reached BLAS or 30GF/s
- F1 device-resident residual blocks are used by default only for 128≤M≤2048. The hidden projection, SiLU, output and skip projections and the addition are placed in the same batch, with only the final value read back; small heads show no net gain and the large-M direction is unstable, so it falls back to the original three projections. The switch TIMESTRAL_VULKAN_RESIDENT_RESIDUAL=0 disables only this leaf; 1 also honors the shape guardrail; illegal values are rejected
- Rotary reuses only the inverse frequencies of this call and the completely identical position rows; no cross-request shape cache was added, and consecutive coordinates are still computed with the original std::cos/sin
- [gemmWide](<../../timestral/vulkan/shaders/gemmWide.comp>) is kept for explicit diagnostics only: N576/1024 improve the microbenchmark by only about 5%, M128 and K512 regress, so it is not selected automatically. The 128-thread candidate of [attentionPacked](<../../timestral/vulkan/shaders/attentionPacked.comp>) is correct but slower than subgroup, so it is not selected automatically. [Register PV](<../../timestral/vulkan/shaders/attentionRegister.comp>) remains diagnostics-only

The `same-process attention comparison` median at time1024: v1 12.37ms, A2 4.00ms, causal 2.47ms, subgroup 2.04ms, packed 3.72ms, including record/submit/fence but not transfers, and not to be treated as device-only. The F1 single-variable standalone A/B went from 46.81/46.90ms down to 40.55/41.05ms (`R1 and the explicit diagnostics section`), but the two four-channel pairs point in opposite directions; the complete data is kept in `first pair` and `sole reverse-order recheck`, and 169ms is not taken as a stable gain.

The early PowerShell label concatenation turned the non-R1 parameters into a single label; the log actually had channels=1, and those rows are void. The original log above takes only the R1 and explicit-diagnostics sections. The multi-channel adoption uses checked records after parameter-count validation, and the final table re-checked the actual channels/case; mislabeled rows are not used as 2/4/8-channel results.

### Accuracy and Default Build

`Final policy acceptance`: AMD and 4060 each pass Tiny/Real 2/2, the original 1/2/3 rows are unchanged in per-layer and full-quantile thresholds, and new 63/513-patch H1 cached/full/reference, read-only page hashes, and 129-patch known-future covariates were added. 63 and 513 respectively straddle the lower and upper bounds of the residual fusion. The new siluGrid covers the 65535-group boundary; the 21-module ABI reflection and dual-GPU validation are in `pre-fusion acceptance`. At the end, only the useless temporary output allocations left before the rollback were cleaned up, then the build and Tiny smoke test passed, and the last matrix was collected.

An explicit build with OpenMP, Eigen, ytensor, BLAS, and Vulkan all disabled (pure standard library) actually passes 24/24 (82.56 seconds), not a copy of the historical 25/25. reference/BLAS keep the original residual default implementation; only the internal virtual numeric hooks were extended.

The existing output budget for the 8-channel public forward/predictBlock passes, but the newly added strict per-layer diagnostics have not passed yet. After disabling all newly selected kernels it still fails at the same layer6/index7459; the per-item RMS candidate also did not fix it and was removed. The `--strict-multi` switch of [test](<../../tests/gpuModelTest.cpp>) is kept to reproduce this, and the default 8-row output test is not written as "all 8 rows pass per-layer". This accuracy boundary still needs to be located; the threshold is not relaxed.

### Remaining Mainline

1. E3 first evaluates whether the 16 K/V readbacks can be read in a single batch after one submission; it is still serial, and the 17 prefill submissions cannot be described as one. Must preserve the liveness and poisoned semantics of command-referenced resources on exception; do not do the arena first
2. K1, after defining pool identity, page generation, and eviction/reuse/modify invalidation, implements the device KV mirror and segmented prefix reads, reducing the per-H1 ~12MiB CPU assembly/re-upload and the 16 mergeKv calls; not yet implemented
3. The decode diagnostics show about 2.5 to 3.4ms outside the CPU API and a GPU submission span of about 4.6 to 5.4ms; each is already above the 0.543ms target, the times overlap and cannot be added, and simply cutting two submissions cannot be presented as the 1% MFU plan
4. The 8-channel strict intermediate-value deviation, and real-time frequency/power/video-memory budgets, still need to be addressed. Fetching the ggml text was again refused by the non-public-IP check; no review of a fixed commit was completed and no source was copied

## Plan Before the Device-Resident Residual Projection Block Experiment (continued 2026-10-24)

After the supplemental test, the 680M single-variable prefill is 45.7700 to 51.8650ms, only 205.32 to 232.66GF/s, not reaching 300; H1 decode is 9.2596 to 9.3902ms, only 1.73 to 1.76GF/s. The data has been written to disk as `final matrix`; the background output lost after a host restart is not reused.

The next step tries only one boundary that already has attribution support: patch/head residual currently does three independent projects, each with an upload/download, and SiLU and the addition run on the CPU; the diagnostics show about 12ms outside the CPU API. Turn the internal Transformer::residual into a protected numeric hook with a default implementation; the default order of reference/BLAS etc. is unchanged; Vulkan completes the hidden projection, SiLU, output projection, skip projection, and addition in one batch, reading back only the final value. Other GpuTransformer APIs are not enabled by default; Vulkan controls it independently; no arena/offset push rework is required.

Keep the per-K, bias, and addition order of the original three projections. Add a correctly 2D-gridded siluGrid, avoiding misuse of the old X-only validation kernel after an 8-channel full forward exceeds 65535 groups. GPU exp and CPU libm differ, so it must first pass the existing operator/per-layer/full-quantile/cache budgets; if it fails, the default enabling is rolled back without relaxing the budget. After passing, A/B with the independent TIMESTRAL_VULKAN_RESIDENT_RESIDUAL=0/1, without changing other optimizations at the same time. If kept, the public header/weight layout is unchanged, but the internal virtual-hook change requires one default std-only regression.

## 300/30GFLOP/s and Multi-variable Push (evening of 2026-10-24)

The user asked to keep exploring 680M prefill 10% (300GF/s) and task decode 1% (30GF/s), while also filling in 2/4/8-channel performance, without using a subagent. The targets are not guarantees; the end-to-end measured gap must be reported, and pure operator peak cannot substitute for it. The old Goal restore interface still refuses; proceed under this round's direct authorization, without claiming that an automatic Goal has been restored.

### Protocol and Required Task Volume

Extend the existing backendCompare: CHANNELS=1/2/4/8, 32768 samples per channel, the same groupId but different channel data, not an independent sample batch. targets are jointly predicted across all channels; per the original cold prefill median of 3, H=1 warmup 5 / timed 25; known is one Target plus R-1 KnownCov and provides real future observations; the current public paged input does not include roles/future, so it cannot pass as a cache hit; the fully conditioned forward is reported separately (warmup 5 / timed 5). Performance is not a forecast-quality improvement; numeric verification is done separately.

From this round, the necessary KV-only task FLOPs are fixed, and the useless trailing attention/MLP on the Vulkan side is not counted as output. The 32k single-Target prefill is 10.648682496G, and the H=1 decode is 0.016289280G, requiring ≤35.496ms and ≤0.543ms respectively to reach the above targets. The earlier 12.260475G was the main arithmetic of the un-trimmed GPU execution graph; it is kept as history but must not inflate the post-trim MFU. The multi-variable time part scales with R and the variate part with R²; the single-variable FLOPs cannot simply be multiplied by R. The known full dual-stream mask has no derived required arithmetic, so no fictional MFU is reported.

### This Batch's Implementation Order and Entry Gates

1. Serially save and measure the cached 1/2/4/8-channel versions and the same-window BLAS, then measure the KnownCov conditioned forward; the performance window must not overlap with this session's compilation
2. G1 adds the real embedding shapes N768/K64, N192/K768, and N192/K64, staying per-K precise; dispatch only after new correctness shapes are added
3. Re-check the causal Empty blocks: trim only in the time encoding region, where all queries in a block are before encoded and not the KnownCov bidirectional role, keeping the real keyCount row stride. This round's verdict is under the cached readback and timestamps; decode does not carry this prefill branch; use an independent shader if needed to avoid register/branch regression
4. The GPU KV-only prefill stops after the K/V of the last time axis is complete; it does not execute attention, gate/O, subsequent variate, MLP, out_norm/hidden readback that have no effect on page content; full forward/trace/decode are not trimmed. The first version still computes the full QKV but does not count the useless Q in task FLOPs; an independent KV projection is considered only afterwards
5. G2 uses 32×4 threads, each 32-lane group reducing one column of K dot products, four columns per group, dispatched per row, aimed at M≤8, with weights loaded contiguous along K. This changes the reduction order; verdict by the original atol/rtol and hidden relL2/full-quantile budgets; byte-identity is not forced and the budget is not relaxed
6. A4 splits a single-query long KV into 128-key chunks, multiple workgroups per head/row, 64 threads jointly load K, PV stays contiguous along the lane dimension, output per-split max/denominator/unnormalized PV, then an independent FP32 merge. Do not assume warp width, and do not average split results as attention; masked/empty splits must have a stable zero-denominator handling. Dispatch only at count1, time/headDim64, and long KV
7. Add a device-capability-gated subgroup reduction candidate: query compute Basic/Vote/Arithmetic; the shader first checks whether each subgroup contains only the same query and is size32/64; if not, the whole workgroup uses the original reduction. When satisfied, use subgroupMax/Add plus a little shared merge, reducing the workgroup barrier per 64 keys; do not depend on lane order or hard-code warp32
8. Compare a gemmWide with 64×64 output tile, 256 threads, 4×4 registers per thread, K block 16, shared 8256B, still per-K precise. Enable only when the resident microbenchmark and end-to-end both benefit on the main model's large-M shapes; keep 32×32 for small grids like M128. Rotary only does exact reuse of the same-call inverse frequencies and identical position rows; no cross-request cache is added
9. For attention still being the top device-side cost, re-evaluate a 128-thread 8-query×64-key block: four QK accumulators and four PV dimensions per thread, 16-lane shared reduction, 37632B shared; avoid the one-thread-one-score serial chain, still using the original mask and online softmax. The reduction order changes, so the budget must be frozen with a model comparison before dispatch
10. For every new numeric path, a minimal oracle/model comparison comes before performance, and only comparisons with directional disputes are repeated once. Record the single/multi-variable and KnownCov actual modes and gaps; do not write "not reached" as "reached"

### ggml Reference Acquisition Status

This round again tried raw.githubusercontent.com fixed modules and api.github.com commit info; both were refused by the non-public-IP check. The search found [fixed mirror mul_mm.comp](<https://tangled.org/okami.mom/llama-cpp-bonsai/blob/01fd9521c92e7882a3fa1083bb933e3bd9305bef/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp>) and [backend notes](<https://deepwiki.com/ggml-org/ggml/3.3-vulkan-backend>); both domains likewise refused; the text was not read, no implementation was copied, and no ggml microkernel review is claimed complete. The above candidates are implementations based on the locally read shaders and GPU access/reduction principles; once the source is obtained later, the tiles and device dispatch will be checked; network waiting must not block other independent work.

## 680M Special Attribution: Low MFU and a Wrong Readback Memory Choice (2026-10-24)

The user pointed out that 680M is about 3TFLOPS yet end-to-end only uses a few percentage points. This judgment holds; the previous round only proved the local shader got faster and did not complete the execution-time attribution; lumping submission and transfer together as the main bottleneck was also unevidenced. This section supersedes those guesses, and the new single-round best values are not treated as optimization completion.

### MFU Recomputed for the Current Execution Scope

Per the user-specified nominal FP32 peak 3000GF/s, do not use the CPU 134.4GF/s denominator to explain GPU efficiency, and do not claim an actual frequency was measured. Single Target, all observations, 1024 patches, D192, FFN512, 8 time axes and 2 variate axes:

- Transformer projections are 8.606711808GFLOPs; the three patch residual projections are 0.427819008GFLOPs
- The effective time/variate visible attention is 3.225944064GFLOPs; total main effective arithmetic 12.260474880GFLOPs
- The H=1 full-prefix decode includes the three head projections; main effective arithmetic 0.016289280GFLOPs; the encode arithmetic must not be reused
- RMS, exp, RoPE, element-wise, cache management, copy, and memory initialization do not count FLOPs, but their time counts toward end-to-end. Vulkan also does partial wasted work on PV where the mask is zero; the actually executed instruction count is not the effective model FLOPs

Therefore the earlier 143.5030ms encode corresponds to only 85.44GF/s, a nominal MFU of about 2.85%. 3TFLOPS is a fixed assumed denominator, not this round's GPU frequency or a hardware busy counter. The ideal pure-arithmetic floor of the current graph is about 4.09ms, not a committable end-to-end latency; an implementation in the tens to hundreds of milliseconds obviously still has a large gap. The old full-forward/J=4 formula cannot be applied directly to KV-only prefill or H=1 decode.

### Attribution Tool and Timing Boundaries

A new vulkanProfile attribution tool (local tool, not in the repository) is added, using outer exclusive-API timing around GpuExecutor, inner submit/wait/recycle timing inside the executor, and Vulkan query timestamps per kernel and copy. Model diagnostics are off by default; enabled only in the explicit tool; no changes to shaders, ABI, sync barriers, or submission boundaries. The queries use the timestampValidBits of the actual compute queue, handle counter wraparound, report dropped_events when the pool is undersized, and do not fabricate zero durations.

The tool uses the same 32k input, cold page-pool encode three times, and five timed diagnostics after five decode warmups, saving the full statistics of the wall-median run; it does not take the per-column medians separately and stitch them into one run. GPU events are timestamped in the same phase as compute or transfer; the submission span is top to bottom; the event sum, GPU span, and CPU fence wait overlap and cannot be added. Timestamps may change scheduling and the local GPU duration fluctuates noticeably; the diagnostic time does not replace the no-diagnostic backendCompare 5/25 baseline. The 11.832656G in the first two logs counted only the Transformer graph; the final formula adds the patch/head projections, giving the above 12.260475G/0.016289G.

### Proven Encode Host Bottleneck

In `pre-subdivision record`, one encode wall was 176.3228ms with a GPU submission span of 85.96935ms; CPU allocation 0.0342ms, dispatch recording 0.5251ms, recycle 0.2878ms, none dominant. The GPU upload+download copy is about 1.92ms, so "slow CPU-to-GPU transfer" cannot explain the whole gap.

`post-subdivision record` confirms the following data, cross-checked against `corrected record`:

| Same diagnostic item | Before fix | After cached readback |
| --- | ---: | ---: |
| encode wall ms | 228.0989 | 143.5917 |
| GPU submission span ms | 120.22921 | 120.94282 |
| CPU map+read+unmap ms | 77.5258 | 1.5239 |
| Read-back bytes | 17.25MiB | 17.25MiB |
| Actual readback memory flags | 6, Visible+Coherent | 14, Visible+Coherent+Cached |

The 680M report's type1 is flags6/heap1, type3 is flags14/heap1. The original implementation always chose the first host-visible coherent type, but used the staging suited for CPU writes for CPU reads as well. CPU reads without HOST_CACHED are very slow; unified physical memory does not mean all Vulkan memory mappings are equally suited for CPU access. This is not an explanation via discrete-GPU PCIe transfer, and all the less a device GEMM peak shortfall.

Fixed: the independent readback reuse pool now prefers HOST_VISIBLE|HOST_COHERENT|HOST_CACHED and is not mixed with upload staging; when there is no cached coherent type it falls back to the original coherent type, without introducing unhandled flush/invalidate semantics. Readback still waits on the fence and transfer→host visibility, all data is genuinely read back, and the public API and accuracy are unchanged.

### End-to-end A/B Without Diagnostics

All tasks are serial, not overlapping with this session's build or other GPU measurements. `First round`: old version 228.5427/10.8646ms, BLAS 181.9911/4.7145ms, clearly proving the old version was slower than BLAS at the time. Then the saved old program and the cached version were run in two orders; the protocol is still cold encode median of 3, decode warmup 5 / timed 25:

| Round | Old encode/decode ms | Cached encode/decode ms | BLAS encode/decode ms |
| --- | --- | --- | --- |
| Cached first, old after | 185.1625 / 16.9553 | 88.0076 / 11.9052 | 182.7768 / 4.8753 |
| Old first, cached after | 196.6105 / 11.9873 | 85.2186 / 16.8519 | 186.6453 / 4.6018 |

The raw reverse-order records are in `final-ab`. Encode improved about 2.10 to 2.31× over two rounds; the cached version is about 2.08 to 2.19× faster than the same-window BLAS, but MFU is only about 4.64% to 4.80%, still far from sufficiently optimized. Decode is inconsistent in direction and still clearly slower than BLAS; no decode speedup is claimed this round, and repeated testing until a satisfactory result is not done. GPU time also fluctuates noticeably; frequency, power, and other system load were not captured, so it cannot be attributed to the driver or downclocking without basis.

### Corrected Optimization Priority

1. The cache-attribute bug for large readback is fixed. Next step: evaluate whether the CPU really needs every intermediate value, and whether the 20 encode submissions (three embedding, one whole graph, sixteen KV) can be reduced; not by rewriting the descriptor pool first
2. Attention is the top GPU hotspot: the corrected diagnostics show eight time attention calls totaling 72.11564ms, about 60% of the GPU span; another round was 47.58ms, consistent ranking but the absolute values cannot be mixed. The current 512-thread/39168B-shared block still has many workgroup barriers and serial 64-dim dot products. Compare a more suitable tile/reduction and causal Empty blocks under GPU timestamps, rather than only looking at end-to-end noise masked by the readback
3. The G1 dispatch missed the real embedding large shapes: M1024/N768/K64 and M1024/N192/K768 still use v1; the diagnostics can reach 3.81 and 12.51ms respectively; the microbenchmark and budget comparison for these shapes takes priority over optimizing already-fast element-wise kernels
4. Decode needs a real GEMV and a long-KV dedicated attention. The M=1 projection still uses a 16×16 GEMM with 15/16 output rows useless; the main shape only reaches about 2 to 8GF/s. A2 at count=1 has only three head workgroups, seven query slots useless and not filling 12CU; it needs separate lane division and split-K/partial merge, not a copy of the prefill kernel
5. Decode has, separately, a 12MiB explicit prefix upload and sixteen merges, about 3ms outside the CPU API including page validation, prefix assembly, and common numerics. Further attribution first, then page mirror/direct read; do not count all unknown CPU time as PCIe copy

The old A3a "unstable end-to-end gain" does not equal proving the GPU kernel has no gain: the original 80ms-scale CPU readback and GPU fluctuation mask the device-side difference. Keep the historical results, but the prefill kernel should be independently re-checked after the memory fix and with timestamps available; the branch cannot be unconditionally carried into decode. The FMA peak and the precise separated multiply-add throughput are different measures, but even accepting the arithmetic ceiling, it does not explain the current few-percent overall efficiency and is not a reason to stop optimizing.

This round did not implement GEMV, split-K, or a new attention tile, and did not relax the numeric budget. New diagnostics-switch byte-consistency, count, and coherent readback tests, plus dual-GPU validation and targeted Tiny/Real, pass. The default CPU code and shaders are both unchanged; the full CPU test suite is not re-run repeatedly for diagnostics.

## This Round's Status and Boundaries

On 2026-10-24, redesigned at the user's request; planning precedes implementation; after planning completes, proceed along the tree; no subagent. The old L1 through L6 no longer serve as the implementation order; failure records are kept at the end of this document. The v1 rollback point and restore method are recorded in the backup notes; that backup directory belongs to local working files and is not published with the repository.

Priority: accuracy, usability, measurable performance gain. The public API, model files, and CPU data layout are unchanged; the default std-only build introduces no Vulkan or system BLAS. Default optimization uses only FP32, keeping the per-K multiply-add and mask semantics; FMA, parallel reduction, and low precision are each accepted separately and cannot be enabled by relaxing the budget.

## What Changes from Failure

- No longer change the arena, descriptor cache, weight upload, and push layout of the ten shaders together. Keep the v1 independent buffers and abiVersion=1; extend with new kernels rather than replacing the whole protocol.
- Executor optimization and operator optimization are independent branches. Pool reuse does not require tensor offsets, dynamic SSBO, or weight migration; shared-memory tiling does not require changing the executor.
- The AMD anomaly has no root-cause conclusion. No validation error does not equal application correctness; passing on 4060 does not substitute for AMD acceptance. On any device failure, disable only that branch; do not block other independent optimizations.
- Do not assume every layer has a gain. First write down a cost assumption and do a small-scope comparison; if numeric requirements are met but end-to-end shows no gain, record it as a candidate or reject it, without accumulating default complexity.
- Do not repeatedly run the whole-repository tests or create a new build tree for a single experiment. Reuse the existing Release tree and tests, verify only at the boundaries involved; run the GPU tree and default-build regression once at the formal-phase close.

## Baseline and Bottleneck Assumptions

Phase-five 32k historical baseline (not a same-window A/B of this round):

| Device | encode ms | decode ms | First output value |
| --- | ---: | ---: | ---: |
| RTX 4060 Laptop | 120.8538 | 7.7143 | 2.3805 |
| AMD iGPU gfx1035 | 236.9136 | 11.8549 | 2.3806 |

The checksum of backendCompare (local tool, not in the repository) is the first output value, not a full numeric validation. Encode is 32768 sample points, 1024-patch cold page-pool prefill, median of 3; decode is full prefix H=1, 5 warmups, median of 25 timed. Gains from re-optimization are judged by same-device, same-window serial A/B against the v1 program saved this round; historical best values cannot be used to compute a speedup.

Costs the code can confirm and hypotheses that still need measurement:

- [Executor](<../../timestral/vulkan/executor.cpp>) creates one descriptor pool and a set of descriptors per dispatch; this is a clear CPU/driver cost, but its share is unknown.
- [gemm](<../../timestral/vulkan/shaders/gemm.comp>) computes one output per thread; the 16×16 workgroup has only 16 threads writing at M=1, and the K loop barriers repeatedly; the W global load does not coalesce along K. Register tiling and coalesced W loading each have independent value.
- [attentionFused](<../../timestral/vulkan/shaders/attentionFused.comp>) uses a group of 64 threads per query/head, each lane serially computes the QK of one key, and each block writes the PV partial sum back to global output and reads it back. V reads are contiguous along j and cannot simply be called all scattered; there is no cross-query K/V sharing. 1024×1024×192×4 ≈ 0.805GB is the per-layer logical V-read upper bound without masking, not the 8GB actual video-memory traffic the old text claimed; cache hits and compiler optimizations also change the actual traffic.
- Paged decode assembles the prefix on the host, uploads K/V, and merges, per time layer. Cross-call residency removes the repeated transfer, but page identity, eviction, and model lifecycle must be solved first; the CPU pageId alone cannot serve as a stable key.

## R Optimization Tree

The figure below is the optimization space and historical implementation order; status is in the implementation records. Each leaf independently keeps a v1 rollback; after the 680M measurements the priority adjusted to cached readback, attention timestamp attribution, the missed embedding shapes, a real GEMV and decode attention, then entering paged mirror and fusion per gain; the descriptor pool is no longer placed at the front of the mainline.

~~~text
R  FP32 deep optimization, v1 independent buffers/abiVersion=1 as root
├── R0  Stable baseline and lightweight observation
│   ├── Save the v1 baseline program, dual-GPU serial A/B
│   └── Collect dispatch, submit, transfer, and hotspot timings only as needed
├── E   Executor CPU/driver cost, independent from the operator branch
│   ├── E1  Multi-set descriptor pool reuse, reset after the fence completes
│   ├── E2  Pre-allocate sets per slot, or cache by physical buffer/range
│   ├── E3  Batch readback/submission, keeping the sync and poisoned contracts
│   └── E4  Separate upload/readback pools, readback prefers HOST_CACHED (fixed)
├── G   FP32 projection, keep the per-K order first
│   ├── G1  32×32 output tile, 16×16 threads, 2×2 accumulators per thread
│   ├── G2  M≤4 narrow kernel, lanes along output columns, serial K keeps accuracy
│   └── G3  Optional K-parallel GEMV, FMA, weight panel layout
├── A   Attention, time and variate dispatched separately
│   ├── A1  headDim=64 register PV, avoiding per-key-block output round trips
│   ├── A2  FP32 multi-query shared K/V blocks, online softmax
│   ├── A3  Full/Empty block trimming, strict dual-stream mask
│   └── A4  Decode split-K and a small-variate dedicated kernel
├── F   Fusion and static data reuse
│   ├── F1  residual+RMSNorm, gate+projection epilogue
│   └── F2  Single-call rotary reuse, then evaluate a bounded cross-call cache
├── K   Paged data residency, without changing the public page-pool semantics
│   ├── K1  First measure the share of host assembly/upload and merge
│   ├── K2  Stable page identity + generation + invalidation contract, device mirror
│   └── K3  Page-table direct-read attention, dropping the prefix merge/upload
└── X   Optional device specialization, not a dependency of the default path
    ├── Subgroup reduction: query the subgroup capability, do not hard-code warp=32
    └── Cooperative matrix: per-component feature probe + independent accuracy budget
~~~

## Branch Design and Entry Conditions

### R0 and E: Remove Pool Creation First, Without Changing Addressing

E1: each pool holds 256 sets, at most 1280 storage-buffer descriptors, with pools added on demand. Each dispatch still allocates an independent set and writes the real buffer/range; after recording, it must not be modified until the fence completes. On a successful submit, release the command buffer first and then reset the used pool for reuse by the next submission. A same-batch continuation after a download may keep recording; scratch handles after end are still invalid. The pool owns plain DescriptorSet handles, does not construct an incomplete UniqueDescriptorSet, and does no per-set release, so eFreeDescriptorSet is not needed.

E1 only changes the executor and the lifecycle regression; it does not change shaders, the manifest, or the shared GPU interface. Verification covers the first dispatch of the same cb, crossing the 256 boundary, upload before dispatch, continuation after download, and cross-batch reuse; both devices must pass. After an error, the pool cannot be reset until drained, keeping the poisoned semantics.

E2 is started only if, after E1, set allocation/update is still seen as a hotspot. If a cache is adopted, the key includes the layout, the real buffer identity, offset, range, and generation; a logical handle that can be invalidated cannot be used to cache. Updating an already-referenced set is forbidden before the command completes. E3 first considers reading back multiple results at once; necessary fences and barriers cannot be removed, and the existing interface contract cannot be changed by an async return.

### G: Fix Access and Register Reuse First, Then Change the Numeric Algorithm

G1: the output tile is 32×32, the workgroup is 16×16 = 256 threads, not 32×32 threads. K block 16, A shared 32×16, W transposed to 16×33 (including padding), 4160B total. Each thread loads two A and two W items, W contiguous along K, computing a 2×2 register result; each output keeps the same K order as v1, precise multiply-add, bias applied after. First use scalar coalesced loads to avoid vec4 alignment and tail-block extra risk. Tail rows/columns and non-16-aligned K are zero-padded and participate in all barriers.

G1 is considered for enabling only when M and N are large enough; the main model shapes are K=192/512, N=192/576/1024; the final boundary is decided by the resident microbenchmark and end-to-end, not by defaulting all M>4 to the new kernel. Untested or regressed shapes keep gemm.

G2, handling M=1 to 4, first tries adjacent lanes responsible for adjacent output columns, each lane serial over K, with shared input blocks, avoiding 15/16 idle rows. If the W layout bandwidth is insufficient, then try transpose-on-load or panel pre-packing. G3's K-parallel reduction and FMA change the accumulation order and must have separate error checks; they are not bundled with G1. Low-precision weights do not belong to this branch.

### A: Reduce Output Round Trips First, Then Do Real Multi-query Blocks

A1 adds a same-ABI kernel for headDim=64, one PV register per lane; the QK and softmax blocks still advance by 64 keys; keep the v1 reduction and mask, writing the result once at the end. Do not assume the compiler has not register-resident-optimized the global output; the gain must be measured. Other headDims use v1.

The first A2 candidate is limited to the time axis, headDim=64, with the output block 8 queries × 64 dims. Use 64×8 = 512 threads; each query's 64 lanes still each compute the QK of one key, the same group of 64 lanes reduces the softmax, and each lane accumulates one output dimension. The K/V block is shared over 64 keys; the K-transposed addressing uses 65-column padding, plus Q and per-query independent scores/reduction; total shared 39168B, below the 48KB common upper limit of the two devices. 512 threads are chosen instead of the first draft's 128 to keep the original 64-key reduction tree and the per-K/PV order, without introducing another division-of-labor/rounding change at the same time; when the device cannot reach 512 threads or the shared limit, keep v1. The variate axis has different K/V per query slot, so the same shared block cannot be applied; it continues to be optimized independently. Choose this one tile candidate first; no full-parameter exhaustive search. K/V block reuse, online max/denominator rescaling stay FP32; fully masked rows output 0; tail queries/keys do not exit before the barrier. The tiling and reduction order of the online softmax change; bit-identity cannot be promised in advance.

A3 follows the original timeAllowed/variateAllowed classification. The first narrowing leaf A3a targets only the case where all queries in the same time block are less than encoded and not the KnownCov bidirectional role: then timeAllowed requires key≤query, so all subsequent blocks past the 64-key block of the largest in-block query are Empty and the K/V loop can be terminated uniformly. Still keep the original keyCount as the row stride, the original tail-key mask, and the per-pair in-block predicate; the real K/V row stride cannot be shortened. No trimming in other cases; enable only after the bit-level comparison and end-to-end gain pass. Other Full/Empty classifications are handled independently later; known-future, missing values, and different groupIds cannot degenerate into pure causal. A4 is done only when the decode attention is proven to be the bottleneck; split-K needs extra partial buffers and a merge, and few queries do not naturally benefit. Small variate avoids the generic 64-key loop and large shared blocks when keyCount is very small.

### F: Fusion Around Stable Intermediate Values, Not Copying the Model Flow

F1 first does the residual output and RMSNorm in the same kernel, strictly keeping the original order of y=x+delta, the sum of squares, and (y*inverse)*scale; calls that need trace or two outputs can still read y. Do not move CPU semantics into the backend. Fusing gate into the projection epilogue is enabled only when numerically identical and without adding duplicate computation.

Re-reading the v1 layer graph confirms rotary is already uploaded once outside the layer loop; the old L5 judgment "re-uploaded per layer" does not hold, so the single-call part of F2 is directly recorded as already present and not re-implemented. A cache is added only when the measured cross-call rotary upload cost is worth optimizing; the key must include the actual rotary content or all position parameters, not just count/contextPatches, because continuousRope and position offsets change the data. The cache has a capacity limit; unlimited persistent uploads are not used to replace resource management.

### K and X: High-risk Items Deferred with Clear Exit Conditions

K2, before designing the page-mirror interface, reads the page pool's generation, reference, and eviction behavior; the key must at least distinguish model, pool identity, pageId, and content generation. scratch, LRU eviction, abort, different channel geometry, shared prefix, and FIFO must all be invalidated correctly. The CPU page pool remains the authoritative source; new-page readback is kept, and the page-hash and shared read-only semantics are unchanged. A capacity overrun may drop the mirror and fall back to upload; not every CPU page is required to be resident on the GPU at the same time. K3 is not implemented without a clear transfer share or lifecycle contract.

X does not promise that the presence of VK_KHR_cooperative_matrix makes some FP16/BF16 combination runnable; the actually supported matrix sizes, component types, and FP32 accumulation capability must be queried. TF32 is not the switch name for BF16. Do not add low-precision options first this round; if the accuracy budget is undetermined, keep it unimplemented.

## Lean but Effective Acceptance

1. Each leaf runs only the affected boundaries: shader changes get glslc+spirv-val+ABI reflection plus a small-set oracle of the new kernel; pure pool reuse does not recompile the ten unchanged shaders.
2. After hooking into the model, run the existing gpuModelTiny/Real once per device, including per-layer, full quantile, and paged; on a failure, locate it first, without relying on dozens of full re-runs.
3. Operator candidates measure only the main model shapes and tail blocks. The resident microbenchmark puts upload/readback outside the timing, or explicitly marks it as host record/submit inclusive; the original per-operator transfer benchmark cannot be treated as pure shader throughput.
4. The performance window serially monopolizes the GPU, A/B against the saved v1 or the last accepted node. A clearly no-gain candidate stops immediately; near noise, at most one reverse-order A/B is added. If still uncertain after two rounds, keep the v1 default, without tuning thresholds to chase a single best value.
5. Default-enabling threshold: all correctness passes, the target path improves stably and the other main path has no clear regression. A directional difference above about 3% is a re-test trigger, not an accuracy or absolute statistical-significance guarantee. Per-device/shape dispatch must be centralized; silently swallowing errors to switch results is forbidden.
6. The numeric budget follows the existing tests, not replaced by the first-output checksum: operator atol=1e-5, rtol=1e-4; hidden scale-aware atol and relL2≤1e-5; final quantile atol=1e-4, rtol=1e-3; cache consistency follows the existing frozen rule. Pure executor changes expect bit-identical output.
7. At the close of a meaningful phase, run the full GPU tree, default std-only regression, and style check once, and update this document and the local progress record. The CPU nominal MFU denominator 134.4GF/s is not used to impersonate GPU hardware utilization.

## Implementation Records

| Node | Status | Correctness | This round's performance |
| --- | --- | --- | --- |
| R0 | Complete | Saved this round's v1 program | Initial 4060 126.6051/7.7350ms, AMD 247.1702/12.1213ms |
| E1 | Experimented, removed | Dual-GPU validation, 257 boundary/cross-submit/first dispatch, Tiny/Real pass | 4060 decode small improvement, AMD encode regression, not enabled globally |
| G1 | Accepted, shape-limited enabling | Dual-GPU 9 shapes × bias switch bit-identical to v1, small tail-block CPU oracle passes, Tiny/Real including 129-patch long context pass | Main 1024-row projection 4060 about 2.27 to 2.32×, AMD about 2.34 to 4.04× |
| A1 | Evaluated, only explicit diagnostics kernel kept, not auto-selected | Dual-GPU bit-identical to v1 | Standalone register PV has no stable significant gain |
| A2 | Accepted, enabled when time/headDim64/keys≥128 and capability satisfied | Tail query/tail key, 2D grid, multi-row multi-head, missing/role/base/encoded bit-identical to v1; dual-GPU model passes | 1024-query time microbenchmark 4060 about 2.96×, AMD about 3.23×, see below |
| A3a | Correctness passes, unstable gain, removed | Dual-GPU bit-level pass | AMD encode improved, 4060 inconsistent direction twice and decode regressed, not enabled by default for now |
| G2, other A3 sub-items, F, K, X | Planned | Not implemented | Not measured |

### E1 No-Net-gain Experiment

Pure pool reuse changes no shader or upload path; dual-GPU targeted correctness and validation all pass. After separately saving the E1 program, two reverse-order 32k comparisons:

| Device/round | v1 encode/decode ms | E1 encode/decode ms |
| --- | ---: | ---: |
| 4060 round 1 | 128.4104 / 8.2558 | 127.9981 / 7.6962 |
| 4060 reverse | 126.7515 / 8.0593 | 123.7811 / 7.8064 |
| AMD round 1 | 248.1191 / 12.0481 | 275.5350 / 12.0973 |
| AMD reverse | 265.5561 / 12.0808 | 275.2834 / 12.0375 |

4060 decode improved about 3% to 7%; AMD decode has no directional improvement and encode regressed about 4% to 11%. Per the threshold E1 is removed, without adding a vendor exception to hold onto a weak gain; the executor still uses an independent pool per dispatch. The pool-boundary regression is kept as a large-batch descriptor lifecycle test; the old arena problem did not reproduce in E1, but the root cause is still unknown. If E2 is studied again in the future, independent evidence of gain must be presented again.

### G1 Resident Microbenchmark

A resident mode is added to the existing vulkanBench (local tool, not in the repository): 5 warmups, 25 timed, each run 8 consecutive dispatches, timed per dispatch, including host recording, barriers, submit, and fence, excluding upload/readback, and not called pure GPU time. The following are this round's medians; all outputs are bit-identical to v1:

| M/N/K | 4060 v1→G1 ms | AMD v1→G1 ms |
| --- | ---: | ---: |
| 1/576/192 | 0.020112→0.020788 | 0.037187→0.035425 |
| 128/192/192 | 0.030113→0.022238 | 0.065200→0.039800 |
| 1024/576/192 | 0.382188→0.165175 | 1.123037→0.277750 |
| 1024/1024/192 | 0.619525→0.266575 | 1.923963→0.822250 |
| 1024/192/512 | 0.327550→0.144425 | 1.069613→0.301212 |

In-model kernel selection is centralized in the Vulkan Executor, without changing the cross-API GpuTransformer contract. G1 is enabled only for the full native gemm grid with M≥128 and (K,N) of (192,192/576/1024) or (512,192); M=1 and untested shapes keep v1. The internal independent Executor has auto kernel selection off by default, so the oracle and benchmark explicitly call the two kernels; the model has it on by default, and TIMESTRAL_VULKAN_OPTIMIZATIONS=0 globally disables auto selection, 1 enables it, and illegal values are explicitly rejected. The switch does not change the file set of the compiled shader package.

### A1/A2 Attention Experiments

The resident attention microbenchmark does 4 dispatches each, 5 warmups, 25 timed, including record/barrier/submit/fence but not transfers; the mask is ordinary causal, three heads, headDim64. A1 is not auto-enabled; A2 is used only on the time axis, keeping the full dual-stream mask.

| Device/shape | v1 ms | A1 ms | A2 ms |
| --- | ---: | ---: | ---: |
| 4060 count1024/base0 | 2.804500 | 2.660175 | 0.946100 |
| 4060 count1/base1024 | 0.226275 | 0.222725 | 0.088600 |
| AMD count1024/base0 | 13.108300 | 13.059775 | 4.052275 |
| AMD count1/base1024 | 0.448200 | 0.444650 | 0.189950 |

A1's other round of 4060 long prefill was 2.654600→2.644050ms, AMD 12.581050→13.006950ms, with no stable large gain; the register rewrite itself is not recorded as the main speedup. A2's improvement also includes K coalesced load and shared reads; a single query benefits without query reuse; the cost is the large workgroup and 39168B shared occupancy, so it is not auto-selected for short keys, variate, or other headDim.

### 32k End-to-end and Variation

G1 alone, first time, was 4060 123.3685/8.4713ms vs v1 130.5715/8.1300ms, AMD 248.7492/14.8189ms vs 314.2363/14.9267ms. This shows the projection microbenchmark multipliers cannot be directly extrapolated to the whole model, and decode is not a G1 target path.

The first G1+A2 same-program switch A/B was 4060 110.7752/7.6756ms vs v1 131.2195/8.2253ms, AMD 199.1684/12.4841ms vs 264.6728/12.0622ms. AMD's decode was slightly slower that round; later re-measurement under the same protocol varied a lot, so the best single round cannot be reported alone.

The last reverse-order window, using the same program and A2 shader, encode/decode respectively as follows; an A3 candidate measurement was inserted adjacent to the v1 comparison, all serial, with no build or other this-session GPU tasks:

| Device | v1 ms | Accepted G1+A2 ms | encode/decode ratio | First output value |
| --- | --- | --- | --- | --- |
| RTX4060 Laptop | 125.9646 / 8.0188 | 109.8991 / 6.7479 | 1.146 / 1.188 | 2.3805 |
| AMD iGPU | 263.0007 / 12.9188 | 143.5030 / 10.5634 | 1.833 / 1.223 | 2.3806 |

Cross-round AMD encode varied from 143 to 199ms and decode from 10.6 to 12.5ms; the last round cannot be treated as the hardware's fixed level. End-to-end includes not only shaders but also CPU normalization, embedding/head, page pool, recording, and readback. This time not all CPU backends were refreshed, and no new definite speedup is computed from the old single CPU values. The first output value is only a fast fingerprint; numeric conclusions come from operator bit-level comparison and per-layer/full-quantile/paged tests.

### A3a Correct but Unstable, Removed

A3a only trims provably Empty encoded future key blocks; the dual-GPU bit-level comparison passes. First round A2→A3: AMD 149.7078/10.6403→130.7113/10.2023ms, 4060 109.1403/6.8161→123.8143/7.6405ms. The reverse-order round: AMD 143.5030/10.5634→132.8497/10.3990ms, 4060 109.8991/6.7479→100.9962/7.2785ms.

AMD encode improved over two rounds; 4060 encode is inconsistent in direction and decode regressed both rounds. Per the plan, repeated testing until satisfactory is not continued; A3a is removed; the candidate source and the pre-trim shader set are kept in a local temporary directory; the formal attentionTile is A2. A separate prefill-dedicated kernel can be split out later for re-evaluation, instead of unconditionally carrying the trim branch into decode.

### Phase Close

The final Release incremental build has zero warnings, the full GPU ctest 29/29 (AMD, 81.88s), 4060 targeted model 2/2 and executor validation pass; the default std-only full 25/25 (77.77s). The enhanced operator comparison uses memcmp for byte-level checking. The real 5M 129-patch dual-row case passes with per-layer and full-quantile budgets unchanged; the old 3-patch short regression is not passed off as covering the new dispatch. The final A2 spv and the reverse-order baseline in the table above have the same module hash.

This is the close of the first accepted nodes, not the completion of the whole R tree; the G2 narrow projection, subsequent fusion, KV mirror, and device specialization still must continue per their entry conditions.

## Research Sources and Material Not Yet Obtained

This round on 2026-10-24 again tried to fetch the ggml fixed commit and shader text; api.github.com and github.com were both refused by web_fetch's non-public-address check. The search only found the [flash_attn_cm2.comp](<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp>) and [shader generator](<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp>) paths; no commit was fixed and no text read; their implementation was not copied. This item can be independently read up and does not block the original FP32 optimization; after network recovery, fix the commit first, review dependencies and license, then cite the specific algorithm, without treating the cooperative-matrix dedicated kernel directly as a generic Vulkan1.1 kernel.

The known device limits are AMD push128B/shared64KB, 4060 push256B/shared48KB; both are designed to the common push128B/shared48KB; the ABI is still v1. Sources are the phase-five device measurements and the current shader/executor; the FA2 method references the [FlashAttention-2 paper](<https://arxiv.org/abs/2307.08691>)'s tiling and online softmax ideas, without claiming paper-level performance has been implemented.

## L1 Implementation Attempt and Rollback Record (2026-10-24)

L1 was once fully implemented per the design above (dual arena, cached descriptor sets, offset push with abiVersion=2, incremental weight-arena rebuild); 4060 passed the whole chain, but the AMD iGPU exposed an unresolved first-dispatch behavior problem. Per the user's decision, that implementation was rolled back as a whole to the v1 backup; the old L1 through L6 are abolished and replaced by this document's R tree. Recorded here for reference in next round's planning.

### Pits Located and Fixed

1. The single-argument constructor `vk::UniqueDescriptorSet(rawHandle)` does not fill the device pointer; at destruction `vkFreeDescriptorSets(nullptr, set)` triggers the loader's Invalid device and an ICD /GS crash (c0000409, the WER crash dump is inside the system vulkan-1.dll). The correct usage is to move the result of `allocateDescriptorSetsUnique` into a container; any bare-handle construction of a unique wrapper is forbidden in this project.
2. A pool that explicitly releases descriptor sets must carry `vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet` (VUID-vkFreeDescriptorSets-descriptorPool-00312; the validation layer reports it). A resident reuse pool combined with explicit invalidation release easily misses this.
3. The recording timing of the weight copy: if recorded only once at command-buffer creation, in a same-batch upload-then-dispatch (the arena is only built inside the dispatch) the copy is never recorded, the arena stays all zero, and no error is reported. The recording point must be hooked per dispatch or triggered on demand before download, or changed to an independent submit on upload.

### Unresolved: AMD First-dispatch Behavior Anomaly (RADV)

Symptom: the weight arena read by the first dispatch in a command buffer is all zero, while the second similar dispatch in the same buffer is completely correct. 4060 has no problem at all. The affected kernels are not limited to attention (unrelated to shared memory); gemm and silu as the cb's first dispatch are equally wrong (maxErr 15.8 and 3.29 respectively; 4060 is within budget).

Excluded hypotheses (all controlled reproduction, source and data line dumps kept in the local workspace):

- Copy not executed: after the weight copy was independently pre-submitted with a fence wait, a probe after the fence still confirmed the arena data present, yet the first dispatch still read zero.
- Missing visibility barrier: an explicit transfer-to-compute barrier was ineffective.
- Pipeline cold start: a silu warmup submit was ineffective; a warmup attention dispatch instead turned the wrong output into non-zero garbage matching no input row (consistent with a combination of arena front data), showing the error form depends on the dispatch history.
- Descriptor and memory object problems: the validation layer reported no error, but that is not enough to exclude an application-side lifecycle, binding, or sync error.

At the time it was suspected that RADV had a defect in push constants for the first dispatch or the first read of the first bound buffer, but no root cause was located, and the driver cannot be judged wrong on that basis. Follow-up must use the v1 unchanged binding as the control, changing pool, descriptor, offset, and upload order one item at a time; it cannot simultaneously refactor and then mask the error with a warmup dispatch. This round avoids the issue by not adopting that arena/offset ABI scheme; AMD remains a must-verify device for every enabled branch.

### Rollback Operations and Verification

Per the BACKUP.md restore method, 19 files were restored byte-for-byte from the backup (ten .comp, kernels, executor, gpuExecutor contract, gpuTransformer, two tests, manifest); the temporary trace.hpp was deleted; build-gpu-vulkan was deleted and rebuilt (zero warnings at /W4). Acceptance: ctest 29/29 (AMD device 0); on 4060 vulkanKernels, vulkanExecutor, vulkanShaderAbi, and the gpuModel full model all pass. At the completion of that rollback the tree was the original v1, with abiVersion back to 1; this round's R tree added optimization kernels on that base, with the ABI version still 1.
