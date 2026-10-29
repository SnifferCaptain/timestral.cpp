# A Design Tutorial for the Dedicated BLAS Backend

This document starts from the compute volume of one forward pass and unfolds, top-down, the whole optimization tree of the Timestral dedicated BLAS backend: first decide which computations are not done at all, then hand the computations that must be done to kernels that fit the caches and the registers. The BLAS backend depends on no external library; it is enabled by the CMake option `TIMESTRAL_USE_BLAS`, is the default on x86 when no other backend is selected, and the default build on other architectures is the pure standard-library reference. The following roofline experiments are records of the AMD 7735H single-threaded, MSVC 19.36; for the current 1/2/4-thread model comparisons see [the benchmark data](<benchmark.md>); cache capacities and peaks are estimates, not hardware-counter measurements; the raw records are in the local acceptance records.

## Threads and Output Isolation

Settings.threads now controls the model's own fixed thread pool: 1 for serial, 0 for automatic selection capped at 4, and N for the number of compute threads; the calling thread also participates. Projections dispatch by row or 16-column panel, attention by independent head and channel group, activations by independent output; background threads are reused across operators. Each task holds its own packed workspace; K-dimension accumulation is not split across threads; exceptions are propagated back to the caller after all tasks finish. Nested calls within the same pool run serially in the current thread; multiple external callers are isolated through a submission lock.

[The parallel test](<../../tests/blasParallelTest.cpp>) checks real two-thread concurrency, multiple callers, nesting, exception recovery, and bit-for-bit consistency at 2/4 threads, without substituting the returned thread count for verification of actual execution.

## The Current Model Path

The model by default uses ordered per-K AVX2 multiply-add, residual addition after the dot, and standard nonlinearities. A single query reads the segmented KV directly, avoiding re-packing the whole prefix. Batched attention chunks over 32 queries, completes QK/PV with packed SGEMM, keeps the two-pass softmax and the per-dimension per-key accumulation order, does not fall back to whole-segment scalar computation, and does not materialize the complete score matrix. This path satisfies the frozen numerical budget of long RoPE queries. Shared prefixes are still read by channel group; weight pre-packing, RMS packed fusion, and the QKV layout are kept. The low-level microbenchmark's Auto mode can still test FMA and online reduction; the historical throughput and fusion records below cannot be taken directly as the current model configuration. New model-level comparisons follow [the benchmark protocol](<benchmark.md>).

## Where the Problem Comes From

### How much does one forward compute

Timestral-5M cuts every 32 samples into one patch, D=192, 3 heads of 64 dimensions each, MLP hidden width F=512, and 8 blocks totaling 8 time-axis sublayers and 2 variate-axis sublayers. An input of T samples corresponds to C=T/32 context patches, plus J forecast patches; the attention sequence length is C+J, not T.

Each linear layer is counted as 2MNK FLOPs. Accumulating per slot: each attention sublayer's QKV, gate, and wO total 368640 FLOPs, 10 sublayers total 3686400, the 8 MLPs total 4718592, and the transformer's main linear layers total 8404992 FLOPs per slot. At T=32768, C=1024, the main linear layers are about 9.07 GFLOPs, and the allowed pairs of time-axis attention contribute another about 3.25 GFLOPs. The reference backend needs 4424 ms to complete this forward, which is about 2.8 GFLOP/s over the above effective matrix FLOPs.

### How MFU Is Computed

Zen3 can issue two 256-bit FMAs per core per cycle, i.e. 2×8×2=32 FP32 FLOPs. At an estimated 4.2 GHz the single-core nominal ceiling is 134.4 GFLOP/s, and all nominal MFU in this document is "measured effective FLOP/s divided by 134.4". The 4.2 GHz is an estimate per user specification, with no frequency locking or measurement of the actual effective frequency; a result above 90% must not be read as evidence of boost. If a result exceeds 100%, it can only mean that at least one of the frequency denominator, the FLOP counting, or the timing does not match the assumptions; one should re-check first, not claim a breakthrough of the hardware ceiling.

The numerator of the model-level MFU counts only the matrix FLOPs in the formulas; RMSNorm, exp, RoPE, packing, bias, and memory movement are all not counted; it is therefore an "effective algorithm MFU", not a hardware utilization.

## Optimization Tree Overview

![BLAS optimization tree](<../../assets/blas-tree.svg>)

Optimization unfolds top-down by layer. The request layer and the model-execution layer decide which computations can be saved or merged; the operator layer decides in what structure the computation is organized; the kernel layer and the instruction-set layer decide the efficiency of each instruction. However thorough the lower-layer optimization, it cannot recover what the upper layers computed extra; conversely, if the computations saved by the upper layers are handed to inefficient kernels, the gain is swallowed. Below, the tree is unfolded layer by layer.

## Request Layer: Decide First What Does Not Need to Be Computed

The request layer reuses the common Service's FCFS batching, whole-group V-channel paging, and prefix content addressing; no separate scheduler is built. The BLAS backend does only two things here.

First, shared prefixes are broadcast by row%V. When several requests in a batch hit the same cached prefix, the computation views directly reference the same read-only KV, instead of materializing a full KV for each request; the new tails are still independent per request.

Second, pure prefill terminates early. A prefill that only writes cache for later decoding does not need the output head; it can return as soon as the last time-axis sublayer has written its K/V. This cutoff must not be moved earlier and must not be used for a complete forward that needs a trace or an output.

## Model Execution Layer: Fusion and Narrow Hooks

### Why Not Copy a Complete Model

The complete forecast flow includes patch normalization, conditional masks, paging, multivariate handling, and quantile post-processing; copying it for BLAS would let the two paths gradually diverge. The approach is to extract reference's model orchestration into a common Transformer executor in utils, and the backend only provides `project`, `normalize`, `prepareQkv`, `attention`, `projectNormalized`, `projectAttention`, `prepareCacheKv`, `applyGate`, `activate`, and `projectResidual` through narrow numerical hooks. reference overrides only the first four; the default implementations of the remaining hooks match the original inline code, and after the migration the real 5M output is unchanged line by line.

### Fusion Checklist

- RMSNorm fused into the A packing: each row's inverse RMS is computed once, and during packing it is written into the micro panel as (input×inverse)×scale, without materializing the normalized matrix. It was not changed to fold scale into the weights, because that would change the rounding order.
- QKV and gate merged into one GEMM: the fused weights are assembled at load time, the output keeps its 4D row width, and later stages read directly by offset, saving one split copy.
- RoPE shares sin/cos: Q and K compute the trig functions only once; the partial-dimension rotation keeps upstream's pairing of j with j+headDim/2; in 5M the dimensions actually rotated are 0–15 and 32–47.
- Residual accumulated with beta=1: wO and the MLP down-projection accumulate directly into the residual tensor, saving a temporary tensor and one pass.
- The sigmoid gate is done in place on the attention output, and SwiGLU writes directly into the compact output the down-projection needs; both use the vector exp below.

Fusion only holds when it reduces allocations and memory passes; nonlinear functions are not simplified across linears, and the accumulation order is not changed for speed.

## GEMM: From Triple Loops to BLIS

### Why the Naive Triple Loop Is Slow

The naive i, j, k triple loop updates one C element at a time, with the innermost loop along K. If the logical matrix B is stored row-major, B[k,j] strides across rows, and each cache line is used for only one float on this dot product. Swapping the loops can make B contiguous, but if C is read and written repeatedly or there is no register blocking, loads, stores, and loop dependencies can still block FMA throughput. The key is to make the read-in A and B be reused multiple times in the registers before writing a block of C back, not merely to reorder the loops in the source code.

### Register Blocking: the 6×16 Microkernel

![6×16 microkernel](<../../assets/blas-microkernel.svg>)

The microkernel computes 6 rows × 16 columns of C at once. Each k-step reads one row of 16 floats of B (2 YMMs), broadcasts each of the 6 values of A in turn, and does 12 FMAs. AVX2 has 16 YMM registers in total: 12 C accumulators, 2 B vectors, 1 A broadcast, leaving 1 for the compiler to schedule.

This way each k-step has only 2 vector loads and 6 broadcasts yet completes 192 FLOPs; the load-to-FMA ratio drops from 1:1 to about 1:1.5; estimating only by 2 FMAs/cycle, the 12 FMAs need at least 6 cycles. Estimating a dependency latency of 4 to 5 cycles, 8 to 10 independent accumulation chains are needed to hide the latency; 12 leaves some margin. The exact port occupancy and latency still follow the target CPU's instruction material and measurement. This is the reason for choosing 6×16, not a proof that other shapes must be worse.

Before the K loop, the 6 rows of C are prefetched, aiming to raise the chance of hitting L1 at the final write-back; the prefetch hint itself does not guarantee a hit. In historical comparisons, M=12 partial-model shapes improved by about 15% because of this. A K unroll of 4 steps with B prefetch was attempted; MSVC pushed the accumulators out of the registers, throughput dropped from 105 to 66 GF/s, and it was reverted.

### Packing: Letting the Microkernel Only Do Sequential Reads

![Packing layout](<../../assets/blas-packing.svg>)

The figure first shows a generic B(K,N) row-major layout; the microbenchmark also uses this input. The real model file stores W(N,K) row-major; at load time `rowStride=1` and `columnStride=K` pass the B=Wᵀ view to the same packer; no complete B is transposed out first. The microkernel wants the 16 B values and the 6 A values read at each k-step to be adjacent in memory. Packing rearranges the original matrix into this order: B is cut into 16-column panels, each panel stored contiguously as K rows × 16 columns; A is one panel per 6 rows, interleaved as [k][6]. Tails shorter than 16 columns or 6 rows are zero-padded; the microkernel needs no boundary handling, and the write-back only writes the valid region.

Packing itself is an O(K·N) gather; it can be amortized by O(M·K·N) computation for very large-M matrix products, but not for the small-M model projections: at M=12, N=1024, K=192, the generic path including B packing is only 16.6 GF/s, while the pre-packed path is 130 GF/s. Weights do not change during inference, so all projection weights are packed into `PackedWeight` at model load time; during inference only A, which varies per request, is packed.

### Should Weights Be Stored Directly in Blocked Format in the File

A natural idea is to store the weights in the file already in blocked format, saving the gather step from strides to blocks. The current implementation already does this in memory: the projection weights are packed at load time, and thereafter each linear projection reads the panels directly, never re-gathering that weight from a stride view. If loading the file yields the same `PackedWeight`, the memory access of the steady-state projection is unchanged, and no further benefit of the same kind should be expected. File pre-packing may reduce cold-start conversion and intermediate copies; the first round of BLAS loading took about 48 ms versus about 33 ms for reference, but this ~16 ms gap also includes costs such as fused-weight construction and allocation; it is not an independently measured packing time or a strict upper bound.

The file format needs extra conventions: panel width 16, dimensions and padding, the order of the fused weights, version, byte order, and a source-weight checksum. SSE or NEON could choose a layout compatible with 16-column panels, or might need a different layout; there is still no cross-ISA format contract. Directly replacing the upstream safetensors would break existing compatibility, so the released weight layout is not changed in this stage. If a later cold start really becomes a bottleneck, one can keep the original model and separately generate a disposable backend cache with a version and a source hash, repacking on mismatch; that is safer than forcing all backends to use the same internal layout, and it is not implemented yet.

Quantized decode is especially worth studying offline panel storage: the weights are then stored as INT8 or FP16, and storing them in micro-panel order lets dequantization directly produce the 16 columns the microkernel needs, avoiding first expanding into a complete FP32 matrix. The interface for this part is already reserved; see below.

### Cache Blocking: Five Loops

![Five loops](<../../assets/blas-loops.svg>)

Register blocking solved the load-to-FMA ratio; cache blocking must ensure these loads hit the nearby cache. Following BLIS's five loops: the outermost jc cuts N by NC, pc cuts K by KC, ic cuts M by MC, and jr and ir walk the micro panels by 16 columns and 6 rows.

At KC=256 a B micro panel is 256×16×4=16 KB, plus the 6 KB A micro panel, which fits the 32 KB L1; at MC=120 the packed A block is about 120 KB, placed in the 512 KB L2 and reused by all B micro panels in the jr loop; B's KC×NC panels stay in L2 or L3, reused by all A blocks in the ic loop. When pre-packing the weights, NC is taken as the whole N, because in 5M the N of the main attention and MLP projections does not exceed 1024, which avoids repeatedly packing A by NC; whether B can stay in the cache still depends on capacity, contention, and execution order. The initial candidates were KC of 128 or 256, MC of 120 or 240, NC of 128 or 256; the current landing is KC=256, MC=120, generic NC=256; the candidate enumeration is the planned range, not a claim of a systematic exhaustive sweep of all combinations. The cache working-set discussion only gives a plausibility analysis, not a claim of hardware-counter hit-rate evidence.

### Small M: Narrow Kernels and Decode

The model has many projections whose M is not a multiple of 6; at decode time M is even 1. Using a 6-row microkernel for M=4 wastes a third of the computation, so a narrow kernel was written for each of the 1 to 5 rows; in the historical records M=4 shapes rose from about 98 to 113 GF/s, which converts to about 73% to 84% of the 4.2 GHz nominal denominator.

The narrow kernel uses a single K accumulation chain. Multi-chain interleaving was tried to hide FMA latency, but it changed the summation order, and the gap between the real model's shared decode and full recompute exceeded the threshold, so it was reverted to a sequential single chain; the follow-up audit also removed an always-1 chain-count template parameter, avoiding keeping a useless abstraction.

At M=1 each weight participates in only one FMA; 2 FLOPs require reading 4 bytes, so compute intensity and memory bandwidth become the bottleneck, and this round's nominal MFU is about 21% to 24%. This is exactly the shape quantized decode is meant to solve: the smaller the weights, the faster they can feed the FMA.

### Accuracy and Accumulation Order

The BLAS backend does not loosen any existing threshold: hidden states max(1e-5, 1e-4×RMS) with relative L2 no more than 1e-5, final forecast 1e-4+1e-3×|ref|, cached decode against full recompute max(1e-6, 1e-5×RMS). Blocking changes the summation order, so the thresholds are set scale-aware; but within the same backend, full recompute and cached decode must follow the same K-direction order, otherwise the gap between the two may be amplified; one must re-verify rather than assume it is still within 1e-5×RMS; the single-chain constraint of the narrow kernel comes from exactly this.

## Attention: Online Softmax and Structured Skipping

### The Problem with the Naive Approach

Standard attention first computes the L×L score matrix, does a per-row softmax, then multiplies by V. At L=1028 the score matrix is about 4 MB, each head writes it out and reads it back, and under a causal mask about half of the scores would be masked anyway.

### Online Softmax

![Online softmax](<../../assets/blas-online-softmax.svg>)

Online softmax (the blocking idea used by the FlashAttention family) splits the keys into blocks; each row keeps only the current maximum m, the denominator l, and the unnormalized output O. After reading the scores S of a new key block, set m'=max(m, max S), α=exp(m−m'), then l'=α·l+Σexp(S−m'), O'=α·O+exp(S−m')·V, and finally output O/l. When an entire row is masked, l=0, the output is set to 0, and no inf−inf appears.

The block size is 6 queries × 64 keys, reusing the GEMM microkernel exactly: Q is packed as an A panel of [d][6], K as a B panel of [d][16], V as [64 keys][16 dims]; QK is a microkernel call with depth=headDim, PV is a microkernel call with depth=64. The probabilities written by the softmax are already laid out as [key][6], becoming the A panel of PV, with no extra transpose in between.

### Mask Block Classification

![Mask block classification](<../../assets/blas-mask.svg>)

Each 6×64 block is first classified into three kinds: an Empty block has no allowed (q,k) pair at all, and both QK and PV are skipped; a Full block is fully allowed, a shared all-1 byte mask is passed in, and no per-element predicate or mask expansion is called; only a Partial block is masked element by element. The speedup comes from structurally computing less, and does not rely on branch prediction.

Timestral's mask is not a pure triangle: missing patches, known-future roles, and variable-group isolation all punch holes below the diagonal; the missing keys 20 to 22 in the figure turn what should be a Full block into a Partial. Therefore the classification must be based on the real predicates. The initial implementation called the predicate per element for each block to classify it, which itself took a large amount of time. The improvement has two steps: without covariates, a block is proven Full geometrically from the three branches of `timeAllowed`; whether a key is valid is decided in O(1) with a prefix sum. A Partial block uses `fill` to expand the predicate into a 64-byte mask in one shot, avoiding calling `std::function` in the inner loop of the softmax.

### Small-Key-Count Path

The variate axis has a sequence length of only the number of variables; when the key count is at most 16, the packing and blocking overhead far exceeds the computation, so a direct two-pass softmax short path is taken.

## Elementwise: Vector exp

### Why exp Deserves Separate Optimization

The online softmax computes exp once per score, and gating and SwiGLU each need it once per element. The initial profile showed that in the 411 ms of a 32768-sample forward, time attention took about 238 ms, the sigmoid gate about 22 ms, and SwiGLU about 18 ms, mainly because of per-element `std::exp`, not because of the microkernel.

### The 2^x Decomposition

![exp decomposition](<../../assets/blas-exp.svg>)

The change-of-base identity is `exp(x)=exp2(x·log2(e))`, correspondingly `log(x)=log2(x)/log2(e)`. But directly switching to standard-library `std::exp2` calls is not guaranteed faster, and AVX2 has no general exp instruction. Here the vector range reduction and bit construction are done ourselves, avoiding per-lane calls to scalar library functions; the power-of-two scaling idea expressed by `ldexp` is unfolded into integer addition, shift, and multiplication.

Exploiting the composition of floating-point numbers, e^x can be rewritten as 2^(x·log2e)=2^n·2^f, where n is an integer and |f|≤0.5. When n is in the normal exponent range, shifting n+127 left by 23 bits into the exponent field and interpreting it bitwise is exactly 2^n. The actual implementation splits n into two halves, constructs normal numbers each, and multiplies them, covering the subnormal and overflow boundaries. The remaining 2^f lies in a very narrow interval, and a 6th-order polynomial achieves about 1 ulp.

The concrete implementation has a few details:

- Rounding uses a magic number: the low bits of the floating-point sum t+1.5·2^23 are exactly round(t); the same sum gives both the floating-point n and, through integer subtraction, the integer n, saving a separate round and cvt. It relies on the default round-to-nearest-even mode; the three Native entry points check MXCSR and call the scalar implementation directly on a non-default rounding mode, without altering the caller's environment on its own.
- f=x·log2e−n is computed in one FMA, with only one rounding; log2e is further split into high and low parts, and a second FMA compensates the constant's own error; otherwise, after the constant error is multiplied by a larger |x|, it may exceed the single-precision 1 ulp budget.
- The range of n is [−150, 128]; it is split into two halves, each assembled into a power of two and multiplied in a chain, so subnormals and outputs near FLT_MAX are all represented correctly.
- The input is first clipped to [−103.97, 88.72]. These two endpoints themselves produce 0 and +inf through the arithmetic, so no blend correction is needed; max and min return the second operand when they meet a NaN; putting x in the second position lets NaN propagate naturally through the arithmetic.

The coefficients come from exp2f.c of Cephes Release 2.2; for the copyright and release notes see [third-party sources](<../../THIRD_PARTY.md>). The original implementation is a Cody-Waite reduction plus the Cephes e^r polynomial, using a three-way blend to handle the endpoints and NaN. Under the default round-to-nearest-even mode with FTZ/DAZ not enabled, all finite floats in [−110,110] were swept, plus ±inf and some NaN payloads (2243428616 test items in total, including alignment zero-padding), compared against double-precision exp. The average ULP here accumulates the real error for every finite reference result, and correctly rounded results are not recorded as zero; overflow and non-finite results are checked separately by category. Three rounds of pure-function throughput experiments of 4096 elements each give the table below:

~~~text
variant                            max error   avg error    time per element
Cody-Waite + e^r poly + 3 blends    1.26ulp     0.0739ulp    0.72 to 0.74ns
2^x decomposition, blend kept       1.18ulp     0.0634ulp    0.70 to 0.75ns
2^x decomposition, endpoint blends removed 1.18ulp     0.0634ulp    0.59 to 0.63ns
2^x decomposition, NaN via operand order    1.18ulp     0.0634ulp    0.53 to 0.62ns
~~~

Simply replacing e^r with 2^f is not stably faster; the two reductions have similar instruction counts, with a slight error improvement; after the full change removed the redundant blends, this round of pure-function time per element dropped by about 16% to 26%. The average ULP dropped by about 14%, but a uniform sweep over bit patterns includes many inputs near zero, which does not represent the average error on real activation distributions. This approximation must not be presented as a correctly rounded standard-library replacement; it still has to pass the softmax, gating, SwiGLU, and full-model regressions.

## Correctness Guardrails

### Runtime Dispatch and Compilation-Unit Isolation

AVX2 code lives only in separately compiled source files, with /arch:AVX2 or -mavx2 -mfma added per file, not enabled globally. The baseline unit confirms AVX2, FMA, and the OS saving YMM state via CPUID, OSXSAVE, and XGETBV before selecting the AVX2 kernel; otherwise it falls back to scalar.

Inside the AVX2 unit, only intrinsics and hand-written loops are used; std inline templates are not instantiated, scalar `std::exp` is not called in vector hot spots, and no exceptions are thrown. The reason is that some standard-library inline functions or templates may form same-name COMDATs in multiple compilation units, and the linker may select the one in the AVX2 unit for the scalar fallback path, causing machines without AVX2 support to execute AVX2 instructions. Argument validation and exceptions both stay in the baseline unit.

### NaN and Whole-Row Masking

When a NaN appears among the allowed keys, the NaN must propagate to the output and must not be swallowed as a whole-row mask; only when the scores of all allowed keys are −inf is it a whole-row mask. This semantics is checked with the same set of tests injecting ±inf and NaN on both the scalar and AVX2 paths.

### Test Coverage

Random GEMM covers KC, MC, NC boundaries, transpose and stride, tail blocks, beta, empty dimensions, out-of-range, aliasing, and concurrency; the narrow kernels 1 to 6 rows, RMS packing, INT8 local panels, vector exp, and softmax rows are checked against scalar; 800 online-attention groups cover causal, sink plus SWA and sparse masks, and the fill and predicate paths; on the real 5M and the miniature model, trace, forward, paging, and shared decode are verified.

## Instruction-Set Layer: SSE4.2 Fallback and the ARM Trade-off

AVX2+FMA is the first target, but not all x86 supports it. The SSE4.2 fallback keeps the pack/microkernel separation boundary, reuses the 16-column panel layout, and adds no new packing format. x64 has only 16 XMMs; the original 6×16 needs 24 YMM accumulators, which do not fit, so the same 6×16 block is split into two 6×8 passes, totaling 12 128-bit accumulators; SSE4.2 has no FMA, so it keeps the per-output single-K-chain mul-then-add without reordering the summation, and the scalar oracle and thresholds are unchanged. The Native symbol still refers to AVX2 only; SSE has its own computation unit and `cpuSupportsSse42` (CPUID leaf 1 ECX bit 20, OSXSAVE not checked); Auto dispatches AVX2+FMA → SSE4.2 → Scalar; forced SSE mode validates availability before dispatch and throws an exception if unavailable. The SSE compilation unit uses per-file flags such as `-msse4.2 -mno-avx -mno-avx2 -mno-fma -ffp-contract=off`; the disassembly confirms no VEX or FMA mixed in; the elementwise path under SSE continues to go scalar.

The forced-SSE GEMM is about 40–41.7 GF/s, about 30% of the 134.4 nominal denominator; the single-head 1028 causal attention is 7.02 ms versus 2.35 ms for AVX2. This is a working correctness fallback, not a performance path.

This machine has no ARM run or cross-validation environment, so unverified NEON is not implemented. The user explicitly allowed ARM to use Eigen, so the build detects the target architecture: TIMESTRAL_TARGET_PROCESSOR first takes CMAKE_GENERATOR_PLATFORM (a VS -A ARM64 cross generation does not change CMAKE_SYSTEM_PROCESSOR); when the target is ARM, Eigen is selected and a STATUS message explains the actual backend. This only verified the configuration routing; it does not mean the numerics were accepted on ARM hardware.

## Quantized Decode: Compressing Only the Weights, Prefill Stays FP32

The bottleneck of decode is different from prefill. Prefill computes the whole history at once and is compute-bound; the CPU only supports physical FP32 computation anyway, so quantization does not help; decode advances only a few patches per step, M is very small, and weight memory access starts to dominate. Therefore quantization targets only decode: the unchanging weights stay compressed and are converted back to FP32 at the registers for accumulation; input activations, outputs, accumulators, and KV are always FP32, with no quantized Level-3 matrix multiply.

The weights are still stored as [K][16-column] panels. FP16 is converted back inside the AVX2 unit with F16C's `_mm256_cvtph_ps`; BF16 shifts the high 16 bits left into FP32 (this is an exact extension, not an approximation); INT8 uses independent symmetric scaling per 32 K values per output column: q=clamp(RNE(w/s), −127, 127), and the scale s of an all-zero group is recorded as 1. INT8 does not use VNNI; after conversion it still goes through FP32 FMA, so none of these three paths is "low-precision computation", only "low-precision storage + FP32 accumulation".

### Phase Gating and Fallback

Quantization takes effect only at true decode. `ProjectionPhase` has two levels, Float32 and Decode, passed through by value; any prefill, trace recording, with covariates, rolling start point, or fresh-history request goes through FP32; `runLayers` directly rejects the Decode phase in these scenarios rather than silently downgrading. Only when the cache pages are non-empty, this step has no fresh history patches (freshPatches==0), and the output head's row count does not exceed 4 do the projections switch to `decodeProject` with the compressed weights; for larger M it falls back to FP32 `gemmPacked`. This means a single decode and a batched decode may take paths of different precision; the difference is accepted against the pre-frozen budget and is not hidden from the outside.

### Acceptance Budget and Measurements

The budget was frozen before running the real model; a failure is recorded as a failure, with no back-filled thresholds: FP16 relative L2 ≤ 0.005 and maxAbs ≤ 0.02×max(reference RMS, 0.1), BF16 and INT8 relative L2 ≤ 0.05 and maxAbs ≤ 0.2×max(reference RMS, 0.1). The bit conversion exhausts all 65536 16-bit patterns, exact-midpoint parity, and four fenv modes; `decodeTest` uses a double dot product as the oracle covering M/N/K boundaries, stride, beta, aliasing, and out-of-range; `decodeModelTest` confirms on tiny and the real 5M that the forward and prefill pages of the four formats are pairwise bit-for-bit identical (the decode format only affects the Decode phase); the quantized decode passes against the reference within the budget; the M>4 fallback is bit-for-bit identical to the Float32 baseline; the page hashes are unchanged before and after decode.

Performance is recorded honestly. On the micro-projection GEMV (M=1/4, real 5M shapes), quantization is consistently slower than FP32: at M=4, fp32 is about 98–112 GF/s, fp16 about 46–80, bf16 about 44–74, int8 about 38–69; these shapes are still compute-bound, and the overhead of conversion + scaling + FMA dominates, honestly recorded as no microkernel benefit. But model-level direct paged decode is actually faster with quantization: H=1 median fp32 0.715 ms vs fp16 0.457 / bf16 0.454 / int8 0.469 ms, H=4 is 1.118 vs 0.853/0.898/0.908, H=16 is 2.723 vs 2.586/2.653/2.590. The costs are increased resident memory (FP32 packed and the compressed copy coexist) and a build-time quantization cost (construct_ms fp32 36.8 vs fp16 110.2 / bf16 87.1 / int8 150.9); no claim that model memory shrinks. If a public request enables mirror averaging, the mirror pass is still a full FP32 recompute, and the decode benefit only covers the direct paged path.

## Experiments Without Benefit

- Microkernel K unroll 4 steps plus B prefetch: accumulator overflow, 1024-cube dropped from 105 to 66 GF/s, reverted.
- FTZ/DAZ: 32768-sample time 461 ms vs 465 ms, no difference, not adopted.
- Residual beta=1 accumulation: 186 ms to 184 ms, within the noise; kept only to save a temporary tensor.
- Narrow-kernel multi-chain interleaving: changed the summation order, numerical out of bounds, reverted.
- exp replacing only e^r with 2^f: better accuracy but not faster; the speed gain comes from removing the blends.

## Acceptance Results and Measurement Fluctuation

![Backend comparison and the attention optimization process](<../../assets/blas-results.svg>)

The figure carries over the data of the first-round same-machine side-by-side acceptance; this round's exp experiment is not mixed with the whole-model difference against the old version as a benefit. All MFU here is uniformly converted to 4.2 GHz × 32 = 134.4 GFLOP/s; the old denominator in the raw logs is kept as historical evidence.

| Object | First-round record | 4.2 GHz nominal MFU | This round's repeated measurement |
| --- | --- | --- | --- |
| pre-packed GEMM of the main model shapes M≥12 | 122 to 130 GF/s | 91% to 97% | full sweep mostly about 105 to 123 GF/s, with lower outliers |
| 1024-cube pre-packed GEMM | 125.2 GF/s | 93.2% | 110 to 112 GF/s, about 82% to 83% |
| M=1 main projections | about 31 GF/s | about 23% | about 28 to 33 GF/s |
| single-head 1028 causal attention | 2.35 ms, 57.6 GF/s effective | 42.8% | 2.38 to 2.53 ms |
| forward T=32768 (with scale) | 169.98 ms, 72.5 GF/s effective | 54.0% | profile 188 to 199 ms, about 46% to 49% |

The first-round model-level effective MFU for T=256/1024/4096/16384/32768 converts to 45.3%, 56.2%, 60.0%, 56.4%, and 54.0% respectively. This round's same sequence is about 43%, 53% to 54%, 56%, 51%, and 46% to 49%; the absolute throughput is below the first round, and even the GEMM whose algorithm was not changed slowed down; there is currently no frequency, temperature, or scheduling counter evidence to assert the cause.

The exp microbenchmark takes the minimum of 15 runs per round, input 4096 elements in [−20,0], repeated 4000 times; this is a pure-function throughput experiment, not the whole-model median protocol. Interleaving old/new rounds, the 32768 model medians are 213.26/216.41 ms and 198.47/196.73 ms, inconsistent in direction, so only the local exp throughput improvement is confirmed, with no claim of a stable whole-model speedup. This round's stageBench is 191.2 ms, batch about 312 req/s, stream about 7.0 ms, decode about 1.19 ms; the 171.7 ms, about 320 req/s, 6.3 ms, and 1.02 ms from a different session in the first round are all kept; the best single run is not used to cover the fluctuation.

The final default, Eigen, and ytensor builds pass 25/25 each, and the BLAS AVX2 build passes 29/29, with no compile warnings; separately, a BLAS scalar build with AVX2 explicitly off and the examples disabled passes 26/26, verifying the renamed Fallback unit. After finally adding the rounding protection, the 32768 profile is 196.13 ms, nominal effective MFU 46.76%; the 176-line comparison against the default reference has 0 differences. For the detailed build and numerical evidence see the local acceptance records. This machine only runs Windows/MSVC; GCC and Clang under a Unix-like environment were not executed; the Full classification of the no-covariate geometry is currently covered only indirectly by the whole-model regression, with no independent per-block predicate cross-check test.

## Code Entry Points and Research Sources

| Layer to read | Code entry |
| --- | --- |
| model fusion and pre-packing | [BLAS model](<../../timestral/blas/model.cpp>) and [common Transformer](<../../timestral/utils/transformer.cpp>) |
| five loops, views and panels | [gemm.cpp](<../../timestral/blas/gemm.cpp>) and [gemm.hpp](<../../timestral/blas/gemm.hpp>) |
| 6×16 and narrow kernels | [full microkernel](<../../timestral/blas/microkernelAvx2.cpp>) and [narrow kernels](<../../timestral/blas/microkernelNarrowAvx2.cpp>) |
| SSE4.2 microkernel | [microkernelSse42.cpp](<../../timestral/blas/microkernelSse42.cpp>) |
| quantized decode weights and projections | [decode.cpp](<../../timestral/blas/decode.cpp>) and [decodeAvx2.cpp](<../../timestral/blas/decodeAvx2.cpp>) |
| online softmax and block classification | [attention.cpp](<../../timestral/blas/attention.cpp>) |
| vector exp, sigmoid and SwiGLU | [elementwiseAvx2.cpp](<../../timestral/blas/elementwiseAvx2.cpp>) |
| CPU/OS detection | [cpuX86.cpp](<../../timestral/blas/cpuX86.cpp>) |
| reproducing throughput and formulas | local tools blasBench and blasProfile (not checked in) |

BLIS is pinned to commit [e8566eb3](<https://github.com/flame/blis/tree/e8566eb3e773fb54d11b33e371d13f22d2941e50>); for the specific designs and source code read, see the local research records. Online softmax can be compared against [FlashAttention](<https://arxiv.org/abs/2205.14135>) and [FlashAttention-2](<https://arxiv.org/abs/2307.08691>); this implementation borrows the blocking and online reduction ideas, and does not claim to replicate the GPU's warp scheduling, asynchronous pipeline, or the same performance characteristics. The tutorial's "basic problem → mathematical structure → engineering improvement → trade-off" writing style references [the pinned tdt-nav-kit version](<https://github.com/T-DT-Algorithm-2026/tdt-nav-kit/tree/814d5ce0237acc7faa8c070e5c5e61e2ddc81996>); the figures were redrawn for this project.

## Future Roadmap

- GPU: currently only the complete Vulkan model backend with GLSL/SPIR-V release is kept; the old OpenGL/OpenCL experiments were deleted and are no longer future backends; see [the trade-off records](<gpuScope.md>).
- ARM NEON: if a verifiable ARM environment is available later, implement with reference to BLIS's ARMv8 kernels, keeping its BSD-3-Clause copyright and license statements when copying the source.
