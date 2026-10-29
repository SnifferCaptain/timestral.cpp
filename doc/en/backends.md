# Compute Backends

The CPU implementations include the default reference, plus the optional Eigen, ytensor, and home-grown BLAS. This document covers the numerical paths, threading, and caching capabilities; for BLAS details see [the BLAS backend](<blas.md>), and for the GPU model see [the Vulkan backend](<vulkan.md>).

## Why Multiple Backends Are Needed

The most direct port would be to write every operator of the upstream PyTorch as a triple loop. That is exactly the reference backend: it depends only on the C++17 standard library, the accumulation order is simple and traceable, and it suits serving as the correctness baseline. For current model-level timings see [the benchmark comparison](<../../README.md#performance>).

The naive way to speed things up is to bring in a matrix library and hand all the computation to it. That runs into two constraints: the default deployment path must remain free of third-party dependencies; and switching libraries changes the floating-point accumulation order, so results are no longer bit-for-bit identical to reference, requiring a tolerance that is fixed in advance and cannot be loosened afterward. This repository's handling: the backend is single-selected by CMake at configure time (mechanism in [model architecture](<architecture.md>)); optional backends only replace the compute-heavy parts, the model semantics stay consistent, reference and BLAS share an executor, and Eigen and ytensor still keep their own complete orchestration; accuracy is judged against a fixed upstream oracle with scale-aware tolerances.

![Backend class hierarchy and capability comparison](<../../assets/backend-layers.svg>)

## reference

reference is a subclass of `Transformer` implementing only the narrow numerical hooks the model orchestration needs: projections, RMSNorm, QKV reshuffling and RoPE, single-head attention; see [reference/model.cpp](<../../timestral/reference/model.cpp>). All the logic of the embedding, masks, paging, and the quantile head is in the [Transformer](<../../timestral/utils/transformer.cpp>) of `utils`, so reference itself is under a hundred lines. The underlying operators are in [kernels.cpp](<../../timestral/reference/kernels.cpp>): the linear layer does a serial dot product per output and adds the bias; attention first computes the full row of scores, fills masked positions with negative infinity, multiplies by V after softmax.

reference has three deliberate choices:

- no aggressive fusions that change the operation order; every step can be matched to a single upstream PyTorch expression, so layers can be compared one by one when something goes wrong.
- fixed single-threaded; `Settings::threads` has no effect on it.
- the compile option is `/fp:precise` and fast-math is not enabled, keeping the serial dot-product order.

The numerical regressions of the other backends all use it as the baseline; before and after the `Transformer` extraction, the referenceTest output of the real 5M is identical line by line (176 lines, diff_count=0).

## Eigen

The Eigen backend uses Eigen 3.4.0 row vectors and Map views for the projections; attention uses the shared ordered numerical kernels.

- Weights are pre-transposed at construction; Eigen row vectors update item by item in K order, with the bias added last. OpenMP parallelizes over input rows and does not reorder the reduction inside the dot product.
- Attention reads the shared prefix and the current round's tail directly; variate tightens the range by request group. The ordered softmax and PV keep the frozen numerical budget.

RMSNorm, RoPE, softmax, SiLU, SwiGLU, and the quantile head call the reference kernels directly; see [eigen/model.cpp](<../../timestral/eigen/model.cpp>). The Eigen backend predates the `Transformer` extraction and still keeps a complete model orchestration and paging implementation; shared-semantics changes must be checked in sync.

The thread count is controlled by `Settings::threads`: 1 for serial (default), 0 for an automatic value capped at 4, and N for an exact thread count. Eigen's thread count is process-level global state and must be set before constructing the model; multiple models in the same process share the last setting. Not taking the automatic value to full cores was measured: 5M's GEMMs are tiny, and in the records the regression suite took about 28 s at 16 threads and about 2.9 s at 4 threads.

## ytensor

The ytensor backend uses a single-header 0.14 from a user-specified source; the repository contains no license file for it; see [the 3rd description](<../../3rd/README.md>) for the factual record. Projections go through ytensor's strided matmul, wrapping this repository's storage with non-owning views. The current attention uses the shared ordered kernels, reading the shared prefix and the independent tail directly, and the elementwise operators also reuse the reference kernels.

- YBLAS handles the projections. The original FLASH_AVX2 and MATH attention implementations are no longer the model's default path for the current long queries; old FA2 configuration records must not be taken as this recipe.
- Only [ytensor/model.cpp](<../../timestral/ytensor/model.cpp>) enters a separate C++20 OBJECT target with AVX2/FMA enabled (MSVC uses `/arch:AVX2`, GCC/Clang use `-mavx2 -mfma`); the factory and core stay ordinary C++17, and IPO/unity is forbidden from crossing the instruction-set boundary. The baseline factory checks the CPU and OS for XMM/YMM support before calling any function of that object; old CPUs are explicitly rejected, and non-x86 configurations are rejected at configure time. For the implementation, verification, and residual numerical failures see [the compatibility records](<linuxCompatibility.md>). MSVC does not define `__FMA__` under `/arch:AVX2` and does not support the GNU `__attribute__` syntax in the single header; that file fills in these two spots in pairs before including ytensor and undoes them after the include; the records note that these attributes are only inlining hints.
- Supports paged incremental; `supportsIncremental()` returns true. Pages store each time layer's K/V and the causal normalization state; append and shared decoding accept the unified page-table contract.

## Accuracy Acceptance Criteria

All backends are checked against the same fixed set of upstream oracles: 14 per-layer block cases and 36 forecast cases (including 5 expected-rejection cases), generated by upstream Python on the fixed baseline. The miniature model has fixed-seed random parameters, not the released weights. The tolerances were fixed in phase one and subsequent backends carry them over without loosening:

| Object | Criterion | Notes |
| --- | --- | --- |
| operator | atol 1e-5, rtol 1e-4 | single-kernel comparison |
| hidden states | max(1e-5, 1e-4×RMS(reference)), and relative L2 no more than 1e-5 | embedding, each layer's output, out_norm |
| final forecast | abs(a−b) ≤ 1e-4 + 1e-3×abs(reference) | per-case judgment, also recording max error and RMSE |
| cached decode | max(1e-6, 1e-5×output RMS) | paged decode against a full recomputation of the same semantics |

The hidden states use a scale-aware atol because the residual stream's magnitude grows layer by layer. The real 5M's layer-7 output RMS is about 120 to 152; a small fixed threshold would misjudge the normal differences from FP32 reduction order as errors; in the records the max in-layer absolute error of the real-weight path is 8.06e-3 and the relative L2 is 5.10e-6. The cached decode threshold is set separately, used to verify the cache, and not mixed with the oracle tolerance; concurrency tests instead configure the baseline and the service under test with the same no-cache path and only check scheduling consistency.

The forecast errors recorded during the initial porting stage are below; they cannot replace the regression of the current ordered implementation:

| Backend | realDefault max absolute error | realFullBlockThenPartial max absolute error |
| --- | --- | --- |
| reference | 9.54e-6 | 0.310 |
| Eigen | 1.34e-5 | 0.083 |
| ytensor (forced Naive) | 9.53674e-6 | 0.310059, bit-for-bit identical to reference |

realFullBlockThenPartial is the 1030-step long-horizon case; the reference-output RMS is about 854; multi-block autoregression amplifies reduction-order differences, so its absolute error is clearly larger than for short cases; all three backends are within tolerance. ytensor's Naive loops have the same accumulation order as reference, and the results match bit for bit. Once YBLAS is enabled the accumulation order changes; in the records the synthetic case max_abs is at most 3.8e-6 and the real 5M multivariate and conditional cases are on the order of about 1e-7, so judgment switched to scale-aware tolerances and bit-for-bit identity is no longer required; the individual numbers of the YBLAS path such as realDefault were not recorded separately.

## Performance Trade-offs

Side-by-side measurements on the same machine (AMD 7735H, MSVC 19.36, Release, single-threaded, stageBench and timestralBench protocol, record date at the phase-three acceptance):

| Item | reference | Eigen | ytensor |
| --- | ---: | ---: | ---: |
| forward T=256 | 32.10ms | 6.83ms | 6.77ms |
| forward T=4096 | 362.21ms | 70.64ms | 55.35ms |
| forward T=32768 | 4424.15ms | 961.20ms | 1185.57ms |
| append | 20.98ms | 8.46ms | not supported |
| decode | 11.79ms | 3.59ms | not supported |
| batch | 15.7 req/s | 88.2 req/s | 123.1 req/s |
| stream | 102.7ms | 21.4ms | 28.0ms |

A few ways to read this:

- For a single-row forward at the three lengths, Eigen and ytensor are about 3.7× to 6.5× faster than reference; Eigen is faster at T=32768, ytensor is faster at T=4096.
- The stream protocol appends data step by step from the same source; Eigen has the paged cache, ytensor recomputes everything each time; so although ytensor's single forward is faster at T=4096, its stream is still slower than Eigen. Whether the paged cache can be used affects streaming scenarios more than how much faster a single GEMM is.
- The multi-threading gain is limited. On stream, 4 threads is the measured overall optimum for Eigen, about 7.4× reference; 8 and 16 threads are actually worse due to OpenMP scheduling overhead of small GEMMs. ytensor's YBLAS gives only about a 1.4× gain at 4 threads for the 32768 context; short contexts plus threads are a net loss. Hence the default is `threads=1`.
- Each configuration reports one round of protocol statistics (5 warmups, 25 timed runs), no frequency locking; there is clear fluctuation between rounds; the numbers in the table are for comparing orders of magnitude, not latency promises.

## BLAS Backend Overview

The home-grown BLAS links no system BLAS or external matrix library, reuses the Transformer orchestration, and covers numerical hooks such as projections, attention, gating, and SwiGLU. Weights are pre-packed at load time; the current model keeps ordered per-K FP32 operations; at runtime microkernels are selected according to AVX2, SSE4.2, and scalar capabilities. Settings.threads controls each model's own fixed thread pool, supporting thread counts of 1, 2, 4, etc.; 0 is automatic, capped at 4, and the parameter is no longer ignored. Parallelism dispatches over complete output blocks, does not split the dot-product reduction, and supports paging, KV-only prefill, and shared prefixes. For current performance see [the benchmark data](<benchmark.md>); for the design see [the BLAS backend](<blas.md>).

## Trade-offs and Lessons

The difference between backends that is most easily underestimated is not speed but the accumulation order. As soon as the GEMM implementation changes, the results are no longer bit-for-bit equal to reference; at that point there must already be a set of tolerances, fixed in advance and given by numerical scale; otherwise it is easy to loosen the threshold on a whim after discovering the differences. This repository fixed the tolerances in phase one; all later backends carry them over; when a bound is exceeded the optimization is rolled back rather than the threshold changed; the BLAS narrow-kernel multi-chain interleaving is an example of such a rollback.

The second lesson is to let a backend do only what a backend should do. Eigen and ytensor each copied a model orchestration, while BLAS implements only the hooks; the former integrates quickly, but every change to a mask or to paging requires checking three places, while the latter concentrates the semantics in `utils`, with a smaller amount of work and a smaller error surface for a new backend. If a new backend is integrated in the future, the BLAS approach should be followed.
