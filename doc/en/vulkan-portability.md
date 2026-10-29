# Vulkan Portable Optimization and GPU Parallel Budget

## Goals and Boundaries

The final deployment targets the 13th- to 15th-generation, smaller-scale Intel iGPUs described by the user; 680M and 4060 are only development-validation devices, and their best results do not decide the Intel default policy. EU, Xe-core, AMD CU, and arithmetic lanes are not the same unit; a device nominating 768 arithmetic lanes does not mean 768 allocable Vulkan threads. Vulkan version compatibility also does not imply identical subgroup, workgroup, SLM, cache, or optional matrix extensions.

Decode drops the 1% MFU acceptance and instead looks at same-task steady-state latency, explicit copy, logical data-traffic estimation, and bandwidth. CPU/GPU, even sharing DRAM, have different caches and replication paths, so latency equality cannot be promised on that basis. The attribution tool has removed the hard-coded 3TFLOPS and MFU percentage; it keeps the required-task GFLOPs/s as a throughput description, not as a memory-task threshold.

## Verified Local References

The user-provided ggml snapshot is directly readable; the snapshot and checksum files belong to local working files and are not published with the repository. No remote commit is fixed, no whole kernel was copied, and no full repository review is claimed. The checksum file records the full SHA256 of the relevant files; the actual review scope is as follows:

| Path | Read scope and useful information | Handling in this project |
| --- | --- | --- |
| `mul_mm.comp` | Tiling/alignment definitions and main loop, BM/BN/BK, warp tile, shared load, and register accumulation | Later evaluation of FP32 panels, aligned vec4 loads and reuse; cannot directly bring in FP16/cooperative requirements |
| `mul_mat_vec_base.glsl`, `mul_mat_vec.comp` | Front-section definitions and reduction structure, row/column parallelism, vector access, subgroup/shared paths | K-parallel reduction must still pass the frozen budget; error is not relaxed for bandwidth |
| `flash_attn_decode_phase_1.comp` | Front-section staged attention and local reduction; this file not fully reviewed | Borrow the split/merge idea, without copying the matrix extension or a specific lane layout |
| `ggml-vulkan.cpp:1468–1478` | Shared overhead includes loads, row indices, coopmat stage, and LUT, filtered by the device limit | Looking only at tile data size is not enough; do not blindly enlarge tiles |
| `Xe decode parameters` | Xe1 branch parameter 8, other Xe paths 16, group size limited by q_len from 64 to 256 | This library's 32/64 fast subgroup kernels are no longer created when the device reports 8/16; the generic reduction is kept; lane width is not guessed from the CPU codename |
| `UMA creation`, `cached host buffer` | UMA prefers mappable device-local, borrowing is constrained by the actual allocation; the host cache has a fallback | Do not pass off a plain std::vector pointer as a GPU buffer; do not treat HostCached as the GPU L1 hit rate |

The online search found the [Intel SLM guide](<https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/shared-local-memory.html>), the [AI Max395-related AMDVLK capability issue](<https://github.com/GPUOpen-Drivers/AMDVLK/issues/420>), and the [RDNA Vulkan inference branch](<https://github.com/SelfRef/llama.cpp-rdna/blob/rdna/README.md>). web_fetch still returns a non-public-IP refusal for these domains; currently only search summaries exist, and their specific implementations or performance numbers are not treated as verified facts. RDNA-specific optimization is not the Intel default; what is truly verified is the above local code.

## Using gpuThreads

The configuration is added at the end of the [public Settings](<../../timestral/timestral.hpp>), read at construction, takes effect only on Vulkan, is ignored by other backends, and does not change CPU threads behavior. After extending the configuration struct, the application and library should be recompiled; the weight and Tensor layouts are unchanged.

~~~cpp
timestral::Timestral::Settings options;
options.threads = 1;
options.gpuThreads = 8192; // example upper bound, not a device recommendation

timestral::Timestral model("model/timestral-5m", options);
~~~

- 0: do not limit this library's original dispatch, the default
- Non-zero: a positive multiple of 256 and within the uint32 range; 256/512/768/4096/8192 are all legal, 384 and 257 are illegal; powers of two only are not the rule
- 256 is the implementation quantization of the currently required GEMM workgroup, not a warp width common to all Vulkan hardware
- Each kernel keeps its verified local_size; the user value is not written into local_size; a budget of 256 does not create a 512-thread attention pipeline and automatically takes the legal original path
- An illegal budget throws invalid_argument before the executor creates the Vulkan instance/device; the fixed LocalSize, the device per-axis/total invocations, and the variant sizes are also validated, avoiding zero stride or mixed modules

**This is a logical shader-invocation upper bound for a single serial dispatch window, not a physical EU/CU quota.** A 768 budget allows at most three workgroups for a 256-thread kernel and at most one for a 512-thread kernel; half a workgroup is not dispatched. The device's nominal 768 lanes cannot be used to explain 512 as 66% occupancy; there may be far more in-flight logical threads than arithmetic lanes. A very small budget severely lowers parallelism and increases command overhead.

The limit applies only to compute shaders issued by this model's executor; it does not limit other processes/models, driver CPU threads, driver-internal copy/migration, shared DRAM bandwidth, and it does not promise a utilization or power percentage. When a hard EU reservation or a fixed duty cycle is needed, this option cannot be treated as an implemented global hardware control. A legal parameter also does not constitute a guarantee that no driver will ever TDR/device-lost.

## Scheduling, Indexing, and ABI

The [executor](<../../timestral/vulkan/executor.cpp>) uses Vulkan1.1 dispatchBase, and the pipeline carries the DISPATCH_BASE flag. There are compute/transfer execution dependencies between windows, avoiding overlap between this library's next window and the previous one; full rows can be merged into a rectangle within the budget, without unconditionally serializing row by row. windowX × windowY × fixed localInvocations is always ≤ gpuThreads.

groupCount cannot simply be shrunk: the flattened kernels depend on gl_NumWorkGroups.x and would compute a wrong address for non-zero Y. The [offline generator](<../../timestral/vulkan/budgetShader.cmake>) generates 26 Budget modules from the 26 compute kernels, keeping the corresponding SSBO/push and floating-point operation order, and provides the flattened width via specialization constant 255. The original 21 GLSL and their ABI are not modified; new shared-prefix, in-group attention, and ordered-precision kernels are added, still abiVersion=1. The [complete manifest](<../../timestral/vulkan/shaderManifest.json>) totals 52 modules; the pre-compile/release directory must be fully updated.

The flattened kernels, per the existing contract, use NumWorkgroups only as the flattened stride, identified at runtime by the Builtin marker of the original module. The budget grid is normalized to a fixed W=min(device X group limit, 65535); the last row dispatches only the original coverage and a partial dispatch does not expand; native GEMM/GEMV/split attention keep the original 2D coordinates. Each operator caches one budget pipeline; no specialization/JIT accumulates with request length, and the pipeline is not released during recording or while in-flight commands reference it. First use may still incur pipeline JIT overhead; the fixed-protocol median does not represent the cold-JIT first request.

## Default Policy and Next Steps

The previous round's residual M upper bound 2048, narrowed on 680M, is withdrawn from the default rationale. The default of TIMESTRAL_VULKAN_RESIDENT_RESIDUAL changes to 0; explicit 1 is only an experiment, keeping the M≥128 small-dispatch amortization condition, and no longer treats the 680M large-M results as an Intel rule. Other model-geometry dispatch and capability legality are separated; no new vendor special case was created by GPU brand.

The current common shapes use an ordered attention with 64 threads and 8200B shared memory, at most 2048 keys, not depending on subgroup or FP64. The shared prefix is uploaded per channel group, and the GPU directly reads one prefix and the independent tail. The G1 projection acceleration is kept; the original A2 and other online-reduction kernels are kept as candidates or out-of-range fallback; the historical kernel selection must not be treated as this round's default dispatch. The numeric range and long-query limits are in the [benchmark protocol](<benchmark.md>).

The buffer physical capacity grows geometrically; descriptors, copy, and download use only logical bytes. Idle device/staging/readback pools are recycled after the fence completes, limited by the device allocation count, avoiding thousands of allocations from ARs that grow per patch.

Follow-up priorities:

1. **Low-SLM attention and the 8/16-wide path**: the existing 512-thread/~39KiB path is not suitable for all small iGPUs; new generic small-block candidates and a frozen budget are needed, not a direct replacement with RDNA-specific kernels
2. **GEMM access**: first do aligned vector loads/panel reuse that do not reorder the K accumulation, then look at the M/N traversal order; L1/L2, SLM banks, and register pressure need verification with target-Intel hardware tools; HostCached or Vulkan timestamps cannot be passed off as a hit rate
3. **Pure copy and KV path**: under a strict budget, mergeKv's many small windows are costly; the range/alias-checked transfer copy and segmented prefix are already used for the current query; the prefix is still assembled and uploaded per layer from the CPU page. Pool identity, generation, modify/eviction invalidation must be defined later before a cross-call KV mirror; zero-copy or device KV residency is not implemented
4. **Decode acceptance**: same-task CPU/GPU latency, copy payload, and logical read/write estimation; the copy payload is not the actual DRAM traffic; real bandwidth is also affected by weight/KV cache hits and extra copies; no 1% MFU threshold is set

## This Round's Verification

- The 42-module ABI reflection passes, including the budget constant 255 detection; original and Budget modules cannot be mixed
- Dual-GPU validation and operator comparison pass: illegal multiple/uint32 overflow, 256/512/768/4096, native rectangle, flat2D, row2D, exceeding 65535 groups, partial-coverage tail not rewritten, 512-thread attention; the same kernel is compared bit-level
- The public Settings path is bit-identical between default and 256 prediction; illegal 1 is rejected; dual-GPU original Tiny/Real 2/2; the 256 and 4096 budget real-model comparisons include continuous coordinates, 129 patches, H1 prefix 63/513, and read-only page hashes
- Pure standard library with OpenMP etc. disabled 24/24 (77.38s); the BLAS factory incremental build passes. The later normalization/rectangle window only changed the GPU-internal indexing/scheduling; the unrelated CPU full run was not repeated
- The 8-channel public output budget passes, but the `--strict-multi` old-path intermediate-value deviation is still not located; it is not written as all 8 channels passing per layer
- `baseline and model`, `standard library`, `normalized grid`, `rectangle window` keep the complete logs
- No Intel real machine yet; the above are AMD/NVIDIA development validation, not Intel performance certification. The optimal budget was not chosen by results

## Budget Overhead Observation

`final observation` selects only 0/512/8192 to show the cost of strict and looser logical budgets; real 5M single channel 32k, prefill cold-page median of 3, decode warmup 5 / timed 25, experimental residual off, not overlapping with the build. This is not auto-tuning or an Intel speed prediction. The timing excludes model loading; the cold JIT is not separately timed; profiling separately takes 5 for window/traffic attribution, not replacing the latency baseline.

| Development device | Logical budget | prefill ms | H1 decode ms |
| --- | ---: | ---: | ---: |
| 680M | 0 | 44.80 | 9.06 |
| 680M | 512 | 461.64 | 21.55 |
| 680M | 8192 | 79.63 | 9.88 |
| 4060 | 0 | 27.40 | 5.59 |
| 4060 | 512 | 291.72 | 24.09 |
| 4060 | 8192 | 39.40 | 6.88 |

The 512 budget is very strict; the cost is far from "using only one third less of the GPU"; 8192 is only a looser upper bound, not a recommendation. The window statistics measure a maximum not exceeding the setting, but the real GPU occupancy/frequency/power was not measured. mergeKv's pure-copy shader forms many small windows under a strict budget; later priority is the range-safe transfer path, without treating these numbers as Intel tuning conclusions.
