# Vulkan GPU Backend

Phase five delivers the ten-operator SPIR-V library and the device-resident model backend; phase six adds FP32 optimized kernels without changing the v1 bindings or push layouts. Vulkan is the first GPU API integrated as a selectable backend in the repository: with TIMESTRAL_USE_VULKAN=ON, loadBackendModel returns a GPU model, weights stay resident on the device, and the layer graph executes in whole batches; the default build links no Vulkan code, and the public API, the Tensor layout, and the CPU backend selection are unchanged. The GPU model keeps only Vulkan; the OpenGL and OpenCL experimental components were deleted at the user's request; see [the trade-off records](<gpuScope.md>) for the migration gaps and the historical boundary.

## Two Build Entries

| Option | Effect |
| --- | --- |
| TIMESTRAL_USE_VULKAN | the selected model backend becomes the Vulkan resident model; mutually exclusive with Eigen, ytensor, and BLAS; enabling any two at the same time errors at the configure stage |
| TIMESTRAL_BUILD_VULKAN_KERNELS | builds only the standalone operator verification component timestralVulkanKernels; does not provide a model backend |
| TIMESTRAL_VULKAN_SPV_DIR | when pointing at a complete 52-module Vulkan 1.1 SPIR-V directory, no shader compiler is needed; missing regular/budget modules or invalid headers error at the configure stage |
| TIMESTRAL_VULKAN_TEST_DEVICE | the Vulkan enumerated device number used by the operator tests, default 0 |
| TIMESTRAL_VULKAN_TEST_VALIDATION | the operator tests require the Khronos validation layers; failing when not installed rather than skipping |
| TIMESTRAL_BUILD_GPU_BENCHMARKS | builds vulkanBench, including the transfer mode by default; the resident/attention modes time the resident recording and submission; default ON |

Model backend build tree (needs the Vulkan SDK headers and the loader):

~~~powershell
cmake -S . -B build-gpu-vulkan -A x64 -DTIMESTRAL_USE_VULKAN=ON "-DVulkan_INCLUDE_DIR=$env:VULKAN_SDK/Include" "-DVulkan_LIBRARY=$env:VULKAN_SDK/Lib/vulkan-1.lib"
cmake --build build-gpu-vulkan --config Release --parallel
ctest --test-dir build-gpu-vulkan -C Release --output-on-failure
~~~

For the general runtime configuration see [runtime configuration](<../../timestral/utils/gpuSettings.hpp>); the Vulkan model additionally provides the optimization fallback switches:

| Variable | Default | Description |
| --- | --- | --- |
| TIMESTRAL_GPU_SHADER_DIR | the shader directory baked in at build time | overrides the pre-compiled spv location |
| TIMESTRAL_GPU_DEVICE | 0 | the Vulkan enumerated device number, strict decimal |
| TIMESTRAL_GPU_VALIDATION | 0 | when 1, requires the Khronos validation layers; failing when not installed |
| TIMESTRAL_VULKAN_OPTIMIZATIONS | 1 | 1 enables the accepted projection acceleration, 0 uses the generic projection without fusion; illegal values are rejected. Both share the ordered precision path, the cached readback, and the KV-only orchestration; 0 is not a rollback to the old source |
| TIMESTRAL_VULKAN_RESIDENT_RESIDUAL | 0 | residual fusion is an explicit experiment only, M≥128; the large-M upper bound exported at 680M is revoked; no Intel default policy was made on that basis; illegal values are rejected |

The device enumeration order does not guarantee a discrete GPU first, and may change after a driver update; the tests and benchmarks print the actual device name. This round's measured device 0 is the AMD Radeon(TM) Graphics (API 1.3.217), device 1 is the NVIDIA GeForce RTX 4060 Laptop GPU (API 1.4.325).

## GPU Parallel Budget

A new gpuThreads was added to the [public Settings](<../../timestral/timestral.hpp>): 0 means unlimited dispatch by this library; non-zero must be a multiple of 256 / in u32 range; it only takes effect on Vulkan. It bounds the logical invocations of each serial dispatch window, not physical EU/CU, the nominal 768 channels, or a GPU utilization percentage; a very small budget may be noticeably slower. The fixed local_size does not change with the user value; illegal multiples are rejected before the GPU object is created; a budget of 256 does not select a 512-thread optional pipeline.

~~~cpp
timestral::Timestral::Settings options;
options.gpuThreads = 8192; // an example budget, not a recommended value for some GPU
timestral::Timestral model("model/timestral-5m", options);
~~~

For the detailed integer constraints, the canonical grid/rectangular window, the native 2D, the exact tail row, the first pipeline JIT, and the Intel verification gap, see [the portability design](<vulkan-portability.md>). Other processes/models, driver copies, and shared DRAM bandwidth are not covered by this budget guarantee; one cannot promise to keep certain EUs or never to go device-lost on this basis.

## Operator Library and ABI

The [source directory](<../../timestral/vulkan/shaders/>) contains 26 compute kernels and the shared exact-math helpers; offline, each generates its own Budget variant; the complete package has 52 modules. The bindings, push, and localSize of the original 21 kernels are unchanged; the 5 new kernels are the shared prefix, the intra-group attention, and the ordered precision path. The Budget variants keep the ABI of the corresponding original kernel. The subgroup fast kernels are created only when size32/64 is reported and the capability is met, not forced onto Intel 8/16.

| Operator | Computation | Workgroup | push |
| --- | --- | --- | --- |
| [gemm](<../../timestral/vulkan/shaders/gemm.comp>) | A[M,K] times transposed W[N,K], optional bias | 16×16 shared-K block | 16 bytes |
| [rmsNorm](<../../timestral/vulkan/shaders/rmsNorm.comp>) | per-row sum-of-squares reduction, (x*inverse)*scale | 64 threads per row | 12 bytes |
| [softmax](<../../timestral/vulkan/shaders/softmax.comp>) | per-row max and sum-of-exponents reduction, negative-infinity masking | 64 threads per row | 8 bytes |
| [silu](<../../timestral/vulkan/shaders/silu.comp>) | x/(1+exp(-x)) | 64 threads, tail check | 4 bytes |
| [prepareQkv](<../../timestral/vulkan/shaders/prepareQkv.comp>) | fused QKV reshuffled by head plus partial RoPE | 64 threads | 20 bytes |
| [mergeKv](<../../timestral/vulkan/shaders/mergeKv.comp>) | prefix K/V and new K/V concatenated by row | 64 threads | 16 bytes |
| [elementwise](<../../timestral/vulkan/shaders/elementwise.comp>) | residual add, sigmoid gating, SwiGLU in three modes | 64 threads | 12 bytes |
| [attentionScores](<../../timestral/vulkan/shaders/attentionScores.comp>) | dual-stream masked QK scores and online softmax | 64 threads | 36 bytes |
| [attentionValues](<../../timestral/vulkan/shaders/attentionValues.comp>) | masked PV reduction | 64 threads | 36 bytes |
| [attentionFused](<../../timestral/vulkan/shaders/attentionFused.comp>) | scores and PV completed within the same workgroup, cross-warp shared reduction | 64 threads | 36 bytes |
| [gemmTile](<../../timestral/vulkan/shaders/gemmTile.comp>) | 32×32 output tile, 2×2 register blocking, merged W load, per-K precise | 16×16 threads, 4160B shared | 16 bytes |
| [attentionTile](<../../timestral/vulkan/shaders/attentionTile.comp>) | time-axis 8-query shared K/V, 64-key online softmax | 64×8 threads, 39168B shared, falls back to v1 when the device does not meet it | 36 bytes |
| [attentionRegister](<../../timestral/vulkan/shaders/attentionRegister.comp>) | headDim64 register PV, diagnosis only | 64 threads | 36 bytes |
| [gemv](<../../timestral/vulkan/shaders/gemm.comp>) | narrow-M per-K exact projection, weight loads merged by K | 64 threads, 16 columns/group | 16 bytes |
| [attentionCausal](<../../timestral/vulkan/shaders/attentionCausal.comp>) | only trims the encoding future key blocks provably fully empty | 64×8, 39168B shared | 36 bytes |
| [attentionSubgroup](<../../timestral/vulkan/shaders/attentionSubgroup.comp>) | subgroup attention with a query alignment check and the original reduction fallback | 64×8, 39300B shared, Basic/Vote/Arithmetic | 36 bytes |
| [attentionDecode](<../../timestral/vulkan/shaders/attentionDecode.comp>) | H1 long KV sharded by 128 keys, keeping max/denominator/PV | 64 threads, 17408B shared | 40 bytes |
| [attentionDecodeMerge](<../../timestral/vulkan/shaders/attentionDecodeMerge.comp>) | stable merge of the shards, not averaging the locally normalized results | 64 threads | 12 bytes |
| [siluGrid](<../../timestral/vulkan/shaders/siluGrid.comp>) | SiLU supporting 2D grid expansion, for the residual fusion | 64 threads | 4 bytes |
| [gemmWide](<../../timestral/vulkan/shaders/gemmWide.comp>) | 64×64 output, 4×4 registers, diagnosis only | 16×16, 8256B shared | 16 bytes |
| [attentionPacked](<../../timestral/vulkan/shaders/attentionPacked.comp>) | four-score/four-PV register candidate, diagnosis only | 16×8, 37632B shared | 36 bytes |

[shaderManifest.json](<../../timestral/vulkan/shaderManifest.json>) is the machine-readable ABI with abiVersion=1: target environment vulkan1.1, entry main/GLCompute, each shader's localSize, binding numbers and access directions, push constant member names, types and offsets, minimum device limits, and the 2D dispatch rules. A resident large-grid linear kernel expands with x=min(G,65535), y=ceil(G/x), with element indices flattened by y first; GEMM uses a native 2D grid. The old silu verification kernel only looked at X; the large-grid resident path uses siluGrid; the old kernel was not directly changed to 2D dispatch.

The local SDK 1.3.283's glslc writes the SPIR-V 1.3 version field as 0x00010300; the same SDK's spirv-dis parses it as 1.3 and spirv-val passes. At build time AUTO prefers glslc, falling back to glslangValidator (or glslang) when missing; the former uses --target-env=vulkan1.1, the latter -V --target-env vulkan1.1. The automatic detection result is not forced back into the user override items; when spirv-val can be found the binary is validated against the same target environment; the inference process does not compile GLSL.

The [ABI reflection test](<../../tests/vulkanShaderAbiTest.cpp>) parses the compiled artifacts without relying on a GPU: the entry point is unique and is main/GLCompute, LocalSize matches the manifest, the descriptor set and binding set of the storage buffers are exactly equal, the push structure member offsets/types/total length match, and four kinds of corrupted modules are rejected: bad magic, truncation inside an instruction, zero wordCount, and out-of-range wordCount. It turns the drift between the manifest and the spv into a build-time failure, no longer relying on a runtime silent mismatch.

## The Resident Executor

[GpuExecutor](<../../timestral/utils/gpuExecutor.hpp>) is the backend-agnostic execution contract: begin, upload (can be marked persistent), allocate, dispatch, download, end. The implementation of the [Vulkan executor](<../../timestral/vulkan/executor.cpp>):

- weights are uploaded once as persistent when the model is constructed; staging uses a host-visible coherent buffer; staging and the device workspace each have one high-water-mark reuse pool
- the whole layer graph of one runLayers is recorded into a primary command buffer; a visibility barrier is inserted before each dispatch, and it is submitted at the first readback; later K/V page readbacks are still submitted/waited serially; no centralized readback was implemented
- in-batch handles become invalid at end; using them across batches throws out_of_range; one cannot reuse the previous batch's scratch addresses

Exception semantics: submit marks the batch poisoned and keeps all resources when the fence creation, queue.submit, or the wait fails; end() drains with a device waitIdle before releasing; later begin() explicitly rejects rather than re-entering while sick. The cleanup helper of [GpuTransformer](<../../timestral/utils/gpuTransformer.cpp>) calls end() only when an exception is already propagating; a cleanup failure does not overwrite the original device error. This semantics fixed a wrong-masking found in a review: the old path, after submit threw, would end an unfinished command buffer again, swallowing the original error and stalling subsequent batches.

## The Model Backend

[GpuTransformer](<../../timestral/utils/gpuTransformer.hpp>) inherits from Transformer, overriding the layer graph and the numerical hooks. The residual block device residency was changed to an explicit experiment; by default the three projections still go through the CPU; post-processing, scheduling, and page management are still on the host. It follows the time/variate dual-stream predicates; RoPE is precomputed by the host with the same libm as reference, only exactly reusing this call's inverse frequencies and completely identical position rows. A pure KV prefill stops after the last time K/V is ready, but a complete forward, trace, and decode are not truncated. The prefix K/V is still assembled/uploaded from the CPU page pool and concatenated by mergeKv; there is no device KV mirror.

Test matrix:

| Test | Coverage |
| --- | --- |
| gpuModelTiny / gpuModelReal | the miniature and the real 5M against an independent reference: per-layer hidden states max(1e-5,1e-4×RMS) plus relL2≤1e-5, output quantiles 1e-4+1e-3×\|ref\|, consistency of the cached and the full, rows 1/2/3, group isolation, roles, missing keys, continuous coordinates, cold/append/shared pages, page hash unchanged |
| vulkanExecutor | G1 twelve shapes × bias bit-for-bit; GEMV ten shapes × bias frozen budget; causal bit-for-bit, subgroup/packed frozen budget; split-K empty shard/tail shard/role; siluGrid exceeding 65535 groups; the original attention seed and lifecycle tests |
| vulkanKernels | four operators, 30 GEMM shape groups, tail blocks, boundaries, concurrency, corrupted spv rejection |
| vulkanShaderAbi | the ABI reflection comparison between the manifest and the spv plus the four corruption rejections |
| time/condition/multivariate Real, sharedDecode | the real 5M regression and the shared paged decode go through the GPU backend |

The 8-channel default only adds the public forward/predictBlock output budget; --strict-multi keeps the strict per-layer diagnostics; currently the v1 kernel selection also fails there, not yet localized. The 63/513 patch H1 cross-residual dispatch upper/lower bounds, the 129 patch KnownCov, and the read-only page hash have passed on both GPUs; the 8-channel output passing is not written as all 8-channel per-layer passing.

## Performance Accounting and the Historical Window

The final deployment targets Intel integrated graphics; the 680M/4060 are only for verification. decode is no longer accepted by 1% MFU; same-task latency, copy, and bandwidth are used; for the latest budget costs and uncovered hardware see [the current design](<vulkan-portability.md>). The following data such as 41.47/8.48 ms belong to the earlier historical strategy with the residual M window enabled; they are not the current defaults, and no Intel latency is inferred from them.

32768 sample points (1024 patches) cold prefill and H=1 decode, backendCompare protocol (local tool, not checked in), kind=paged, encode median of 3, decode warmup 5 timed 25 median, the machine with no other builds or GPU tasks:

| Device | encode ms | decode ms | checksum |
| --- | ---: | ---: | --- |
| RTX4060 Laptop (device 1) | 120.8538 | 7.7143 | 2.3805 |
| AMD iGPU (device 0) | 236.9136 | 11.8549 | 2.3806 |

The same round's CPU historical re-measurement (see the local acceptance records): blas-avx2 181.15/4.39 ms, reference 4769.09/8.16 ms, checksum both 2.3806. This can only state the end-to-end behavior of that window; it cannot prove sufficient GPU utilization. The old text applied the complete-forward formula to the paged prefill and discussed GPU efficiency with the CPU's 134.4 GF/s denominator; that accounting is now withdrawn; the unattributed judgment that H=1 is "dominated by submission and orchestration" is also withdrawn. The 680M was recomputed at the user-given 3 TFLOPS and the actual execution scope; the analysis is in [the 680M dedicated attribution](<vulkan-opt.md>); the CPU nominal MFU is still used for the CPU only.

The table above is the phase-five historical baseline and does not represent the current optimized version. In the same call, rotary and metadata are uploaded once outside the layer loop; the old text's judgment of "precompute/upload per layer" has been corrected. Currently it still uses an independent descriptor pool per dispatch, serial page readback, and a scratch pool that only keeps the high water mark without a budget.

After the phase-six redesign, the G1 projection and the A2 shared K/V attention were accepted; the last same-window v1 → optimized version: 4060 encode 125.9646→109.8991 ms, decode 8.0188→6.7479 ms; AMD encode 263.0007→143.5030 ms, decode 12.9188→10.5634 ms; the first output value is still 2.3805/2.3806. The AMD fluctuation across rounds is large; for all rounds, the operator microbenchmarks, the unaccepted pool reuse, and the empty-block trimming see [the optimization tree](<vulkan-opt.md>); no fixed multiple is promised.

The current kernel selection is concentrated in the Vulkan Executor; the orchestration only adds the verified KV-only stop and the M-window residual fusion. G1 covers the tested embedding shapes; M1 to 8 can use the per-K GEMV; long time attention picks causal/subgroup by capability; H1 long KV can split-K. FP16, projection FMA, and projection K parallel reduction are not enabled; the arena was not restored; the attention reduction reordering was accepted against the original numerical budget.

## 680M Readback Correction and Diagnostics

The 680M dedicated timestamps and CPU timing found: the old readback used Visible+Coherent memory without Cached; the CPU map/read/unmap of 17.25 MiB took about 77.53 ms, while the GPU-executed copy took about 2 ms. The current readback's independent reuse pool prefers HOST_CACHED|HOST_COHERENT, keeping the existing coherent fallback when that type is absent; the upload staging is unchanged, and no fence or visibility barrier was removed. After the correction the corresponding CPU read is about 1.52 ms.

The above is the historical attribution at the time of the cached correction; the old window encode 85 to 88 ms, 12.260475 G counted per the pre-trim figure. The current MFU fixes the necessary task amount at 10.648682496 G, not counting the useless trailing computations later deleted as output. For the latest numbers, the per-round fluctuation, and the unmet items see [the optimization tree](<vulkan-opt.md>).

The vulkanProfile attribution tool (local tool, not checked in) is built only when TIMESTRAL_USE_VULKAN and the GPU benchmarks are enabled; it does not change the default model behavior. Usage:

~~~powershell
.\build-gpu-vulkan\timestral\vulkan\Release\vulkanProfile.exe build-gpu-vulkan/timestral/vulkan/shaders model/timestral-5m 0 1 1 0 8192
~~~

The last five parameters are the device number, diagnostics 0/1, optimizations 0/1, experimental residual 0/1, and the GPU budget; the last two default to 0. The tool reports the host API, the GPU span, the copy, the actual window count, and the maximum window invocations; the hardcoded 3 TFLOPS/MFU is removed; throughput is described as necessary-task GF/s. The diagnostic decode has only 5 runs; the CPU wait and the GPU time are not added; latency uses the fixed protocol with diagnostics off. The KV mirror is still not implemented.

## Known Limitations

- each dispatch binds one buffer to a single SSBO, constrained by maxStorageBufferRange; the 32k path of timestral-5m is far below the typical 128 MiB limit; larger models or larger batches may hit the ceiling
- the scratch reuse pool is only kept at the high water mark, neither reclaimed nor memory-budgeted
- the residual fusion is off by default, an explicit experiment only; the Intel default policy is not decided by the 680M M2048 upper bound
- FP32 only; FP16/cooperative matrix is not required; the subgroup is an optional optimization, falling back when the capability is missing
- the small-shape operators with transfer are slower than the CPU, see the section above; the resident batch submission only removed the repeated synchronization between layers, not the allocation and submission overhead inside a single dispatch

## The SPIR-V Cross-API Boundary

This library only releases Shader SPIR-V targeting Vulkan 1.1 and does not promise it can be consumed by other APIs. The historical OpenGL experiment lacked attentionFused and a cross-platform context; the OpenCL experiment only measured four operators, the resident model was not accepted, the device IL version was empty, and the AMD queue had failed; it cannot be migrated by changing the extension. Both components are now deleted, with no enabled entry kept; see [GPU backend trade-offs](<gpuScope.md>).

## Binary Release

~~~powershell
cmake --install build-gpu-vulkan --config Release --component VulkanShaders --prefix .agents/data/vulkan-package
~~~

The install component puts the 26 regular and 26 Budget spv files, 52 in total, [shaderManifest.json](<../../timestral/vulkan/shaderManifest.json>), and [gpuKernels.md](<../../timestral/utils/gpuKernels.md>) into share/timestral/vulkan. Pointing TIMESTRAL_VULKAN_SPV_DIR at that directory allows an offline build, without looking for glslc or glslangValidator. For explicitly selecting a tool, set the TIMESTRAL_VULKAN_SHADER_COMPILER path and TIMESTRAL_VULKAN_SHADER_COMPILER_KIND=AUTO/GLSLC/GLSLANGVALIDATOR. The pre-compiled inputs must be trusted, validated files whose descriptor and push layouts match; at runtime only a basic SPIR-V header check is done, and the library is not responsible for proving the safety of arbitrary external shaders.
