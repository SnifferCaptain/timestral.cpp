# GPU Backend Scope

## Current Decision

The current complete GPU model backend is Vulkan only. OpenGL and OpenCL are both ❌, removed at user request, and are no longer optional components or future migration commitments. Past operator experiments do not equal a delivered complete backend; historical observations are retained in this document and are not rewritten as current support.

## Why This Is Not a Small Migration

| Item | OpenGL legacy component | OpenCL legacy component |
| --- | --- | --- |
| Measured boundary | Windows WGL, nine basic kernels and partial resident chain | Four standalone OpenCL C 1.2 operators, saved resident executor never accepted |
| Current shared model gap | Missing attentionFused, incomplete with existing model dispatch contract | Similarly missing attentionFused and subsequent resident kernels, saved factory not wired into build |
| Execution environment | OpenGL4.5 target / SPIR-V 1.0, UBO/SSBO and context binding, not Vulkan target binary | Kernel execution environment differs from Vulkan Shader execution environment, cannot directly reuse GLSL-generated Vulkan SPIR-V |
| Driver facts | Only AMD WGL path has legacy experiments, no Linux/EGL or complete 5M model acceptance delivered | Both devices had empty IL version strings at the time; AMD queue creation failed, cannot claim portability just because loader has IL entry points |
| Work still needed | New critical kernels, platform context, resource/sync adaptation and complete model/cache/quantile numeric and performance acceptance | New critical kernels, compile/IL path, resource/sync/driver issues and equivalent complete acceptance |

The shared Transformer and some operator math can be reused, but simply changing API names or a set of shader compile parameters does not yield a complete backend. Reverting to separated attention operators to dodge the gap, silent CPU fallback, or claiming whole-model support after running only four operators are all not delivery options.

## Removed Scope

- All host, loader, executor, factory drafts, shader/embedded kernels, directory docs and CMake from both experimental component directories
- Two dedicated operator tests, without touching other backend tests or frozen budgets
- Two optional component entries in the root build and the OpenCL experimental executor entry
- The OpenCL static component build task in the Linux workflow
- Links and enablement instructions in formal documentation pointing to removed components

No separate formal "experimental backend" directory was created, and no empty factories or pseudo-implementations are retained. Old build caches that explicitly enable removed options will fail at configure time with a message to use Vulkan, rather than silently ignoring the old option and letting the user believe that API is still selected.

Vulkan and shared GPU contracts, the four CPU backends, and original model/measurement data are not removed. Performance charts are not re-filled due to component scope changes. The current support matrix is in the [main README](<../../README.md#platforms-and-backends>); known numeric stress failures for existing GPUs remain in local numeric records.
