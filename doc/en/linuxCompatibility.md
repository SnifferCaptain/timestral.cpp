# Linux Compatibility Fixes and the Verification Boundary

The test counts of this document record the compatibility fixes of that time. The user later decided to remove the OpenGL/OpenCL components; the related code, build, and CI tasks are now deleted; the current platform status follows [the main description matrix](<../../README.md#platforms-and-backends>), and for the trade-offs see [GPU backend scope](<gpuScope.md>).

This round corrected the confirmable build problems item by item according to the audit report. The default deployment is still C++17, no new runtime third-party dependencies, and the forecast algorithm, weights, default decode step size, and frozen error budget are not changed. The current host is Windows/MSVC, with no available Linux distribution, GCC, or Clang; the local results in the table below cannot be written as Linux measurements passing.

## Fix Comparison

| Reported problem | Current handling | Evidence entry |
| --- | --- | --- |
| Nested Settings default arguments block GCC/Clang | Removed the in-class default arguments, added a single-parameter delegating constructor and a loadEngine overload; Settings is still an aggregate, with no user-defined constructor | [public declaration](<../../timestral/timestral.hpp>), [default loading](<../../timestral/utils/engine.cpp>) |
| Threads appear only in the test/BLAS branches | The root configuration discovers Threads unconditionally; timestral and the Vulkan kernels each link Threads::Threads PUBLIC; the old OpenCL component has been deleted | [root configuration](<../../CMakeLists.txt>), [standalone downstream test](<../../tests/embedding/CMakeLists.txt>) |
| ytensor source file COMPILE_FEATURES did not actually raise the standard | A separate C++20 OBJECT target is used; the core and factory stay C++17; CI asserts the actual compile_commands | [build configuration](<../../CMakeLists.txt>), [Linux CI](<../../.github/workflows/linux.yml>) |
| Whole-unit AVX2/FMA of ytensor is an illegal instruction on old CPUs | Only the model object enables AVX2/FMA; the baseline factory first checks CPUID, OSXSAVE, and the XMM/YMM state and explicitly rejects when not satisfied; IPO/unity is disabled on both sides to avoid optimization crossing the safety boundary | [CPU check](<../../timestral/utils/cpu.cpp>), [factory](<../../timestral/ytensor/backend.cpp>) |
| Vulkan only looks for glslc | AUTO prefers glslc, falls back to glslangValidator/glslang when not found, supports explicit path and kind override; both command sets target Vulkan 1.1 | [compiler selection](<../../timestral/vulkan/shaderCompiler.cmake>) |
| Shader compiler -NOTFOUND cache pollution | The user override items and the automatic detection results are kept separate; the automatic result is not forced back into the user item; tests cover missing, later appearing, and forced kind switching | [reproducible selection test](<../../tests/shaderCompilerTest.cmake>) |
| Offline SPIR-V incomplete/mismatched | Each of the 26 regular + 26 budget modules is checked, the SPIR-V magic is checked, and if spirv-val exists it validates at configure time; a complete offline directory does not look for a compiler | [Vulkan build](<../../timestral/vulkan/CMakeLists.txt>) |
| C++20 char8_t path concatenation failure | UTF-8 bytes are extracted uniformly, without narrowing via the local code page; the implementation and the test fixture share the conversion; the raw UTF-8 JSON test no longer treats char8_t as char | [path conversion](<../../timestral/utils/path.hpp>), [dual-standard regression](<../../tests/portabilityTest.cpp>) |
| config.cpp range-loop temporary string warning | Literal traversal uses const char*; error concatenation uses an explicit std::string; the message meaning is unchanged | [config parsing](<../../timestral/utils/config.cpp>) |
| OpenGL/OpenCL platform boundary unclear | Later evaluation showed a complete migration was not low-cost; per the user's decision the experimental code and the build/CI entries were deleted, not kept as backends | [trade-off basis](<gpuScope.md>) |

Choosing ytensor still requires x86 AVX2+FMA and OS support. Non-x86 is rejected at configure time; old x86 throws an explainable exception before entering the optimized model, suggesting the user choose reference or Eigen; there is no silent backend fallback masquerading as ytensor. BLAS's existing multi-ISA runtime dispatch is unchanged.

## Local Actual Verification

| Configuration/test | Result and scope |
| --- | --- |
| reference full suite | 29/29 passed |
| BLAS full suite | 39/39 passed |
| Eigen full suite | 30/30 passed |
| ytensor build and new compatibility cases | build passed; the C++17/C++20 headers, aggregate initialization, UTF-8 paths, CPU capability bits, and shader selection tests passed |
| ytensor full suite | 30/31, parallelDecodeReal still has the frozen numerical budget failing, not written as all green |
| the whole core built with C++20 | build passed; the dual-standard public header, JSON, and safetensors targeted regressions passed |
| standalone consumer with tests/examples off | links only timestral, verifying the transitive thread dependency; compile/link/run passed |
| glslc | 52-module generation and spirv-val, 52 shader ABI items, and real dispatch of the four small operators passed |
| glslangValidator | the same 52 modules, validation/ABI, and low-load real dispatch passed |
| offline SPIR-V | the complete 52-module directory still builds and passes the ABI even when an invalid compiler path is deliberately provided; missing regular/budget modules are rejected at configure time |
| AUTO/cache contract | no Vulkan device needed; missing fallback, NOTFOUND recovery, explicit kind switching, and illegal arguments passed, reproducible |

The real Vulkan dispatch here comes from the local AMD Radeon(TM) Graphics of device 0, not a Linux llvmpipe or Intel integrated graphics result. Only 2×2 linear/rmsNorm and 3-element softmax/silu are called; no GPU stress test or complete model benchmark was run.

### The Numerical Failure ytensor Still Keeps

The failing position is single-frame/cached-full index 13712: actual=2.294824, expected=2.294836, original limit about 0.000011. That budget was not loosened, no assertion was deleted, and no case was excluded from CTest. To separate build isolation from the pre-existing numerical problem, this round only, in a temporary archived copy, swapped the new model object back to the original ytensor model object saved on October 25; the same regression still fails at the same index with the same actual/expected/limit. The original library was not replaced, and the formal backend was not changed to Naive/reference.

Therefore, this round can state that the build blocks in the report have been handled, but it cannot state that all ytensor real numerical scenarios pass. The previously failing scope of other long forecasts, extra GPU shapes, and device 1 does not disappear because of this round's fixes either; see the local numerical acceptance records.

## Regressions Pending on Linux

The [Linux workflow](<../../.github/workflows/linux.yml>) now pins GCC11 and Clang14 on Ubuntu 22.04, covering reference, full-core C++20, BLAS, Eigen, and ytensor; the ytensor actual commands must show -std=c++20/-mavx2/-mfma only in the model unit, and the baseline factory/CPU check must not carry AVX2. In addition:

- a QEMU Nehalem process actually runs [the no-AVX2 guard test](<../../tests/ytensorGuardTest.cpp>), not merely simulating the CPUID bits
- compile/link of the standalone consumer with tests off, not relying on the test targets to supply pthread
- the RISC-V cross build must actually select Eigen; the old command that masked the failure is removed
- glslc/glslangValidator each build 52 modules and validate with spirv-val; using the llvmpipe software ICD, run only the ABI and a small smoke
- the complete offline directory must not depend on a compiler; an incomplete directory must fail
- the original OpenCL static-component build task was removed with the component and is no longer in CI or the support plan

The workflow syntax was parsed locally, but it has not yet been triggered on remote CI; "new CI" was not used as a substitute for Linux results.

~~~bash
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-11
cmake --build build-linux --parallel 2
ctest --test-dir build-linux --output-on-failure --parallel 2

cmake -S . -B build-vk -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_VULKAN_KERNELS=ON -DTIMESTRAL_VULKAN_SHADER_COMPILER_KIND=GLSLANGVALIDATOR
cmake --build build-vk --target vulkanShaderAbiTest vulkanSmokeTest --parallel 2
ctest --test-dir build-vk -R '^(vulkanShaderAbi|vulkanSmoke)$' --output-on-failure
~~~

The smoke and the static library must be built from the same CMake configuration. Some types of Vulkan-Hpp are affected by NDEBUG; Debug headers/objects and the Release static library must not be mixed; the current fixes do not promise such cross-configuration ABI compatibility, nor use hand-mismatched wrappers to verify the GPU.

This round did not re-measure performance. The existing figures and CSVs keep the program/environment records of the original measurements; the post-fix binary SHA or latency cannot be back-filled into the old results.
