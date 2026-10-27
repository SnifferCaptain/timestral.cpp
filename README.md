# Timestral C++

[中文](<README_zh.md>) · English

A local C++ inference library for the [Timestral](<https://github.com/mistralai/timestral>) time-series model: give it timestamped numeric series and it returns the median, quantiles and prediction intervals for future steps.

- The default build uses only the C++17 standard library, without Python or PyTorch
- Joint multivariate forecasting with missing values and conditional completion
- Block autoregressive decoding with a content-addressed KV cache
- Optional backends: built-in BLAS, Eigen, ytensor, Vulkan, one per build
- Models load from local files and are never downloaded

## Build

CMake 3.16 or newer, with a C++17 compiler.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
~~~

The default build produces only the `timestral` library, without tests or examples. Tests are local regression tools not shipped with the repository; add `-DTIMESTRAL_BUILD_TESTS=ON` at configure time to build them and run `ctest --test-dir build -C Release --output-on-failure`.

## Usage

Put config.json and model.safetensors from the same release in one model directory; checksums are in the local checkpoint list (not shipped with the repository). Link the `timestral` target and include the [entry header](<timestral/timestral.hpp>).

~~~cpp
#include "timestral/timestral.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    timestral::Timestral model{"model/timestral-5m"};

    std::vector<timestral::Sample> history;
    for (std::size_t t = 0; t < 128; ++t) {
        history.push_back({static_cast<double>(t) * 0.01, std::sin(static_cast<double>(t) * 0.05)});
    }

    timestral::Timestral::Request request;
    request.variables = {history};
    request.sampleInterval = 0.01;
    request.horizonSeconds = 0.32;

    const auto result = model.predict(request);
    for (std::size_t t = 0; t < result.steps(); ++t) {
        std::cout << result.time(t) << ',' << result.median(t) << '\n';
    }
}
~~~

Each channel is a `vector<Sample>` with a time in seconds and a value; NaN marks a missing reading. Put channels in one Request to forecast them jointly. A sampleInterval of 0 is inferred from the input, and horizonSeconds counts from the latest valid observation.

### Multivariate

When one channel runs longer, the extra part completes the shorter channels before the joint forecast, and known positions keep their original values.

![History, completion and forecast regions for two channels](<assets/multivariate-regions.png>)

- Result.quantiles stores values in channel, step, level order. The model determines the levels; 5M has 21.
- median, quantile, lower and upper default to channel 0.
- isObserved separates real readings from generated ones; the output is per-channel marginal quantiles.

### Scheduling

predict returns synchronously, submit returns a future, and the weights load once.

- Fifo runs requests in submission order.
- Mailbox keeps only the newest pending request per streamId.
- Continuous batches compatible requests: maxBatch 4 and maxVariables 8.
- abort cancels pending requests without preempting a running forecast.

### KV cache

When channels end at different observations, conditional completion runs in stages without reusing the common history as paged KV. Eligible streaming requests cache one patch per page. For 5M, a page holds 32 points; cachePages defaults to 64 pages per channel count, or 0 to disable caching. A single-variable 16k history needs 512 pages.

## Performance

### Backend timing

Timestral-5M, median of 3 runs, with symmetric averaging and non-negative clipping off; ar is 1 patch per round and par is one 32-patch forward.

![32k prefill and 16k-context-to-1k ar/par decode time per backend](<assets/backend-compare-1k-20261026.png>)

Raw data and recipes are in the [measurement notes](<doc/benchmark.md>).

### Prefill time

Single variable, in milliseconds.

| Length | reference | Eigen 1 thread | Eigen 2 threads | ytensor 1 thread | ytensor 2 threads | BLAS AVX2 1 thread | BLAS AVX2 2 threads | BLAS AVX2 4 threads | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan normal | Vulkan accelerated | Vulkan accelerated 512 | Vulkan accelerated 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 51.50 | 15.12 | 10.46 | 6.18 | 6.39 | 2.36 | 2.19 | 1.47 | 4.27 | 2.38 | 2.50 | 2.46 | 4.20 | 3.60 | 14.40 | 4.21 |
| 1k | 90.21 | 30.12 | 19.42 | 10.98 | 10.88 | 4.32 | 2.78 | 2.63 | 8.07 | 4.32 | 4.76 | 4.61 | 4.71 | 4.50 | 26.21 | 6.93 |
| 2k | 183.01 | 63.29 | 44.87 | 24.60 | 23.70 | 8.72 | 5.72 | 4.52 | 16.19 | 8.93 | 9.34 | 9.22 | 6.38 | 6.41 | 50.53 | 10.60 |
| 4k | 391.11 | 148.48 | 104.13 | 63.57 | 61.81 | 19.85 | 15.62 | 13.06 | 37.43 | 20.79 | 20.76 | 20.69 | 11.64 | 8.89 | 57.68 | 13.16 |
| 8k | 835.35 | 371.16 | 292.20 | 200.56 | 191.16 | 51.30 | 36.75 | 27.85 | 87.78 | 50.47 | 52.79 | 52.77 | 24.65 | 18.57 | 147.99 | 28.98 |
| 16k | 1954.58 | 1086.64 | 889.81 | 784.45 | 819.54 | 152.42 | 112.14 | 73.33 | 229.22 | 146.71 | 154.96 | 153.84 | 60.03 | 45.15 | 435.50 | 76.36 |
| 32k | 5035.31 | 3284.99 | 3066.65 | 2906.21 | 2580.37 | 488.94 | 354.55 | 224.23 | 706.29 | 499.09 | 529.33 | 501.28 | 168.30 | 138.33 | 1672.28 | 229.07 |

[2/4/8-variable results](<doc/benchmark.md#多变量prefill>).

### Sliding window

A two-channel signal advances to 128k, forecasting 512 points each time. The frozen FIFO is an explicit approximation and is off by default.

![Sliding window RMSE curves and update time once the window is full](<assets/swa-rmse.png>)

### Short forecasts and decode block size

Pooled 4k forecasts on six functions; 54 points exceed the numerical comparison budget. [Data and measurement details](<doc/benchmark.md#短预测精度与块大小>)

![Pooled RMSE and 4k decode time](<assets/block-ar-4k-summary-20261026.png>)

## Backends and configuration

reference is the default and needs only the standard library. The built-in BLAS also links no external library and picks AVX2, SSE4.2 or scalar kernels at run time on x86, while ARM and RISC-V switch to Eigen when BLAS is selected. Eigen and ytensor use the optional dependencies in the repository, and ytensor needs C++20 and AVX2/FMA. Vulkan needs headers, a loader and a compute driver; shaders can be compiled or supplied as a complete precompiled set. See the [Vulkan guide](<doc/vulkan.md>). Optional backends are mutually exclusive.

~~~bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_USE_BLAS=ON
~~~

The other switches are TIMESTRAL_USE_EIGEN, TIMESTRAL_USE_YTENSOR and TIMESTRAL_USE_VULKAN, and TIMESTRAL_BLAS_DECODE selects FP16, BF16 or INT8 for the BLAS backend. [Settings](<timestral/timestral.hpp#L38-L55>) are fixed at construction: threads 1, patchBatch 16 and cachePages 64 by default.

## Support

✅ measured · ✅✅ measured with dedicated optimization · ⚠ not fully tested on hardware · 🛠️ unresolved failure · 🗓️ planned · ❌ not provided

### Platforms and backends

| CPU architecture | OS / toolchain | reference | Built-in BLAS | Eigen | ytensor | Vulkan model |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| x86-64 | Windows / MSVC | ✅ | ✅✅ | ✅ | 🛠️ | ✅✅¹ |
| x86-64 | GNU/Linux / GCC, Clang | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| x86-64 | macOS / Apple Clang | ⚠ | ⚠ | ⚠ | ⚠ | ❌² |
| x86 (32-bit) | Windows, GNU/Linux | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| ARM64 | Windows | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |
| ARM64 | GNU/Linux | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |
| ARM64 | macOS / Apple Silicon | ⚠ | ❌³ | ⚠ | ❌ | ❌² |
| RISC-V64 | GNU/Linux | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |

1. Vulkan covers only the GPU families below and the usual 5M path; 32-bit x86 is not inferred from 64-bit results.
2. The portability enumeration and subset support that MoltenVK needs is not implemented.
3. The built-in BLAS has x86 kernels only, and ytensor requires x86 AVX2+FMA.

Linux build issues were addressed from an audit report, but the GCC/Clang CI has not run; ARM and RISC-V have no hardware acceptance. ytensor is measured on Windows, but parallelDecodeReal still exceeds the frozen numerical budget; see the [compatibility notes](<doc/linuxCompatibility.md>). The real 5M has been checked layer by layer and on forecasts, while 28M, 123M and 209M were only checked against their config and weight contract.

### Vulkan devices

| Platform | GPU family | Status | Scope |
| --- | --- | :---: | --- |
| Windows x86-64 | AMD RDNA 2 | ✅✅ | Usual 5M model, paged and shared prefixes, base operators, dedicated GLSL optimization. The 8-variable stress test still fails |
| Windows x86-64 | NVIDIA Ada (SM 8.9) | ✅✅ | Usual model, operators and dedicated optimization. A long-query budget failure is known |
| Windows, GNU/Linux | Intel Xe integrated graphics | ⚠ | Generic capability and subgroup handling in place, no hardware results. 🗓️ Target hardware acceptance |
| GNU/Linux | Software Vulkan, AMD, NVIDIA | ⚠ | Only an early small software-ICD smoke test. Current CI has not run |

RDNA 2 and Ada were each tested on one device only (Radeon 680M, RTX 4060 Laptop). Vulkan is the only GPU backend; see [GPU backend scope](<doc/gpuScope.md>).

## Documentation and license

The interface is documented in the [API reference](<doc/api.md>), inference details are in the [inference notes](<doc/inference.md>), backend selection in [compute backends](<doc/backends.md>), other guides in the [documentation index](<doc/README.md>), and command-line options in the [example notes](<example/README.md>).

The ported Timestral parts derive from the upstream Apache-2.0 project, with sources in the [third-party notes](<THIRD_PARTY.md>). This repository is licensed under [MIT](<LICENSE>); the released weights use a separate Modified MIT license and are not distributed here.
