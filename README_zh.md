# Timestral C++

简体中文 · [English](<README.md>)

[Timestral](<https://github.com/mistralai/timestral>)时间序列模型的本地C++推理库：输入带时间戳的数值序列，输出未来各时刻的中位数、分位数和预测区间。

- 默认只依赖C++17标准库，不需要Python或PyTorch
- 多变量联合预测，支持缺测与条件补全
- 块自回归解码，内容寻址KV缓存
- 可选后端：自写BLAS、Eigen、ytensor、Vulkan，构建时单选
- 模型本地加载，不自动下载

## 构建

CMake 3.16及以上，编译器支持C++17。

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
~~~

默认构建只产出库`timestral`，不含测试与示例。测试是本地回归工具，不随仓库分发；配置时加`-DTIMESTRAL_BUILD_TESTS=ON`构建并运行`ctest --test-dir build -C Release --output-on-failure`。

## 使用

模型目录放同一发布版本的config.json与model.safetensors，校验值见本地模型清单（不随仓库分发）。链接目标timestral并包含[入口头文件](<timestral/timestral.hpp>)。

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

每条通道是`vector<Sample>`，time以秒计，NaN为缺测。多条通道放进同一个Request联合预测。sampleInterval为0时从输入推断，horizonSeconds从最晚有效观测起算。

### 多变量

某条通道观测更长时，多出的部分用于补全较短通道，再联合预测；已知位置保留原值。

![双通道历史、补全与预测区间](<assets/multivariate-regions.png>)

- Result.quantiles按channel、step、level存储，分位点由模型决定，5M为21个。
- median、quantile、lower、upper省略下标时取第0通道。
- isObserved区分真实读数与生成值；输出为各通道边缘分位数。

### 调度

predict同步返回，submit返回future，权重只加载一次。

- Fifo按提交顺序执行。
- Mailbox只留同一streamId最新的待执行请求。
- Continuous自动拼批，maxBatch为4，maxVariables为8。
- abort作废待执行请求，不抢占正在执行的预测。

### KV缓存

末次观测不一致时，条件补全按观测布局分阶段计算，不复用共同历史的分页KV。支持分页的流式请求按patch缓存。以5M为例，每页32点，cachePages默认按通道数各分配64页，设为0关闭；单变量16k历史需512页。

## 性能

### 后端耗时

Timestral-5M，3次取中位，关闭对称平均与非负截断；ar每轮1patch，par一次32patch前向。

![各后端32k prefill与16k上下文到1k的ar/par解码耗时](<assets/backend-compare-1k-20261026.png>)

原始数据与配方见[测量口径](<doc/benchmark.md>)。

### Prefill耗时

单变量，单位毫秒。

| 序列长度 | reference | Eigen 1线程 | Eigen 2线程 | ytensor 1线程 | ytensor 2线程 | BLAS AVX2 1线程 | BLAS AVX2 2线程 | BLAS AVX2 4线程 | BLAS SSE4.2 | BLAS FP16 | BLAS BF16 | BLAS INT8 | Vulkan普通 | Vulkan加速 | Vulkan加速512 | Vulkan加速4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 51.50 | 15.12 | 10.46 | 6.18 | 6.39 | 2.36 | 2.19 | 1.47 | 4.27 | 2.38 | 2.50 | 2.46 | 4.20 | 3.60 | 14.40 | 4.21 |
| 1k | 90.21 | 30.12 | 19.42 | 10.98 | 10.88 | 4.32 | 2.78 | 2.63 | 8.07 | 4.32 | 4.76 | 4.61 | 4.71 | 4.50 | 26.21 | 6.93 |
| 2k | 183.01 | 63.29 | 44.87 | 24.60 | 23.70 | 8.72 | 5.72 | 4.52 | 16.19 | 8.93 | 9.34 | 9.22 | 6.38 | 6.41 | 50.53 | 10.60 |
| 4k | 391.11 | 148.48 | 104.13 | 63.57 | 61.81 | 19.85 | 15.62 | 13.06 | 37.43 | 20.79 | 20.76 | 20.69 | 11.64 | 8.89 | 57.68 | 13.16 |
| 8k | 835.35 | 371.16 | 292.20 | 200.56 | 191.16 | 51.30 | 36.75 | 27.85 | 87.78 | 50.47 | 52.79 | 52.77 | 24.65 | 18.57 | 147.99 | 28.98 |
| 16k | 1954.58 | 1086.64 | 889.81 | 784.45 | 819.54 | 152.42 | 112.14 | 73.33 | 229.22 | 146.71 | 154.96 | 153.84 | 60.03 | 45.15 | 435.50 | 76.36 |
| 32k | 5035.31 | 3284.99 | 3066.65 | 2906.21 | 2580.37 | 488.94 | 354.55 | 224.23 | 706.29 | 499.09 | 529.33 | 501.28 | 168.30 | 138.33 | 1672.28 | 229.07 |

[2/4/8变量结果](<doc/benchmark.md#多变量prefill>)。

### 滑动窗口

双通道信号推进到128k，每次预测512点。冻结FIFO是显式近似，默认不启用。

![滑动窗口RMSE曲线和窗口满后的更新耗时](<assets/swa-rmse.png>)

### 短预测精度与解码块大小

六种函数的4k预测；数值对照有54个点超出误差预算。[数据与测量口径](<doc/benchmark.md#短预测精度与块大小>)

![综合RMSE与4k解码耗时](<assets/block-ar-4k-summary-20261026.png>)

## 后端与配置

默认reference只依赖标准库。自写BLAS不链接外部库，x86按AVX2、SSE4.2、标量选核，ARM与RISC-V选BLAS时改选Eigen。Eigen与ytensor用仓库内可选依赖，ytensor需C++20与AVX2/FMA。Vulkan需要头文件、loader与计算驱动；shader可使用编译器或完整预编译包，见[Vulkan使用说明](<doc/vulkan.md>)。可选后端互斥。

~~~bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_USE_BLAS=ON
~~~

其余开关：TIMESTRAL_USE_EIGEN、TIMESTRAL_USE_YTENSOR、TIMESTRAL_USE_VULKAN，BLAS可用TIMESTRAL_BLAS_DECODE选FP16、BF16或INT8。[Settings](<timestral/timestral.hpp#L38-L55>)在构造时固定：threads默认1，patchBatch默认16，cachePages默认64。

## 支持范围

✅ 已实测 · ✅✅ 专项优化 · ⚠ 未完成实机验收 · 🛠️ 有未解决失败 · 🗓️ 后续计划 · ❌ 不提供

### 平台与后端矩阵

| CPU架构 | 操作系统/工具链 | reference | 自写BLAS | Eigen | ytensor | Vulkan模型 |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| x86-64 | Windows / MSVC | ✅ | ✅✅ | ✅ | 🛠️ | ✅✅¹ |
| x86-64 | GNU/Linux / GCC、Clang | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| x86-64 | macOS / Apple Clang | ⚠ | ⚠ | ⚠ | ⚠ | ❌² |
| x86（32位） | Windows、GNU/Linux | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| ARM64 | Windows | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |
| ARM64 | GNU/Linux | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |
| ARM64 | macOS / Apple Silicon | ⚠ | ❌³ | ⚠ | ❌ | ❌² |
| RISC-V64 | GNU/Linux | ⚠ | ❌³ | ⚠ | ❌ | ⚠ |

1. Vulkan只覆盖下表列出的GPU架构族与常规5M路径，32位x86不由64位推断。
2. 未实现MoltenVK所需的portability enumeration与subset适配。
3. 自写BLAS只有x86内核；ytensor要求x86 AVX2+FMA。

Linux构建问题已按审计报告修正，GCC/Clang CI尚未执行；ARM与RISC-V无实机推理验收。ytensor在Windows有实测，但parallelDecodeReal仍超出冻结数值预算，见[兼容性记录](<doc/linuxCompatibility.md>)。真实5M已完成逐层与预测对照，28M、123M、209M只核对配置与权重契约。

### Vulkan设备

| 平台 | GPU架构族 | 状态 | 范围 |
| --- | --- | :---: | --- |
| Windows x86-64 | AMD RDNA 2 | ✅✅ | 常规5M模型、分页与共享前缀、基础算子、专用GLSL优化；8变量压力测试仍有失败 |
| Windows x86-64 | NVIDIA Ada（SM 8.9） | ✅✅ | 常规模型、算子与专用优化；长查询预算有已知失败 |
| Windows、GNU/Linux | Intel Xe核显 | ⚠ | 通用能力与subgroup适配已就位，无实机结果，🗓️目标硬件验收 |
| GNU/Linux | 软件Vulkan、AMD、NVIDIA | ⚠ | 只有早期软件ICD小规模smoke，当前CI待执行 |

RDNA 2与Ada各只测过一款设备（Radeon 680M、RTX 4060 Laptop）。GPU后端仅Vulkan，见[GPU后端范围](<doc/gpuScope.md>)。

## 文档与许可

接口参考见[API参考](<doc/api.md>)，推理细节见[推理说明](<doc/inference.md>)，后端选用见[计算后端](<doc/backends.md>)，其它教程见[文档索引](<doc/README.md>)，命令行参数见[示例说明](<example/README.md>)。

Timestral移植部分源自上游Apache-2.0项目，来源见[第三方说明](<THIRD_PARTY.md>)。本仓库整体许可为[MIT](<LICENSE>)；发布权重使用单独的Modified MIT许可，不在本仓库分发。
