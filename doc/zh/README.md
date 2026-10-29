# 设计文档

[English](<../en/README.md>) | [Français](<../fr/README.md>)

本目录按“模型算什么、推理怎样省、后端怎样算”的顺序说明Timestral C++推理库的设计。使用方法与构建命令见[项目说明](<../../README_zh.md>)，示例的参数与运行方法见[示例说明](<../../example/README.md>)，数值证据见本地验收记录（不入库）。

## 更新历史

[版本更新记录](<changelog.md>)按版本简述全部更新，后续新版本追加到文件末尾。

## 建议阅读顺序

1. [模型架构与仓库模块](<architecture.md>)：以Timestral-5M的真实维度介绍patch嵌入、因果归一化、time与variate注意力轴、掩码、部分RoPE、门控注意力、SwiGLU和分位数头，再说明仓库的模块边界与链接期后端选择。
2. [推理侧设计](<inference.md>)：一条请求怎样经过调度、分块自回归和后处理，分页KV缓存为什么在标准路径上成立而在条件路径上不成立，以及示例层的冻结前缀滑窗实验。
3. [计算后端](<backends.md>)：reference、Eigen、ytensor的实现思路，精度验收的容差口径，以及实测的性能取舍。
4. [BLAS后端](<blas.md>)：从请求层到微内核逐层展开优化树，推导预打包、缓存分块、在线softmax、块掩码和2^x向量exp，并记录性能口径与失败实验。

前一篇的概念会在后一篇直接使用，例如推理篇的缓存条件依赖架构篇的掩码规则。只关心某个后端时，可以先读架构篇的模块一节，再直接跳到对应文档。

需要查具体接口的参数、返回值、异常与调度策略时，直接看[API参考](<api.md>)。

## 跨平台构建

[Linux兼容性修复与验证边界](<linuxCompatibility.md>)逐项记录公开头文件、线程依赖、ytensor局部C++20/ISA防护、UTF-8路径和双Vulkan编译器的修复，以及本机结果和待执行Linux CI。已知数值失败不因构建通过而忽略。

## GPU开发关卡

[Vulkan GPU后端](<vulkan.md>)说明第五期的十算子SPIR-V库、ABI manifest与反射测试、常驻执行器、模型后端与32k实测性能，当前GPU后端仅Vulkan，旧OpenGL/OpenCL组件移除原因见[GPU后端取舍](<gpuScope.md>)。

[Vulkan可移植性与并行预算](<vulkan-portability.md>)记录Intel核显目标、本地ggml部分审读、GPU逻辑窗口配置、52模块派发契约、严格预算的代价及尚未覆盖的Intel实机验证。decode按延迟/流量验收，不把开发设备的3TFLOPS分母用于最终部署。

## 性能与质量复现

[基准协议与原始数据](<benchmark.md>)记录16种配方和按变量数拆开的prefill结果。主说明只保留单变量表、速度图及一组RMSE，多变量表与AR默认步长核验集中在此。数据按允许日常轻负载、无外部重负载的条件实测，扩展用例的未通过范围单独列明。

## 配图

配图放在仓库根目录的[assets](<../../assets>)，以arch-、infer-、backend-为前缀的SVG分别对应前三篇，blas-为前缀的SVG对应BLAS教程，SWA扫描图与原始CSV来自滑窗实验。
