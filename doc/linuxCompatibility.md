# Linux兼容性修复与验证边界

本文的测试计数记录当时的兼容性修复。后续用户决定移除OpenGL/OpenCL组件，相关代码、构建和CI任务现已删除；现行平台状态以[主说明矩阵](<../README_zh.md#平台与后端矩阵>)为准，取舍见[GPU后端范围](<gpuScope.md>)。

本次按审计报告逐项修正可确认的构建问题。默认部署仍是C++17，不新增运行时第三方依赖，不修改预测算法、权重、默认解码步长或冻结误差预算。当前主机为Windows/MSVC，没有可用Linux发行版、GCC或Clang，不能把下表的本机结果写成Linux实测通过。

## 修复对照

| 报告问题 | 当前处理 | 证据入口 |
| --- | --- | --- |
| 嵌套Settings默认参数阻断GCC/Clang | 去掉类内默认实参，新增单参数委托构造及loadEngine重载；Settings仍为聚合，没有用户构造函数 | [公开声明](<../timestral/timestral.hpp>)、[默认加载](<../timestral/utils/engine.cpp>) |
| 线程只在测试/BLAS分支出现 | 根配置无条件发现Threads，timestral与Vulkan kernels各自PUBLIC链接Threads::Threads；旧OpenCL组件现已删除 | [根配置](<../CMakeLists.txt>)、[独立下游测试](<../tests/embedding/CMakeLists.txt>) |
| ytensor源文件COMPILE_FEATURES没有实际升级标准 | 使用独立C++20 OBJECT目标，核心与工厂继续C++17；CI断言实际compile_commands | [构建配置](<../CMakeLists.txt>)、[Linux CI](<../.github/workflows/linux.yml>) |
| ytensor整单元AVX2/FMA会在旧CPU非法指令 | 仅模型对象启用AVX2/FMA，基线工厂先检查CPUID、OSXSAVE和XMM/YMM状态，不满足时明确拒绝；关闭两侧IPO/unity，避免优化越过安全边界 | [CPU检查](<../timestral/utils/cpu.cpp>)、[工厂](<../timestral/ytensor/backend.cpp>) |
| Vulkan只找glslc | AUTO优先glslc，找不到时使用glslangValidator/glslang，支持路径与种类显式覆盖，两套命令均以Vulkan1.1为目标 | [编译器选择](<../timestral/vulkan/shaderCompiler.cmake>) |
| shader编译器-NOTFOUND缓存污染 | 用户覆盖项与自动探测结果分开，不将自动结果强写回用户项；测试覆盖缺失、后来出现和强制切换种类 | [可重复选择测试](<../tests/shaderCompilerTest.cmake>) |
| 离线SPIR-V不完整/不匹配 | 逐一检查26普通+26预算模块，检查SPIR-V magic，存在spirv-val时在配置期验证；完整离线目录不寻找编译器 | [Vulkan构建](<../timestral/vulkan/CMakeLists.txt>) |
| C++20 char8_t路径拼接失败 | 统一提取UTF-8字节，不使用本地代码页窄化；实现及测试夹具共用转换，JSON原始UTF-8测试不再把char8_t当char | [路径转换](<../timestral/utils/path.hpp>)、[双标准回归](<../tests/portabilityTest.cpp>) |
| config.cpp范围循环临时string警告 | 字面量遍历用const char*，错误拼接时显式std::string，消息含义不变 | [配置解析](<../timestral/utils/config.cpp>) |
| OpenGL/OpenCL平台边界不清 | 后续评估不是低成本完整迁移，按用户决定删除实验代码和构建/CI入口，不作为后端保留 | [取舍依据](<gpuScope.md>) |

选择ytensor仍要求x86 AVX2+FMA及OS支持。非x86配置期拒绝，旧x86在进入优化模型前抛出可解释异常，建议用户选择reference或Eigen；没有静默冒充ytensor的后端回退。BLAS原有多ISA运行期分派不变。

## 本机实际验证

| 配置/测试 | 结果与范围 |
| --- | --- |
| reference全套 | 29/29通过 |
| BLAS全套 | 39/39通过 |
| Eigen全套 | 30/30通过 |
| ytensor构建与新兼容性用例 | 构建通过；C++17/C++20头、聚合初始化、UTF-8路径、CPU能力位与shader选择测试通过 |
| ytensor全套 | 30/31，parallelDecodeReal仍有冻结数值预算失败，不写成全绿 |
| 整个核心用C++20构建 | 构建通过；双标准公开头、JSON和safetensors针对性回归通过 |
| tests/examples关闭的独立消费者 | 仅链接timestral，验证传递线程依赖，编译/链接/执行通过 |
| glslc | 52模块生成与spirv-val、52项shader ABI及四小算子真实派发通过 |
| glslangValidator | 同样52模块、验证/ABI及低负载真实派发通过 |
| 离线SPIR-V | 完整52模块目录在故意提供无效编译器路径时仍可构建并通过ABI；缺少普通/预算模块在配置时拒绝 |
| AUTO/cache契约 | 无需Vulkan设备；缺失回退、NOTFOUND恢复、显式种类切换及非法参数通过，可重复运行 |

这里的真实Vulkan派发来自本机设备0的AMD Radeon(TM) Graphics，不是Linux llvmpipe或Intel核显结果。只调用2×2 linear/rmsNorm、3元素softmax/silu，没有运行GPU压力测试或完整模型基准。

### ytensor仍保留的数值失败

失败位置为single-frame/cached-full索引13712：actual=2.294824，expected=2.294836，原limit约0.000011。没有放宽该预算、删断言或从CTest排除用例。为了区分构建隔离和既有数值问题，本次仅在临时归档副本中把新模型对象换回10月25日保存的原ytensor模型对象，同一回归仍在同索引、同actual/expected/limit失败。原始库没有被替换，正式后端没有改为Naive/reference。

因此，本次可以说明报告中的构建阻塞已处理，但不能说明ytensor所有真实数值场景通过。此前其它长预测、GPU额外形状和设备1的未通过范围也不因本轮修复而消失，见本地数值验收记录。

## Linux待执行的回归

[Linux工作流](<../.github/workflows/linux.yml>)现固定Ubuntu22.04上的GCC11与Clang14，覆盖reference、全核心C++20、BLAS、Eigen、ytensor；ytensor实际命令必须只在模型单元出现-std=c++20/-mavx2/-mfma，基线工厂/CPU检查不能带AVX2。另有：

- QEMU Nehalem进程实际运行[无AVX2防护测试](<../tests/ytensorGuardTest.cpp>)，不是只模拟CPUID位
- tests关闭的独立消费者编译/链接，不依赖测试目标补上pthread
- RISC-V交叉构建必须实际选中Eigen，移除旧的掩盖失败命令
- glslc/glslangValidator各构建52模块并以spirv-val验证，使用llvmpipe软件ICD只运行ABI和小smoke
- 完整离线目录不依赖编译器，不完整目录必须失败
- 原OpenCL静态组件构建任务已随组件移除，不再列入CI或支持计划

工作流语法已在本机解析检查，但尚未在远程CI触发；没有用“新增CI”替代Linux结果。

~~~bash
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-11
cmake --build build-linux --parallel 2
ctest --test-dir build-linux --output-on-failure --parallel 2

cmake -S . -B build-vk -DCMAKE_BUILD_TYPE=Release -DTIMESTRAL_BUILD_VULKAN_KERNELS=ON -DTIMESTRAL_VULKAN_SHADER_COMPILER_KIND=GLSLANGVALIDATOR
cmake --build build-vk --target vulkanShaderAbiTest vulkanSmokeTest --parallel 2
ctest --test-dir build-vk -R '^(vulkanShaderAbi|vulkanSmoke)$' --output-on-failure
~~~

smoke与静态库必须由同一个CMake配置构建。Vulkan-Hpp的部分类型受NDEBUG影响，不能把Debug头/对象和Release静态库混用；当前修复不承诺这种跨配置ABI兼容，也不使用手工不匹配的wrapper验证GPU。

本轮未重测性能。既有图和CSV保留原测量时的程序/环境记录，不能把兼容性修复后的二进制SHA或时延倒填入旧结果。
