# Vulkan GPU后端

第五期交付十算子SPIR-V库和设备常驻模型后端，第六期在不改v1绑定与push布局的前提下新增FP32优化核。Vulkan是仓库第一个接入选中后端的GPU API：TIMESTRAL_USE_VULKAN=ON时loadBackendModel返回GPU模型，权重常驻设备、层图整批执行；默认构建不链接任何Vulkan代码，公开API、Tensor布局与CPU后端选择不变。GPU模型只保留Vulkan；OpenGL与OpenCL实验组件已按用户要求删除，迁移缺口和历史边界见[取舍记录](<gpuScope.md>)。

## 两个构建入口

| 选项 | 效果 |
| --- | --- |
| TIMESTRAL_USE_VULKAN | 选中的模型后端改为Vulkan常驻模型，与Eigen、ytensor、BLAS互斥，同时开启任意两个在配置阶段报错 |
| TIMESTRAL_BUILD_VULKAN_KERNELS | 只构建独立算子验证组件timestralVulkanKernels，不提供模型后端 |
| TIMESTRAL_VULKAN_SPV_DIR | 指向完整52模块Vulkan1.1 SPIR-V目录时不需要shader编译器，缺少普通/预算模块或头无效时在配置阶段报错 |
| TIMESTRAL_VULKAN_TEST_DEVICE | 算子测试使用的Vulkan枚举设备号，默认0 |
| TIMESTRAL_VULKAN_TEST_VALIDATION | 算子测试要求Khronos validation层，未安装会失败而不是跳过 |
| TIMESTRAL_BUILD_GPU_BENCHMARKS | 构建vulkanBench，默认含传输模式，resident/attention模式计常驻录制与提交，默认ON |

模型后端构建树（需要Vulkan SDK头文件与loader）：

~~~powershell
cmake -S . -B build-gpu-vulkan -A x64 -DTIMESTRAL_USE_VULKAN=ON "-DVulkan_INCLUDE_DIR=$env:VULKAN_SDK/Include" "-DVulkan_LIBRARY=$env:VULKAN_SDK/Lib/vulkan-1.lib"
cmake --build build-gpu-vulkan --config Release --parallel
ctest --test-dir build-gpu-vulkan -C Release --output-on-failure
~~~

通用运行配置见[运行期配置](<../timestral/utils/gpuSettings.hpp>)，Vulkan模型另提供优化回退开关：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| TIMESTRAL_GPU_SHADER_DIR | 构建期烧入的shader目录 | 覆盖预编译spv位置 |
| TIMESTRAL_GPU_DEVICE | 0 | Vulkan枚举设备号，严格十进制 |
| TIMESTRAL_GPU_VALIDATION | 0 | 1时要求Khronos validation层，未安装失败 |
| TIMESTRAL_VULKAN_OPTIMIZATIONS | 1 | 1启用已验收投影加速，0使用通用投影且不融合，非法值拒绝。两者共享有序精度路径、cached回读与KV-only编排，0不是旧源码回退 |
| TIMESTRAL_VULKAN_RESIDENT_RESIDUAL | 0 | 残差融合仅显式实验，M≥128；撤销680M导出的大M上界，未据此制定Intel默认策略，非法值拒绝 |

设备枚举顺序不保证独显优先，换驱动后可能改变，测试与基准都会打印实际设备名。本轮实测设备0为AMD Radeon(TM) Graphics（API1.3.217），设备1为NVIDIA GeForce RTX 4060 Laptop GPU（API1.4.325）。

## GPU并行预算

[公开Settings](<../timestral/timestral.hpp>)新增gpuThreads，0不限制本库派发，非0须为256倍数/u32范围，只对Vulkan生效。它限制每个串行dispatch窗口的逻辑invocations，不是物理EU/CU、标称768通道或GPU利用率百分比；很小预算可能明显变慢。固定local_size不随用户值改变，非法倍数在创建GPU对象前拒绝，预算256不选择512线程可选管线。

~~~cpp
timestral::Timestral::Settings options;
options.gpuThreads = 8192; // 示例预算，不是某个GPU的推荐值
timestral::Timestral model("model/timestral-5m", options);
~~~

详细整数约束、规范网格/矩形窗口、原生二维、精确尾行、首次管线JIT和Intel验证缺口见[可移植设计](<vulkan-portability.md>)。其它进程/模型、驱动copy及共享DRAM带宽不受此预算保证，不能据此承诺保留某些EU或永不device-lost。

## 算子库与ABI

[源目录](<../timestral/vulkan/shaders/>)包含26个计算核及共享精确数学辅助，离线生成各自Budget变体，完整包共52模块。原21核的绑定、push和localSize不变，新增5核为共享前缀、组内注意力和有序精度路径。Budget变体保持对应原核ABI。subgroup快核只在报告size32/64且能力满足时创建，不强加给Intel8/16。

| 算子 | 计算 | 工作组 | push |
| --- | --- | --- | --- |
| [gemm](<../timestral/vulkan/shaders/gemm.comp>) | A[M,K]乘W[N,K]转置，可选bias | 16×16共享K分块 | 16字节 |
| [rmsNorm](<../timestral/vulkan/shaders/rmsNorm.comp>) | 每行平方和归约，(x*inverse)*scale | 每行64线程 | 12字节 |
| [softmax](<../timestral/vulkan/shaders/softmax.comp>) | 每行最大值与指数和归约，负无穷屏蔽 | 每行64线程 | 8字节 |
| [silu](<../timestral/vulkan/shaders/silu.comp>) | x/(1+exp(-x)) | 64线程，尾部检查 | 4字节 |
| [prepareQkv](<../timestral/vulkan/shaders/prepareQkv.comp>) | 融合QKV按头重排加partial RoPE | 64线程 | 20字节 |
| [mergeKv](<../timestral/vulkan/shaders/mergeKv.comp>) | 前缀K/V与新增K/V按行拼接 | 64线程 | 16字节 |
| [elementwise](<../timestral/vulkan/shaders/elementwise.comp>) | 残差加、sigmoid门控、SwiGLU三模式 | 64线程 | 12字节 |
| [attentionScores](<../timestral/vulkan/shaders/attentionScores.comp>) | 双流掩码QK分数与在线softmax | 64线程 | 36字节 |
| [attentionValues](<../timestral/vulkan/shaders/attentionValues.comp>) | 掩码PV归约 | 64线程 | 36字节 |
| [attentionFused](<../timestral/vulkan/shaders/attentionFused.comp>) | 分数与PV同一workgroup内完成，跨warp共享归约 | 64线程 | 36字节 |
| [gemmTile](<../timestral/vulkan/shaders/gemmTile.comp>) | 32×32输出tile、2×2寄存器分块、合并W加载、逐K precise | 16×16线程，4160B共享 | 16字节 |
| [attentionTile](<../timestral/vulkan/shaders/attentionTile.comp>) | time轴8-query共享K/V，64键在线softmax | 64×8线程，39168B共享，设备不满足时回退v1 | 36字节 |
| [attentionRegister](<../timestral/vulkan/shaders/attentionRegister.comp>) | headDim64寄存器PV，仅诊断 | 64线程 | 36字节 |
| [gemv](<../timestral/vulkan/shaders/gemv.comp>) | 窄M逐K精确投影，按K合并权重加载 | 64线程，16列/组 | 16字节 |
| [attentionCausal](<../timestral/vulkan/shaders/attentionCausal.comp>) | 仅裁剪可证明全空的编码未来键块 | 64×8，39168B共享 | 36字节 |
| [attentionSubgroup](<../timestral/vulkan/shaders/attentionSubgroup.comp>) | 带query对齐检查和原归约回退的subgroup注意力 | 64×8，39300B共享，Basic/Vote/Arithmetic | 36字节 |
| [attentionDecode](<../timestral/vulkan/shaders/attentionDecode.comp>) | H1长KV按128键分片，保存max/分母/PV | 64线程，17408B共享 | 40字节 |
| [attentionDecodeMerge](<../timestral/vulkan/shaders/attentionDecodeMerge.comp>) | 稳定合并分片，不平均局部归一化结果 | 64线程 | 12字节 |
| [siluGrid](<../timestral/vulkan/shaders/siluGrid.comp>) | 支持二维grid展开的SiLU，供残差融合 | 64线程 | 4字节 |
| [gemmWide](<../timestral/vulkan/shaders/gemmWide.comp>) | 64×64输出、4×4寄存器，仅诊断 | 16×16，8256B共享 | 16字节 |
| [attentionPacked](<../timestral/vulkan/shaders/attentionPacked.comp>) | 四score/四PV寄存器候选，仅诊断 | 16×8，37632B共享 | 36字节 |

[shaderManifest.json](<../timestral/vulkan/shaderManifest.json>)是abiVersion=1的机器可读ABI：目标环境vulkan1.1、入口main/GLCompute、每个shader的localSize、binding序号与访问方向、push常量成员名、类型与偏移、最低设备限制和二维dispatch规则。常驻大网格线性kernel按x=min(G,65535)、y=ceil(G/x)展开，元素下标先展平y，GEMM用原生二维网格。旧silu验证核只看X，大网格常驻路径使用siluGrid，不把旧核直接改成二维派发。

本机SDK1.3.283的glslc把SPIR-V 1.3的版本字写成0x00010300，同SDK的spirv-dis解析为1.3、spirv-val通过。构建期AUTO优先glslc，缺失时使用glslangValidator（或glslang）；前者用--target-env=vulkan1.1，后者用-V --target-env vulkan1.1。自动探测结果不强写回用户覆盖项，能找到spirv-val时按同目标环境校验二进制，推理进程不编译GLSL。

[ABI反射测试](<../tests/vulkanShaderAbiTest.cpp>)解析编译产物而不依赖GPU：入口点唯一且为main/GLCompute、LocalSize与manifest一致、storage buffer的descriptor set与binding集合精确相等、push结构成员偏移/类型/总长一致，并拒绝坏magic、指令内部截断、零wordCount与越界wordCount四类损坏模块。它把manifest与spv的漂移变成构建期失败，不再依赖运行时静默错配。

## 常驻执行器

[GpuExecutor](<../timestral/utils/gpuExecutor.hpp>)是后端无关的执行契约：begin、upload（可标记persistent）、allocate、dispatch、download、end。[Vulkan执行器](<../timestral/vulkan/executor.cpp>)的实现：

- 权重在模型构造时一次上传为persistent，staging用host-visible coherent缓冲，staging与device工作区各有一个高水位复用池
- 一次runLayers的整段层图录入primary command buffer，每次dispatch前插入可见性barrier，在首个回读处提交；后续K/V页回读仍串行提交/等待，没有实现集中回读
- 批内句柄在end时失效，跨批使用会抛out_of_range，不能复用上一批的scratch地址

异常语义：submit在fence创建、queue.submit或等待失败时把本批标记poisoned并保留全部资源，end()用device waitIdle排空后释放，之后的begin()明确拒绝而不是带病重入。[GpuTransformer](<../timestral/utils/gpuTransformer.cpp>)的清理助手只在已有异常传播时调用end()，清理失败不覆盖最初的设备错误。这条语义修掉了一次审查发现的错误掩盖：旧路径在submit抛错后仍会对未结束的command buffer再次end，把原始错误吞掉并把后续批次卡死。

## 模型后端

[GpuTransformer](<../timestral/utils/gpuTransformer.hpp>)继承Transformer，覆写层图和数值钩子。残差块设备常驻改为显式实验，默认三投影间仍经过CPU；后处理、调度、页管理仍在host。沿用time/variate双流谓词，RoPE由host按reference相同libm预计算，只精确复用本调用inverse频率和完全相同的位置行。纯KV prefill最后time K/V就绪后停止，但完整forward、trace、decode不截断。前缀K/V仍从CPU页池装配/上传，经mergeKv拼接，没有设备KV镜像。

测试矩阵：

| 测试 | 覆盖 |
| --- | --- |
| gpuModelTiny / gpuModelReal | 微型与真实5M对独立reference逐层隐状态max(1e-5,1e-4×RMS)加relL2≤1e-5、输出分位数1e-4+1e-3×|ref|、缓存与全量一致、行1/2/3、组隔离、角色、缺键、连续坐标、冷/追加/共享页、页哈希不变 |
| vulkanExecutor | G1十二形状×bias逐位；GEMV十形状×bias冻结预算；causal逐位、subgroup/packed冻结预算；split-K空片/尾片/角色；siluGrid超过65535组；原attention种子与生命周期测试 |
| vulkanKernels | 四算子30组GEMM形状、尾块、边界、并发、损坏spv拒绝 |
| vulkanShaderAbi | manifest与spv的ABI反射对照加四类损坏拒绝 |
| time/condition/multivariate Real、sharedDecode | 真实5M回归与共享分页decode走GPU后端 |

8通道默认只增加公开forward/predictBlock输出预算；--strict-multi保留严格逐层诊断，目前v1选核也失败，尚未定位。63/513patch H1跨残差分派上下界、129patch KnownCov、只读页哈希已通过双GPU，不将8通道输出通过写成8通道逐层全部通过。

## 性能口径与历史窗口

最终部署面向Intel核显，680M/4060只是验证。decode不再以1%MFU验收，使用同任务时延、copy与带宽；最新预算代价与未覆盖硬件见[当前设计](<vulkan-portability.md>)。下列41.47/8.48ms等数据属于此前启用残差M窗口的历史策略，不是当前默认值，不推断Intel时延。

32768采样点（1024个patch）冷prefill与H=1 decode，backendCompare协议（本地工具，不入库），kind=paged，encode3次中位、decode预热5次计时25次中位，机器无其它构建与GPU任务：

| 设备 | encode ms | decode ms | checksum |
| --- | ---: | ---: | --- |
| RTX4060 Laptop（设备1） | 120.8538 | 7.7143 | 2.3805 |
| AMD集显（设备0） | 236.9136 | 11.8549 | 2.3806 |

同一轮CPU历史复测（见本地验收记录）：blas-avx2 181.15/4.39ms，reference 4769.09/8.16ms，checksum均2.3806。这只能说明该窗口的端到端表现，不能证明GPU利用充分。旧文把完整forward公式套入分页prefill，并以CPU的134.4GF/s分母讨论GPU效率，现撤回这一口径；也撤回未经归因就说H=1“由提交与编排主导”的判断。680M按用户给定3TFLOPS及实际执行范围重新计算，分析见[680M专项归因](<vulkan-opt.md>)，CPU名义MFU仍只用于CPU。

上表为第五期历史基线，不代表当前优化版。同调用rotary与metadata在层循环外只上传一次，旧文“每层预计算/上传”的判断已纠正。当前仍逐dispatch独立descriptor pool、页回读串行、scratch池只保留高水位不设预算。

第六期重新设计后接受G1投影与A2共享K/V注意力，末次同窗口v1→优化版：4060 encode 125.9646→109.8991ms、decode 8.0188→6.7479ms；AMD encode 263.0007→143.5030ms、decode 12.9188→10.5634ms，首输出值仍2.3805/2.3806。AMD跨轮波动较大，全部轮次、算子微基准、未接受的池复用与空块裁剪见[优化树](<vulkan-opt.md>)，不承诺固定倍数。

当前选核集中在Vulkan Executor，编排只增加已验证KV-only停止与M窗口残差融合。G1涵盖已测embedding形状，M1至8可用逐K GEMV，长time注意力按能力选causal/subgroup，H1长KV可split-K。没有启用FP16、投影FMA或投影K并行归约，没有恢复arena；attention归约重排按原数值预算验收。

## 680M回读修正与诊断

680M专项时间戳与CPU计时发现：旧readback使用Visible+Coherent但没有Cached的内存，17.25MiB的CPU map/读取/unmap约77.53ms，而GPU执行copy约2ms。当前readback独立复用池优先HOST_CACHED|HOST_COHERENT，无该类型时保持既有coherent回退；upload staging不变，不移除fence或可见性屏障。修正后对应CPU读取约1.52ms。

以上为cached修正当时的历史归因，旧窗口encode85至88ms、12.260475G按裁剪前图计。当前MFU固定必要任务量10.648682496G，不把后续删除的无用末尾计算当产出。最新数字、各轮波动和未达目标项见[优化树](<vulkan-opt.md>)。

vulkanProfile归因工具（本地工具，不入库）仅在TIMESTRAL_USE_VULKAN与GPU基准开启时构建，不改变默认模型行为。用法：

~~~powershell
.\build-gpu-vulkan\timestral\vulkan\Release\vulkanProfile.exe build-gpu-vulkan/timestral/vulkan/shaders model/timestral-5m 0 1 1 0 8192
~~~

末尾五个参数为设备号、诊断0/1、优化0/1、实验残差0/1、GPU预算，最后两项默认0。工具报告主机API、GPU跨度、copy、实际窗口数与最大窗口invocations，去掉硬编码3TFLOPS/MFU；吞吐以必要任务GF/s描述。诊断decode只有5次，CPU等待/GPU时间不相加，延迟用关闭诊断的固定协议。KV镜像仍未实现。

## 已知限制

- 每个dispatch绑定一个buffer到单一SSBO，受maxStorageBufferRange约束，timestral-5m的32k路径远低于典型128MiB上限，更大模型或更大批量可能触顶
- scratch复用池只按高水位保留，不回收也不设内存预算
- 残差融合默认关闭、仅显式实验，不按680M的M2048上界决定Intel默认策略
- 只做FP32，不要求FP16/cooperative matrix；subgroup是可选优化，缺能力回退
- 小形状算子含传输时慢于CPU，见上节；常驻批提交只消除了层间的重复同步，没有消除单dispatch内的分配与提交开销

## SPIR-V跨API边界

本库只发布Vulkan1.1目标的Shader SPIR-V，不承诺它能被其它API消费。历史OpenGL实验缺少attentionFused与跨平台上下文；OpenCL实验仅四算子实测、未验收常驻模型，设备IL版本为空且AMD队列曾失败，不能靠改扩展名迁移。两套组件现已删除，不保留启用入口，见[GPU后端取舍](<gpuScope.md>)。

## 二进制发布

~~~powershell
cmake --install build-gpu-vulkan --config Release --component VulkanShaders --prefix .agents/data/vulkan-package
~~~

安装组件把26个普通与26个Budget共52个spv、[shaderManifest.json](<../timestral/vulkan/shaderManifest.json>)与[gpuKernels.md](<../timestral/utils/gpuKernels.md>)放入share/timestral/vulkan。TIMESTRAL_VULKAN_SPV_DIR指向该目录即可离线构建，不寻找glslc或glslangValidator。显式选择工具可设置TIMESTRAL_VULKAN_SHADER_COMPILER路径与TIMESTRAL_VULKAN_SHADER_COMPILER_KIND=AUTO/GLSLC/GLSLANGVALIDATOR。预编译输入必须是可信、已验证且descriptor与push布局匹配的文件，运行时只做基本SPIR-V头检查，不负责证明任意外部shader安全。
