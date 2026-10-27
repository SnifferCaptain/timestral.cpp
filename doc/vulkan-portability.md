# Vulkan可移植优化与GPU并行预算

## 目标与边界

最终部署面向用户所述13至15代、较小规模Intel核显，680M与4060只作开发验证，不按它们的最好成绩决定Intel默认策略。EU、Xe-core、AMD CU与算术通道不是同一单位，设备标称768个算术通道不等于768个可分配的Vulkan线程。Vulkan版本兼容也不代表subgroup、workgroup、SLM、缓存或可选矩阵扩展相同。

decode撤销1% MFU验收，改看同任务稳态延迟、显式copy、逻辑数据流量估算与带宽。CPU/GPU即使共享DRAM也有不同缓存和复制路径，不能据此承诺延迟必定相等。归因工具已移除硬编码3TFLOPS和MFU百分比，保留必要任务GFLOPs/s作为吞吐描述，不作为内存任务门槛。

## 已核验的本地参考

用户提供的ggml快照可直接读取，快照与校验文件属于本地工作文件，不随仓库发布。没有确定远程commit，没有复制整个内核，未声称完整审读仓库。校验文件记录了相关文件的完整SHA256，实际审读范围如下：

| 路径 | 已读范围与有用信息 | 本项目的处理 |
| --- | --- | --- |
| `mul_mm.comp` | 分块/对齐定义与主循环，BM/BN/BK、warp tile、共享加载和寄存器累加 | 后续评估FP32面板、对齐vec4加载与复用，不能直接带入FP16/cooperative要求 |
| `mul_mat_vec_base.glsl`、`mul_mat_vec.comp` | 前段定义与归约结构，行/列并行、向量访存、subgroup/共享路径 | K并行归约仍须过冻结预算，不为带宽而放宽误差 |
| `flash_attn_decode_phase_1.comp` | 前段分阶段注意力及局部归约，未完整审读该文件 | 借鉴分片/合并，不照抄矩阵扩展或特定lane布局 |
| `ggml-vulkan.cpp:1468–1478` | 共享开销包含加载、行索引、coopmat stage和LUT，按设备上限筛选 | 只看tile数据大小不够，不能盲目扩大tile |
| `Xe decode参数` | Xe1分支参数8、其它Xe路径16，group size按q_len限制64至256 | 本库32/64快速subgroup核不再在报告8/16时创建，保留通用归约；不按CPU代号猜lane宽度 |
| `UMA创建`、`cached主机缓冲` | UMA优先可映射device-local，借用受实际分配约束；主机缓存有fallback | 不把普通std::vector指针冒充GPU buffer，不把HostCached当作GPU L1命中率 |

在线检索找到[Intel SLM指南](<https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/shared-local-memory.html>)、[AI Max395相关AMDVLK能力议题](<https://github.com/GPUOpen-Drivers/AMDVLK/issues/420>)与[RDNA Vulkan推理分支](<https://github.com/SelfRef/llama.cpp-rdna/blob/rdna/README.md>)。web_fetch仍对这些域名返回非公网IP拒绝，目前仅有搜索摘要，不将其具体实现或性能数字当作已核验事实。RDNA专用优化不是Intel默认方案；真正已核验的是上述本地代码。

## 使用gpuThreads

配置在[公开Settings](<../timestral/timestral.hpp>)末尾新增，构造时读取，只对Vulkan生效，其它后端忽略，不改变CPU threads行为。配置结构扩展后应重新编译应用和库，权重与Tensor布局未变。

~~~cpp
timestral::Timestral::Settings options;
options.threads = 1;
options.gpuThreads = 8192; // 示例上限，不是设备推荐值

timestral::Timestral model("model/timestral-5m", options);
~~~

- 0：不限制本库原派发，默认值
- 非0：256的正整数倍，且不超过uint32范围，256/512/768/4096/8192均合法，384和257非法；不是只允许2的幂
- 256是当前必需GEMM工作组的实现量化，不是所有Vulkan硬件通用的warp宽度
- 每个kernel维持已验证local_size，不把用户数值写进local_size；预算256不创建512线程attention管线，自动走合法原路径
- 非法预算在执行器创建Vulkan实例/设备前抛invalid_argument；模块固定LocalSize、设备各轴/总invocations和变体尺寸也校验，避免零步长或混装模块

**这是单个串行dispatch窗口的逻辑shader invocations上限，不是物理EU/CU配额。** 768预算对256线程kernel最多三个工作组，对512线程kernel最多一个工作组，不派发半个工作组。不能拿设备标称768通道把512解释成66%占用；可能存在远多于算术通道数的在途逻辑线程。很小的预算会严重降低并行度并增加命令开销。

限制只作用于本模型执行器发出的compute shader，不限制其他进程/模型、驱动CPU线程、驱动内部copy/迁移、共享DRAM带宽，也不承诺利用率或功耗百分比。需要硬保留EU或固定占空比时不能把本选项当成已实现的全局硬件控制。合法参数也不构成所有驱动永不TDR/device-lost的保证。

## 调度、索引与ABI

[执行器](<../timestral/vulkan/executor.cpp>)使用Vulkan1.1 dispatchBase，管线带DISPATCH_BASE标志。窗口间有compute/transfer执行依赖，避免后一个本库窗口与前一个重叠；完整行可在预算内合并成矩形，不无条件逐行串行。始终满足windowX乘windowY乘固定localInvocations不超过gpuThreads。

不能直接缩小groupCount：展平核依赖gl_NumWorkGroups.x，会算错非零Y的地址。[离线生成器](<../timestral/vulkan/budgetShader.cmake>)从26个计算核生成26个Budget模块，保留对应SSBO/push和浮点运算顺序，通过specialization constant255提供展平宽度。原21个GLSL及其ABI不修改，新增共享前缀、组内注意力及有序精度核，仍为abiVersion=1。[完整manifest](<../timestral/vulkan/shaderManifest.json>)共52模块，预编译/发布目录必须完整更新。

展平核按既有契约只把NumWorkgroups用作展平步距，运行时由原模块的Builtin标记识别。预算网格规范化为固定W=min(设备X组上限,65535)，末行只派原覆盖范围，部分dispatch不扩张；native GEMM/GEMV/拆分attention保留原二维坐标。每个operator只缓存一个预算管线，不随请求长度累积特化/JIT，也不在录制或在途命令引用期间释放管线。首次使用仍可能有管线JIT开销，固定协议中位数不代表冷JIT首请求。

## 默认策略与下一步

上一轮基于680M收窄的residual M上界2048撤出默认依据。TIMESTRAL_VULKAN_RESIDENT_RESIDUAL默认改0，显式1仅作为实验，保留M≥128的小派发摊薄条件，不再把680M大M结果当作Intel规则。其它模型几何分派与能力合法性分开，未按GPU品牌新建特例。

当前常用形状采用64线程、8200B共享内存的有序注意力，最多2048键，不依赖subgroup或FP64。共享前缀按通道组上传，GPU直接读取一份前缀和独立尾部。G1投影加速保留，原A2等在线归约核保留作候选或超范围回退，不能把历史选核当作本次默认派发。数值范围及长查询限制见[基准协议](<benchmark.md>)。

缓冲物理容量按几何级增长，描述符、copy和download只使用逻辑bytes。空闲device/staging/readback池在fence完成后按设备分配数限制回收，避免逐patch变长的AR积累数千份分配。

后续优先级：

1. **低SLM注意力和8/16-wide路径**：现有512线程/约39KiB路径不适合所有小核显；需新通用小块候选与冻结预算，不直接替换成RDNA专核
2. **GEMM访存**：先做不重排K累加的对齐向量加载/面板复用，再看M/N遍历顺序；L1/L2、SLM bank、寄存器压力需目标Intel实机工具核验，不能用HostCached或Vulkan时间戳冒称命中率
3. **纯复制与KV路径**：预算较严时mergeKv大量小窗口成本高，范围/别名检查后的transfer copy和分段prefix已用于当前查询，前缀仍按层从CPU页装配上传。后续须定义pool身份、代际、修改/淘汰失效，才能做跨调用KV镜像，未实现zero-copy或设备KV常驻
4. **decode验收**：同任务CPU/GPU延迟、copy payload与逻辑读写估算；copy payload不是实际DRAM流量，真实带宽还受权重/KV缓存命中和额外复制影响，不设1%MFU门槛

## 本轮验证

- 42模块ABI反射通过，包含预算常量255检测；原模块与Budget模块不能混装
- 双GPUvalidation与算子对照通过：非法倍数/uint32溢出、256/512/768/4096、native矩形、flat2D、row2D、超过65535组、部分覆盖尾部不改写、512线程注意力；相同内核按逐位比较
- 公开Settings路径默认与256预测逐位一致，非法1拒绝；双GPU原Tiny/Real2/2，预算256和4096的真实模型对照包含连续坐标、129patch、H1前缀63/513及只读页哈希
- 纯标准库关闭OpenMP等24/24（77.38秒），BLAS工厂增量编译通过。后续规范化/矩形窗口仅改GPU内部索引/调度，未重复无关CPU全量
- 8通道公开输出预算通过，但--strict-multi旧路径中间值偏差仍未定位，不写成8通道逐层全部达标
- `基础与模型`、`标准库`、`规范网格`、`矩形窗口`保存完整日志
- 尚无Intel实机，以上是AMD/NVIDIA开发验证，不是Intel性能认证。未用结果选最优预算

## 预算开销观测

`最终观测`只选0/512/8192展示严格与较宽逻辑预算的代价，真实5M单通道32k，prefill冷页中位3、decode预热5/计时25，实验残差关闭，不与构建重叠。不是自动调优或Intel速度预测。计时不含模型加载，冷JIT未单独计时；profiling另取5次用于窗口/流量归因，不替代延迟基准。

| 开发设备 | 逻辑预算 | prefill ms | H1 decode ms |
| --- | ---: | ---: | ---: |
| 680M | 0 | 44.80 | 9.06 |
| 680M | 512 | 461.64 | 21.55 |
| 680M | 8192 | 79.63 | 9.88 |
| 4060 | 0 | 27.40 | 5.59 |
| 4060 | 512 | 291.72 | 24.09 |
| 4060 | 8192 | 39.40 | 6.88 |

512预算非常严格，代价远非“只少用三分之一GPU”；8192也只是更宽上限，不是推荐值。窗口统计实测最大不超过设置，但未测真实GPU占用率/频率/功耗。mergeKv的纯复制shader在严格预算下形成大量小窗口，后续优先研究范围安全的transfer路径，而非把这些数字作为Intel调参结论。
