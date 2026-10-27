# 深度Vulkan后端优化树

## 当前模型精度路径

模型常用形状采用有序注意力、RMS归约和补偿FP32非线性，共享前缀不再按每条查询复制。G1投影加速保留，A2及其它在线归约核仍在独立基准和超范围回退中使用。当前52模块的部署及数值限制见[可移植设计](<vulkan-portability.md>)，性能图和长decode表的协议见[基准说明](<benchmark.md>)。以下优化树中的历史成绩不代表当前默认派发。

本文引用的记录、日志和转储都在本地工作区，不随仓库发布。

## 最新方向：Intel可移植性与GPU预算（2026-10-25）

当前目标已更新：面向13至15代、较小规模Intel核显，不按680M特调；decode不再用1% MFU验收，改看同任务时延、copy与带宽。之前300/30数字只保留作历史开发记录，不是Intel部署门槛。

已读用户提供的本地ggml关键路径，远程commit未知，部分审读范围与SHA256见[可移植设计](<vulkan-portability.md>)。AI Max395/RDNA资料已搜索，但外部正文仍被非公网IP校验拦截，未冒称完整代码审读。没有使用subagent或复制RDNA专核。

已新增Timestral::Settings::gpuThreads：0不限制；非0须为256倍数且在uint32范围内，约束每个串行dispatch窗口的逻辑invocations，不是物理EU配额。固定local_size、合法性与变体尺寸在host检查，预算小于512退避可选512线程管线；按设备报告8/16 subgroup时不创建仅32/64快速核。当前完整包52模块，原21源码/绑定/push不改。

残差融合原M≤2048是680M实验护栏，已撤出默认依据；现在默认关闭，显式开启实验不保留该大M上界。未宣称低SLM注意力、GEMM新面板或UMA KV镜像已经实现。完整接口、安全边界、验证和未覆盖Intel硬件见[新设计](<vulkan-portability.md>)。

## 历史：300/30GFLOP/s与多变量结果（2026-10-25早段）

本节是本轮目标调整前的采用状态与实测，当前默认策略以顶部可移植设计为准。未达到用户提出的680M prefill 300GF/s、H1 decode 30GF/s，不将第六期标记完成。无subagent，未改公开入口、权重/Tensor布局，未恢复arena/偏移push。

### 最终默认策略实测

`末次完整矩阵`对应当前源码，另有`结构化汇总`。每通道32768采样、同一组内不同数据，不是独立batch；targets全部共同预测。cold prefill三次中位，H1 decode预热5次/计时25次中位，所有任务串行且不与构建并行。记录的是这轮中位数，不是所有轮次中最小值，未测实时频率/功耗，不能把跨轮波动定性为驱动问题。

| 同组通道数 | 680M prefill ms | 680M H1 ms | prefill GF/s | 680M名义MFU | 4060 prefill ms | 4060 H1 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.47 | 8.48 | 256.8 | 8.56% | 21.68 | 5.11 |
| 2 | 76.31 | 17.67 | 279.1 | 9.30% | 42.20 | 11.60 |
| 4 | 165.99 | 28.23 | 256.7 | 8.56% | 98.37 | 22.07 |
| 8 | 328.71 | 55.03 | 259.3 | 8.64% | 189.49 | 48.93 |

680M decode仅约1.84至2.37GF/s，即0.061%至0.079%名义MFU，远低于1%。单变量prefill为256.8GF/s、8.56%，仍需约16.8%的吞吐提升，或把41.47ms降至35.50ms。2/4/8通道prefill也未达300GF/s。4060不使用680M的3TFLOPS分母推算自身MFU。

同机本轮较早`CPU参考窗口`的BLAS encode依次170.85/336.81/683.07/1424.89ms，decode为4.21/7.85/14.90/36.04ms；不是最后一次GPU逐项配对的CPU重测。GPU prefill明显快于该参考，decode仍较慢。8通道总输入为262144采样，不能把耗时增长解释为同工作量回退，也没有出现随通道数增加而成倍提高的总吞吐。

### 已知未来协变量

一个Target加其余KnownCov，提供真实未来观测，同组总通道数为2/4/8。当前分页追加输入不含roles/future，此模式为完整条件forward，预热5次、计时5次中位，不能标成缓存decode。以下取日志中metric=full_conditioned_forward的稳态计时；兼容输出的decode_ms字段在此模式仅是旧字段名，不代表增量解码。

| 条件 | 总通道数 | 680M完整forward ms |
| --- | ---: | ---: |
| 1 Target + 1 KnownCov | 2 | 294.44 |
| 1 Target + 3 KnownCov | 4 | 644.15 |
| 1 Target + 7 KnownCov | 8 | 1344.05 |

这些forward返回归一化分位数，不能拿首输出0.50与分页反归一化的2.38直接比较精度。完整双流必要FLOPs尚未单独推导，因此不报告该场景MFU，也不声称协变量一定提高预测质量。

### 采用、回退与无收益叶子

- G1补齐实际embedding投影(K,N)=(64,768)/(768,192)/(64,192)，仍逐K precise，12形状×bias与v1逐位一致
- A3用独立causal核裁剪能证明全空的编码未来键块，不改变真实K/V步距；KnownCov双向和双流后缀保留原谓词
- A5 subgroup在compute Basic/Vote/Arithmetic和39300B共享能力满足时启用。shader验证每subgroup只含同一query，size32/64才走快速路径，否则整个工作组回退；64-lane直接归约，32-lane共享合并，不写死warp32。完整条件forward也经过长上下文模型对照
- P1 KV-only prefill在最后time的K/V就绪后停止，无用的末尾attention/gate/O、后续variate/MLP、out_norm与hidden回读不执行；完整forward、trace、decode不截断。最后QKV投影仍多算Q，不把这部分无用Q计入任务产出
- G2当前是64线程、16列/组、合并权重加载的逐K顺序GEMV，用于M1至8的完整原生grid。32-lane并行K候选在M1/N672/K768算子及共享缓存/完整前向预算失败，已撤下，未放宽预算
- A4 count1、time/headDim64、键数至少512时按128键split，每片输出max/分母/未归一化PV，再稳定合并，空split不会产生NaN。通过尾片、缺失、角色、多行及H1长前缀模型对照，但不据此声称整个decode已达到BLAS或30GF/s
- F1残差块设备常驻只默认用于128≤M≤2048。hidden投影、SiLU、output与skip投影及相加放在同batch，只回读最终值；小head没有净收益，大M方向不稳定，所以回退原三投影。开关TIMESTRAL_VULKAN_RESIDENT_RESIDUAL=0仅关闭此叶子，1也遵守形状护栏，非法值拒绝
- Rotary只复用本调用的inverse频率和完全相同position行，未增加跨请求shape缓存，连续坐标仍按原std::cos/sin计算
- [gemmWide](<../timestral/vulkan/shaders/gemmWide.comp>)只留显式诊断：N576/1024仅约5%微基准改善，M128及K512退化，不自动选用。[attentionPacked](<../timestral/vulkan/shaders/attentionPacked.comp>)128线程候选正确但慢于subgroup，不自动选用。[寄存器PV](<../timestral/vulkan/shaders/attentionRegister.comp>)仍仅诊断

`同进程注意力对照`的time1024中位：v1 12.37ms、A2 4.00ms、causal2.47ms、subgroup2.04ms、packed3.72ms，含录制/提交/fence、不含传输，不当作device-only。F1单变量独立A/B由46.81/46.90ms降至40.55/41.05ms（`R1及显式诊断有效段`），但四通道两对结果方向相反，完整数据保留在`首对`与`唯一反序复核`，不取169ms当稳定收益。

早期PowerShell标签拼接让非R1参数成为单个label，日志实际channels=1，那些行已作废；上面的原日志仅取R1与显式诊断段。多通道采用参数数目校验后的checked记录，最终表又核对了实际channels/case，不拿错标行作2/4/8通道结果。

### 精度与默认构建

`最终策略验收`：AMD与4060各自Tiny/Real 2/2，原1/2/3行逐层与全分位数阈值不变，新增63/513patch H1缓存/完整/reference、只读页哈希、129patch已知未来协变量。63与513分别跨越残差融合下界与上界。新增siluGrid覆盖65535组边界，21模块ABI反射和双GPU validation见`融合前置验收`。最后只清理回退前无用的临时输出分配，再构建及Tiny冒烟通过，随后采集末次矩阵。

显式关闭OpenMP、Eigen、ytensor、BLAS、Vulkan的纯标准库构建，实际24/24通过（82.56秒），不是复制历史25/25。reference/BLAS保留原residual默认实现，只有内部虚拟数值钩子扩展。

8通道公开forward/predictBlock的既有输出预算通过，但新增严格逐层诊断尚未通过。关闭所有新选核后仍在同一layer6/index7459失败；逐项RMS的候选也未解决，已撤下。保留[测试](<../tests/gpuModelTest.cpp>)的--strict-multi开关重现，不把默认8行输出测试写成8行逐层全部达标。该精度边界仍待定位，不放宽阈值。

### 剩余主线

1. E3先评估16个K/V回读能否同一提交后集中读取，当前仍串行，不能把17次prefill提交说成1次。需保持异常时command引用资源的存活和poisoned语义，不先做arena
2. K1在定义pool身份、页代际、淘汰/复用/修改失效后，才实现设备KV镜像和分段prefix读取，减少每次H1约12MiB CPU装配/再上传与16个mergeKv；目前未实现
3. decode诊断CPU API外约2.5至3.4ms、GPU提交跨度约4.6至5.4ms，二者分别已高于0.543ms目标，时间有重叠不能相加，不能只把少提交两次当成1% MFU方案
4. 8通道严格中间值偏差、实时频率/功耗/显存预算仍需处理。ggml正文获取再次被非公网IP校验拒绝，未完成固定commit审读，未复制源码

## 残差投影块设备常驻试验前计划（2026-10-24续）

补测后的680M单变量prefill45.7700至51.8650ms，仅205.32至232.66GF/s，未达到300；H1 decode9.2596至9.3902ms，仅1.73至1.76GF/s。数据已落盘到`最终矩阵`，不沿用宿主重启后丢失的后台输出。

下一步只试一个已有归因支持的边界：patch/head residual目前三次独立project，每次上传/下载，SiLU与加法在CPU，诊断CPU API外约12ms。把内部Transformer::residual变成有默认实现的protected数值钩子，reference/BLAS等默认顺序不变；Vulkan通过一次batch完成hidden投影、SiLU、output投影、skip投影与加法，仅回读最终值。GpuTransformer的其它API默认不启用，Vulkan单独可控，不要求arena/偏移push改造。

保留原三个projection的逐K、bias与加法顺序。新增二维grid正确的siluGrid，避免8通道完整forward超过65535组后误用只看X的旧验证核。GPU exp与CPU libm存在差异，必须先过既有算子/逐层/全分位数/缓存预算；不过关则撤下默认启用，不放宽预算。通过后用独立TIMESTRAL_VULKAN_RESIDENT_RESIDUAL=0/1做A/B，不同时改其它优化。若保留，公共头/权重布局不变，但内部虚拟钩子变化需要默认std-only回归一次。

## 300/30GFLOP/s与多变量推进（2026-10-24晚间）

用户要求继续探索680M prefill 10%（300GF/s）、任务decode 1%（30GF/s），同时补2/4/8通道表现，不使用subagent。目标不是保证，必须报告端到端实测差距，不能拿纯算子峰值替代。旧Goal恢复接口仍拒绝，按本次直接授权推进，不伪称自动Goal已恢复。

### 协议与必要任务量

扩展既有backendCompare：CHANNELS=1/2/4/8、每通道32768采样、相同groupId但不同通道数据，不是独立样本batch。targets为所有通道共同预测，按原cold prefill中位3、H=1预热5/计时25；known为一个Target加R-1个KnownCov且提供真实未来观测，目前公开分页输入不含roles/future，不能冒充缓存命中，单独报告完整条件forward（预热5/计时5）。性能不等于预测质量提升，数值验证另做。

从本轮开始固定必要KV-only任务FLOPs，不把Vulkan无用的最后attention/MLP算作产出。32k单Target prefill为10.648682496G，H=1 decode为0.016289280G，分别需≤35.496ms、≤0.543ms才能达到上述目标。此前12.260475G是未裁剪GPU执行图的主要运算量，保留历史但不能据此让裁剪后的MFU虚高。多变量time部分按R扩展，variate部分按R²计，不能只把单变量FLOPs乘R。known完整双流mask未推导必要运算量，不报告虚构MFU。

### 本批实施顺序与进入门槛

1. 串行保存并测量cached版1/2/4/8通道与同窗BLAS，再测KnownCov条件forward，性能窗口不与本会话编译重叠
2. G1补实际embedding形状N768/K64、N192/K768与N192/K64，保持逐K precise；新增正确性形状后再分派
3. 复查causal Empty块，仅time编码区、块内所有query都在encoded之前且非KnownCov双向时裁剪，保留真实keyCount行步距。此次在cached回读和时间戳下裁决；decode不带此prefill分支，必要时用独立shader避免寄存器/分支退化
4. GPU KV-only prefill在最后time轴K/V完成后停止，不执行对页内容无作用的attention、gate/O、后续variate、MLP、out_norm/隐藏回读，完整forward/trace/decode不裁剪。初版仍算完整QKV但不把无用Q计入任务FLOPs，之后才考虑独立KV投影
5. G2采用32×4线程，每个32-lane组归约一列K点积，四列/组、按row派发，面向M≤8，权重按K连续加载。会改变归约顺序，以原atol/rtol与hidden relL2/全分位数预算裁决，不强求字节相同，也不放宽预算
6. A4单query长KV按128键拆分，每head/row多工作组，64线程共同加载K、PV保持lane维度连续，输出每split的max/denominator/unnormalized PV，再独立FP32合并。不假设warp宽度，不把split结果平均当注意力；masked/empty split必须有稳定零分母处理。只在count1、time/headDim64且长KV时派发
7. 补一个设备能力门控的subgroup归约候选：查询compute Basic/Vote/Arithmetic，shader先检查每个subgroup是否只含同一query及size32/64；不成立整个workgroup走原归约。符合时用subgroupMax/Add加少量共享合并，减少每64键的workgroup barrier，不依赖lane顺序或写死warp32
8. 对照64×64输出tile、256线程、每线程4×4寄存器的gemmWide，K块16、共享8256B，仍逐K precise。只在主模型大M形状的常驻微基准与端到端都获益时启用，M128等小grid保留32×32。Rotary只做同调用inverse频率和相同position行的精确复用，不新增跨请求缓存
9. 针对attention仍占设备端首位，再评估128线程的8-query×64-key块：每线程四个QK累加与四个PV维度，16-lane共享归约、37632B共享；避免一线程一score的串行链，仍按原mask与在线softmax。归约顺序改变，需冻结预算与模型对照，通过后才分派
10. 每个新数值路径先最小oracle/模型对照再性能，只重复有方向性争议的对照一次。记录单/多变量和KnownCov实际模式与差距，未达目标不写成达标

### ggml参考取得情况

本轮再次尝试raw.githubusercontent.com固定模块与api.github.com提交信息，均被非公网IP校验拒绝。搜索找到[固定镜像mul_mm.comp](<https://tangled.org/okami.mom/llama-cpp-bonsai/blob/01fd9521c92e7882a3fa1083bb933e3bd9305bef/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp>)和[后端说明](<https://deepwiki.com/ggml-org/ggml/3.3-vulkan-backend>)，两个域名同样拒绝，未读到正文，未复制实现或声称完成ggml微内核审读。以上候选为基于本地已读shader与GPU访存/归约原理的实现，后续取得源码再核对tile与设备分派，不让网络等待阻塞其它独立工作。

## 680M专项归因：低MFU与错误的回读内存选择（2026-10-24）

用户指出680M约3TFLOPS而端到端只利用到几个百分点。这个判断成立，前一轮只证明局部shader变快，没有完成执行时间归因；把提交、传输笼统称为主瓶颈也缺乏证据。本节取代这些猜测，不把新出现的单轮最好值当作优化完成。

### MFU重新按当前执行范围计算

按用户指定名义FP32峰值3000GF/s，不用CPU的134.4GF/s分母解释GPU效率，也不声称测到了实际频率。单Target、全观测、1024patch、D192、FFN512、8个time轴与2个variate轴：

- Transformer投影为8.606711808GFLOPs，patch residual的三次投影为0.427819008GFLOPs
- time/variate可见对的有效attention为3.225944064GFLOPs，合计主要有效运算12.260474880GFLOPs
- H=1完整前缀decode包含head三次投影，主要有效运算0.016289280GFLOPs，不可沿用encode的运算量
- RMS、exp、RoPE、逐元素、缓存管理、拷贝和内存初始化不计FLOPs，但时间计入端到端。Vulkan还对mask为零的PV做部分无效工作，实际执行指令量不等于有效模型FLOPs

因此此前143.5030ms encode只相当于85.44GF/s、名义MFU约2.85%。3TFLOPS是固定假设分母，不是本轮GPU频率或硬件busy counter。当前图理想纯算术下限约4.09ms，不是可承诺的端到端延迟；几十至上百毫秒的实现显然还有很大差距。旧的完整forward/J=4公式不能直接套到KV-only prefill或H=1 decode。

### 归因工具与计时边界

新增vulkanProfile归因工具（本地工具，不入库），通过GpuExecutor外层独占API计时、执行器内部提交/等待/回收计时，以及Vulkan query timestamps统计每核和copy。默认模型诊断关闭；只在显式工具里开启，不改shader、ABI、同步屏障或提交边界。查询使用实际计算队列的timestampValidBits，处理计数器回绕，池容量不足报告dropped_events，不伪造零耗时。

工具用相同32k输入、冷页池encode三次、decode预热五次后诊断计时五次，按wall中位那次保存完整统计，不分别取每列中位拼成一次执行。GPU事件在compute或transfer同阶段打点，提交跨度为top到bottom；事件和、GPU跨度、CPU fence等待相互有重叠，不能相加。时间戳可能改变调度且本机GPU耗时波动明显，诊断时间不替代无诊断的backendCompare 5/25基准。早期两份日志的11.832656G只计Transformer图，最终公式已补入patch/head投影为上述12.260475G/0.016289G。

### 已证实的encode主机瓶颈

`细分前记录`中一次encode wall176.3228ms，GPU提交跨度85.96935ms；CPU分配0.0342ms、dispatch录制0.5251ms、回收0.2878ms，均不是主导项。GPU upload+download copy约1.92ms，不能拿“CPU到GPU传输慢”解释整个差距。

`细分后的记录`确认以下数据，并与`修正后记录`对照：

| 同一项诊断 | 修正前 | cached回读后 |
| --- | ---: | ---: |
| encode wall ms | 228.0989 | 143.5917 |
| GPU提交跨度 ms | 120.22921 | 120.94282 |
| CPU map+读取+unmap ms | 77.5258 | 1.5239 |
| 读回字节 | 17.25MiB | 17.25MiB |
| 实际回读memory flags | 6，Visible+Coherent | 14，Visible+Coherent+Cached |

680M报告的type1为flags6/heap1、type3为flags14/heap1。原实现总选第一个host-visible coherent类型，却把适合CPU写入的staging也拿来给CPU读取。没有HOST_CACHED的CPU读取非常慢；统一物理内存不意味着所有Vulkan内存映射都同样适合CPU访问。这里不是独显PCIe传输解释，更不是设备GEMM峰值不够。

已修正：独立readback复用池优先选择HOST_VISIBLE|HOST_COHERENT|HOST_CACHED，不和upload staging混用；无cached coherent类型时回退原coherent类型，不引入未处理的flush/invalidate语义。回读仍等fence和transfer→host可见性，所有数据真实回读，公开API与精度不变。

### 不带诊断的端到端A/B

所有任务串行，没有与本会话构建或其它GPU测量重叠。`首轮`旧版228.5427/10.8646ms、BLAS181.9911/4.7145ms，明确证明旧版当时比BLAS慢。随后用保存的旧程序与cached版做两种顺序，协议仍为cold encode中位3、decode预热5计时25：

| 轮次 | 旧版encode/decode ms | cached版encode/decode ms | BLAS encode/decode ms |
| --- | --- | --- | --- |
| cached先、旧版后 | 185.1625 / 16.9553 | 88.0076 / 11.9052 | 182.7768 / 4.8753 |
| 旧版先、cached后 | 196.6105 / 11.9873 | 85.2186 / 16.8519 | 186.6453 / 4.6018 |

原始反序记录见`final-ab`。encode两轮改善约2.10至2.31倍，cached版快于同窗BLAS约2.08至2.19倍，但MFU也只有约4.64%至4.80%，仍远未优化充分。decode方向不一致且仍明显慢于BLAS，本轮不宣称decode加速，不为得到满意结果继续反复测。GPU时间也有明显波动，未采集频率、功耗或其它系统负载，不能擅自归因为驱动或降频。

### 修正后的优化优先级

1. 已修复大回读的缓存属性错误。下一步评估CPU是否真的需要每个中间值，以及20次encode提交（embedding三次、整图一次、KV十六次）能否减少；不是先重写描述符池
2. attention是GPU第一热点：修正后诊断time attention八次合计72.11564ms，占GPU跨度约60%；另一轮为47.58ms，排名一致但绝对值不能混用。当前512线程/39168B共享的块仍有大量workgroup barrier和串行64维点积。应在GPU时间戳下比较更合适的tile/归约和causal Empty块，而不是只看被回读掩盖的端到端噪声
3. G1分派漏了实际embedding大形状：M1024/N768/K64与M1024/N192/K768仍用v1，诊断分别可到3.81和12.51ms；补这些形状的微基准和预算对照优先于优化已很快的逐元素核
4. decode必须有真正的GEMV和长KV专用attention。M=1投影仍用16×16 GEMM，15/16输出行无效，主形状只有约2至8GF/s。A2在count=1时只有三个head工作组，七个query槽无效且不能铺满12CU，需要单独的lane分工与split-K/partial合并，而不是照搬prefill核
5. decode另有12MiB显式prefix上传和十六次merge，CPU API外约3ms含页校验、prefix装配和公共数值。先进一步归因，再做页镜像/直读，不把未知CPU时间全算成PCIe拷贝

旧A3a“端到端收益不稳”不等于已经证明GPU核无收益：原80ms级CPU回读和GPU波动会掩盖设备侧差异。保留历史结果，但应在修正内存与具备时间戳以后独立复查prefill核，不能把分支无条件带进decode。FMA峰值与precise分离乘加的吞吐口径也不同，但即使承认算术上限约束，也解释不了当前几个百分点的整体效率，不作为停止优化的理由。

本次没有实现GEMV、split-K或新attention tile，也没有放宽数值预算。新增诊断开关字节一致性、计数与coherent回读测试，双GPUvalidation和Tiny/Real定向通过。默认CPU代码与shader均未改，不为诊断反复重跑全部CPU测试。

## 本轮状态与边界

2026-10-24按用户要求重新设计，规划先于实现，完成规划后沿树推进，不使用subagent。旧L1至L6不再作为实施顺序，失败记录保留在本文末尾。v1回退点及恢复方法记在备份说明中；该备份目录属于本地工作文件，不随仓库发布。

优先级为精度、易用性、可测的性能收益。公开API、模型文件和CPU数据布局不变，默认std-only构建不引入Vulkan或系统BLAS。默认优化只用FP32，保留逐K乘加与掩码语义；FMA、并行归约和低精度分别验收，不能借放宽预算启用。

## 从失败中改变什么

- 不再把arena、描述符缓存、权重上传、十个shader的push布局一起改。保留v1独立缓冲与abiVersion=1，以新增kernel扩展，而非替换全部协议。
- 执行器优化与算子优化是独立分支。池复用不要求tensor偏移、动态SSBO或权重搬迁；共享内存分块也不要求改执行器。
- AMD异常没有根因结论。validation无报错不等于应用正确，4060通过不替代AMD验收。任何设备失败，仅禁用该分支，不阻断其它独立优化。
- 不预设每层有收益。先写成本假设，做小范围对照；达到数值要求但端到端无收益，记录为候选或拒绝，不累积默认复杂度。
- 不反复跑整仓测试或为单个实验新建构建树。复用现有Release树和测试，只在涉及的边界验证；正式阶段收口再跑一次GPU树与默认构建回归。

## 基线与瓶颈假设

第五期32k历史基线（非本轮同时段A/B）：

| 设备 | encode ms | decode ms | 首输出值 |
| --- | ---: | ---: | ---: |
| RTX 4060 Laptop | 120.8538 | 7.7143 | 2.3805 |
| AMD集显gfx1035 | 236.9136 | 11.8549 | 2.3806 |

backendCompare（本地工具，不入库）的checksum是首输出值，不是完整数值校验。encode为32768采样点、1024个patch冷页池prefill，3次中位；decode为完整前缀H=1，5次预热、25次计时中位。重新优化的收益用本轮保存的v1程序同设备、同窗口串行A/B判定，不能用历史最快值算加速。

代码能确认的成本与仍需测量的假设：

- [Executor](<../timestral/vulkan/executor.cpp>)逐dispatch创建一个描述符池和一组描述符；这是明确的CPU/驱动成本，但占比未知。
- [gemm](<../timestral/vulkan/shaders/gemm.comp>)一个线程计算一个输出，16×16工作组在M=1时只有16个线程写出，K循环反复barrier；W的全局加载在K方向不合并。寄存器tile和合并W加载都有独立价值。
- [attentionFused](<../timestral/vulkan/shaders/attentionFused.comp>)每query/head一组64线程，每lane串行算一个key的QK，每块把PV部分和写回全局输出再读回。V读取沿j连续，不能简单称为全部散读；没有跨query的K/V共享。1024×1024×192×4约0.805GB是单层不计掩码的V逻辑读取上界，不是旧文所称8GB实际显存流量，缓存命中和编译器优化还会改变实际流量。
- 分页decode每个time层在host装配前缀、上传K/V并merge。跨调用驻留能去掉重复传输，但必须先解决页身份、淘汰和模型生命周期，不能把CPU pageId单独当稳定键。

## R优化树

下图是优化空间与历史实施顺序，状态见实施记录。各叶子独立保留v1回退；680M实测后优先级调整为cached回读、attention时间戳归因、遗漏的embedding形状、真正的GEMV与decode注意力，再按收益进入分页镜像与融合，不再把描述符池放在主线前面。

~~~text
R  FP32深度优化，v1独立缓冲/abiVersion=1为根
├── R0  稳定基线与轻量观测
│   ├── 保存v1基准程序，双GPU串行A/B
│   └── 只按需要统计dispatch、提交、传输与热点时间
├── E   执行器CPU/驱动成本，与算子分支独立
│   ├── E1  多set描述符池复用，fence完成后reset
│   ├── E2  按槽预分配set，或按物理buffer/range缓存
│   ├── E3  批量回读/提交，保留同步与poisoned契约
│   └── E4  upload/readback分池，回读优先HOST_CACHED（已修复）
├── G   FP32投影，先保持逐K顺序
│   ├── G1  32×32输出tile，16×16线程，每线程2×2累加
│   ├── G2  M≤4窄核，lane沿输出列，串行K保持精度
│   └── G3  可选K并行GEMV、FMA、权重面板布局
├── A   注意力，time与variate分开派发
│   ├── A1  headDim=64寄存器PV，避免每键块输出往返
│   ├── A2  FP32多query共享K/V块，在线softmax
│   ├── A3  Full/Empty块裁剪，严格双流mask
│   └── A4  decode split-K与小variate专用核
├── F   融合与静态数据复用
│   ├── F1  residual+RMSNorm、gate+投影epilogue
│   └── F2  单调用rotary复用，再评估有界跨调用缓存
├── K   分页数据驻留，不改公共页池语义
│   ├── K1  先测host装配/上传与merge占比
│   ├── K2  稳定页身份+代际+失效契约，设备镜像
│   └── K3  页表直读attention，取消前缀merge/上传
└── X   可选设备特化，不是默认路径的依赖
    ├── subgroup归约：查询subgroup能力，不写死warp=32
    └── cooperative matrix：逐组件特性探测+独立精度预算
~~~

## 分支设计与进入条件

### R0与E：先消除池创建，不改寻址

E1每池容纳256个set、最多1280个storage-buffer描述符，按需增加池。每dispatch仍分配独立set并写真实buffer/range，录制后直到fence完成不可修改。成功submit先释放command buffer再reset已用池，供下个提交复用。download后的同batch可以继续录制；end后的scratch句柄仍失效。池拥有普通DescriptorSet句柄，不构造不完整的UniqueDescriptorSet，不做逐set释放，因此不需要eFreeDescriptorSet。

E1只改执行器与生命周期回归，不改shader、manifest或共享GPU接口。验证覆盖同cb首dispatch、跨256边界、先上传后dispatch、download后继续、跨batch复用；两设备都必须通过。出错后未排空前不可reset池，沿用poisoned语义。

E2仅在E1之后仍看到set分配/更新是热点时启动。若采用缓存，键包含layout、真实buffer身份、offset、range与代际，不能用会失效的逻辑句柄缓存。命令未完成前禁止更新已引用set。E3先考虑多个结果一次回读，不能删除必要fence、屏障或借异步返回改变现有接口契约。

### G：先修访存与寄存器复用，再改数值算法

G1输出tile为32×32，工作组为16×16共256线程，不是32×32个线程。K块16，A共享32×16，W转置到16×33（含padding），共4160B。每线程加载两项A与两项W，W沿K连续，计算2×2寄存器结果；每个输出保持与v1相同的K顺序、precise乘加、bias后置。先用标量合并加载避免vec4对齐与尾块另加风险。尾行/列/非16整K补零并参与所有barrier。

G1仅在M和N足够大时考虑启用，主要模型形状为K=192/512、N=192/576/1024；最终边界由常驻微基准与端到端裁决，而不是默认M>4全走新核。未测或退化形状保留gemm。

G2处理M=1至4时优先尝试相邻lane负责相邻输出列、每lane串行K，输入块共享，避免15/16的行空转。若W布局带宽不足，再试加载时转置或面板预打包。G3的K并行归约和FMA会改变累加顺序，必须单独做误差检查，不与G1绑在一起。低精度权重不属于本分支。

### A：先减少输出往返，再做真正的多query块

A1为headDim=64新增同ABI核，每lane一个PV寄存器，QK与softmax块仍按64键推进，保留v1归约和掩码，最后一次写结果。禁止假设global输出一定没有被编译器驻寄存器优化，收益必须实测。其它headDim走v1。

A2首候选限定time轴、headDim=64，输出块为8个query×64维。采用64×8共512线程，每个query的64lane仍各算一个key的QK、同组64lane归约softmax、每lane累计一个输出维度。K/V块64键共享，K转置寻址用65列padding，另加Q与每query独立scores/reduction，总共享39168B，低于两设备共同上限48KB。选择512线程而非初稿128线程，是为了先保持原64键归约树和逐K/PV顺序，不同时引入另一种分工/舍入变化；设备达不到512线程或共享上限时保持v1。variate轴不同query槽位对应不同K/V，不可套用同一共享块，继续独立优化。先选择这一tile候选，不做全参数穷举。K/V块复用、在线最大值/分母重标定保持FP32，全屏蔽行输出0，尾query/key不在barrier前退出。在线softmax的分块与归约顺序会变，不能预先承诺逐位一致。

A3沿原timeAllowed/variateAllowed分类。首个窄化叶子A3a仅针对同一time块内全部query都小于encoded且不属于KnownCov双向角色：此时timeAllowed要求key≤query，因此超过块内最大query所在64键块的后续块全部Empty，可统一终止K/V循环。仍保留原keyCount作行步距、原尾键掩码和块内逐对谓词，不能缩短实际K/V行stride。其它情况不裁剪；只在逐位对照与端到端收益通过后启用。其它Full/Empty分类后续独立处理，known-future、缺失值、不同groupIds不能退化成纯causal。A4只在decode注意力确为瓶颈时做，split-K需要额外partial缓冲与合并，少query并不天然受益。小variate在keyCount很小时避免通用64键循环和大块共享。

### F：融合围绕稳定中间值，不复制模型流程

F1先做residual输出与RMSNorm同核，严格保持y=x+delta、平方和、(y*inverse)*scale的原顺序；需要trace或两份输出的调用仍能读取y。不把CPU语义搬入后端。gate融合到投影epilogue仅在数值一致且不增加重复计算时启用。

复读v1层图确认rotary已经在层循环外单次上传，旧L5“每层重复上传”的判断不成立，因此F2的单调用部分直接记为已有，不再重复实现。只有测到跨调用rotary上传成本值得优化时才加缓存，键必须包括实际rotary内容或全部位置参数，不能仅用count/contextPatches，因为continuousRope和位置偏移会改变数据。缓存设容量上限，不以无限persistent upload代替资源管理。

### K与X：高风险项延后且有明确退出条件

K2在设计页镜像接口前先读页池的代际、引用和淘汰行为，键至少区分模型、pool身份、pageId与内容代际。scratch、LRU淘汰、abort、不同通道几何、共享前缀、FIFO都必须正确失效。CPU页池仍是权威来源，新页回读保留，不改变page-hash和共享只读语义。容量超限可丢弃镜像并回退上传，不要求每个CPU页同时驻GPU。无明确传输占比或生命周期契约时不实现K3。

X不承诺VK_KHR_cooperative_matrix存在就能运行某种FP16/BF16组合，须查询实际支持的矩阵尺寸、组件类型和FP32累加能力。TF32不是BF16的开关名。本轮不先增加低精度选项；精度预算若未确定，保持未实施。

## 精简而有效的验收

1. 每个叶子只跑受影响的边界：shader改动做glslc+spirv-val+ABI反射与新核小集合oracle；纯池复用不重复编译十个未改shader。
2. 接入模型后运行现有gpuModelTiny/Real，两设备各一次，包含逐层、全分位数与分页；发现失败先定位，不用几十次全量重跑碰运气。
3. 算子候选只测模型主形状及尾块。常驻微基准把上传/读回放在计时外，或明确标为host录制/提交含量；原逐算子传输基准不能当纯shader吞吐。
4. 性能窗口串行独占GPU，与保存v1或上一接受节点做A/B。明显无收益直接停止该候选；接近噪声时最多补一次反序A/B。两轮仍不确定，保持v1默认，不调阈值追逐单次最好值。
5. 默认启用门槛：正确性全部通过，目标路径稳定改善且另一主要路径无明确退化。超过约3%的方向性差异作为复测触发值，不作为精度或绝对统计显著性保证。按设备/形状分派须集中，禁止静默吞错再换结果。
6. 数值预算沿用既有测试，不以首输出checksum代替：算子atol=1e-5、rtol=1e-4；hidden尺度感知atol与relL2≤1e-5；最终分位数atol=1e-4、rtol=1e-3；缓存一致按现有冻结规则。纯执行器变动预期输出逐位一致。
7. 一个有意义阶段收口才跑一次完整GPU树、默认std-only回归和风格检查，更新本文与本地进度记录。CPU名义MFU分母134.4GF/s不拿来冒充GPU硬件利用率。

## 实施记录

| 节点 | 状态 | 正确性 | 本轮性能 |
| --- | --- | --- | --- |
| R0 | 完成 | 保存本轮v1程序 | 初始4060 126.6051/7.7350ms，AMD 247.1702/12.1213ms |
| E1 | 已实验，撤下 | 双GPU validation、257边界/跨提交/首dispatch、Tiny/Real通过 | 4060 decode小幅改善，AMD encode退化，未全局启用 |
| G1 | 接受并限定形状启用 | 双GPU9形状×偏置开关与v1逐位一致，小尾块CPU oracle通过，Tiny/Real含129patch长上下文通过 | 主要1024行投影4060约2.27至2.32倍，AMD约2.34至4.04倍 |
| A1 | 已评估，只保留显式诊断核，不自动选用 | 双GPU与v1逐位一致 | 单独寄存器PV没有稳定显著收益 |
| A2 | 接受，time/headDim64/键数≥128且能力满足时启用 | 尾query/尾key、二维grid、多行多头、缺失/角色/base/encoded与v1逐位一致；双GPU模型通过 | 1024query time微基准4060约2.96倍、AMD约3.23倍，见下 |
| A3a | 正确性通过，收益不稳定，撤下 | 双GPU逐位通过 | AMD encode改善，4060两次方向不一致且decode退化，暂不默认启用 |
| G2、A3其它子项、F、K、X | 规划 | 未实现 | 未测 |

### E1无净收益实验

纯池复用不改任何shader或上传路径，双GPU全部定向正确性与validation通过。单独保存E1程序后做两次反序32k对照：

| 设备/轮次 | v1 encode/decode ms | E1 encode/decode ms |
| --- | ---: | ---: |
| 4060第1轮 | 128.4104 / 8.2558 | 127.9981 / 7.6962 |
| 4060反序 | 126.7515 / 8.0593 | 123.7811 / 7.8064 |
| AMD第1轮 | 248.1191 / 12.0481 | 275.5350 / 12.0973 |
| AMD反序 | 265.5561 / 12.0808 | 275.2834 / 12.0375 |

4060 decode改善约3%至7%，AMD decode无方向性改善且encode退化约4%至11%。按门槛撤下E1，不增加厂商特例保住弱收益；执行器仍逐dispatch独立池。池边界回归保留为大批量描述符生命周期测试，旧arena问题没有在E1复现，但根因仍未知。未来E2若再研究，必须重新提出独立收益证据。

### G1常驻微基准

在既有vulkanBench（本地工具，不入库）新增resident模式：5次预热、25次计时，每次连续8个dispatch，以每dispatch计时，含host录制、屏障、提交与fence，不含上传/回读，不称纯GPU时间。以下为本轮median，所有输出与v1逐位一致：

| M/N/K | 4060 v1→G1 ms | AMD v1→G1 ms |
| --- | ---: | ---: |
| 1/576/192 | 0.020112→0.020788 | 0.037187→0.035425 |
| 128/192/192 | 0.030113→0.022238 | 0.065200→0.039800 |
| 1024/576/192 | 0.382188→0.165175 | 1.123037→0.277750 |
| 1024/1024/192 | 0.619525→0.266575 | 1.923963→0.822250 |
| 1024/192/512 | 0.327550→0.144425 | 1.069613→0.301212 |

模型内选核集中在Vulkan Executor，不修改跨API的GpuTransformer契约。只对M≥128且(K,N)为(192,192/576/1024)或(512,192)的完整原生gemm grid启用G1，M=1及未测形状保持v1。内部独立Executor默认关闭自动选核，供oracle与基准明确调用两个核；模型默认开启，TIMESTRAL_VULKAN_OPTIMIZATIONS=0可整体关闭自动选核，1为开启，非法值明确拒绝。该开关不改变已编译shader包的文件集合。

### A1/A2注意力实验

常驻attention微基准每次4个dispatch，5预热、25计时，包含录制/屏障/提交/fence而不含传输；mask为普通causal，三头、headDim64。A1不自动启用，A2仅用于time轴，保留完整双流mask。

| 设备/形状 | v1 ms | A1 ms | A2 ms |
| --- | ---: | ---: | ---: |
| 4060 count1024/base0 | 2.804500 | 2.660175 | 0.946100 |
| 4060 count1/base1024 | 0.226275 | 0.222725 | 0.088600 |
| AMD count1024/base0 | 13.108300 | 13.059775 | 4.052275 |
| AMD count1/base1024 | 0.448200 | 0.444650 | 0.189950 |

A1此前另一轮4060长prefill为2.654600→2.644050ms、AMD为12.581050→13.006950ms，没有稳定大收益，不把寄存器改写本身记成主要加速。A2的改进还包含K合并加载与共享读取，单query不靠query复用也能受益；代价是大工作组与39168B共享占用，因此不对短键、variate或其它headDim自动选用。

### 32k端到端与波动

G1单独第一次为4060 123.3685/8.4713ms对v1 130.5715/8.1300ms，AMD 248.7492/14.8189ms对314.2363/14.9267ms。说明投影微基准倍数不能直接外推整模型，decode也不是G1目标路径。

G1+A2第一次同程序开关A/B为4060 110.7752/7.6756ms对v1 131.2195/8.2253ms，AMD 199.1684/12.4841ms对264.6728/12.0622ms。AMD当轮decode略慢，后续同协议复测变化较大，不能只报告最佳一轮。

末次反序窗口，采用相同程序与A2 shader，encode/decode分别如下；紧邻v1对照期间插入一次A3候选测量，均串行、无构建与其它本会话GPU任务：

| 设备 | v1 ms | 接受的G1+A2 ms | encode/decode比值 | 首输出值 |
| --- | --- | --- | --- | --- |
| RTX4060 Laptop | 125.9646 / 8.0188 | 109.8991 / 6.7479 | 1.146 / 1.188 | 2.3805 |
| AMD集显 | 263.0007 / 12.9188 | 143.5030 / 10.5634 | 1.833 / 1.223 | 2.3806 |

跨轮AMD encode出现143至199ms、decode10.6至12.5ms的变化，不能把最后一轮当作硬件固定水平。端到端不只包含shader，还包括CPU归一化、embedding/head、页池、录制与回读。此次没有刷新所有CPU后端，不以旧CPU单次值计算新的确定加速比。首输出值仅作快速指纹，数值结论来自算子逐位比较与逐层/全分位数/分页测试。

### A3a正确但不稳定，撤下

A3a只裁掉可证明Empty的编码区未来键块，双GPU逐位对照通过。第一轮A2→A3为AMD 149.7078/10.6403→130.7113/10.2023ms，4060 109.1403/6.8161→123.8143/7.6405ms。反序一轮为AMD 143.5030/10.5634→132.8497/10.3990ms，4060 109.8991/6.7479→100.9962/7.2785ms。

AMD encode两轮改善，4060 encode方向不一致、decode两轮退化。按计划不继续反复测到满意为止，撤下A3a；候选源码与裁剪前shader集保留在本地临时目录，正式attentionTile为A2。以后可拆独立prefill专用核再评估，而不是把裁剪分支无条件带进decode。

### 本阶段收口

最终Release增量构建零警告，GPU完整ctest29/29（AMD，81.88秒），4060定向模型2/2与执行器validation通过；默认std-only完整25/25（77.77秒）。增强后的算子对照使用memcmp逐字节检查。真实5M追加129patch双行用例，逐层、全分位数预算不变；不以原来只有3patch的短回归冒充覆盖新分派。最终A2 spv与上表反序基准保存模块哈希相同。

这是首批接受节点的收口，不是整棵R树全部完成；G2窄投影、后续融合、KV镜像与设备特化仍须按进入条件继续。

## 研究来源与尚未取得的资料

本轮2026-10-24再次尝试获取ggml固定commit与shader正文，api.github.com及github.com均被web_fetch的非公网地址校验拒绝。搜索只发现[flash_attn_cm2.comp](<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp>)与[shader生成器](<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp>)路径，尚未固定commit或读到正文，未复制其实现。此项可独立补读，不阻塞原创FP32优化；网络恢复后先固定commit、审读依赖与许可，再引用具体算法，不把合作矩阵专用核直接当通用Vulkan1.1核。

设备已知上限为AMD push128B/shared64KB、4060 push256B/shared48KB，两者共同按push128B/shared48KB设计；ABI仍为v1。来源为第五期设备实测与当前shader/执行器，FA2方法参考[FlashAttention-2论文](<https://arxiv.org/abs/2307.08691>)的分块与在线softmax思想，不声称已经实现论文级性能。

## L1实现尝试与回退记录（2026-10-24）

L1曾按上述设计完整实现（双arena、缓存描述符集、abiVersion=2的偏移push、权重arena增量重建），4060全链路通过，但AMD集显暴露一个未解决的首dispatch行为问题。经用户决策，该实现整体回退到v1备份，旧L1至L6已废止，由本文R树取代。记录如下，供下轮规划参考。

### 已定位并修复的坑

1. `vk::UniqueDescriptorSet(rawHandle)`单参构造不填充device指针，析构时`vkFreeDescriptorSets(nullptr, set)`触发loader的Invalid device与ICD的/GS崩溃（c0000409，WER崩溃转储在系统vulkan-1.dll内）。正确用法是`allocateDescriptorSetsUnique`后move进容器，任何裸句柄构造唯一包装器的写法在本项目禁用。
2. 显式释放描述符集的池必须带`vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet`（VUID-vkFreeDescriptorSets-descriptorPool-00312，validation层会报）。常驻复用池加显式失效释放的组合容易漏掉。
3. 权重拷贝的录制时机：若只在命令缓冲创建时录一次，同batch先上传后dispatch（arena在dispatch内才建成）时拷贝永远不会被录制，arena保持全零且无任何报错。录制点必须挂在每次dispatch或下载前按需触发，或改为上传即独立提交。

### 未解决：AMD首dispatch行为异常（RADV）

现象：命令缓冲内第一个dispatch读到的权重arena为全零，同缓冲第二个同类dispatch完全正确。4060无任何问题。受影响的kernel不限定attention（与共享内存无关），gemm与silu作cb首dispatch同样错（maxErr分别为15.8与3.29，4060在预算内）。

已排除的假设（全部受控复现，源码与数据行转储保留在本地工作区）：

- 拷贝未执行：权重拷贝独立预提交加fence等待后，fence后探测仍确认arena数据在场，首dispatch依然读零。
- 可见性缺屏障：显式transfer到compute屏障无效。
- 管线冷启动：silu预热提交无效；预热attention dispatch反而让错误输出变成不匹配任何输入行的非零乱值（与arena前段数据组合一致），说明错误形态与dispatch历史相关。
- 描述符与内存对象问题：validation层未报告错误，但这不足以排除应用侧的生命周期、绑定或同步错误。

当时怀疑RADV对首dispatch的push常量或首个绑定缓冲首读存在缺陷，但没有定位到根因，不能据此认定驱动有错。后续必须以v1不变的绑定方式为对照，逐项改变池、描述符、偏移与上传顺序；不能同时重构后再靠预热dispatch遮盖错误。本轮通过不采用该arena/偏移ABI方案规避，AMD仍是每个启用分支的必验设备。

### 回退操作与验证

按BACKUP.md恢复方法，19个文件从备份逐字节恢复（十个.comp、kernels、executor、gpuExecutor契约、gpuTransformer、两个测试、manifest），删除临时trace.hpp，build-gpu-vulkan删除重建（/W4零警告）。验收：ctest 29/29（AMD设备0），4060上vulkanKernels、vulkanExecutor、vulkanShaderAbi与gpuModel全模型均通过。该次回退完成时树为v1原样，abiVersion回到1；本轮R树在此基础上新增优化核，ABI版本仍为1。
