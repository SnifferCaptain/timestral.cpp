# Timestral模型架构与仓库模块

本文按一次预测经过的顺序介绍Timestral的网络结构，再说明C++仓库怎样切分模块。维度以Timestral-5M为例，取自本地发布模型的config.json（模型文件不进入版本管理）；按这些维度推算的权重总数为5208960，与本地权重记录中5M权重的F32值个数一致。网络语义移植自上游固定基线a534bdbd，对应关系见各源文件的文件头。

## 要解决的问题

时间序列预测的输入是一条或几条通道的历史读数，输出是未来若干步的取值。基础模型还要额外面对三件事：不同序列的量纲可以相差几个数量级，历史里会有缺测，调用方需要的往往不只是一个点预测，而是分位数区间（不保证在任意数据上校准）。

最朴素的做法是把每个采样点当成一个token，用语言模型的方式逐点自回归。这条路在时间序列上很快遇到瓶颈：5M配置的上下文是32768个采样点，逐点token会让注意力矩阵达到32768×32768的规模；预测512步就要做512轮前向；点预测也表达不了不确定性。

Timestral的回答可以概括成四点，后文逐一展开：

- 按patch切分：每32个采样点及其缺测标志压成一个token，32768点的上下文变成1024个patch token。
- 因果归一化：每个patch用截至自身末尾的累计均值与标准差缩放，再取asinh，消除量纲差异且不偷看未来。
- 独立的预测查询：每个待预测patch用一个学习得到的查询向量占位，一次前向输出该patch全部32步的21个分位数。
- 两类注意力轴：time轴在同一通道的patch之间做注意力，variate轴在同一时刻的不同通道之间做注意力，二者按配置交替。

## 5M配置

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `patch_size` | 32 | 每个patch的采样点数P |
| `d_model` | 192 | 隐藏维 |
| `num_heads` | 3 | 注意力头数，单头维度64 |
| `mlp_hidden_dim` | 512 | SwiGLU中间维 |
| `block_schedule` | 8项 | time×3、time_variate、time×3、time_variate |
| `quantile_knots` | 21个 | 0.01、0.05至0.95每隔0.05、0.99 |
| `block_size` | 32 | 单次前向最多解码的patch数 |
| `context_length` | 32768 | 上下文采样点数，即1024个patch |
| `rope_partial_factor` | 0.5 | 单头64维中32维参与旋转 |
| `time_attn_gate`、`variate_attn_gate` | `sdpa_sigmoid` | 注意力输出乘sigmoid门 |
| `quantile_width_factor` | 3.0 | 分位数头的间距系数 |
| `use_asinh` | true | 归一化后取asinh |
| `attend_missing_patches` | true | 缺测patch在出现首个观测后仍可被关注 |
| `attn_bias`、`mlp_bias` | false | 注意力与MLP投影不带偏置 |

字段读取与校验见[config.hpp](<../../timestral/utils/config.hpp>)，未知字段和缺失必填项都会被拒绝。

![Timestral-5M一次前向的结构](<../../assets/arch-overview.svg>)

## 因果patch归一化

每一行（一条通道）沿时间维做Welford累计，只累计观测到的点。第p个patch结束时的状态给出该patch所有点共用的loc和scale：

$$
\mu_p=\frac{\text{sum}}{\text{count}},\qquad
\sigma_p=\max\left(\sqrt{\frac{M_2}{\max(\text{count}-1,1)}},\ \text{minimum\_scale}\right)
$$

$$
\tilde{x}_t=\operatorname{asinh}\left(\frac{x_t-\mu_p}{\max(\sigma_p,\varepsilon_{\text{FP32}})}\right)
$$

缺测点的归一化值置0，缺测信息另由指示位携带。累计量在[patch.cpp](<../../timestral/utils/patch.cpp>)中用double保存，源码注释说明累加顺序、asinh与minimum_scale与上游causal_patch_loc_scale一致。

这里的关键是“因果”：第p个patch的缩放只依赖它和它之前的观测。后面追加数据不会改变前面patch的输入，这是后文分页KV缓存成立的前提之一。每页因此会保存patch末尾的count、sum和M2，增量解码从这里继续累计，不必重扫整条前缀，见[推理设计](<inference.md>)。

known-future协变量行是例外：它按整个窗口（上下文加horizon）的总体方差归一化，取值只有0和1的二值行固定loc=0、scale=1且不取asinh，实现在[transformer.cpp](<../../timestral/utils/transformer.cpp#L502-L544>)。

## patch嵌入与预测查询

每个patch的输入是2P=64维：前32维是归一化值，后32维是缺测指示（缺测为1）。嵌入层`representation.patch_proj`是一个带残差支路的两层MLP，三个投影都带偏置：

$$
h=W_o\,\operatorname{SiLU}(W_h u+b_h)+b_o+W_r u+b_r,\qquad u\in\mathbb{R}^{64},\ W_h\in\mathbb{R}^{768\times 64},\ W_o\in\mathbb{R}^{192\times 768}
$$

待预测的patch不经过嵌入层，而是统一放一个学习得到的192维向量`representation.forecast_query`。没有协变量时，一行的token序列是N个上下文patch之后接H个查询槽；查询槽靠位置编码和注意力区分彼此。这与记录中官方仓库的描述一致：预测来自读取因果编码历史的独立prediction token（该描述来自官方仓库的搜索摘录，没有完整抓取在线文章）。

存在协变量时序列改为双流：编码流包含上下文加horizon的全部patch（known-future的未来值也进入编码），解码流为每个编码槽各放一个查询，只取最后H个解码输出送入分位数头，对应代码中的`encodedCount`与`decodedCount`。

## block schedule与两类注意力轴

`block_schedule`的每一项展开成一到两个注意力子层，见[patch.cpp](<../../timestral/utils/patch.cpp#L20-L34>)：

| 取值 | 子层顺序 |
| --- | --- |
| `time` | time |
| `variate` | variate |
| `time_variate` | time，再variate |
| `variate_time` | variate，再time |

5M共8层，展开后有8个time子层和2个variate子层。每个子层各有自己的`attn_norms`、`wQKV`、`wO`和`_gate`，每层只有一个MLP。

time轴注意力在同一行内跨slot进行，承担时间上的依赖。variate轴注意力在同一slot内跨行进行，只允许同一`groupIds`的行互相看见，承担通道之间的信息交换；它不需要跨patch的K/V。分页缓存因此只保存time子层的K/V，5M每页8份。

## 因果掩码与缺失patch

time轴掩码由`timeAllowed`给出，见[transformer.cpp](<../../timestral/utils/transformer.cpp#L706-L739>)。先定义有效key：一个patch有观测，或者允许关注缺测且该行此前已出现过观测，就可以被后续槽读取（[buildKeyValid](<../../timestral/utils/patch.cpp#L167-L188>)）。在没有协变量时规则如下：

- 编码query只读不晚于自身的有效编码key，自身总是可见。
- 解码query读全部有效的上下文编码key，以及不晚于自身的解码槽。
- 编码query不读解码流。

![time轴掩码示例](<../../assets/arch-mask.svg>)

加入协变量后有三条附加规则：known行的编码query在时间上双向可见；解码流中位于上下文段的query不读编码流；未来段的解码query只读上下文段的编码key。variate轴另有一道“防火墙”：协变量行不能读目标行，known行不能读非known行，信息只从协变量流向目标。

known行双向可见这一条直接影响缓存：known的未来值能改变历史patch的隐藏状态。记录中的固定上游探针显示，只把一个known未来值加3，第4层之后目标历史的hidden就发生变化，见本地验收记录中的known-future核验。所以条件路径不复用跨请求的历史KV。

## 部分RoPE

只有time子层对Q、K施加旋转位置编码，V和variate子层不旋转。参与旋转的维度为单头维度乘`rope_partial_factor`，5M是64×0.5=32。配对方式不是相邻两维，而是第j维与第j+32维成对，j取0到15：

$$
\theta_j=10000^{-2j/32},\qquad
\begin{bmatrix}x'_j\\ x'_{j+32}\end{bmatrix}=
\begin{bmatrix}\cos p\theta_j & -\sin p\theta_j\\ \sin p\theta_j & \cos p\theta_j\end{bmatrix}
\begin{bmatrix}x_j\\ x_{j+32}\end{bmatrix}
$$

![部分RoPE布局](<../../assets/arch-rope.svg>)

实现见[kernels.cpp](<../../timestral/reference/kernels.cpp#L128-L162>)。发布权重的fused QKV按(qkv, headDim, head)排列，注意力需要(head, headDim)，重排在`prepareQkv`中完成，见[reference/model.cpp](<../../timestral/reference/model.cpp#L35-L60>)。

位置p是slot下标减去同组首个有观测patch的下标。没有协变量时查询槽接在上下文之后继续编号；有协变量时解码流的位置与编码流对齐，即slot减去编码槽数。`Settings::continuousRope`可以把整数slot换成由真实时间换算的实数坐标，这是默认关闭的实验选项，记录中的样例没有显示精度改善。

## 门控注意力

每个注意力子层采用pre-norm结构：

$$
\hat{x}=\operatorname{RMSNorm}(x),\quad
o=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{64}}+M\right)V,\quad
x\leftarrow x+W_O\left(o\odot\sigma(W_g\hat{x})\right)
$$

门`_gate`与QKV读取同一个归一化输入，逐元素作用在拼接后的192维注意力输出上。配置为`none`时没有门，权重契约也不要求`_gate`。RMSNorm的epsilon为1e-6。

## SwiGLU MLP

MLP同样是pre-norm：`w12`把192维投影到1024维，按前一半逐元素乘SiLU(后一半)得到512维，再经`w3`投影回192维加到残差上。这个“两半中只有后一半经过SiLU”的顺序与上游一致，交换会得到完全不同的结果，见[transformer.cpp](<../../timestral/utils/transformer.cpp#L35-L51>)。

## 分位数头

最后一层后接`out_norm`，解码槽的隐藏状态经过`head.projection`（与嵌入层同构的残差MLP，192→768→672，不带偏置），重排成32步×21个分位数的原始值r。为了保证分位数单调，中位数通道r_m直接输出，其余通道以softplus累加间距：

$$
q_j=r_m-\sum_{i=j}^{m-1}w\,(\tau_{i+1}-\tau_i)\,\operatorname{softplus}(r_i),\quad j<m
$$

$$
q_j=r_m+\sum_{i=m+1}^{j}w\,(\tau_i-\tau_{i-1})\,\operatorname{softplus}(r_i),\quad j>m
$$

其中w是`quantile_width_factor`，τ是分位点。结果夹到FP32有限范围内。`predictBlock`再做反归一化：先sinh，再乘最后一个上下文点的scale、加loc；sinh溢出的无穷值替换为该位置有限分位数的最大或最小值，都不是有限值时取0。

## 参数量核算

按上面的结构逐项计数，可以验证对结构的理解没有漏项：

| 组件 | 权重 | 参数数 |
| --- | --- | ---: |
| patch嵌入 | patch_proj三个投影，含偏置 | 210048 |
| 预测查询 | forecast_query | 192 |
| 注意力子层×10 | attn_norm、wQKV、wO、_gate | 1845120 |
| MLP×8 | mlp_norm、w12、w3 | 2360832 |
| 输出归一化 | out_norm | 192 |
| 分位数头 | head.projection三个投影 | 792576 |
| 合计 | | 5208960 |

[Transformer构造函数](<../../timestral/utils/transformer.cpp#L146-L214>)按同样的清单校验每个权重的名称、形状和有限性，多出或缺少任何一个都会拒绝加载。

## C++仓库模块

~~~text
timestral/
├── timestral.hpp/.cpp          # 公开入口，转发给内部调度服务
├── utils/                      # 与后端无关的基础设施与公共模型语义
│   ├── service.hpp/.cpp        # 请求队列、三种调度策略、共享前缀树
│   ├── engine.hpp/.cpp         # 张量级引擎：分块自回归、refine、后处理编排
│   ├── condition.hpp/.cpp      # 交错观测的网格对齐与分段条件补全
│   ├── forecast.hpp/.cpp       # 请求校验与结果提取
│   ├── time.hpp/.cpp           # 实际时间到实数patch坐标
│   ├── transformer.hpp/.cpp    # 公共模型编排、掩码、分页执行
│   ├── patch.hpp/.cpp          # 因果归一化、patch装配、位置、页键
│   ├── cache.hpp/.cpp          # 页池、FNV页键、引用与LRU候选
│   ├── model.hpp               # 后端接口timestral::Model
│   ├── backend.hpp             # loadBackendModel声明
│   ├── attention.hpp           # 注意力视图与掩码描述
│   └── 其余基础文件            # tensor、config、json、safetensors、postprocess、stats
├── reference/                  # 标量内核，非x86目标的默认后端
├── eigen/                      # Eigen可选后端
├── ytensor/                    # ytensor可选后端
└── blas/                       # 自写BLAS可选后端
~~~

职责边界按仓库内部代码规范执行：`timestral/`根目录只有入口两个文件；`utils/`承载所有与后端无关的东西，包括调度、缓存、归一化、后处理和统计；后端目录只放该后端自己的线性代数与内核，不放缓存管理和后处理。没有独立的ops层，跨后端共用的“算子”实际是模型语义，归`utils`。

![模块划分与依赖方向](<../../assets/arch-modules.svg>)

公开头[timestral.hpp](<../../timestral/timestral.hpp>)只暴露`Sample`和嵌套在`Timestral`里的`Settings`、`Request`、`Result`、`Policy`，内部类型只以`Service`前置声明出现。张量、页池和角色表都留在内部，调用方只处理采样点。

## 后端在链接期选择

后端选择不经过宏。[backend.hpp](<../../timestral/utils/backend.hpp>)只声明一个工厂函数`loadBackendModel`，四个后端目录各有一个`backend.cpp`给出定义，CMake在配置期只把其中一个加入库目标：

~~~cmake
set(TIMESTRAL_SELECTED_BACKEND timestral/reference/backend.cpp)

if(TIMESTRAL_USE_EIGEN)
    set(TIMESTRAL_SELECTED_BACKEND timestral/eigen/backend.cpp)
endif()
~~~

上面是[CMakeLists.txt](<../../CMakeLists.txt#L19-L35>)的节选，ytensor与BLAS的分支同理；任意两个可选后端同时开启会在配置阶段以FATAL_ERROR退出。这样做的结果是：整个构建里`loadBackendModel`只有一个定义，公开头和调度代码里看不到任何后端条件编译，默认构建也不会包含Eigen或ytensor的头文件。代价是一次构建只能有一个后端，对比后端要分别配置构建目录。

## 一次预测的数据流

1. 构造`Timestral`时，`loadEngine`读取config.json和model.safetensors，调用链接进来的`loadBackendModel`得到只读模型，再启动`Service`的工作线程。
2. `submit`在调用线程完成全部校验：历史不少于2个patch、时间戳严格递增、推断采样间隔和预测步数、按`maxContext`裁剪历史，并判断能否使用分页网格。
3. 工作线程按调度策略取出一轮请求，命中前缀页的走分页路径，其余全量重算或冷拼批。
4. [Engine](<../../timestral/utils/engine.cpp>)把上下文左填充到patch整数倍，按每块`patchBatch`个patch循环：前向得到分位数，做advancing refine，把目标中位数回灌为下一块的上下文。
5. 全部块结束后依次做范围截断、对称平均和非负截断，最后组装成`Result`，附带计时和页命中统计。

第2至5步的调度、缓存和后处理细节见[推理设计](<inference.md>)。

## 取舍

模型编排只写一份。[Transformer](<../../timestral/utils/transformer.hpp>)持有嵌入、掩码、分页和分位数头的全部逻辑，reference和BLAS只覆盖投影、RMSNorm、QKV布局、注意力等窄钩子，掩码或分页的修改不必在多个后端同步。Eigen与ytensor是更早接入的后端，仍各自保留一份完整编排，只把矩阵乘法换成各自的库、逐元素算子调用reference内核；这部分重复是已知的历史成本，修改公共语义时要三处一起核对。

`split_context_horizon_attn`字段会被解析并保存，但在当前C++源码中没有找到读取它的计算分支；记录中“关闭split”的微型模型对照通过，但该开关在上游的确切语义尚未逐行核对，这里标为未验证。

从这套结构里能得到的经验是：一个模型能不能做增量推理，主要取决于它的依赖结构，而不是工程实现。因果归一化、因果time掩码和独立预测查询让历史patch的K/V只依赖历史本身，分页缓存才有意义；known行的双向注意力和全窗口统计打破了这个性质，对应的路径就只能全量重算。读懂掩码，比优化任何一个GEMM都先决定了推理侧能做什么。
