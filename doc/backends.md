# 计算后端

CPU实现包括默认reference，以及可选的Eigen、ytensor和自写BLAS。本文介绍数值路径、线程与缓存能力，BLAS细节见[BLAS后端](<blas.md>)，GPU模型见[Vulkan后端](<vulkan.md>)。

## 为什么需要多个后端

最直接的移植方式是把上游PyTorch的每个算子写成三重循环。这正是reference后端：只依赖C++17标准库，累加顺序简单可追溯，适合作为正确性基线。当前模型级耗时见[基准对照](<../README_zh.md#性能>)。

加速的朴素做法是引入矩阵库，把计算全部交给它。这样会遇到两个约束：默认部署路径必须保持零第三方依赖；换库后浮点累加顺序改变，结果不再与reference逐位相同，需要一套事先确定、不能事后放宽的容差。本仓库的处理方式是：后端由CMake在配置期单选（机制见[模型架构](<architecture.md>)），可选后端只替换计算密集的部分，模型语义保持一致，reference与BLAS共用执行器，Eigen与ytensor仍保留各自编排；精度用固定的上游oracle和尺度感知容差判定。

![后端类层次与能力对照](<../assets/backend-layers.svg>)

## reference

reference是`Transformer`的子类，只实现模型编排需要的窄数值钩子：投影、RMSNorm、QKV重排与RoPE、单头注意力，见[reference/model.cpp](<../timestral/reference/model.cpp>)。嵌入、掩码、分页、分位数头等全部逻辑在`utils`的[Transformer](<../timestral/utils/transformer.cpp>)中，因此reference本身不到一百行。底层算子在[kernels.cpp](<../timestral/reference/kernels.cpp>)：线性层逐输出做串行点积，再加偏置；注意力先算出整行分数，被掩码的位置填负无穷，softmax后再与V相乘。

reference有三个刻意的选择：

- 不引入改变运算顺序的激进融合，每一步都能与上游的一个PyTorch表达式对应，出问题时可以逐层对照。
- 固定单线程，`Settings::threads`对它不生效。
- 编译选项为`/fp:precise`且未启用fast-math，保持串行点积顺序。

其它后端的数值回归都以它为对照，BLAS引入`Transformer`抽取前后，真实5M的referenceTest输出逐行一致（176行，diff_count=0）。

## Eigen

Eigen后端使用Eigen 3.4.0的行向量与Map视图完成投影，注意力采用公共有序数值内核。

- 权重在构造时预转置，Eigen行向量按K顺序逐项更新，最后加偏置。OpenMP按输入行并行，不在点积内部重排归约。
- 注意力直接读取共享前缀和本轮尾部，variate按请求组收紧范围。有序softmax与PV保持冻结数值预算。

RMSNorm、RoPE、softmax、SiLU、SwiGLU和分位数头直接调用reference的内核，见[eigen/model.cpp](<../timestral/eigen/model.cpp>)。Eigen后端早于`Transformer`抽取，仍保留一份完整的模型编排和分页实现，修改公共语义时需要同步核对。

线程数由`Settings::threads`控制：1为串行（默认），0为自动值且上限4，N为精确线程数。Eigen的线程数是进程级全局状态，必须在构造模型前设置，同一进程里多个模型共享最后一次设置。自动值不取满核是测出来的：5M的GEMM很小，记录中回归套件16线程约28秒，4线程约2.9秒。

## ytensor

ytensor后端使用用户指定来源的single-header 0.14，仓库内没有它的许可文件，事实记录见[3rd说明](<../3rd/README.md>)。投影走ytensor的strided matmul，以非拥有视图包装本仓库存储。当前注意力使用公共有序内核，直接读共享前缀和独立尾部，逐元素算子也沿用reference内核。

- YBLAS负责投影。原FLASH_AVX2与MATH注意力实现不再作为当前长查询的模型默认路径，不能把旧FA2配置记录当成此次配方。
- 只有[ytensor/model.cpp](<../timestral/ytensor/model.cpp>)进入独立C++20 OBJECT目标并开启AVX2/FMA（MSVC为`/arch:AVX2`，GCC/Clang为`-mavx2 -mfma`）；工厂与核心保持普通C++17，禁止IPO/unity跨过指令集边界。基线工厂在调用该对象的任何函数前检查CPU与OS的XMM/YMM支持，旧CPU明确拒绝，非x86配置期拒绝。实现、验证与残留数值失败见[兼容性记录](<linuxCompatibility.md>)。MSVC在`/arch:AVX2`下不定义`__FMA__`，也不支持单头里的GNU `__attribute__`语法，该文件在包含ytensor前成对补齐这两处、包含后撤销，记录中说明这些属性只是内联提示。
- 支持分页增量，`supportsIncremental()`返回true。页内保存各time层K/V和因果归一化状态，追加与共享解码接受统一页表契约。

## 精度验收口径

所有后端对照同一组固定的上游oracle：14个逐层block用例、36个预测用例（含5个预期拒绝用例），由上游Python在固定基线上生成。微型模型是固定种子的随机参数，不是发布权重。容差在第一期确定，之后的后端沿用，没有放宽：

| 对象 | 判据 | 说明 |
| --- | --- | --- |
| 算子 | atol 1e-5，rtol 1e-4 | 单个内核对照 |
| 隐藏状态 | max(1e-5, 1e-4×RMS(参考))，且相对L2不超过1e-5 | embedding、各层输出、out_norm |
| 最终预测 | abs(a−b) ≤ 1e-4 + 1e-3×abs(参考) | 逐用例判定，另记最大误差与RMSE |
| cached decode | max(1e-6, 1e-5×输出RMS) | 分页解码对相同语义的全量重算 |

隐藏状态采用尺度感知的atol，是因为残差流的幅值逐层增长。真实5M的第7层输出RMS约120至152，固定的小阈值会把FP32归约顺序带来的正常差异误判为错误；记录中真实权重路径的层内最大绝对误差为8.06e-3，相对L2为5.10e-6。cached decode的阈值单独设定，用来验证缓存，不与oracle容差混用；并发测试则把基线与被测服务配置成相同的无缓存路径，只检查调度一致性。

最初移植阶段记录的预测误差如下，不能替代当前有序实现的回归：

| 后端 | realDefault最大绝对误差 | realFullBlockThenPartial最大绝对误差 |
| --- | --- | --- |
| reference | 9.54e-6 | 0.310 |
| Eigen | 1.34e-5 | 0.083 |
| ytensor（强制Naive） | 9.53674e-6 | 0.310059，与reference逐位相同 |

realFullBlockThenPartial是1030步的长horizon用例，参考输出RMS约854，多块自回归会放大归约顺序差异，所以它的绝对误差明显大于短用例；三个后端都在容差内。ytensor的Naive循环与reference累加顺序相同，结果逐位一致。启用YBLAS后累加顺序改变，记录中合成用例max_abs不超过3.8e-6、真实5M多变量与条件用例约1e-7量级，改为按尺度感知容差判定，不再要求逐位一致；YBLAS路径的realDefault等单项数字没有单独记录。

## 性能取舍

同机并排测量（AMD 7735H、MSVC 19.36、Release、单线程，stageBench与timestralBench协议，记录日期为第三期验收时）：

| 项目 | reference | Eigen | ytensor |
| --- | ---: | ---: | ---: |
| forward T=256 | 32.10ms | 6.83ms | 6.77ms |
| forward T=4096 | 362.21ms | 70.64ms | 55.35ms |
| forward T=32768 | 4424.15ms | 961.20ms | 1185.57ms |
| append | 20.98ms | 8.46ms | 不支持 |
| decode | 11.79ms | 3.59ms | 不支持 |
| batch | 15.7 req/s | 88.2 req/s | 123.1 req/s |
| stream | 102.7ms | 21.4ms | 28.0ms |

几点读法：

- 单行前向在三个长度上，Eigen与ytensor相对reference约为3.7至6.5倍；T=32768时Eigen更快，T=4096时ytensor更快。
- stream协议是同一来源逐步追加数据，Eigen有分页缓存，ytensor每次全量重算，所以ytensor的单次前向虽然在T=4096更快，stream仍慢于Eigen。分页缓存能否使用，比单个GEMM快多少更影响流式场景。
- 多线程收益有限。Eigen在stream上4线程是测得的综合最优点，约为reference的7.4倍；8和16线程反而因小GEMM的OpenMP调度开销变差。ytensor的YBLAS在4线程下只有32768上下文约1.4倍收益，短上下文加线程是净损失。因此默认`threads=1`。
- 每组配置报告一轮协议统计（预热5次、计时25次），未做锁频，多轮之间存在明显波动，表中数字用于比较量级，不作为延迟承诺。

## BLAS后端概述

自写BLAS不链接系统BLAS或外部矩阵库，复用Transformer编排，覆盖投影、注意力、门控和SwiGLU等数值钩子。权重加载时预打包，当前模型保持逐K有序FP32运算，运行期按AVX2、SSE4.2和标量能力选择微内核。Settings.threads控制每个模型自己的固定线程池，支持1、2、4等线程数，0自动选择且最多4，不再忽略参数。并行按完整输出块分派，不拆分点积归约，支持分页、KV-only prefill及共享前缀。当前性能见[基准数据](<benchmark.md>)，设计见[BLAS后端](<blas.md>)。

## 取舍与经验

后端之间最容易被低估的差别不是速度，而是累加顺序。只要换了GEMM实现，结果就不再逐位等于reference，此时必须已经有一套事先定好的、按数值尺度给出的容差，否则很容易在发现差异后顺手放宽阈值。本仓库把容差定在第一期，后续后端一律沿用，超界时回退优化而不是改阈值，BLAS窄核多链交错就是因此回退的例子。

第二个经验是让后端只做后端该做的事。Eigen和ytensor各自复制了一份模型编排，BLAS则只实现钩子；前者接入快，但每次改掩码或分页都要三处核对，后者把语义集中在`utils`里，新增后端的工作量和出错面都更小。如果以后再接新后端，应当沿用BLAS的方式。
