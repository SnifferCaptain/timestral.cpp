# API参考

公共接口只有一个入口头文件[timestral/timestral.hpp](<../../timestral/timestral.hpp>)，链接CMake目标timestral即可使用，默认构建只依赖C++17标准库。

~~~cpp
#include "timestral/timestral.hpp"
~~~

## timestral::Sample

一个采样点，时间单位秒，原点任意。

**成员**
| 成员变量 | 类型 | 描述 |
| --- | --- | --- |
| time | double | 时间戳，单位秒 |
| value | double | 读数；NaN表示该时刻缺测 |

## timestral::Timestral

本地Timestral推理引擎：加载一次模型，支持并发提交，共享前缀缓存按内容寻址，零引用页按LRU淘汰。

### Timestral::Timestral()

从本地模型目录加载配置与权重，并启动调度线程。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| modelDirectory | const std::filesystem::path& | - | 含config.json与权重文件的目录 |
| settings | Settings | 默认聚合初始化 | 构造后固定的实例配置 |
| policy | Policy | Policy::Fifo | 初始调度策略 |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::runtime_error | 目录、配置或权重缺失、字段非法 |

**示例**
~~~cpp
timestral::Timestral model{"model/timestral-5m"};
~~~

不提供拷贝构造与拷贝赋值；析构时停止调度线程，尚未开始的请求的future以std::runtime_error结束。

### Timestral::predict()

提交一次预测并等待结果，等价于submit后立即get。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| request | Request | - | 请求内容，见Request成员表 |

**返回**
| 类型 | 描述 |
| --- | --- |
| Result | 预测结果 |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | 请求参数非法 |
| RequestSuperseded | Mailbox策略下被同源新请求或abort作废 |

**示例**
~~~cpp
timestral::Timestral model{"model/timestral-5m"};
std::vector<timestral::Sample> history;
for (std::size_t t = 0; t < 128; ++t)
    history.push_back({static_cast<double>(t) * 0.01, std::sin(static_cast<double>(t) * 0.05)});
timestral::Timestral::Request request;
request.variables = {history};
request.sampleInterval = 0.01;
request.horizonSeconds = 0.32;
auto result = model.predict(request);
for (std::size_t t = 0; t < result.steps(); ++t)
    std::cout << result.time(t) << ',' << result.median(t) << '\n';
~~~

### Timestral::submit()

提交一次预测并立即返回，参数校验与会话追加在提交线程完成。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| request | Request | - | 空streamId按匿名请求处理，服务内部分配标识 |

**返回**
| 类型 | 描述 |
| --- | --- |
| std::future<Result> | 就绪后得到结果；被作废时get()抛RequestSuperseded，引擎失败时抛std::runtime_error |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | 请求参数非法、会话不存在或新读数不晚于会话末尾时在提交线程抛出，请求不进入队列，会话不变 |

### Timestral::abort()

结束一个来源：作废其尚未开始的请求，并释放它持有的前缀页引用。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| streamId | const std::string& | - | 来源标识，被作废请求的future抛RequestSuperseded |

**注意**

仍被其他来源引用的页继续共享，无人引用的页立即摘除页键并释放。

### Timestral::close()

删除一个会话保存的历史，并作废该会话尚未开始的请求。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| session | const std::string& | - | 会话标识，取自Result::session，不存在时忽略 |

**注意**

会话用过的KV页不立即释放，留在LRU中按需淘汰。

### Timestral::setPolicy()

切换调度策略，只影响尚未被调度的请求。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| policy | Policy | - | 切换为Mailbox时，队列中同源请求只保留最后一条未开始的 |

### Timestral::policy()

当前调度策略。

**返回**
| 类型 | 描述 |
| --- | --- |
| Policy | 当前策略 |

### Timestral::pendingRequests()

队列中尚未开始执行的请求数。

**返回**
| 类型 | 描述 |
| --- | --- |
| std::size_t | 待执行请求数 |

### Timestral::patchSize()

模型的patch长度，5M模型为32。

**返回**
| 类型 | 描述 |
| --- | --- |
| std::size_t | patch包含的采样槽位数 |

### Timestral::contextLength()

模型支持的最长上下文。

**返回**
| 类型 | 描述 |
| --- | --- |
| std::size_t | 上下文上限，按步计 |

### Timestral::Policy

调度策略枚举。

| 值 | 描述 |
| --- | --- |
| Fifo | 按提交顺序执行 |
| Mailbox | 保留同源最新待执行请求，同源旧请求作废 |
| Continuous | 自动拼批，受Settings::maxBatch与maxVariables约束 |

### Timestral::Settings

实例配置，构造时读取，构造后不可修改。全部取默认值时是与上游一致的完整注意力，不做滑窗近似。

**成员**
| 成员变量 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| maxContext | std::size_t | 0 | 上下文样本数上限，0表示用模型contextLength，超出部分丢弃最早读数 |
| patchBatch | std::size_t | 16 | 块AR每轮patch数，如1/2/4/8/16/32，构造时钳制到blockSize |
| maxBatch | std::size_t | 4 | 每轮请求槽位数 |
| maxVariables | std::size_t | 8 | 每条请求的目标与协变量总数上限 |
| cachePages | std::size_t | 2048 | 每种通道数的KV页数上限，一页存一个patch，按需分配，0禁用缓存 |
| maxSessions | std::size_t | 16 | 保留历史的会话数，超出时淘汰最久未用的，0不建立会话 |
| fullPrefillThreshold | std::size_t | 512 | 会话本轮保留历史不超过此点数时使用全部新读数，允许重新prefill；0始终固定边界 |
| threads | std::size_t | 1 | BLAS/Eigen/ytensor线程数，默认1串行，0自动（最多4），N为精确线程数 |
| continuousRope | bool | false | 实验性连续patch时间位置 |
| ropeReferenceInterval | double | 0.0 | 参考采样间隔，0采用请求间隔 |
| gpuThreads | std::size_t | 0 | Vulkan逻辑派发窗口预算，0不限制，正数须为256倍数，不是物理CU配额 |
| attentionMask | std::function<bool(std::size_t, std::size_t)> | 空 | time轴附加可见性谓词，参数为(query, key)，空为原生掩码 |

**注意**

attentionMask的参数是time轴token下标：一个token是patchSize个采样点，从模型上下文网格的最早patch起计，预测patch接续计数。返回false的(query, key)对在原生因果掩码之外再被屏蔽，token对自身恒可见；谓词会被并发调用，须可重入。只有reference与BLAS后端原生执行，Eigen、ytensor与Vulkan后端设置后改用reference模型计算，速度明显下降。因为attentionMask是std::function，Settings仍是聚合，但不再能用于constexpr。

### Timestral::Request

一条预测请求。末尾不齐时按观测可用性分段补全，horizon从最晚有效观测起算；所有通道共用一条时间轴，条件与目标都由variables自身的观测位置决定。session为空时variables是完整历史；非空时variables只放该会话各通道末尾之后的新读数，可以为空。

**成员**
| 成员变量 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| session | std::string | 空串 | 续接的会话标识，取自上次Result::session |
| streamId | std::string | 空串 | 来源标识，空串是匿名请求，续接会话时按会话去重 |
| variables | std::vector<std::vector<Sample>> | - | 各通道按时间递增的读数，续接时通道数须与会话一致 |
| horizonSeconds | double | 0.0 | 预测跨度，单位秒，0取patchBatch个patch（5M默认512点），负数非法 |
| sampleInterval | double | 0.0 | 采样间隔，单位秒，0表示从时间戳推断 |
| symmetricAveraging | bool | true | 加做上下文取负的镜像前向并平均 |
| clipNonnegative | bool | true | 观测全非负的轨迹截断到非负 |

同步通道的会话默认在短历史阶段使用全部新读数：按maxContext裁剪后的本轮保留历史不超过fullPrefillThreshold时，允许因patch重新对齐而重做prefill，bufferedSamples为0。阈值默认512个采样点，与5M默认一轮解码长度相同，但不随patchBatch或本次horizon改变。超过阈值后，保留最后一次完整接纳读数时的patch边界及左填充，续接时只将凑满的patch送入模型；未满patch的尾部读数保存在会话中，不参与本轮预测。下一次追加会先与尾部拼接，完整旧页不因追加点数的余数变化而重编码。这一输入规则不依赖是否启用KV缓存，设阈值为0可始终固定边界。首次请求仍使用全部读数，保持上游左填充语义。

暂存期间，内部预测长度为所需步数加暂存点数，再跳过开头对应的跨度。因此Result::origin仍为最新读数的时间，返回步数和horizon不变，但预测未使用暂存读数，不能视为与全部最新观测的无状态请求等价。Result::bufferedSamples给出每通道暂存点数，范围为0至patchSize减1。以5M默认配置为例，历史达到512点后追加31点时复用16页、暂存31点，下一次追加1点才新增1页；160点追加31点仍属于短历史阶段，191点全部参与预测。

超过maxContext后按整patch移动窗口，历史归一化与上下文发生变化，仍需重建窗口KV；这里没有启用冻结KV的近似滑窗。通道不同步的条件补全仍走分阶段全量路径，不套用尾部暂存规则；会话保存辅助通道覆盖最早待预测位置所需的历史及已知未来，不按各通道末尾独立截断。需要每次使用全部最新读数时，提交不带session的完整历史。

### Timestral::Result

一次预测的输出，时间单位秒，分位数按(channel, step, level)行主序存储。

**成员**
| 成员变量 | 类型 | 描述 |
| --- | --- | --- |
| channels | std::size_t | 通道数，顺序与输入一致 |
| sampleInterval | double | 生效的采样间隔 |
| origin | double | 预测窗口前一时刻 |
| levels | std::vector<double> | 模型分位点 |
| quantiles | std::vector<double> | (channels, steps, levels)行主序 |
| observed | std::vector<std::uint8_t> | (channels, steps)，1为原始观测 |
| prefillSeconds | double | 前缀页编码耗时 |
| decodeSeconds | double | 解码耗时 |
| totalSeconds | double | 整次执行耗时 |
| reusedPages | std::size_t | 命中的前缀页数，不含对称平均镜像pass的页 |
| computedPages | std::size_t | 新计算的页数 |
| bufferedSamples | std::size_t | 每通道暂存、尚未参与本轮预测的尾部读数数，少于patchSize |
| session | std::string | 保存本次完整历史的会话，maxSessions为0时为空 |

#### Result::steps()

每个通道的预测窗口步数。

**返回**
| 类型 | 描述 |
| --- | --- |
| std::size_t | 步数 |

#### Result::time()

第step步的时间戳。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| step | std::size_t | - | 从0起的步下标，须小于steps() |

**返回**
| 类型 | 描述 |
| --- | --- |
| double | origin加(step+1)个采样间隔，单位秒 |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | step不小于steps() |

#### Result::median()

第step步的中位数，即0.5分位。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| step | std::size_t | - | 从0起的步下标，须小于steps() |
| channel | std::size_t | 0 | 通道下标 |

**返回**
| 类型 | 描述 |
| --- | --- |
| double | 该步中位数的原始量纲值 |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | step或channel越界 |

#### Result::quantile()

第step步在任意水平上的分位数。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| step | std::size_t | - | 从0起的步下标，须小于steps() |
| level | double | - | 目标水平，取值(0,1)，落在模型分位点网格外时取网格端点 |
| channel | std::size_t | 0 | 通道下标 |

**返回**
| 类型 | 描述 |
| --- | --- |
| double | 该步在level上的分位数，原始量纲 |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | step或channel越界，或level不在(0,1) |

**示例**
~~~cpp
// 0.10与0.90分位
double p10 = result.quantile(t, 0.10);
double p90 = result.quantile(t, 0.90);
~~~

#### Result::lower()

中心区间的下界。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| step | std::size_t | - | 从0起的步下标，须小于steps() |
| level | double | - | 区间覆盖率，取值(0,1)，level=0.9取0.05分位 |
| channel | std::size_t | 0 | 通道下标 |

**返回**
| 类型 | 描述 |
| --- | --- |
| double | 下界，原始量纲 |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | step或channel越界，或level不在(0,1) |

#### Result::upper()

中心区间的上界。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| step | std::size_t | - | 从0起的步下标，须小于steps() |
| level | double | - | 区间覆盖率，取值(0,1)，level=0.9取0.95分位 |
| channel | std::size_t | 0 | 通道下标 |

**返回**
| 类型 | 描述 |
| --- | --- |
| double | 上界，原始量纲 |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | step或channel越界，或level不在(0,1) |

#### Result::isObserved()

该点是否为原始观测，已知位置不代表模型预测分布。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| step | std::size_t | - | 从0起的步下标 |
| channel | std::size_t | 0 | 通道下标 |

**返回**
| 类型 | 描述 |
| --- | --- |
| bool | 原始观测返回true，预测返回false |

**抛出**
| 异常 | 条件 |
| --- | --- |
| std::invalid_argument | step或channel越界 |

#### Result::cacheHitRate()

分页缓存命中率。

**返回**
| 类型 | 描述 |
| --- | --- |
| double | reusedPages/(reusedPages+computedPages)，无页统计时返回0 |

### Timestral::RequestSuperseded

请求被Mailbox策略的同源新请求或abort作废时，其future抛出此异常。

**继承**

std::runtime_error
