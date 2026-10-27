# API参考

公共接口只有一个入口头文件[timestral/timestral.hpp](<../timestral/timestral.hpp>)，链接CMake目标timestral即可使用，默认构建只依赖C++17标准库。

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

提交一次预测并立即返回，参数校验在提交线程完成。

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
| std::invalid_argument | 请求参数非法时在提交线程抛出，请求不进入队列 |

### Timestral::abort()

结束一个来源：作废其尚未开始的请求，并释放它持有的前缀页引用。

**参数**
| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| streamId | const std::string& | - | 来源标识，被作废请求的future抛RequestSuperseded |

**注意**

仍被其他来源引用的页继续共享，无人引用的页立即摘除页键并释放。

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

实例配置，构造时读取，构造后不可修改。

**成员**
| 成员变量 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| maxContext | std::size_t | 0 | 上下文样本数上限，0表示用模型contextLength |
| patchBatch | std::size_t | 16 | 块AR每轮patch数，如1/2/4/8/16/32，构造时钳制到blockSize |
| maxBatch | std::size_t | 4 | 每轮请求槽位数 |
| maxVariables | std::size_t | 8 | 每条请求的目标与协变量总数上限 |
| cachePages | std::size_t | 64 | 每种通道数的预分配页数，0禁用缓存 |
| threads | std::size_t | 1 | Eigen/ytensor线程数，默认1串行，0自动（最多4），N为精确线程数 |
| continuousRope | bool | false | 实验性连续patch时间位置 |
| ropeReferenceInterval | double | 0.0 | 参考采样间隔，0采用请求间隔 |
| gpuThreads | std::size_t | 0 | Vulkan逻辑派发窗口预算，0不限制，正数须为256倍数，不是物理CU配额 |

### Timestral::Request

一条预测请求。末尾不齐时按观测可用性分段补全，horizon从最晚有效观测起算；所有通道共用一条时间轴，条件与目标都由variables自身的观测位置决定。

**成员**
| 成员变量 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| streamId | std::string | 空串 | 来源标识，空串是匿名请求 |
| variables | std::vector<std::vector<Sample>> | - | 各通道按时间递增的已知读数 |
| horizonSeconds | double | 0.0 | 预测跨度，单位秒，必须为正 |
| sampleInterval | double | 0.0 | 采样间隔，单位秒，0表示从时间戳推断 |
| symmetricAveraging | bool | true | 加做上下文取负的镜像前向并平均 |
| clipNonnegative | bool | true | 观测全非负的轨迹截断到非负 |

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
| reusedPages | std::size_t | 命中的前缀页数 |
| computedPages | std::size_t | 新计算的页数 |

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
