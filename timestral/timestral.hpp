/***************
 * @file timestral.hpp
 * @brief 本地Timestral推理入口：采样点进，分位数预测出
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "timestral/timestralExport.hpp"

namespace timestral {

class Service;

// 时间单位秒，原点任意，NaN表示缺测
struct TIMESTRAL_API Sample {
    double time = 0.0;   // 时间戳
    double value = 0.0;  // 读数
};

// 支持并发提交，共享前缀缓存按内容寻址，零引用页按LRU淘汰
class TIMESTRAL_API Timestral {
public:
    // Fifo顺序执行，Mailbox保留同源最新待执行请求，Continuous自动拼批
    enum class Policy {
        Fifo,
        Mailbox,
        Continuous
    };

    // 实例配置，在构造时读取
    struct Settings {
        std::size_t maxContext = 0;          // 上下文样本数上限，0表示用模型contextLength
        std::size_t patchBatch = 16;         // 块AR每轮patch数，如1/2/4/8/16/32，构造时钳制到blockSize
        std::size_t maxBatch = 4;            // 每轮请求槽位数
        std::size_t maxVariables = 8;        // 每条请求的目标与协变量总数上限
        std::size_t cachePages = 64;         // 每种通道数的预分配页数，0禁用缓存
        std::size_t threads = 1;             // Eigen/ytensor线程数，默认1串行，0自动（最多4），N为精确线程数
        bool continuousRope = false;         // 实验性连续patch时间位置
        double ropeReferenceInterval = 0.0;  // 参考采样间隔，0采用请求间隔
        std::size_t gpuThreads = 0;          // Vulkan逻辑派发窗口预算，0不限制，正数须为256倍数，不是物理CU配额
    };

    // 末尾不齐时按观测可用性分段补全，horizon从最晚有效观测起算
    // 所有通道共用一条时间轴，条件与目标都由variables自身的观测位置决定
    struct Request {
        std::string streamId;                        // 来源标识，空串是匿名请求
        std::vector<std::vector<Sample>> variables;  // 各通道按时间递增的已知读数
        double horizonSeconds = 0.0;                 // 预测跨度，单位秒，必须为正
        double sampleInterval = 0.0;                 // 采样间隔，单位秒，0表示从时间戳推断
        bool symmetricAveraging = true;              // 加做上下文取负的镜像前向并平均
        bool clipNonnegative = true;                 // 观测全非负的轨迹截断到非负
    };

    // 时间单位秒，分位数按(channel,step,level)行主序存储
    struct Result {
        std::size_t channels = 1;            // 通道数，顺序与输入一致
        double sampleInterval = 0.0;         // 生效的采样间隔
        double origin = 0.0;                 // 预测窗口前一时刻
        std::vector<double> levels;          // 模型分位点
        std::vector<double> quantiles;       // (channels, steps, levels)行主序
        std::vector<std::uint8_t> observed;  // (channels, steps)，1为原始观测
        double prefillSeconds = 0.0;         // 前缀页编码耗时
        double decodeSeconds = 0.0;          // 解码耗时
        double totalSeconds = 0.0;           // 整次执行耗时
        std::size_t reusedPages = 0;         // 命中的前缀页数
        std::size_t computedPages = 0;       // 新计算的页数

        // @brief 每个通道的预测窗口步数
        std::size_t steps() const;

        // @brief 第step步的时间戳
        // @param step 从0起的步下标，须小于steps()
        // @return origin加(step+1)个采样间隔，单位秒
        // @throw std::invalid_argument step不小于steps()时抛出
        double time(std::size_t step) const;

        // @brief 第step步的中位数，即0.5分位
        // @param step 从0起的步下标，须小于steps()
        // @param channel 通道下标，默认第0通道
        // @return 该步中位数的原始量纲值
        // @throw std::invalid_argument step或channel越界时抛出
        double median(std::size_t step, std::size_t channel = 0) const;

        // @brief 第step步在任意水平上的分位数
        // @param step 从0起的步下标，须小于steps()
        // @param level 目标水平，取值(0,1)，落在模型分位点网格外时取网格端点
        // @param channel 通道下标，默认第0通道
        // @return 该步在level上的分位数，原始量纲
        // @throw std::invalid_argument step或channel越界，或level不在(0,1)时抛出
        double quantile(std::size_t step, double level, std::size_t channel = 0) const;

        // @brief 中心区间的下界
        // @param step 从0起的步下标，须小于steps()
        // @param level 区间覆盖率，取值(0,1)，level=0.9取0.05分位
        // @param channel 通道下标，默认第0通道
        // @throw std::invalid_argument step或channel越界，或level不在(0,1)时抛出
        double lower(std::size_t step, double level, std::size_t channel = 0) const;

        // @brief 中心区间的上界
        // @param step 从0起的步下标，须小于steps()
        // @param level 区间覆盖率，取值(0,1)，level=0.9取0.95分位
        // @param channel 通道下标，默认第0通道
        // @throw std::invalid_argument step或channel越界，或level不在(0,1)时抛出
        double upper(std::size_t step, double level, std::size_t channel = 0) const;

        // @brief 该点是否为原始观测，已知位置不代表模型预测分布
        // @param step 从0起的步下标
        // @param channel 通道下标，默认第0通道
        // @return 原始观测返回true，预测返回false
        // @throw std::invalid_argument step或channel越界时抛出
        bool isObserved(std::size_t step, std::size_t channel = 0) const;

        // @brief 分页缓存命中率
        // @return reusedPages/(reusedPages+computedPages)，无页统计时返回0
        double cacheHitRate() const;
    };

    // 请求被Mailbox策略的同源新请求或abort作废时，其future抛出此异常
    class TIMESTRAL_API RequestSuperseded : public std::runtime_error {
    public:
        RequestSuperseded();
    };

    // @brief 从本地模型目录加载配置与权重，并启动调度线程
    // @param modelDirectory 含config.json与权重文件的目录
    // @param settings 构造后固定的配置
    // @param policy 初始调度策略
    // @throw std::runtime_error 目录、配置或权重缺失、字段非法时抛出
    explicit Timestral(
        const std::filesystem::path& modelDirectory,
        Settings settings,
        Policy policy = Policy::Fifo
    );

    // @brief 使用默认Settings加载，保留Settings的聚合初始化语义
    explicit Timestral(const std::filesystem::path& modelDirectory);

    // @brief 停止调度线程，尚未开始的请求的future以std::runtime_error结束
    ~Timestral();

    Timestral(const Timestral&) = delete;
    Timestral& operator=(const Timestral&) = delete;

    // @brief 切换调度策略，只影响尚未被调度的请求
    // @param policy 新策略，切换为Mailbox时，队列中同源请求只保留最后一条未开始的
    void setPolicy(Policy policy);

    // @brief 当前调度策略
    Policy policy() const;

    // @brief 提交一次预测并立即返回，参数校验在提交线程完成
    // @param request 请求内容，空streamId按匿名请求处理，服务内部分配标识
    // @return 就绪后得到结果的future，被作废时get()抛RequestSuperseded，引擎失败时抛std::runtime_error
    // @throw std::invalid_argument 请求参数非法时在提交线程抛出，请求不进入队列
    std::future<Result> submit(Request request);

    // @brief 提交一次预测并等待结果，等价于submit后立即get
    // @param request 请求内容，要求同submit
    // @return 预测结果
    // @throw std::invalid_argument 请求参数非法时抛出
    // @throw RequestSuperseded Mailbox策略下被同源新请求或abort作废时抛出
    Result predict(Request request);

    // @brief 结束一个来源：作废其尚未开始的请求，并释放它持有的前缀页引用
    // @param streamId 来源标识，被作废请求的future抛RequestSuperseded
    // @note 仍被其他来源引用的页继续共享，无人引用的页立即摘除页键并释放
    void abort(const std::string& streamId);

    // @brief 队列中尚未开始执行的请求数
    std::size_t pendingRequests() const;

    // @brief 模型的patch长度，5M模型为32
    std::size_t patchSize() const;

    // @brief 模型支持的最长上下文（按步计）
    std::size_t contextLength() const;

private:
    std::shared_ptr<Service> service;
};

} // namespace timestral
