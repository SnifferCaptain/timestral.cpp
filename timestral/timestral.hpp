/***************
 * @file timestral.hpp
 * @brief 本地Timestral推理入口：采样点进，分位数预测出
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
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

    // 实例配置，在构造时读取，无状态请求默认采用与上游一致的完整注意力
    // attentionMask的参数是time轴token下标：一个token是patchSize个采样点，从模型上下文网格的最早patch起计，
    // 预测patch接续计数。返回false的(query,key)对在原生因果掩码之外再被屏蔽，token对自身恒可见。
    // reference与BLAS后端原生执行，Eigen、ytensor与Vulkan后端设置后改用reference计算，速度明显下降
    struct Settings {
        std::size_t maxContext = 0;          // 上下文样本数上限，0表示用模型contextLength，超出部分按滑窗丢弃最早读数
        std::size_t patchBatch = 16;         // 块AR每轮patch数，如1/2/4/8/16/32，构造时钳制到blockSize
        std::size_t maxBatch = 4;            // 每轮请求槽位数
        std::size_t maxVariables = 8;        // 每条请求的目标与协变量总数上限
        std::size_t cachePages = 2048;       // 每种通道数的KV页数上限，一页存一个patch，按需分配，0禁用缓存
        std::size_t maxSessions = 16;        // 保留历史的会话数，超出时淘汰最久未用的，0不建立会话
        std::size_t threads = 1;             // BLAS/Eigen/ytensor线程数，默认1串行，0自动（最多4），N为精确线程数
        bool continuousRope = false;         // 实验性连续patch时间位置
        double ropeReferenceInterval = 0.0;  // 参考采样间隔，0采用请求间隔
        std::size_t gpuThreads = 0;          // Vulkan逻辑派发窗口预算，0不限制，正数须为256倍数，不是物理CU配额
        std::function<bool(std::size_t query, std::size_t key)> attentionMask;  // time轴附加可见性，空为原生掩码
        std::size_t fullPrefillThreshold = 512;  // 会话保留历史不超过此点数时使用全部新读数，允许重新prefill，0始终固定边界
    };

    // 末尾不齐时按观测可用性分段补全，horizon从最晚有效观测起算
    // 所有通道共用一条时间轴，条件与目标都由variables自身的观测位置决定
    // session为空时variables是完整历史，非空时variables只含追加到该会话各通道末尾之后的新读数
    // 同步通道的会话在保留历史不超过fullPrefillThreshold时使用全部读数，超过后固定最后一次完整接纳的patch边界
    // 固定边界后，未满patch的追加读数暂存，凑满后才参与预测
    // 暂存期间从最后完整patch预测并跳过暂存点对应的跨度，返回窗口仍紧接最新读数
    struct Request {
        std::string session;                         // 续接的会话标识，取自上次Result::session
        std::string streamId;                        // 来源标识，空串是匿名请求，续接会话时默认取会话标识
        std::vector<std::vector<Sample>> variables;  // 各通道按时间递增的读数，通道数须与会话一致
        double horizonSeconds = 0.0;                 // 预测跨度，单位秒，0取patchBatch个patch
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
        std::size_t bufferedSamples = 0;     // 每通道暂存、尚未参与预测的尾部读数数，少于patchSize
        std::string session;                 // 保存本次完整历史的会话，maxSessions为0时为空

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

    // @brief 提交一次预测并立即返回，参数校验与会话追加在提交线程完成
    // @param request 请求内容，空streamId按匿名请求处理，服务内部分配标识
    // @return 就绪后得到结果的future，被作废时get()抛RequestSuperseded，引擎失败时抛std::runtime_error
    // @throw std::invalid_argument 请求参数非法、会话不存在或新读数不晚于会话末尾时在提交线程抛出，请求不进入队列，会话不变
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

    // @brief 删除一个会话保存的历史，并作废其尚未开始的请求
    // @param session 会话标识，不存在时忽略
    // @note 会话用过的KV页不立即释放，留在LRU中按需淘汰
    void close(const std::string& session);

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
