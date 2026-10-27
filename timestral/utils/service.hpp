/***************
 * @file service.hpp
 * @brief 内部调度服务：请求队列、三种调度策略与共享分页前缀树
 ***************/

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "timestral/timestral.hpp"
#include "timestral/utils/cache.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

class Engine;

// 调度线程关闭时，尚未开始执行的请求在此抛出
class EngineShutdown : public std::runtime_error {
public:
    EngineShutdown();
};

// 唯一的内部调度实现：一个工作线程按策略取请求，执行后填写future
// 参数校验在提交线程完成，工作线程只做数值计算与页缓存维护
// 调度约定参考vLLM/sglang的serve路径：等待队列FCFS、只从队首准入、
// Continuous以maxBatch为槽位数、没有空位就等待且不抢占
// 页缓存参考vLLM的prefix caching：页键是父键与patch输入的链式哈希，
// 共享页按内容命中并维护引用计数，页池满时只淘汰引用计数为0的LRU页
class Service {
public:
    // @brief 持有引擎并启动工作线程
    // @param engine 已加载的引擎，不得为空
    // @param settings 载入期参数，只读取maxBatch
    // @param policy 初始调度策略
    // @throw std::invalid_argument 引擎为空时抛出
    Service(
        std::shared_ptr<const Engine> engine,
        const Timestral::Settings& settings,
        Timestral::Policy policy
    );

    // @brief 停止工作线程，队列中未开始的请求以EngineShutdown结束
    ~Service();

    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;

    // @brief 校验并提交一次预测
    // @param request 请求内容，校验在调用线程完成
    // @return 就绪后得到结果的future
    // @throw std::invalid_argument 历史、步数、间隔或协变量非法时抛出
    std::future<Timestral::Result> submit(Timestral::Request request);

    // @brief 提交并等待结果
    // @param request 请求内容，要求同submit
    // @return 预测结果
    // @throw std::invalid_argument 请求参数非法时抛出
    Timestral::Result predict(Timestral::Request request);

    // @brief 切换调度策略，只影响尚未被调度的请求
    // @param policy 新策略
    void setPolicy(Timestral::Policy policy);

    // @brief 当前调度策略
    // @return 当前策略
    Timestral::Policy policy() const;

    // @brief 作废一个来源尚未开始的请求，并释放其页表引用、摘除无人引用的页键
    // @param streamId 来源标识
    void abort(const std::string& streamId);

    // @brief 队列中尚未开始执行的请求数
    // @return 请求数
    std::size_t pendingRequests() const;

    // @brief 模型patch长度
    // @return 模型的patchSize
    std::size_t patchSize() const;

    // @brief 模型训练上下文长度
    // @return 模型配置的contextLength
    std::size_t contextLength() const;

private:
    // 排队请求：公开请求、结果承诺与提交线程算好的网格
    struct QueuedRequest {
        Timestral::Request request;               // 原始输入
        std::promise<Timestral::Result> promise;  // 结果承诺
        double interval = 0.0;                    // 预测间隔，单位秒
        std::size_t steps = 0;                    // 预测步数
        std::size_t begin = 0;                    // 保留历史的起点
        std::size_t kept = 0;                     // 保留历史的样本数
        std::size_t gridLength = 0;               // 模型输入网格长度
        bool streamingGrid = false;               // 采用分页网格
        bool anonymous = false;                   // 执行完归还匿名视图
        Tensor observations;                      // 条件补全的公共观测网格
        std::vector<std::size_t> observedEnds;    // 各通道末次有效观测之后的下标
        double gridOrigin = 0.0;                  // 公共网格起点
    };

    // 内容寻址前缀树的一个节点：页号、父页键与当前patch输入指纹
    struct PrefixNode {
        std::size_t page = 0;                // 物理页下标
        std::uint64_t parentKey = 0;         // 父节点页键
        std::uint64_t inputFingerprint = 0;  // 当前patch指纹
    };

    // 每种通道数各有固定几何的页池和前缀树
    struct PrefixCache {
        PagePool pool;                                                 // 固定几何缓冲
        std::map<std::uint64_t, PrefixNode> nodes;                     // 内容寻址节点
        std::map<std::uint64_t, std::vector<std::uint64_t>> children;  // 前缀树边
        std::map<std::string, std::vector<std::size_t>> views;         // 来源页表引用
    };

    // @brief 工作线程主循环，无请求时在条件变量上阻塞
    void run();

    // @brief 在提交线程算好步数、上下文区间与页池网格
    // @param queued 待补全的排队请求
    void prepare(QueuedRequest& queued);

    // @brief 取本次上下文的公共时间轴，离散模式返回空数组
    // @param queued 已校验请求
    // @param length 需要的采样点数
    // @return 连续模式使用的时间戳
    std::vector<double> timeGrid(const QueuedRequest& queued, std::size_t length) const;

    // @brief 取队首一条请求
    // @param round 输出，容量为1
    void takeFront(std::vector<QueuedRequest>& round);

    // @brief 按FCFS准入最多maxBatch条可拼批请求
    // @param round 输出
    void takeContinuousRound(std::vector<QueuedRequest>& round);

    // @brief 从Mailbox索引中移除被取走的请求
    // @param entry 队列节点
    void removeMailboxEntry(std::list<QueuedRequest>::iterator entry);

    // @brief 判断两条请求能否共用一次前向
    // @param first 本轮首条请求
    // @param next 候选请求
    // @return 步数、间隔、历史长度、协变量与网格一致时返回true
    bool batchable(const QueuedRequest& first, const QueuedRequest& next) const;

    // @brief 执行一轮取出的请求
    // @param round 本轮请求，至少一条
    void executeRound(std::vector<QueuedRequest>& round);

    // @brief 逐条执行一条请求并填写future
    // @param queued 待执行请求
    void executeOne(QueuedRequest& queued);

    // @brief 用一次(B,V,T)前向算完本轮请求
    // @param round 本轮请求，至少两条
    void executeBatch(std::vector<QueuedRequest>& round);

    // @brief 用共享整组页表的一次前向解码前缀相同的请求
    // @param group 组内请求，按提交顺序，至少两条
    // @param tables 每条的命中页表，成功时引用转交来源视图
    // @note 调用方持有streamMutex，异常时负责归还尚未转交视图的引用
    void executeShared(
        std::vector<QueuedRequest>& group,
        std::vector<std::vector<std::size_t>>& tables
    );

    // @brief 组装请求的整patch前缀输入
    // @param queued 请求
    // @param prefixContext 输出形状(V,fullPatches*patch)的前缀
    // @param prefixObserved 输出同形掩码
    // @return 整patch数
    std::size_t buildPrefixInput(
        const QueuedRequest& queued,
        Tensor& prefixContext,
        std::vector<std::uint8_t>& prefixObserved,
        std::vector<double>& coordinates
    ) const;

    // @brief 从根沿内容寻址键扫描前缀
    // @param cache 对应通道几何的缓存
    // @param prefixContext 形状(V,fullPatches*patchSize)的前缀输入
    // @param prefixObserved 与prefixContext同形的观测掩码
    // @param fullPatches 整patch数
    // @param acquire 为true时对命中页增加引用并写入table
    // @param table 输出页表，仅acquire为true时追加
    // @return 从第一个patch起连续命中的页数
    std::size_t scanPrefix(
        PrefixCache& cache,
        const Tensor& prefixContext,
        const std::vector<std::uint8_t>& prefixObserved,
        std::size_t fullPatches,
        bool acquire,
        std::vector<std::size_t>& table,
        const std::vector<double>& coordinates
    );

    // @brief 取得固定通道几何的页池，调用方须持有streamMutex
    // @param channels 每组通道数
    // @return 该几何的缓存状态
    PrefixCache& ensurePagePool(std::size_t channels);

    // @brief 登记一个页键节点
    // @param cache 对应通道几何的缓存
    // @param key 页键
    // @param parentKey 父页键
    // @param inputFingerprint 输入指纹
    // @param page 页号
    void registerNode(
        PrefixCache& cache,
        std::uint64_t key,
        std::uint64_t parentKey,
        std::uint64_t inputFingerprint,
        std::size_t page
    );

    // @brief 从树中摘除一个键及其后代
    // @param cache 对应通道几何的缓存
    // @param key 根键
    // @param discardPages 为true时释放引用计数为0的页
    // @return 摘除的节点数
    std::size_t pruneSubtree(PrefixCache& cache, std::uint64_t key, bool discardPages);

    // @brief 淘汰LRU页直到能容纳needed个新页
    // @param cache 对应通道几何的缓存
    // @param needed 新增整patch数
    // @return 容量足够时返回true
    bool ensureRoom(PrefixCache& cache, std::size_t needed);

    // @brief 释放一个来源的页表引用，不摘除页键
    // @param cache 对应通道几何的缓存
    // @param streamId 来源标识
    void releaseView(PrefixCache& cache, const std::string& streamId);

    // @brief 释放一个来源的页表引用并摘除无人引用的页键，调用方须持有streamMutex
    // @param cache 对应通道几何的缓存
    // @param streamId 来源标识
    void dropView(PrefixCache& cache, const std::string& streamId);

    // @brief 把本次页表登记为该来源的新引用视图
    // @param cache 对应通道几何的缓存
    // @param streamId 来源标识
    // @param table 本次页表，引用所有权转移给视图
    void replaceView(
        PrefixCache& cache,
        const std::string& streamId,
        const std::vector<std::size_t>& table
    );

    // @brief 归还页表中的引用
    // @param pool 页表所属页池
    // @param table 页表
    static void releaseTable(PagePool& pool, std::vector<std::size_t>& table);

    // @brief 全量重算一条请求
    // @param queued 请求
    // @param start 本轮计时起点
    // @return 预测结果，页统计为0
    Timestral::Result runFull(const QueuedRequest& queued, std::chrono::steady_clock::time_point start) const;

    // @brief 用给定的对齐上下文做全量重算
    // @param queued 请求
    // @param context 形状(1,V,T)的上下文
    // @param start 本轮计时起点
    // @return 预测结果，页统计为0
    Timestral::Result runFullContext(
        const QueuedRequest& queued,
        const Tensor& context,
        std::chrono::steady_clock::time_point start
    ) const;

    // @brief 走共享前缀树执行一条请求
    // @param queued 请求
    // @param start 本轮计时起点
    // @return 预测结果与页统计
    Timestral::Result runStreamed(
        QueuedRequest& queued,
        std::chrono::steady_clock::time_point start
    );

    // @brief 把引擎返回值组装成Result
    // @param queued 请求
    // @param quantiles 形状(B,V,steps,Q)的分位数
    // @param batchIndex 批下标
    // @param prefillSeconds 页编码耗时
    // @param decodeSeconds 解码耗时
    // @param totalSeconds 整次执行耗时
    // @return 公开结果
    Timestral::Result makeResult(
        const QueuedRequest& queued,
        const Tensor& quantiles,
        std::size_t batchIndex,
        double prefillSeconds,
        double decodeSeconds,
        double totalSeconds
    ) const;

    std::shared_ptr<const Engine> engine;
    ModelConfig config;
    std::size_t maxBatch = 1;
    std::size_t maxVariables = 8;
    std::size_t pageCapacity = 64;
    std::size_t anonymousCount = 0;  // 已分配的匿名来源数

    mutable std::mutex queueMutex;
    std::condition_variable condition;
    std::list<QueuedRequest> queue;
    std::map<std::string, std::list<QueuedRequest>::iterator> mailbox;
    Timestral::Policy policyValue = Timestral::Policy::Fifo;
    bool stopping = false;
    std::thread worker;

    // 共享页池与前缀树只在工作线程内使用，abort可来自任意线程，因此单独保护
    mutable std::mutex streamMutex;
    std::map<std::size_t, PrefixCache> caches;
};

} // namespace timestral
