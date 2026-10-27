/***************
 * @file service.cpp
 * @brief 调度服务的实现：FCFS队列、按槽位准入、共享前缀树与LRU淘汰
 ***************/

#include "timestral/utils/service.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "timestral/utils/condition.hpp"
#include "timestral/utils/engine.hpp"
#include "timestral/utils/forecast.hpp"
#include "timestral/utils/patch.hpp"
#include "timestral/utils/tensor.hpp"
#include "timestral/utils/time.hpp"

namespace timestral {

static double secondsSince(const std::chrono::steady_clock::time_point& start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

static std::size_t roundPatch(std::size_t length, std::size_t patch) {
    const std::size_t extra = length % patch == 0 ? 0 : patch - length % patch;
    if (length > std::numeric_limits<std::size_t>::max() - extra) {
        throw std::invalid_argument("sample count overflow while aligning patches");
    }
    return length + extra;
}

// 对称平均需要镜像分位点，在提交线程校验，非法请求不进入队列
static void validateControls(const Timestral::Request& request, const ModelConfig& config) {
    if (!request.symmetricAveraging) {
        return;
    }
    const std::size_t levels = config.quantileKnots.size();
    for (std::size_t q = 0; q < levels; ++q) {
        if (std::abs(config.quantileKnots[q] + config.quantileKnots[levels - q - 1] - 1.0f) > 1e-7f) {
            throw std::invalid_argument("symmetric averaging requires symmetric quantile levels");
        }
    }
}

EngineShutdown::EngineShutdown()
    : std::runtime_error("engine shutdown") {
}

Service::Service(
    std::shared_ptr<const Engine> engine,
    const Timestral::Settings& settings,
    Timestral::Policy policy
)
    : engine(std::move(engine)),
      config(this->engine ? this->engine->config() : ModelConfig{}),
      maxBatch(std::max<std::size_t>(1, settings.maxBatch)),
      maxVariables(settings.maxVariables),
      pageCapacity(settings.cachePages),
      policyValue(policy) {
    if (!this->engine) {
        throw std::invalid_argument("service requires a loaded engine");
    }
    if (maxVariables == 0) {
        throw std::invalid_argument("maxVariables must be positive");
    }
    worker = std::thread(&Service::run, this);
}

Service::~Service() {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stopping = true;
        for (QueuedRequest& queued : queue) {
            try {
                queued.promise.set_exception(std::make_exception_ptr(EngineShutdown()));
            } catch (...) {
                // 调用方已丢弃future时承诺可能失效，关闭流程继续
            }
        }
        queue.clear();
        mailbox.clear();
    }
    condition.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

std::future<Timestral::Result> Service::submit(Timestral::Request request) {
    QueuedRequest queued;
    queued.request = std::move(request);
    prepare(queued);
    std::future<Timestral::Result> future = queued.promise.get_future();
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (stopping) {
            queued.promise.set_exception(std::make_exception_ptr(EngineShutdown()));
            return future;
        }
        if (queued.request.streamId.empty()) {
            // 匿名与具名来源使用不同命名空间，序号保证哈希碰撞时仍不混源
            queued.anonymous = true;
            if (anonymousCount == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error("anonymous request identifier exhausted");
            }
            const std::size_t ordinal = anonymousCount++;
            FnvHash hash;
            hash.addUnsigned(ordinal);
            queued.request.streamId = "anonymous:" + std::to_string(hash.value())
                + ":" + std::to_string(ordinal);
        } else {
            queued.request.streamId = "named:" + queued.request.streamId;
        }
        // Mailbox只在提交时作废同源旧请求，匿名标识唯一，互不去重
        if (policyValue == Timestral::Policy::Mailbox) {
            const auto found = mailbox.find(queued.request.streamId);
            if (found != mailbox.end()) {
                found->second->promise.set_exception(std::make_exception_ptr(Timestral::RequestSuperseded()));
                queue.erase(found->second);
                mailbox.erase(found);
            }
        }
        queue.push_back(std::move(queued));
        if (policyValue == Timestral::Policy::Mailbox) {
            mailbox[queue.back().request.streamId] = --queue.end();
        }
    }
    condition.notify_one();
    return future;
}

Timestral::Result Service::predict(Timestral::Request request) {
    return submit(std::move(request)).get();
}

void Service::setPolicy(Timestral::Policy policy) {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (policyValue == policy) {
        return;
    }
    policyValue = policy;
    mailbox.clear();
    if (policy != Timestral::Policy::Mailbox) {
        return;
    }
    // 切换为Mailbox时，队列里同源请求只保留最后一条未开始的
    std::map<std::string, std::list<QueuedRequest>::iterator> latest;
    for (auto entry = queue.begin(); entry != queue.end(); ++entry) {
        const std::string& streamId = entry->request.streamId;
        if (streamId.empty()) {
            continue;
        }
        const auto previous = latest.find(streamId);
        if (previous != latest.end()) {
            previous->second->promise.set_exception(std::make_exception_ptr(Timestral::RequestSuperseded()));
            queue.erase(previous->second);
        }
        latest[streamId] = entry;
    }
    mailbox = std::move(latest);
}

Timestral::Policy Service::policy() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return policyValue;
}

void Service::abort(const std::string& streamId) {
    const std::string identity = "named:" + streamId;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        for (auto entry = queue.begin(); entry != queue.end(); ) {
            if (entry->request.streamId == identity) {
                try {
                    entry->promise.set_exception(std::make_exception_ptr(Timestral::RequestSuperseded()));
                } catch (...) {
                    // 调用方已丢弃future时承诺可能失效，作废流程继续
                }
                entry = queue.erase(entry);
            } else {
                ++entry;
            }
        }
        mailbox.erase(identity);
    }
    std::lock_guard<std::mutex> lock(streamMutex);
    for (auto& entry : caches) {
        dropView(entry.second, identity);
    }
}

std::size_t Service::pendingRequests() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return queue.size();
}

std::size_t Service::patchSize() const {
    return config.patchSize;
}

std::size_t Service::contextLength() const {
    return config.contextLength;
}

void Service::prepare(QueuedRequest& queued) {
    const std::string name = "submit";
    if (queued.request.variables.empty() || queued.request.variables.size() > maxVariables) {
        throw std::invalid_argument("submit: variables count must be between 1 and maxVariables");
    }
    const auto& first = queued.request.variables.front();
    bool sameGrid = true;
    for (const auto& channel : queued.request.variables) {
        validateHistory(channel, config.patchSize, name);
        sameGrid = sameGrid && channel.size() == first.size() && std::isfinite(channel.back().value);
        for (std::size_t i = 0; i < std::min(channel.size(), first.size()); ++i) {
            sameGrid = sameGrid && channel[i].time == first[i].time;
        }
    }
    if (!sameGrid) {
        queued.observations = alignObservations(
            queued.request,
            config.patchSize,
            queued.observedEnds,
            queued.interval,
            queued.gridOrigin
        );
        const std::size_t firstEnd = *std::min_element(queued.observedEnds.begin(), queued.observedEnds.end());
        const std::size_t lastEnd = *std::max_element(queued.observedEnds.begin(), queued.observedEnds.end());
        if (std::min(firstEnd, engine->maxContext()) < 2 * config.patchSize) {
            throw std::invalid_argument("submit: conditional context requires at least 2 * patchSize steps");
        }
        if (firstEnd != lastEnd && !config.supportsKnownCovariates()) {
            throw std::invalid_argument("submit: checkpoint cannot condition on known-future covariates");
        }
        const std::size_t future = stepCount(queued.request.horizonSeconds, queued.interval, name);
        if (lastEnd > std::numeric_limits<std::size_t>::max() - future) {
            throw std::invalid_argument("submit: conditional prediction length overflow");
        }
        queued.steps = lastEnd - firstEnd + future;
        elementCount({queued.request.variables.size(), queued.steps, config.quantileKnots.size()});
        validateControls(queued.request, config);
        return;
    }
    queued.interval = resolveInterval(
        queued.request.variables.front(),
        queued.request.sampleInterval,
        name
    );
    queued.steps = stepCount(queued.request.horizonSeconds, queued.interval, name);
    const std::pair<std::size_t, std::size_t> range = contextRange(
        queued.request.variables.front(),
        engine->maxContext(),
        config.patchSize,
        name
    );
    queued.begin = range.first;
    queued.kept = range.second;
    // FP32范围同样属于提交期校验
    for (const auto& channel : queued.request.variables) {
        for (std::size_t i = queued.begin; i < queued.begin + queued.kept; ++i) {
            fp32Sample(channel, i, name);
        }
    }
    validateControls(queued.request, config);
    const std::size_t gridLength = roundPatch(queued.kept, config.patchSize);
    // 后端支持分页且补齐后不越过maxContext时才用流式网格
    queued.streamingGrid = engine->supportsIncremental()
        && (!engine->continuousRope() || queued.kept % config.patchSize == 0)
        && gridLength <= engine->maxContext()
        && queued.kept / config.patchSize <= pageCapacity;
    queued.gridLength = queued.streamingGrid ? gridLength : queued.kept;
}

void Service::run() {
    std::unique_lock<std::mutex> lock(queueMutex);
    while (!stopping) {
        condition.wait(lock, [this] {
            return stopping || !queue.empty();
        });
        if (stopping) {
            break;
        }
        std::vector<QueuedRequest> round;
        if (policyValue == Timestral::Policy::Continuous) {
            takeContinuousRound(round);
        } else {
            takeFront(round);
        }
        lock.unlock();
        try {
            executeRound(round);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (QueuedRequest& queued : round) {
                try {
                    queued.promise.set_exception(error);
                } catch (...) {
                    // 已填写的future不再重复设置
                }
            }
        }
        lock.lock();
    }
}

std::vector<double> Service::timeGrid(const QueuedRequest& queued, std::size_t length) const {
    if (!engine->continuousRope()) {
        return {};
    }
    std::vector<double> times(length);
    const auto& channel = queued.request.variables.front();
    for (std::size_t i = 0; i < length; ++i) {
        times[i] = i < queued.kept ? channel[queued.begin + i].time
            : channel[queued.begin + queued.kept - 1].time + static_cast<double>(i - queued.kept + 1) * queued.interval;
    }
    return times;
}

void Service::takeFront(std::vector<QueuedRequest>& round) {
    auto entry = queue.begin();
    removeMailboxEntry(entry);
    round.push_back(std::move(*entry));
    queue.erase(entry);
}

void Service::takeContinuousRound(std::vector<QueuedRequest>& round) {
    // FCFS准入：只从队首连续取可拼批请求，遇到不可拼批的留到下一轮
    // maxBatch限制请求槽位，一轮内的兼容请求共同执行
    // 没有逐token的中间状态，因此一轮准入的请求一起算完并一起释放槽位
    auto entry = queue.begin();
    removeMailboxEntry(entry);
    round.push_back(std::move(*entry));
    queue.erase(entry);
    while (round.size() < maxBatch && !queue.empty()) {
        auto next = queue.begin();
        if (!batchable(round.front(), *next)) {
            break;
        }
        removeMailboxEntry(next);
        round.push_back(std::move(*next));
        queue.erase(next);
    }
}

void Service::removeMailboxEntry(std::list<QueuedRequest>::iterator entry) {
    if (policyValue != Timestral::Policy::Mailbox) {
        return;
    }
    const auto found = mailbox.find(entry->request.streamId);
    if (found != mailbox.end() && found->second == entry) {
        mailbox.erase(found);
    }
}

bool Service::batchable(const QueuedRequest& first, const QueuedRequest& next) const {
    const bool conditioned = !first.observations.values.empty();
    if (conditioned != !next.observations.values.empty()) {
        return false;
    }
    if (conditioned) {
        if (first.observations.shape != next.observations.shape || first.observedEnds != next.observedEnds
            || first.steps != next.steps || first.interval != next.interval
            || first.request.symmetricAveraging != next.request.symmetricAveraging
            || first.request.clipNonnegative != next.request.clipNonnegative) {
            return false;
        }
        const std::size_t start = *std::min_element(first.observedEnds.begin(), first.observedEnds.end());
        const std::size_t time = first.observations.shape[1];
        for (std::size_t channel = 0; channel < first.observations.shape[0]; ++channel) {
            for (std::size_t i = start; i < time; ++i) {
                const std::size_t index = channel * time + i;
                if (std::isfinite(first.observations.values[index]) != std::isfinite(next.observations.values[index])) {
                    return false;
                }
            }
        }
        return true;
    }
    if (first.request.variables.size() != next.request.variables.size()
        || first.request.variables.front().size() != next.request.variables.front().size()) {
        return false;
    }
    if (first.steps != next.steps || first.interval != next.interval) {
        return false;
    }
    if (first.streamingGrid != next.streamingGrid || first.gridLength != next.gridLength) {
        return false;
    }
    // 这两个标志决定每行的后处理，拼批时必须一致才能共用一次前向
    if (first.request.symmetricAveraging != next.request.symmetricAveraging
        || first.request.clipNonnegative != next.request.clipNonnegative) {
        return false;
    }
    return true;
}

void Service::executeRound(std::vector<QueuedRequest>& round) {
    if (round.empty()) {
        return;
    }
    if (round.size() == 1) {
        executeOne(round.front());
        return;
    }
    if (!round.front().streamingGrid) {
        executeBatch(round);
        return;
    }
    // 命中前缀的请求按页表内容分组：页号序列相同意味着复用的页逐位相同，
    // 组内R条可以共用单行页表的一次前向，未命中的请求仍按冷拼批执行
    std::vector<std::size_t> cold;
    std::vector<std::vector<std::size_t>> tables(round.size());
    std::map<std::vector<std::size_t>, std::vector<std::size_t>> groups;
    std::vector<QueuedRequest> retry;
    {
        std::lock_guard<std::mutex> lock(streamMutex);
        PrefixCache& cache = ensurePagePool(round.front().request.variables.size());
        for (std::size_t i = 0; i < round.size(); ++i) {
            if (!round[i].streamingGrid) {
                cold.push_back(i);
                continue;
            }
            Tensor prefixContext;
            std::vector<std::uint8_t> prefixObserved;
            std::vector<double> coordinates;
            const std::size_t fullPatches = buildPrefixInput(
                round[i],
                prefixContext,
                prefixObserved,
                coordinates
            );
            const std::size_t hits = scanPrefix(
                cache,
                prefixContext,
                prefixObserved,
                fullPatches,
                true,
                tables[i],
                coordinates
            );
            if (hits == 0) {
                cold.push_back(i);
                continue;
            }
            groups[tables[i]].push_back(i);
        }
        for (auto& entry : groups) {
            const std::vector<std::size_t>& members = entry.second;
            if (members.size() < 2) {
                // 单条命中沿用逐条流式路径：先归还这里取的引用，runStreamed会重新命中
                releaseTable(cache.pool, tables[members.front()]);
                retry.push_back(std::move(round[members.front()]));
                continue;
            }
            std::vector<QueuedRequest> groupMembers;
            std::vector<std::vector<std::size_t>> memberTables;
            groupMembers.reserve(members.size());
            memberTables.reserve(members.size());
            for (std::size_t index : members) {
                groupMembers.push_back(std::move(round[index]));
                memberTables.push_back(std::move(tables[index]));
            }
            bool done = false;
            try {
                executeShared(groupMembers, memberTables);
                done = true;
            } catch (...) {
                done = false;
            }
            if (!done) {
                // 引擎级失败只影响本组：归还引用后逐条执行，单条失败仍只影响该条
                for (std::vector<std::size_t>& table : memberTables) {
                    releaseTable(cache.pool, table);
                }
                for (QueuedRequest& queued : groupMembers) {
                    retry.push_back(std::move(queued));
                }
            }
        }
    }
    for (QueuedRequest& queued : retry) {
        executeOne(queued);
    }
    std::vector<QueuedRequest> batch;
    batch.reserve(cold.size());
    for (std::size_t index : cold) {
        batch.push_back(std::move(round[index]));
    }
    if (batch.empty()) {
        return;
    }
    if (batch.size() == 1) {
        executeOne(batch.front());
        return;
    }
    try {
        executeBatch(batch);
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        for (QueuedRequest& queued : batch) {
            try {
                queued.promise.set_exception(error);
            } catch (...) {
                // 已填写的future不再重复设置
            }
        }
    }
}

void Service::executeShared(
    std::vector<QueuedRequest>& group,
    std::vector<std::vector<std::size_t>>& tables
) {
    const std::size_t channels = group.front().request.variables.size();
    PrefixCache& cache = ensurePagePool(channels);
    const auto start = std::chrono::steady_clock::now();
    const std::vector<std::size_t>& sharedTable = tables.front();
    const std::size_t reusedPages = sharedTable.size();
    const std::size_t computedPages = 0;
    const double prefillSeconds = 0.0;
    // 只共享已命中的页，未入页的各通道尾部在本轮独立编码，不改写共享页
    const std::size_t rows = group.size();
    const std::size_t gridLength = group.front().gridLength;
    Tensor context({rows, channels, gridLength}, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (std::size_t i = 0; i < group[row].kept; ++i) {
                context.values[(row * channels + channel) * gridLength + i] = fp32Sample(
                    group[row].request.variables[channel],
                    group[row].begin + i,
                    "submit"
                );
            }
        }
    }
    std::vector<double> times;
    if (engine->continuousRope()) {
        for (const auto& queued : group) {
            const auto rowTimes = timeGrid(queued, gridLength);
            times.insert(times.end(), rowTimes.begin(), rowTimes.end());
        }
    }
    TimingStats timing;
    const Tensor quantiles = engine->decodeShared(
        context,
        group.front().steps,
        group.front().request,
        cache.pool,
        sharedTable,
        timing,
        times,
        group.front().interval
    );
    const double total = secondsSince(start);
    for (std::size_t row = 0; row < rows; ++row) {
        Timestral::Result result;
        try {
            result = makeResult(
                group[row],
                quantiles,
                row,
                prefillSeconds,
                timing.decodeSeconds,
                total
            );
        } catch (...) {
            // 结果组装失败只影响该条：归还它尚未转交视图的页引用
            releaseTable(cache.pool, tables[row]);
            try {
                group[row].promise.set_exception(std::current_exception());
            } catch (...) {
                // 已填写的future不再重复设置
            }
            continue;
        }
        result.reusedPages = reusedPages;
        result.computedPages = computedPages;
        // 每条请求只接收本次命中的页表引用
        replaceView(cache, group[row].request.streamId, tables[row]);
        if (group[row].anonymous) {
            // 匿名请求的视图只服务本次执行：完成后归还引用，页键留给LRU回收
            releaseView(cache, group[row].request.streamId);
        }
        try {
            group[row].promise.set_value(std::move(result));
        } catch (...) {
            // 已填写的future不再重复设置
        }
    }
}

std::size_t Service::buildPrefixInput(
    const QueuedRequest& queued,
    Tensor& prefixContext,
    std::vector<std::uint8_t>& prefixObserved,
    std::vector<double>& coordinates
) const {
    const std::size_t patch = config.patchSize;
    const std::size_t fullPatches = queued.kept / patch;
    const std::size_t prefixLength = fullPatches * patch;
    const std::size_t channels = queued.request.variables.size();
    prefixContext = Tensor({channels, prefixLength}, std::numeric_limits<float>::quiet_NaN());
    prefixObserved.assign(prefixContext.size(), 0);
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t i = 0; i < prefixLength; ++i) {
            const float value = fp32Sample(queued.request.variables[channel], queued.begin + i, "submit");
            const std::size_t index = channel * prefixLength + i;
            prefixContext.values[index] = value;
            prefixObserved[index] = std::isfinite(value) ? 1 : 0;
        }
    }
    coordinates.clear();
    if (engine->continuousRope()) {
        coordinates = rotaryCoordinates(
            timeGrid(queued, prefixLength),
            1,
            channels,
            prefixLength,
            0,
            patch,
            queued.interval,
            engine->ropeReferenceInterval()
        );
    }
    return fullPatches;
}

void Service::executeOne(QueuedRequest& queued) {
    const auto start = std::chrono::steady_clock::now();
    try {
        Timestral::Result result;
        if (!queued.observations.values.empty()) {
            queued.observations.shape.insert(queued.observations.shape.begin(), 1);
            auto results = predictConditionedBatch(
                *engine,
                {std::cref(queued.request)},
                queued.observations,
                queued.observedEnds,
                queued.interval,
                {queued.gridOrigin}
            );
            result = std::move(results.front());
            std::lock_guard<std::mutex> lock(streamMutex);
            for (auto& entry : caches) {
                releaseView(entry.second, queued.request.streamId);
            }
        } else if (queued.streamingGrid) {
            std::lock_guard<std::mutex> lock(streamMutex);
            result = runStreamed(queued, start);
        } else {
            result = runFull(queued, start);
        }
        queued.promise.set_value(std::move(result));
    } catch (...) {
        queued.promise.set_exception(std::current_exception());
    }
}

void Service::executeBatch(std::vector<QueuedRequest>& round) {
    const auto start = std::chrono::steady_clock::now();
    const std::size_t batch = round.size();
    if (!round.front().observations.values.empty()) {
        const std::size_t channels = round.front().observations.shape[0];
        const std::size_t time = round.front().observations.shape[1];
        Tensor observations({batch, channels, time});
        std::vector<std::reference_wrapper<const Timestral::Request>> requests;
        std::vector<double> origins;
        for (std::size_t b = 0; b < batch; ++b) {
            requests.emplace_back(round[b].request);
            origins.push_back(round[b].gridOrigin);
            std::copy(
                round[b].observations.values.begin(),
                round[b].observations.values.end(),
                observations.values.begin() + b * channels * time
            );
        }
        auto results = predictConditionedBatch(
            *engine,
            requests,
            observations,
            round.front().observedEnds,
            round.front().interval,
            origins
        );
        {
            std::lock_guard<std::mutex> lock(streamMutex);
            for (const auto& queued : round) {
                for (auto& entry : caches) {
                    releaseView(entry.second, queued.request.streamId);
                }
            }
        }
        for (std::size_t b = 0; b < batch; ++b) {
            round[b].promise.set_value(std::move(results[b]));
        }
        return;
    }
    const std::size_t kept = round.front().kept;
    const std::size_t gridLength = round.front().gridLength;
    const std::size_t channels = round.front().request.variables.size();
    Tensor context({batch, channels, gridLength}, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t b = 0; b < batch; ++b) {
        const QueuedRequest& queued = round[b];
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (std::size_t i = 0; i < kept; ++i) {
                context.values[(b * channels + channel) * gridLength + i] = fp32Sample(
                    queued.request.variables[channel],
                    queued.begin + i,
                    "submit"
                );
            }
        }
    }
    std::vector<double> times;
    if (engine->continuousRope()) {
        for (const auto& queued : round) {
            const auto rowTimes = timeGrid(queued, gridLength);
            times.insert(times.end(), rowTimes.begin(), rowTimes.end());
        }
    }
    const Tensor noCovariates;
    const Tensor quantiles = engine->predict(
        context,
        round.front().steps,
        round.front().request,
        noCovariates,
        noCovariates,
        times,
        round.front().interval
    );
    const double elapsed = secondsSince(start);
    for (std::size_t b = 0; b < batch; ++b) {
        try {
            round[b].promise.set_value(makeResult(
                round[b],
                quantiles,
                b,
                0.0,
                elapsed,
                elapsed
            ));
        } catch (...) {
            round[b].promise.set_exception(std::current_exception());
        }
    }
}

Service::PrefixCache& Service::ensurePagePool(std::size_t channels) {
    const auto found = caches.find(channels);
    if (found != caches.end()) {
        return found->second;
    }
    PrefixCache cache;
    FnvHash fingerprint(engine->pageFingerprint());
    fingerprint.addUnsigned(channels);
    cache.pool = PagePool(engine->pageGeometry(channels), fingerprint.value(), pageCapacity);
    return caches.emplace(channels, std::move(cache)).first->second;
}

std::size_t Service::scanPrefix(
    PrefixCache& cache,
    const Tensor& prefixContext,
    const std::vector<std::uint8_t>& prefixObserved,
    std::size_t fullPatches,
    bool acquire,
    std::vector<std::size_t>& table,
    const std::vector<double>& coordinates
) {
    const std::size_t patch = config.patchSize;
    const std::size_t prefixLength = fullPatches * patch;
    const std::size_t channels = prefixContext.shape[0];
    const std::vector<std::size_t> groups(channels, 0);
    const std::map<std::size_t, std::size_t> first = firstObservedPatches(
        prefixObserved,
        groups,
        channels,
        prefixLength,
        patch
    );
    std::vector<std::uint8_t> keyValid;
    buildKeyValid(
        prefixObserved,
        channels,
        prefixLength,
        patch,
        config.attendMissingPatches,
        keyValid
    );
    std::size_t count = 0;
    std::uint64_t parentKey = 0;
    for (std::size_t p = 0; p < fullPatches; ++p) {
        std::vector<std::uint8_t> maskBits(channels, 0);
        for (std::size_t channel = 0; channel < channels; ++channel) {
            maskBits[channel] = keyValid[channel * fullPatches + p];
        }
        std::vector<double> positions;
        buildPositions(channels, 1, p, fullPatches, false, groups, first, positions, coordinates);
        std::uint64_t key = 0;
        std::uint64_t inputFingerprint = 0;
        pageKeys(
            prefixContext,
            prefixObserved,
            channels,
            prefixLength,
            p,
            patch,
            maskBits,
            positions,
            parentKey,
            cache.pool.fingerprint(),
            key,
            inputFingerprint
        );
        const auto node = cache.nodes.find(key);
        if (node == cache.nodes.end()) {
            break;
        }
        if (acquire) {
            const std::size_t index = cache.pool.find(key, parentKey, inputFingerprint);
            if (index == cache.pool.capacity()) {
                // 节点仍在但页已被淘汰或改写：摘除失效子树后重新编码
                pruneSubtree(cache, key, true);
                break;
            }
            node->second.page = index;
            table.push_back(index);
        } else {
            const std::size_t index = cache.pool.peek(key, parentKey, inputFingerprint);
            if (index == cache.pool.capacity() || index != node->second.page) {
                break;
            }
        }
        ++count;
        parentKey = key;
    }
    return count;
}

void Service::registerNode(
    PrefixCache& cache,
    std::uint64_t key,
    std::uint64_t parentKey,
    std::uint64_t inputFingerprint,
    std::size_t page
) {
    if (key == 0) {
        return;
    }
    const auto found = cache.nodes.find(key);
    if (found != cache.nodes.end()) {
        if (found->second.parentKey == parentKey
            && found->second.inputFingerprint == inputFingerprint) {
            found->second.page = page;
        }
        return;
    }
    cache.nodes.emplace(key, PrefixNode{page, parentKey, inputFingerprint});
    cache.children[parentKey].push_back(key);
}

std::size_t Service::pruneSubtree(PrefixCache& cache, std::uint64_t key, bool discardPages) {
    const auto root = cache.nodes.find(key);
    if (root == cache.nodes.end()) {
        return 0;
    }
    const std::uint64_t parentKey = root->second.parentKey;
    std::vector<std::uint64_t> pending(1, key);
    std::size_t removed = 0;
    while (!pending.empty()) {
        const std::uint64_t current = pending.back();
        pending.pop_back();
        const auto node = cache.nodes.find(current);
        if (node == cache.nodes.end()) {
            continue;
        }
        const std::size_t page = node->second.page;
        const auto child = cache.children.find(current);
        if (child != cache.children.end()) {
            for (std::uint64_t next : child->second) {
                pending.push_back(next);
            }
            cache.children.erase(child);
        }
        if (discardPages && page < cache.pool.capacity()) {
            const CachePage& entry = cache.pool.page(page);
            if (entry.occupied && entry.references == 0 && entry.key == current) {
                cache.pool.discard(page);
            }
        }
        cache.nodes.erase(node);
        ++removed;
    }
    const auto siblings = cache.children.find(parentKey);
    if (siblings != cache.children.end()) {
        std::vector<std::uint64_t>& list = siblings->second;
        list.erase(std::remove(list.begin(), list.end(), key), list.end());
        if (list.empty()) {
            cache.children.erase(siblings);
        }
    }
    return removed;
}

bool Service::ensureRoom(PrefixCache& cache, std::size_t needed) {
    // 依据vLLM的block evictor：只淘汰引用计数为0的LRU页，在跑请求引用的页不抢占
    // 淘汰顺序由调度器决定：PagePool只报告LRU候选，Service负责摘键与释放
    while (needed > cache.pool.freePages()) {
        const std::size_t victim = cache.pool.leastRecentlyUsed();
        if (victim == cache.pool.capacity()) {
            return false;
        }
        const std::uint64_t key = cache.pool.page(victim).key;
        const std::size_t freeBefore = cache.pool.freePages();
        const auto node = cache.nodes.find(key);
        if (node != cache.nodes.end() && node->second.page == victim) {
            pruneSubtree(cache, key, true);
        }
        if (cache.pool.freePages() == freeBefore) {
            // 未登记的页没有树节点可摘，直接释放该候选
            cache.pool.discard(victim);
        }
    }
    return true;
}

void Service::releaseView(PrefixCache& cache, const std::string& streamId) {
    const auto found = cache.views.find(streamId);
    if (found == cache.views.end()) {
        return;
    }
    for (auto entry = found->second.rbegin(); entry != found->second.rend(); ++entry) {
        cache.pool.release(*entry);
    }
    cache.views.erase(found);
}

void Service::replaceView(
    PrefixCache& cache,
    const std::string& streamId,
    const std::vector<std::size_t>& table
) {
    for (auto& entry : caches) {
        releaseView(entry.second, streamId);
    }
    if (!table.empty()) {
        cache.views[streamId] = table;
    }
}

void Service::dropView(PrefixCache& cache, const std::string& streamId) {
    const auto found = cache.views.find(streamId);
    if (found == cache.views.end()) {
        return;
    }
    const std::vector<std::size_t> table = std::move(found->second);
    cache.views.erase(found);
    // 先归还子页再归还父页，父页引用归零时整条子树都已无人引用
    for (auto entry = table.rbegin(); entry != table.rend(); ++entry) {
        const std::size_t index = *entry;
        cache.pool.release(index);
        const CachePage& page = cache.pool.page(index);
        if (page.occupied && page.references == 0 && page.key != 0) {
            // 手动丢弃：摘键并释放零引用页，避免appendPages按旧键重新命中
            pruneSubtree(cache, page.key, true);
        }
    }
}

void Service::releaseTable(PagePool& pool, std::vector<std::size_t>& table) {
    for (std::size_t index : table) {
        pool.release(index);
    }
    table.clear();
}

Timestral::Result Service::runFull(
    const QueuedRequest& queued,
    std::chrono::steady_clock::time_point start
) const {
    const std::size_t channels = queued.request.variables.size();
    Tensor context({1, channels, queued.kept}, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t i = 0; i < queued.kept; ++i) {
            context.values[channel * queued.kept + i] = fp32Sample(
                queued.request.variables[channel],
                queued.begin + i,
                "submit"
            );
        }
    }
    return runFullContext(queued, context, start);
}

Timestral::Result Service::runFullContext(
    const QueuedRequest& queued,
    const Tensor& context,
    std::chrono::steady_clock::time_point start
) const {
    const auto decodeStart = std::chrono::steady_clock::now();
    const Tensor noCovariates;
    const Tensor quantiles = engine->predict(
        context,
        queued.steps,
        queued.request,
        noCovariates,
        noCovariates,
        timeGrid(queued, context.shape.back()),
        queued.interval
    );
    const double decode = secondsSince(decodeStart);
    return makeResult(queued, quantiles, 0, 0.0, decode, secondsSince(start));
}

Timestral::Result Service::runStreamed(
    QueuedRequest& queued,
    std::chrono::steady_clock::time_point start
) {
    const std::size_t channels = queued.request.variables.size();
    PrefixCache& cache = ensurePagePool(channels);
    const std::size_t patch = config.patchSize;
    const std::size_t gridLength = roundPatch(queued.kept, patch);
    Tensor context({1, channels, gridLength}, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t i = 0; i < queued.kept; ++i) {
            context.values[channel * gridLength + i] = fp32Sample(
                queued.request.variables[channel],
                queued.begin + i,
                "submit"
            );
        }
    }

    Tensor prefixContext;
    std::vector<std::uint8_t> prefixObserved;
    std::vector<double> coordinates;
    const std::size_t fullPatches = buildPrefixInput(queued, prefixContext, prefixObserved, coordinates);

    // 内容寻址（vLLM prefix caching）：同内容不同来源命中同一批页，引用计数加一，
    // 页键由父页键与当前patch输入指纹链式得到，不按streamId隔离缓存
    std::vector<std::size_t> table;
    table.reserve(fullPatches);
    std::size_t reusedPages = scanPrefix(
        cache,
        prefixContext,
        prefixObserved,
        fullPatches,
        true,
        table,
        coordinates
    );
    std::size_t computedPages = 0;
    TimingStats timing;
    if (table.size() < fullPatches) {
        if (!ensureRoom(cache, fullPatches - table.size())) {
            // 页池被在跑请求的引用占满时不做抢占，退回全量重算
            releaseTable(cache.pool, table);
            return runFullContext(queued, context, start);
        }
        CacheAppendInput input;
        input.context = prefixContext;
        input.observed = prefixObserved;
        input.coordinates = coordinates;
        input.groupIds.assign(channels, 0);
        const std::size_t before = table.size();
        const std::size_t hitsBefore = cache.pool.hitCount();
        const auto prefillStart = std::chrono::steady_clock::now();
        try {
            engine->appendPages(input, cache.pool, table);
        } catch (...) {
            releaseTable(cache.pool, table);
            throw;
        }
        timing.prefillSeconds = secondsSince(prefillStart);
        for (std::size_t i = before; i < table.size(); ++i) {
            const CachePage& page = cache.pool.page(table[i]);
            registerNode(cache, page.key, page.parentKey, page.inputFingerprint, table[i]);
        }
        const std::size_t appended = table.size() - before;
        const std::size_t poolHits = cache.pool.hitCount() - hitsBefore;
        reusedPages += poolHits;
        computedPages = appended - poolHits;
    }

    Tensor quantiles;
    try {
        quantiles = engine->predictStreamed(
            context,
            queued.steps,
            queued.request,
            cache.pool,
            table,
            timing,
            timeGrid(queued, gridLength),
            queued.interval
        );
    } catch (...) {
        releaseTable(cache.pool, table);
        throw;
    }
    if (table.size() != fullPatches) {
        // 引擎在上下文截断时已自行释放页表，这里只同步本地视图
        table.clear();
    }
    replaceView(cache, queued.request.streamId, table);
    if (queued.anonymous) {
        // 匿名请求的视图只服务本次执行：完成后归还引用，页键留给LRU回收
        releaseView(cache, queued.request.streamId);
    }

    Timestral::Result result = makeResult(
        queued,
        quantiles,
        0,
        timing.prefillSeconds,
        timing.decodeSeconds,
        secondsSince(start)
    );
    result.reusedPages = reusedPages;
    result.computedPages = computedPages;
    return result;
}

Timestral::Result Service::makeResult(
    const QueuedRequest& queued,
    const Tensor& quantiles,
    std::size_t batchIndex,
    double prefillSeconds,
    double decodeSeconds,
    double totalSeconds
) const {
    Timestral::Result result;
    result.channels = queued.request.variables.size();
    result.sampleInterval = queued.interval;
    result.origin = queued.request.variables.front().back().time;
    result.levels = quantileLevels(config);
    result.quantiles = quantileRow(quantiles, batchIndex, queued.steps, result.levels.size());
    result.observed.assign(elementCount({result.channels, queued.steps}), 0);
    result.prefillSeconds = prefillSeconds;
    result.decodeSeconds = decodeSeconds;
    result.totalSeconds = totalSeconds;
    result.reusedPages = 0;
    result.computedPages = 0;
    return result;
}

} // namespace timestral
