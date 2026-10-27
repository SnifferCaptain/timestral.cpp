/***************
 * @file engine.cpp
 * @brief 协变量校验、分块自回归及advancing frame预测的张量级实现
 * 移植自Mistral AI Timestral，Apache-2.0
 * 固定基线：a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff，
 * forecaster.py、scaler.py及covariate_encoding.py
 ***************/

#include "timestral/utils/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "timestral/utils/backend.hpp"
#include "timestral/utils/forecast.hpp"
#include "timestral/utils/postprocess.hpp"
#include "timestral/utils/safetensors.hpp"
#include "timestral/utils/stats.hpp"
#include "timestral/utils/tensor.hpp"
#include "timestral/utils/time.hpp"

namespace timestral {

// 固定的分位数后处理阈值：输出范围按上下文尺度的1e4倍截断，自回归反馈丢弃超过1e30的
// 极端中位数。两者属于模型后处理语义，不随调用方变化，公开选项不再暴露
static constexpr float quantileRealCapK = 1e4f;
static constexpr float extremeValueThreshold = 1e30f;

// 计时统一用steady_clock，返回秒，统计对象由调用者持有，不使用全局状态
static double secondsSince(const std::chrono::steady_clock::time_point& start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

static std::size_t checkedAdd(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::invalid_argument("forecast dimension overflow");
    }
    return a + b;
}

static std::size_t roundPatch(std::size_t length, std::size_t patch) {
    const std::size_t extra = length % patch == 0 ? 0 : patch - length % patch;
    return checkedAdd(length, extra);
}

// 协变量只作为模型行输入，NaN换成0，±inf换成同号最大有限值
static float finiteCovariate(float value) {
    if (std::isnan(value)) {
        return 0.0f;
    }
    if (std::isinf(value)) {
        return std::copysign(std::numeric_limits<float>::max(), value);
    }
    return value;
}

// 协变量行是(B,C,L)张量，空shape表示不提供，长度按上下文或上下文加horizon
static void validateCovariateTensor(
    const Tensor& values,
    std::size_t batch,
    std::size_t length,
    const ModelConfig& config,
    const std::string& name
) {
    if (values.size() == 0) {
        return;
    }
    if (!config.covariateEnabled) {
        throw std::invalid_argument(name + ": checkpoint does not support covariates");
    }
    values.validate();
    if (values.shape.size() != 3 || values.shape[0] != batch || values.shape[1] == 0
        || values.shape[2] != length) {
        throw std::invalid_argument(
            name + ": expected (batch, channels, " + std::to_string(length) + ")"
        );
    }
}

// advancing refine frame：按Hillis-Steele前缀合并重算每个refine步的Welford状态，
// 把归一化中位数折叠回目标空间，再按该块尺度反归一化其余分位点
static void refine(
    Tensor& quantiles,
    const Tensor& context,
    const std::vector<std::uint8_t>& mask,
    const std::vector<int>& roles,
    const Model& model
) {
    const auto& config = model.config();
    const std::size_t patch = config.patchSize;
    const std::size_t horizon = quantiles.shape[1];
    const std::size_t patches = horizon / patch;
    if (patches <= 1) {
        return;
    }
    const auto frames = model.scale(context, mask);
    const std::size_t time = context.shape[1];
    const std::size_t levels = quantiles.shape[2];
    const std::size_t median = config.medianIndex();
    const double clip = std::sqrt(static_cast<double>(std::numeric_limits<float>::max())) / 1e3;
    for (std::size_t r = 0; r < context.shape[0]; ++r) {
        if (roles[r] != 0) {
            continue;
        }
        // Hillis-Steele每轮读取上一轮快照，保持上游两两合并的结合顺序
        std::vector<Welford> states(time / patch + patches);
        for (std::size_t p = 0; p < time / patch; ++p) {
            auto& state = states[p];
            for (std::size_t j = 0; j < patch; ++j) {
                const std::size_t index = r * time + p * patch + j;
                state.count += mask[index] != 0 ? 1.0 : 0.0;
                state.mean += static_cast<double>(context.values[index]) * (mask[index] != 0 ? 1.0 : 0.0);
            }
            state.mean /= std::max(state.count, 1.0);
            for (std::size_t j = 0; j < patch; ++j) {
                const std::size_t index = r * time + p * patch + j;
                const double delta = static_cast<double>(context.values[index]) - state.mean;
                state.m2 += (mask[index] != 0 ? 1.0 : 0.0) * delta * delta;
            }
        }
        for (std::size_t distance = 1; distance < states.size();) {
            const auto previous = states;
            for (std::size_t p = 0; p < states.size(); ++p) {
                states[p] = combine(p >= distance ? previous[p - distance] : Welford{}, previous[p]);
            }
            if (distance > states.size() / 2) {
                break;
            }
            distance *= 2;
        }
        Welford state = states[time / patch - 1];
        const double offset = state.mean;
        const float loc0 = frames.loc.values[(r + 1) * time - 1];
        const float scale0 = frames.scale.values[(r + 1) * time - 1];
        std::vector<double> folded(patch, 0.0);
        for (std::size_t p = 0; p < patches; ++p) {
            const double scale = std::max(
                std::sqrt(std::max(state.m2 / std::max(state.count - 1.0, 1.0), 0.0)),
                static_cast<double>(config.minimumScale)
            );
            const float outputLoc = p == 0 ? loc0 : static_cast<float>(state.mean);
            const float outputScale = p == 0 ? scale0 : static_cast<float>(scale);
            Welford block;
            block.count = static_cast<double>(patch);
            for (std::size_t j = 0; j < patch; ++j) {
                const std::size_t base = (r * horizon + p * patch + j) * levels;
                const double u = (static_cast<double>(quantiles.values[base + median]) - static_cast<double>(loc0))
                    / static_cast<double>(scale0);
                folded[j] = foldAdvancingFrame(u, scale, state.mean, offset, clip);
                block.mean += folded[j];
                for (std::size_t q = 0; q < levels; ++q) {
                    const float normalized = (quantiles.values[base + q] - loc0) / scale0;
                    quantiles.values[base + q] = normalized * outputScale + outputLoc;
                }
                clampInfiniteRange(quantiles.values, base, levels);
            }
            block.mean /= static_cast<double>(patch);
            for (double value : folded) {
                const double delta = value - block.mean;
                block.m2 += delta * delta;
            }
            // 首patch只强制输出anchor帧，内部统计仍使用原始advancing帧
            state = combine(state, block);
        }
    }
}

// 单次分块自回归，sign为-1时用于对称平均的镜像输入
static Tensor rawForecast(
    const Model& model,
    std::size_t maxContext,
    std::size_t patchBatch,
    const Tensor& filled,
    const std::vector<std::uint8_t>& targetMask,
    std::size_t batch,
    std::size_t variates,
    std::size_t time,
    std::size_t predictionLength,
    const Tensor& past,
    const Tensor& known,
    float sign,
    PagePool& pool,
    std::vector<std::size_t>& table,
    TimingStats& timing,
    bool incremental,
    bool sharedPages,
    const std::vector<double>& times,
    double interval,
    double referenceInterval
) {
    const auto& config = model.config();
    const std::size_t patch = config.patchSize;
    const std::size_t levels = config.quantileKnots.size();
    const std::size_t pastCount = past.size() == 0 ? 0 : past.shape[1];
    const std::size_t knownCount = known.size() == 0 ? 0 : known.shape[1];
    const std::size_t channels = checkedAdd(checkedAdd(variates, pastCount), knownCount);
    const std::size_t rowCount = elementCount({batch, channels});
    const std::size_t kept = std::min(time, maxContext);
    const std::size_t initialTime = roundPatch(kept, patch);
    const std::size_t padding = initialTime - kept;
    const std::size_t horizon = roundPatch(predictionLength, patch);
    Tensor context({rowCount, initialTime});
    std::vector<std::uint8_t> mask(context.size(), 0);
    Tensor future({rowCount, horizon});
    std::vector<std::uint8_t> futureMask(future.size(), 0);
    std::vector<int> roles(rowCount, 0);
    std::vector<std::size_t> groups(rowCount, 0);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t c = 0; c < channels; ++c) {
            const std::size_t r = b * channels + c;
            groups[r] = b;
            roles[r] = c < variates ? 0 : (c < variates + pastCount ? 1 : 2);
            for (std::size_t j = 0; j < kept; ++j) {
                const std::size_t sourceTime = time - kept + j;
                const std::size_t destination = r * initialTime + padding + j;
                if (c < variates) {
                    const std::size_t source = (b * variates + c) * time + sourceTime;
                    context.values[destination] = sign * filled.values[source];
                    mask[destination] = targetMask[source];
                } else {
                    const Tensor& cov = roles[r] == 1 ? past : known;
                    const std::size_t covChannel = roles[r] == 1 ? c - variates : c - variates - pastCount;
                    const float value = cov.values[(b * cov.shape[1] + covChannel) * cov.shape[2] + sourceTime];
                    context.values[destination] = finiteCovariate(sign * value);
                    mask[destination] = !std::isnan(value);
                }
            }
            if (roles[r] == 2) {
                const std::size_t covChannel = c - variates - pastCount;
                for (std::size_t j = 0; j < predictionLength; ++j) {
                    const std::size_t source = (b * knownCount + covChannel) * (time + predictionLength) + time + j;
                    const float value = known.values[source];
                    future.values[r * horizon + j] = finiteCovariate(sign * value);
                    futureMask[r * horizon + j] = !std::isnan(value);
                }
            }
        }
    }
    std::vector<double> contextTimes;
    if (!times.empty()) {
        contextTimes.resize(elementCount({batch, initialTime}));
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t i = 0; i < initialTime; ++i) {
                contextTimes[b * initialTime + i] = i < padding
                    ? times[b * time + time - kept] - static_cast<double>(padding - i) * interval
                    : times[b * time + time - kept + i - padding];
            }
        }
    }
    const Tensor initialContext = context;
    const auto initialMask = mask;
    Tensor result({batch, variates, predictionLength, levels});
    bool useCache = incremental;
    // 生成分支仅在本次预测内缓存，绝不把回灌值加入服务的真实观测页表
    const bool cacheFeedback = incremental && !sharedPages && !table.empty()
        && pool.geometry().rowCount == rowCount && table.size() <= initialTime / patch
        && horizon / patch > patchBatch && initialTime <= maxContext
        && horizon <= maxContext - initialTime;
    PagePool generatedPool;
    std::vector<std::size_t> generatedTable;
    if (cacheFeedback) {
        generatedPool = PagePool(pool.geometry(), pool.fingerprint(), (initialTime + horizon) / patch);
        for (const auto page : table) {
            const auto target = generatedPool.acquire();
            generatedPool.page(target) = pool.page(page);
            generatedPool.page(target).references = 1;
            generatedTable.push_back(target);
        }
    }
    PagePool& decodePool = cacheFeedback ? generatedPool : pool;
    auto& decodeTable = cacheFeedback ? generatedTable : table;
    std::size_t offset = 0;
    while (offset < horizon) {
        const std::size_t blockPatches = std::min(patchBatch, (horizon - offset) / patch);
        const std::size_t blockLength = blockPatches * patch;
        Tensor blockFuture;
        std::vector<std::uint8_t> blockFutureMask;
        if (knownCount > 0) {
            blockFuture = Tensor({rowCount, blockLength});
            blockFutureMask.resize(blockFuture.size());
            for (std::size_t r = 0; r < rowCount; ++r) {
                std::copy_n(
                    future.values.begin() + r * horizon + offset,
                    blockLength,
                    blockFuture.values.begin() + r * blockLength
                );
                std::copy_n(
                    futureMask.begin() + r * horizon + offset,
                    blockLength,
                    blockFutureMask.begin() + r * blockLength
                );
            }
        }
        ForwardInput blockInput;
        blockInput.context = context;
        blockInput.observed = mask;
        blockInput.groupIds = groups;
        blockInput.horizonPatches = blockPatches;
        blockInput.roles = roles;
        blockInput.future = blockFuture;
        blockInput.hasFuture = knownCount > 0;
        if (!contextTimes.empty()) {
            blockInput.coordinates = rotaryCoordinates(
                contextTimes,
                batch,
                channels,
                context.shape[1],
                blockPatches,
                patch,
                interval,
                referenceInterval
            );
        }
        blockInput.futureObserved = blockFutureMask;
        Tensor quantiles;
        if (useCache) {
            // 页表覆盖整patch前缀，尾部未入页的patch与horizon一起按页解码
            CacheAppendInput cacheInput;
            cacheInput.context = context;
            cacheInput.observed = mask;
            cacheInput.groupIds = groups;
            cacheInput.horizonPatches = blockPatches;
            cacheInput.coordinates = blockInput.coordinates;
            const auto start = std::chrono::steady_clock::now();
            if (cacheFeedback && offset > 0) {
                model.appendPages(cacheInput, decodePool, decodeTable);
            }
            quantiles = sharedPages
                ? model.decodePagesShared(cacheInput, decodePool, decodeTable)
                : model.decodePages(cacheInput, decodePool, decodeTable);
            timing.decodeSeconds += secondsSince(start);
        } else {
            const auto start = std::chrono::steady_clock::now();
            quantiles = model.predictBlock(blockInput);
            timing.decodeSeconds += secondsSince(start);
        }
        refine(quantiles, context, mask, roles, model);
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t c = 0; c < variates; ++c) {
                for (std::size_t j = 0; j < blockLength && offset + j < predictionLength; ++j) {
                    const std::size_t source = ((b * channels + c) * blockLength + j) * levels;
                    const std::size_t destination = ((b * variates + c) * predictionLength + offset + j) * levels;
                    std::copy_n(quantiles.values.begin() + source, levels, result.values.begin() + destination);
                }
            }
        }
        offset += blockLength;
        if (offset >= horizon) {
            break;
        }
        const std::size_t oldTime = context.shape[1];
        const std::size_t extended = checkedAdd(oldTime, blockLength);
        const std::size_t newTime = extended > maxContext
            ? std::max(patch, maxContext / patch * patch)
            : extended;
        if (useCache && newTime != extended) {
            // 上下文被maxContext截断后网格左移，页表失效：释放页并让剩余块全量重算
            // 共享解码的页表引用归调用方所有，这里只丢弃本地副本，不归还引用
            if (!sharedPages) {
                for (std::size_t index : table) {
                    pool.release(index);
                }
            }
            table.clear();
            useCache = false;
        }
        Tensor next({rowCount, newTime});
        std::vector<std::uint8_t> nextMask(next.size(), 0);
        for (std::size_t r = 0; r < rowCount; ++r) {
            for (std::size_t j = 0; j < newTime; ++j) {
                const std::size_t source = extended - newTime + j;
                const std::size_t destination = r * newTime + j;
                if (source < oldTime) {
                    next.values[destination] = context.values[r * oldTime + source];
                    nextMask[destination] = mask[r * oldTime + source];
                } else if (roles[r] == 2) {
                    next.values[destination] = blockFuture.values[r * blockLength + source - oldTime];
                    nextMask[destination] = blockFutureMask[r * blockLength + source - oldTime];
                } else if (roles[r] == 0) {
                    const std::size_t medianIndex = (r * blockLength + source - oldTime) * levels + config.medianIndex();
                    const float median = quantiles.values[medianIndex];
                    const bool valid = std::isfinite(median) && std::abs(median) < extremeValueThreshold;
                    next.values[destination] = valid ? median : 0.0f;
                    nextMask[destination] = 1;
                }
            }
        }
        if (!contextTimes.empty()) {
            std::vector<double> nextTimes(elementCount({batch, newTime}));
            for (std::size_t b = 0; b < batch; ++b) {
                for (std::size_t i = 0; i < newTime; ++i) {
                    const std::size_t source = extended - newTime + i;
                    nextTimes[b * newTime + i] = source < oldTime
                        ? contextTimes[b * oldTime + source]
                        : contextTimes[(b + 1) * oldTime - 1] + static_cast<double>(source - oldTime + 1) * interval;
                }
            }
            contextTimes = std::move(nextTimes);
        }
        context = std::move(next);
        mask = std::move(nextMask);
    }
    // 每条轨道各自cap，范围只取初始目标窗，绝不改变AR反馈
    const auto frames = model.scale(initialContext, initialMask);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t c = 0; c < variates; ++c) {
            const std::size_t r = b * channels + c;
            float low = std::numeric_limits<float>::infinity();
            float high = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < initialTime; ++j) {
                if (initialMask[r * initialTime + j]) {
                    low = std::min(low, initialContext.values[r * initialTime + j]);
                    high = std::max(high, initialContext.values[r * initialTime + j]);
                }
            }
            if (!std::isfinite(low)) {
                low = 0.0f;
                high = 0.0f;
            }
            capQuantileRange(
                result.values,
                (b * variates + c) * predictionLength * levels,
                predictionLength * levels,
                low,
                high,
                frames.scale.values[(r + 1) * initialTime - 1] * quantileRealCapK
            );
        }
    }
    for (float value : result.values) {
        if (std::isnan(value)) {
            throw std::runtime_error("prediction produced NaN values; input is degenerate");
        }
    }
    return result;
}

Engine::Engine(std::shared_ptr<const Model> model, const Timestral::Settings& settings)
    : model(std::move(model)) {
    if (!this->model) {
        throw std::invalid_argument("engine requires a loaded model");
    }
    contextLimit = settings.maxContext == 0
        ? this->model->config().contextLength
        : settings.maxContext;
    if (settings.patchBatch == 0) {
        throw std::invalid_argument("engine patchBatch must be positive");
    }
    patchBatch = std::min(settings.patchBatch, this->model->config().blockSize);
    if (!std::isfinite(settings.ropeReferenceInterval) || settings.ropeReferenceInterval < 0.0) {
        throw std::invalid_argument("ropeReferenceInterval must be finite and nonnegative");
    }
    continuousTime = settings.continuousRope;
    referenceInterval = settings.ropeReferenceInterval;
}

const ModelConfig& Engine::config() const {
    return model->config();
}

std::size_t Engine::maxContext() const {
    return contextLimit;
}

bool Engine::continuousRope() const {
    return continuousTime;
}

double Engine::ropeReferenceInterval() const {
    return referenceInterval;
}

bool Engine::supportsIncremental() const {
    return model->supportsIncremental();
}

PageGeometry Engine::pageGeometry(std::size_t rowCount) const {
    return model->pageGeometry(rowCount);
}

std::uint64_t Engine::pageFingerprint() const {
    if (!continuousTime) {
        return model->pageFingerprint();
    }
    FnvHash hash(model->pageFingerprint());
    hash.addString("continuous-time-rope-v1");
    hash.addPosition(referenceInterval);
    return hash.value();
}

void Engine::appendPages(
    const CacheAppendInput& input,
    PagePool& pool,
    std::vector<std::size_t>& table
) const {
    model->appendPages(input, pool, table);
}

Tensor Engine::decodePages(
    const CacheAppendInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table
) const {
    return model->decodePages(input, pool, table);
}

Tensor Engine::computePrediction(
    const Tensor& input,
    std::size_t predictionLength,
    const Timestral::Request& request,
    const Tensor& pastCovariates,
    const Tensor& knownCovariates,
    PagePool& pool,
    std::vector<std::size_t>& table,
    TimingStats& timing,
    bool incremental,
    bool sharedPages,
    const std::vector<double>& times,
    double interval
) const {
    input.validate();
    if (predictionLength == 0 || input.shape.empty() || input.shape.size() > 3) {
        throw std::invalid_argument("predict requires positive horizon and context rank 1, 2 or 3");
    }
    const std::size_t batch = input.shape.size() == 1 ? 1 : input.shape[0];
    const std::size_t variates = input.shape.size() == 3 ? input.shape[1] : 1;
    const std::size_t time = input.shape.back();
    if (batch == 0 || variates == 0 || time == 0) {
        throw std::invalid_argument("predict context dimensions must be nonzero");
    }
    if (continuousTime) {
        interval = interval == 0.0 ? request.sampleInterval : interval;
        if (times.size() != elementCount({batch, time}) || !std::isfinite(interval) || interval <= 0.0) {
            throw std::invalid_argument("continuous RoPE requires timestamps and a positive sampling interval");
        }
    }
    const std::vector<double> noTimes;
    const auto& inputTimes = continuousTime ? times : noTimes;
    const std::size_t totalLength = checkedAdd(time, predictionLength);
    std::vector<std::uint8_t> mask(input.size(), 0);
    Tensor filled({batch, variates, time});
    for (std::size_t r = 0; r < batch * variates; ++r) {
        float previous = 0.0f;
        for (std::size_t t = 0; t < time; ++t) {
            const std::size_t index = r * time + t;
            mask[index] = std::isfinite(input.values[index]) ? 1 : 0;
            if (mask[index] != 0) {
                previous = input.values[index];
            }
            filled.values[index] = previous;
        }
    }
    const std::vector<std::uint8_t> nonnegative = nonnegativeTrajectories(
        input.values,
        mask,
        batch * variates,
        time
    );
    validateCovariateTensor(pastCovariates, batch, time, model->config(), "pastCovariates");
    validateCovariateTensor(knownCovariates, batch, totalLength, model->config(), "knownCovariates");
    if (incremental && (pastCovariates.size() != 0 || knownCovariates.size() != 0)) {
        throw std::invalid_argument("predictStreamed requires a target-only context without covariates");
    }
    if (knownCovariates.size() != 0 && !model->config().supportsKnownCovariates()) {
        throw std::invalid_argument("checkpoint cannot condition on known-future covariates");
    }
    const std::size_t levels = model->config().quantileKnots.size();
    if (request.symmetricAveraging) {
        for (std::size_t q = 0; q < levels; ++q) {
            if (std::abs(model->config().quantileKnots[q] + model->config().quantileKnots[levels - q - 1] - 1.0f) > 1e-7f) {
                throw std::invalid_argument("symmetric averaging requires symmetric quantile levels");
            }
        }
    }
    Tensor quantiles = rawForecast(
        *model,
        contextLimit,
        patchBatch,
        filled,
        mask,
        batch,
        variates,
        time,
        predictionLength,
        pastCovariates,
        knownCovariates,
        1.0f,
        pool,
        table,
        timing,
        incremental,
        sharedPages,
        inputTimes,
        interval,
        referenceInterval
    );
    if (request.symmetricAveraging) {
        // 镜像pass把上下文取负，已缓存的页属于正号上下文，不能复用：用空页池全量重算
        PagePool mirroredPool;
        std::vector<std::size_t> mirroredTable;
        const Tensor mirrored = rawForecast(
            *model,
            contextLimit,
            patchBatch,
            filled,
            mask,
            batch,
            variates,
            time,
            predictionLength,
            pastCovariates,
            knownCovariates,
            -1.0f,
            mirroredPool,
            mirroredTable,
            timing,
            false,
            false,
            inputTimes,
            interval,
            referenceInterval
        );
        averageMirrored(quantiles.values, mirrored.values, levels);
    }
    if (request.clipNonnegative) {
        for (std::size_t r = 0; r < batch * variates; ++r) {
            if (nonnegative[r] != 0) {
                clampMinRange(
                    quantiles.values,
                    r * predictionLength * levels,
                    predictionLength * levels,
                    0.0f
                );
            }
        }
    }
    return quantiles;
}

Tensor Engine::predictParallel(
    const Tensor& context,
    std::size_t predictionLength,
    const Timestral::Request& request,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    TimingStats& timing
) const {
    context.validate();
    const auto& config = model->config();
    const std::size_t patch = config.patchSize;
    if (continuousTime || context.shape.size() != 3 || context.shape[0] != 1
        || context.shape.back() % patch != 0 || predictionLength == 0 || predictionLength % patch != 0
        || context.shape.back() > contextLimit || predictionLength > contextLimit - context.shape.back()
        || table.size() != context.shape.back() / patch || table.empty() || !model->supportsIncremental()) {
        throw std::invalid_argument("parallel forecast requires fully cached aligned target history within maxContext");
    }
    const auto start = std::chrono::steady_clock::now();
    const std::size_t channels = context.shape[1];
    const std::size_t time = context.shape[2];
    const std::size_t prefix = time / patch;
    const std::size_t framePatches = config.blockSize;
    const std::size_t frameLength = elementCount({patch, framePatches});
    if (predictionLength % frameLength != 0) {
        throw std::invalid_argument("parallel forecast requires complete native block_size frames");
    }
    const std::size_t frameCount = predictionLength / frameLength;
    const std::size_t levels = config.quantileKnots.size();
    Tensor history({channels, time});
    std::vector<std::uint8_t> observed(history.size(), 0);
    for (std::size_t row = 0; row < channels; ++row) {
        float previous = 0.0f;
        for (std::size_t t = 0; t < time; ++t) {
            const std::size_t index = row * time + t;
            if (std::isinf(context.values[index])) {
                throw std::invalid_argument("parallel forecast does not accept infinite observations");
            }
            observed[index] = std::isfinite(context.values[index]) ? 1 : 0;
            if (observed[index] != 0) {
                previous = context.values[index];
            }
            history.values[index] = previous;
        }
    }
    const auto run = [&](float sign, PagePool& pages, const std::vector<std::size_t>& prefixTable) {
        CacheAppendInput input;
        const std::size_t rows = elementCount({frameCount, channels});
        input.context = Tensor({rows, time});
        input.observed.resize(input.context.size());
        input.groupIds.resize(rows);
        input.coordinates.resize(elementCount({rows, prefix + framePatches}));
        input.horizonPatches = framePatches;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t channel = row % channels;
            const std::size_t frame = row / channels;
            input.groupIds[row] = frame;
            for (std::size_t t = 0; t < time; ++t) {
                input.context.values[row * time + t] = sign * history.values[channel * time + t];
                input.observed[row * time + t] = observed[channel * time + t];
            }
            for (std::size_t slot = 0; slot < prefix; ++slot) {
                input.coordinates[row * (prefix + framePatches) + slot] = static_cast<double>(slot);
            }
            // 帧内保留原生32patch因果query链，仅将整帧移动到对应未来位置
            for (std::size_t local = 0; local < framePatches; ++local) {
                input.coordinates[row * (prefix + framePatches) + prefix + local]
                    = static_cast<double>(prefix + frame * framePatches + local);
            }
        }
        Tensor decoded = model->decodePagesShared(input, pages, prefixTable);
        refine(decoded, input.context, input.observed, std::vector<int>(rows, 0), *model);
        Tensor output({1, channels, predictionLength, levels});
        for (std::size_t frame = 0; frame < frameCount; ++frame) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
                const std::size_t source = (frame * channels + channel) * frameLength * levels;
                const std::size_t target = (channel * predictionLength + frame * frameLength) * levels;
                std::copy_n(decoded.values.begin() + source, frameLength * levels, output.values.begin() + target);
            }
        }
        return output;
    };
    Tensor output = run(1.0f, pool, table);
    if (request.symmetricAveraging) {
        CacheAppendInput mirrored;
        mirrored.context = history;
        mirrored.observed = observed;
        mirrored.groupIds.assign(channels, 0);
        for (float& value : mirrored.context.values) {
            value = -value;
        }
        PagePool mirrorPool(model->pageGeometry(channels), model->pageFingerprint(), prefix);
        std::vector<std::size_t> mirrorTable;
        model->appendPages(mirrored, mirrorPool, mirrorTable);
        const Tensor mirror = run(-1.0f, mirrorPool, mirrorTable);
        averageMirrored(output.values, mirror.values, levels);
    }
    const auto frames = model->scale(history, observed);
    const auto nonnegative = nonnegativeTrajectories(history.values, observed, channels, time);
    for (std::size_t channel = 0; channel < channels; ++channel) {
        float low = std::numeric_limits<float>::infinity();
        float high = -std::numeric_limits<float>::infinity();
        for (std::size_t t = 0; t < time; ++t) {
            const auto index = channel * time + t;
            if (observed[index] != 0) {
                low = std::min(low, history.values[index]);
                high = std::max(high, history.values[index]);
            }
        }
        if (!std::isfinite(low)) {
            low = 0.0f;
            high = 0.0f;
        }
        capQuantileRange(output.values, channel * predictionLength * levels, predictionLength * levels,
            low, high, frames.scale.values[(channel + 1) * time - 1] * quantileRealCapK);
        if (request.clipNonnegative && nonnegative[channel] != 0) {
            clampMinRange(output.values, channel * predictionLength * levels, predictionLength * levels, 0.0f);
        }
    }
    timing.decodeSeconds += secondsSince(start);
    return output;
}


Tensor Engine::predict(
    const Tensor& input,
    std::size_t predictionLength,
    const Timestral::Request& request,
    const Tensor& pastCovariates,
    const Tensor& knownCovariates,
    const std::vector<double>& times,
    double interval
) const {
    PagePool emptyPool;
    std::vector<std::size_t> emptyTable;
    TimingStats timing;
    return computePrediction(
        input,
        predictionLength,
        request,
        pastCovariates,
        knownCovariates,
        emptyPool,
        emptyTable,
        timing,
        false,
        false,
        times,
        interval
    );
}

Tensor Engine::predictStreamed(
    const Tensor& context,
    std::size_t predictionLength,
    const Timestral::Request& request,
    PagePool& pool,
    std::vector<std::size_t>& table,
    TimingStats& timing,
    const std::vector<double>& times,
    double interval
) const {
    if (!supportsIncremental()) {
        throw std::runtime_error("predictStreamed: backend does not support incremental page cache");
    }
    const Tensor noCovariates;
    return computePrediction(
        context,
        predictionLength,
        request,
        noCovariates,
        noCovariates,
        pool,
        table,
        timing,
        true,
        false,
        times,
        interval
    );
}

Tensor Engine::decodeShared(
    const Tensor& context,
    std::size_t predictionLength,
    const Timestral::Request& request,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    TimingStats& timing,
    const std::vector<double>& times,
    double interval
) const {
    if (!supportsIncremental()) {
        throw std::runtime_error("decodeShared: backend does not support incremental page cache");
    }
    context.validate();
    if (context.shape.size() != 2 && context.shape.size() != 3) {
        throw std::invalid_argument("decodeShared requires a (B,T) or (B,V,T) context");
    }
    if (context.shape[0] == 0 || context.shape[1] == 0 || context.shape.back() == 0) {
        throw std::invalid_argument("decodeShared requires a nonempty (B,V,T) context");
    }
    if (predictionLength == 0) {
        throw std::invalid_argument("decodeShared requires a positive horizon");
    }
    if (table.empty()) {
        throw std::invalid_argument("decodeShared requires a nonempty page table");
    }
    const std::size_t channels = context.shape.size() == 3 ? context.shape[1] : 1;
    const PageGeometry shared = model->pageGeometry(channels);
    const PageGeometry& actual = pool.geometry();
    if (actual.rowCount != shared.rowCount || actual.headCount != shared.headCount
        || actual.headDim != shared.headDim || actual.timeLayers != shared.timeLayers) {
        throw std::invalid_argument("decodeShared page pool must match the variable group");
    }
    // 共享页表引用归调用方所有：传副本，上下文截断时只丢弃副本上的页表
    std::vector<std::size_t> localTable = table;
    const Tensor noCovariates;
    return computePrediction(
        context,
        predictionLength,
        request,
        noCovariates,
        noCovariates,
        pool,
        localTable,
        timing,
        true,
        true,
        times,
        interval
    );
}

std::shared_ptr<const Engine> loadEngine(
    const std::filesystem::path& directory,
    const Timestral::Settings& settings
) {
    const ModelConfig config = ModelConfig::load(directory / "config.json");
    utils::Safetensors weights = utils::Safetensors::load(directory / "model.safetensors");
    return std::make_shared<const Engine>(
        loadBackendModel(config, std::move(weights), settings.threads, settings.gpuThreads),
        settings
    );
}

std::shared_ptr<const Engine> loadEngine(const std::filesystem::path& directory) {
    return loadEngine(directory, Timestral::Settings{});
}

} // namespace timestral
