/***************
 * @file patch.cpp
 * @brief 因果归一化、patch装配、位置/掩码与页键实现
 ***************/

#include "timestral/utils/patch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace timestral {

// 页键与输入指纹用不同的种子，避免两者退化为同一个哈希
static constexpr std::uint64_t pageKeySeed = 0xc2b2ae3d27d4eb4fULL;
static constexpr std::uint64_t inputFingerprintSeed = 0x9e3779b97f4a7c15ULL;

std::vector<bool> attentionAxes(const std::string& kind) {
    if (kind == "time") {
        return {true};
    }
    if (kind == "variate") {
        return {false};
    }
    if (kind == "time_variate") {
        return {true, false};
    }
    if (kind == "variate_time") {
        return {false, true};
    }
    throw std::invalid_argument("unknown block_schedule kind: " + kind);
}

std::vector<std::pair<std::size_t, std::size_t>> attentionGroups(const std::vector<std::size_t>& groupIds) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    std::set<std::size_t> seen;
    for (std::size_t begin = 0; begin < groupIds.size();) {
        if (!seen.insert(groupIds[begin]).second) {
            return {{0, groupIds.size()}};
        }
        std::size_t end = begin + 1;
        while (end < groupIds.size() && groupIds[end] == groupIds[begin]) {
            ++end;
        }
        ranges.emplace_back(begin, end - begin);
        begin = end;
    }
    return ranges;
}

std::size_t timeLayerCount(const ModelConfig& config) {
    std::size_t count = 0;
    for (const std::string& kind : config.blockSchedule) {
        for (bool isTime : attentionAxes(kind)) {
            count += isTime ? 1 : 0;
        }
    }
    return count;
}

float scaledValue(float value, float loc, float scale, bool useAsinh) {
    float result = (value - loc) / std::max(scale, std::numeric_limits<float>::epsilon());
    if (useAsinh) {
        result = std::asinh(result);
    }
    return result;
}

void normalizePatches(
    const Tensor& context,
    const std::vector<std::uint8_t>& observed,
    const std::vector<PageScaleState>& startStates,
    std::size_t startPatch,
    std::size_t endPatch,
    const ModelConfig& config,
    std::vector<float>& scaled,
    std::vector<float>& loc,
    std::vector<float>& scale,
    std::vector<PageScaleState>& states
) {
    const std::size_t rowCount = context.shape[0];
    const std::size_t time = context.shape[1];
    const std::size_t patchSize = config.patchSize;
    const std::size_t count = endPatch - startPatch;
    scaled.assign(rowCount * count * patchSize, 0.0f);
    loc.assign(rowCount * count * patchSize, 0.0f);
    scale.assign(rowCount * count * patchSize, 0.0f);
    states.assign(rowCount * count, PageScaleState{});
    std::vector<PageScaleState> running(rowCount, PageScaleState{});
    if (startStates.size() == rowCount) {
        running = startStates;
    }
    for (std::size_t row = 0; row < rowCount; ++row) {
        PageScaleState& state = running[row];
        for (std::size_t patch = startPatch; patch < endPatch; ++patch) {
            for (std::size_t t = 0; t < patchSize; ++t) {
                const std::size_t index = row * time + patch * patchSize + t;
                if (observed[index] == 0) {
                    continue;
                }
                const double value = static_cast<double>(context.values[index]);
                const double previous = state.count > 0.0 ? state.sum / state.count : 0.0;
                state.count += 1.0;
                state.sum += value;
                const double mean = state.sum / state.count;
                state.m2 += (value - previous) * (value - mean);
            }
            const double mean = state.count > 0.0 ? state.sum / state.count : 0.0;
            const float patchLoc = static_cast<float>(mean);
            const float patchScale = static_cast<float>(std::max(
                std::sqrt(state.m2 / std::max(state.count - 1.0, 1.0)),
                static_cast<double>(config.minimumScale)
            ));
            const std::size_t local = patch - startPatch;
            states[row * count + local] = state;
            for (std::size_t t = 0; t < patchSize; ++t) {
                const std::size_t index = row * time + patch * patchSize + t;
                const std::size_t out = (row * count + local) * patchSize + t;
                loc[out] = patchLoc;
                scale[out] = patchScale;
                scaled[out] = observed[index] != 0
                    ? scaledValue(context.values[index], patchLoc, patchScale, config.useAsinh)
                    : 0.0f;
            }
        }
    }
}

void assemblePatchInput(
    const std::vector<float>& scaled,
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time,
    std::size_t startPatch,
    std::size_t count,
    std::size_t patchSize,
    std::vector<float>& patchInput
) {
    patchInput.assign(rowCount * count * 2 * patchSize, 0.0f);
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t patch = startPatch + local;
            for (std::size_t t = 0; t < patchSize; ++t) {
                const std::size_t index = row * time + patch * patchSize + t;
                const std::size_t out = (row * count + local) * 2 * patchSize + t;
                patchInput[out] = scaled[(row * count + local) * patchSize + t];
                patchInput[out + patchSize] = observed[index] != 0 ? 0.0f : 1.0f;
            }
        }
    }
}

std::map<std::size_t, std::size_t> firstObservedPatches(
    const std::vector<std::uint8_t>& observed,
    const std::vector<std::size_t>& groupIds,
    std::size_t rowCount,
    std::size_t time,
    std::size_t patchSize
) {
    const std::size_t patchCount = time / patchSize;
    std::map<std::size_t, std::size_t> first;
    for (std::size_t row = 0; row < rowCount; ++row) {
        auto entry = first.emplace(groupIds[row], patchCount).first;
        for (std::size_t patch = 0; patch < patchCount; ++patch) {
            bool any = false;
            for (std::size_t t = 0; t < patchSize; ++t) {
                any = any || observed[row * time + patch * patchSize + t] != 0;
            }
            if (any) {
                entry->second = std::min(entry->second, patch);
            }
        }
    }
    for (auto& entry : first) {
        if (entry.second == patchCount) {
            entry.second = 0;
        }
    }
    return first;
}

void buildKeyValid(
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time,
    std::size_t patchSize,
    bool attendMissingPatches,
    std::vector<std::uint8_t>& keyValid
) {
    const std::size_t patchCount = time / patchSize;
    keyValid.assign(rowCount * patchCount, 0);
    for (std::size_t row = 0; row < rowCount; ++row) {
        bool started = false;
        for (std::size_t patch = 0; patch < patchCount; ++patch) {
            bool any = false;
            for (std::size_t t = 0; t < patchSize; ++t) {
                any = any || observed[row * time + patch * patchSize + t] != 0;
            }
            started = started || any;
            keyValid[row * patchCount + patch] = any || (attendMissingPatches && started) ? 1 : 0;
        }
    }
}

// 坐标和首观测锚点都按请求内的同一时间轴解释
static double relativeCoordinate(
    std::size_t row,
    std::size_t slot,
    std::size_t rowCount,
    const std::vector<std::size_t>& groupIds,
    const std::map<std::size_t, std::size_t>& first,
    const std::vector<double>& coordinates
) {
    const std::size_t anchor = first.at(groupIds.at(row));
    if (coordinates.empty()) {
        return static_cast<double>(slot) - static_cast<double>(anchor);
    }
    if (rowCount == 0 || coordinates.size() % rowCount != 0) {
        throw std::invalid_argument("model: invalid real coordinate shape");
    }
    const std::size_t width = coordinates.size() / rowCount;
    if (slot >= width || anchor >= width) {
        throw std::invalid_argument("model: real coordinate range does not cover tokens");
    }
    const double position = coordinates[row * width + slot] - coordinates[row * width + anchor];
    if (!std::isfinite(position)) {
        throw std::invalid_argument("model: rotary coordinates must be finite");
    }
    return position;
}

void buildPositions(
    std::size_t rowCount,
    std::size_t count,
    std::size_t firstSlot,
    std::size_t encoded,
    bool covariates,
    const std::vector<std::size_t>& groupIds,
    const std::map<std::size_t, std::size_t>& first,
    std::vector<double>& positions,
    const std::vector<double>& coordinates
) {
    positions.assign(rowCount * count, 0);
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t slot = firstSlot + local;
            const std::size_t position = covariates && slot >= encoded ? slot - encoded : slot;
            positions[row * count + local] = relativeCoordinate(row, position, rowCount, groupIds, first, coordinates);
        }
    }
}

void verifyPoolGeometry(const PagePool& pool, const ModelConfig& config, std::size_t rowCount) {
    const PageGeometry& geometry = pool.geometry();
    if (geometry.rowCount != rowCount || geometry.headCount != config.headCount
        || geometry.headDim != config.headDim()
        || geometry.timeLayers != timeLayerCount(config)) {
        throw std::invalid_argument("model: page pool geometry does not match model");
    }
}

void verifyTableGrid(
    const PagePool& pool,
    const std::vector<std::size_t>& table,
    std::size_t rowCount,
    const std::vector<std::uint8_t>& keyValid,
    const std::vector<std::size_t>& groupIds,
    const std::map<std::size_t, std::size_t>& first,
    const std::vector<double>& coordinates,
    double positionOffset
) {
    const std::size_t patchCount = rowCount == 0 ? 0 : keyValid.size() / rowCount;
    for (std::size_t slot = 0; slot < table.size(); ++slot) {
        const CachePage& entry = pool.page(table[slot]);
        for (std::size_t row = 0; row < rowCount; ++row) {
            const double expected = relativeCoordinate(row, slot, rowCount, groupIds, first, coordinates);
            if (entry.positions[row] != expected + positionOffset) {
                throw std::invalid_argument("model: page table grid does not match context");
            }
            if (entry.maskBits[row] != keyValid[row * patchCount + slot]) {
                throw std::invalid_argument("model: cached page mask does not match context");
            }
        }
    }
}

void verifySharedTableGrid(
    const PagePool& pool,
    const std::vector<std::size_t>& table,
    std::size_t rowCount,
    const std::vector<std::uint8_t>& keyValid,
    const std::vector<std::size_t>& groupIds,
    const std::map<std::size_t, std::size_t>& first,
    const std::vector<double>& coordinates
) {
    const std::size_t cachedRows = pool.geometry().rowCount;
    if (cachedRows == 0 || rowCount == 0 || rowCount % cachedRows != 0 || groupIds.size() != rowCount) {
        throw std::invalid_argument("model: shared page group geometry does not match context");
    }
    std::map<std::size_t, std::size_t> chunks;
    for (std::size_t row = 0; row < rowCount; ++row) {
        const std::size_t chunk = row / cachedRows;
        if (groupIds[row] != groupIds[chunk * cachedRows]) {
            throw std::invalid_argument("model: shared pages require complete variable groups");
        }
        const auto inserted = chunks.emplace(groupIds[row], chunk);
        if (!inserted.second && inserted.first->second != chunk) {
            throw std::invalid_argument("model: shared request groups must be isolated");
        }
    }
    const std::size_t patchCount = keyValid.size() / rowCount;
    for (std::size_t slot = 0; slot < table.size(); ++slot) {
        const CachePage& entry = pool.page(table[slot]);
        for (std::size_t row = 0; row < rowCount; ++row) {
            const double expected = relativeCoordinate(row, slot, rowCount, groupIds, first, coordinates);
            const std::size_t pageRow = row % cachedRows;
            if (entry.positions[pageRow] != expected) {
                throw std::invalid_argument("model: shared page table grid does not match context");
            }
            if (entry.maskBits[pageRow] != keyValid[row * patchCount + slot]) {
                throw std::invalid_argument("model: shared page mask does not match context");
            }
        }
    }
}

void pageKeys(
    const Tensor& context,
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time,
    std::size_t patch,
    std::size_t patchSize,
    const std::vector<std::uint8_t>& maskBits,
    const std::vector<double>& positions,
    std::uint64_t parentKey,
    std::uint64_t configFingerprint,
    std::uint64_t& key,
    std::uint64_t& inputFingerprint
) {
    std::vector<std::uint8_t> observedBits(rowCount * patchSize, 0);
    std::vector<float> values(rowCount * patchSize, 0.0f);
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t t = 0; t < patchSize; ++t) {
            const std::size_t index = row * time + patch * patchSize + t;
            const std::size_t out = row * patchSize + t;
            observedBits[out] = observed[index] != 0 ? 1 : 0;
            values[out] = observed[index] != 0 ? context.values[index] : 0.0f;
        }
    }
    FnvHash fingerprint(inputFingerprintSeed);
    fingerprint.addBytes(observedBits);
    fingerprint.addValues(values);
    fingerprint.addBytes(maskBits);
    for (double position : positions) {
        fingerprint.addPosition(position);
    }
    inputFingerprint = fingerprint.value();
    FnvHash hash(pageKeySeed);
    hash.addUnsigned(parentKey);
    hash.addBytes(observedBits);
    hash.addValues(values);
    hash.addBytes(maskBits);
    for (double position : positions) {
        hash.addPosition(position);
    }
    hash.addUnsigned(configFingerprint);
    key = hash.value();
}

} // namespace timestral
