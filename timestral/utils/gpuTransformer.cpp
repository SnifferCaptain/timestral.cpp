/***************
 * @file gpuTransformer.cpp
 * @brief 共享GPU层图、精确双流掩码与分页KV传输
 ***************/

#include "timestral/utils/gpuTransformer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "timestral/utils/patch.hpp"

namespace timestral {

// 只在已有异常传播时清理，清理失败不能覆盖最初的设备或分配错误
static void gpuCleanup(GpuExecutor& executor) noexcept {
    try {
        executor.end();
    } catch (...) {
    }
}

static std::uint32_t gpuCount(std::size_t count) {
    if (count > (std::numeric_limits<std::uint32_t>::max)()) {
        throw std::invalid_argument("gpu: shader index overflow");
    }
    return static_cast<std::uint32_t>(count);
}

static std::uint32_t gpuBits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static std::uint32_t gpuGroups(std::size_t count) {
    return gpuCount((count + 63) / 64);
}

static std::array<std::uint32_t, 3> gpuGrid(std::size_t groups) {
    const std::size_t x = std::min<std::size_t>(groups, 65535);
    return {gpuCount(x), gpuCount((groups + x - 1) / x), 1};
}

GpuTransformer::GpuTransformer(
    ModelConfig config,
    utils::Safetensors weights,
    std::shared_ptr<GpuExecutor> executor,
    bool residentResidual
)
    : Transformer(std::move(config), std::move(weights)),
      executor(std::move(executor)),
      residentResidual(residentResidual) {
    if (!this->executor) {
        throw std::invalid_argument("gpu: missing executor");
    }
    this->executor->begin();
    try {
        for (const auto& entry : this->weights.tensors()) {
            deviceWeights.emplace(entry.first, this->executor->upload(entry.second.values, true));
        }
        zero = this->executor->upload(std::vector<float>(1, 0.0f), true);
        this->executor->end();
    } catch (...) {
        gpuCleanup(*this->executor);
        throw;
    }
}

std::size_t GpuTransformer::deviceProject(
    std::size_t input,
    std::size_t rows,
    const std::string& name,
    bool bias
) const {
    const auto& shape = weights.at(name + ".weight").shape;
    const std::size_t output = executor->allocate(elementCount({rows, shape[0]}));
    executor->dispatch(
        "gemm",
        {input, deviceWeights.at(name + ".weight"), bias ? deviceWeights.at(name + ".bias") : zero, output},
        {gpuCount(rows), gpuCount(shape[0]), gpuCount(shape[1]), bias ? 1u : 0u},
        {gpuCount((shape[0] + 15) / 16), gpuCount((rows + 15) / 16), 1}
    );
    return output;
}

std::size_t GpuTransformer::deviceNormalize(
    std::size_t input,
    std::size_t rows,
    const std::string& scale,
    float epsilon
) const {
    const std::size_t width = weights.at(scale).values.size();
    const std::size_t output = executor->allocate(elementCount({rows, width}));
    executor->dispatch(
        executor->supportsOrderedAttention() ? "rmsNormOrdered" : "rmsNorm", {input, deviceWeights.at(scale), output},
        {gpuCount(rows), gpuCount(width), gpuBits(epsilon)}, gpuGrid(rows)
    );
    return output;
}

std::size_t GpuTransformer::elementwise(
    std::size_t first,
    std::size_t second,
    std::size_t count,
    std::size_t width,
    std::uint32_t mode
) const {
    const std::size_t output = executor->allocate(count);
    executor->dispatch(
        executor->supportsOrderedAttention() ? "elementwiseOrdered" : "elementwise", {first, second, output},
        {gpuCount(count), gpuCount(width), mode}, gpuGrid(gpuGroups(count))
    );
    return output;
}

Tensor GpuTransformer::residual(
    const Tensor& input,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    if (!residentResidual) {
        return Transformer::residual(input, name, bias, phase);
    }
    const std::lock_guard<std::recursive_mutex> lock(executionMutex);
    input.validate();
    const auto& hidden = weights.at(name + ".hidden_layer.weight").shape;
    const auto& projected = weights.at(name + ".output_layer.weight").shape;
    const auto& skipped = weights.at(name + ".residual_layer.weight").shape;
    if (input.shape.empty() || input.shape.back() != hidden[1] || hidden[0] != projected[1]
        || hidden[1] != skipped[1] || projected[0] != skipped[0]) {
        throw std::invalid_argument("gpu: residual projection shape mismatch: " + name);
    }
    const std::size_t rows = input.size() / hidden[1];
    // 实验融合仅摊薄小派发开销，不把680M测得的大M上界当作其它GPU规则
    if (rows < 128) {
        return Transformer::residual(input, name, bias, phase);
    }
    auto outputShape = input.shape;
    outputShape.back() = projected[0];
    Tensor output(outputShape);
    const std::size_t hiddenCount = elementCount({rows, hidden[0]});
    executor->begin();
    try {
        const auto source = executor->upload(input.values);
        const auto linear = deviceProject(source, rows, name + ".hidden_layer", bias);
        const auto activated = executor->allocate(hiddenCount);
        executor->dispatch(
            "siluGrid",
            {linear, activated},
            {gpuCount(hiddenCount)},
            gpuGrid(gpuGroups(hiddenCount))
        );
        const auto value = deviceProject(activated, rows, name + ".output_layer", bias);
        const auto skip = deviceProject(source, rows, name + ".residual_layer", bias);
        const auto result = elementwise(value, skip, output.size(), projected[0], 0);
        output.values = executor->download(result);
        executor->end();
    } catch (...) {
        gpuCleanup(*executor);
        throw;
    }
    return output;
}

Tensor GpuTransformer::project(const Tensor& input, const std::string& name, bool bias) const {
    const std::lock_guard<std::recursive_mutex> lock(executionMutex);
    input.validate();
    const auto& shape = weights.at(name + ".weight").shape;
    if (input.shape.empty() || input.shape.back() != shape[1]) {
        throw std::invalid_argument("gpu: projection shape mismatch: " + name);
    }
    auto outputShape = input.shape;
    outputShape.back() = shape[0];
    Tensor output(outputShape);
    if (input.size() == 0) {
        return output;
    }
    executor->begin();
    try {
        const auto source = executor->upload(input.values);
        const auto result = deviceProject(source, input.size() / shape[1], name, bias);
        output.values = executor->download(result);
        executor->end();
    } catch (...) {
        gpuCleanup(*executor);
        throw;
    }
    return output;
}

Tensor GpuTransformer::normalize(const Tensor& input, const utils::Weight& scale, float epsilon) const {
    const std::lock_guard<std::recursive_mutex> lock(executionMutex);
    Tensor output(input.shape);
    if (input.size() == 0) {
        return output;
    }
    executor->begin();
    try {
        const auto source = executor->upload(input.values);
        const auto scaling = executor->upload(scale.values);
        const auto result = executor->allocate(input.size());
        executor->dispatch(
            executor->supportsOrderedAttention() ? "rmsNormOrdered" : "rmsNorm", {source, scaling, result},
            {gpuCount(input.size() / scale.values.size()), gpuCount(scale.values.size()), gpuBits(epsilon)},
            gpuGrid(input.size() / scale.values.size())
        );
        output.values = executor->download(result);
        executor->end();
    } catch (...) {
        gpuCleanup(*executor);
        throw;
    }
    return output;
}

void GpuTransformer::prepareQkv(
    const Tensor&,
    Tensor&,
    Tensor&,
    Tensor&,
    const std::vector<double>&,
    bool
) const {
    throw std::logic_error("gpu: host QKV hook must not be entered by resident execution");
}

void GpuTransformer::attention(const AttentionView&, const AttentionMask&) const {
    throw std::logic_error("gpu: host attention hook must not be entered by resident execution");
}

void GpuTransformer::runLayers(
    Tensor& x,
    std::size_t base,
    std::size_t encoded,
    std::size_t contextPatches,
    const std::vector<double>& positions,
    const std::vector<std::uint8_t>& keyValid,
    bool covariates,
    const std::vector<int>& roles,
    const std::vector<std::size_t>& groupIds,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    const std::vector<std::size_t>& newPages,
    bool sharedPages,
    bool prefill,
    bool recordTrace,
    ModelTrace& trace,
    ProjectionPhase phase
) const {
    const std::lock_guard<std::recursive_mutex> lock(executionMutex);
    const auto& config = modelConfig;
    const std::size_t rows = x.shape[0];
    const std::size_t count = x.shape[1];
    const std::size_t slots = base + count;
    const std::size_t dim = config.modelDim;
    const std::size_t tokens = elementCount({rows, count});
    const std::size_t elements = elementCount({tokens, dim});
    if (table.size() != base || newPages.size() != count || positions.size() != tokens
        || keyValid.size() != rows * encoded || roles.size() != rows || groupIds.size() != rows) {
        throw std::invalid_argument("gpu: inconsistent incremental layer input");
    }
    if (phase == ProjectionPhase::Decode
        && (prefill || recordTrace || covariates || base == 0 || encoded != base)) {
        throw std::invalid_argument("gpu: invalid decode phase");
    }
    // 外部groupId先压缩为连续小整数，避免大编号经FP32转换丢失等价关系
    std::vector<float> metadata(std::max<std::size_t>(1, keyValid.size() + 4 * rows), 0.0f);
    for (std::size_t i = 0; i < keyValid.size(); ++i) {
        metadata[i] = keyValid[i] != 0 ? 1.0f : 0.0f;
    }
    std::map<std::size_t, std::size_t> groups;
    for (std::size_t row = 0; row < rows; ++row) {
        groups.emplace(groupIds[row], groups.size());
        metadata[keyValid.size() + row] = static_cast<float>(roles[row]);
        metadata[keyValid.size() + rows + row] = static_cast<float>(groups.at(groupIds[row]));
    }
    for (const auto& range : attentionGroups(groupIds)) {
        for (std::size_t row = range.first; row < range.first + range.second; ++row) {
            metadata[keyValid.size() + 2 * rows + row] = static_cast<float>(range.first);
            metadata[keyValid.size() + 3 * rows + row] = static_cast<float>(range.second);
        }
    }
    // 与reference使用相同FP32角度及libm，GPU只执行旋转乘加，连续位置亦适用
    const std::size_t pairs = config.ropeDim() / 2;
    std::vector<float> rotations(elementCount({tokens, pairs, 2}));
    std::vector<float> inverseFrequencies(pairs, 0.0f);
    for (std::size_t j = 0; j < pairs; ++j) {
        inverseFrequencies[j] = 1.0f / std::pow(config.ropeTheta,
            static_cast<float>(2 * j) / static_cast<float>(config.ropeDim()));
    }
    const std::size_t rowRotationSize = elementCount({count, pairs, 2});
    for (std::size_t row = 0; row < rows; ++row) {
        const auto firstPosition = positions.begin() + static_cast<std::ptrdiff_t>(row * count);
        if (row != 0 && std::equal(firstPosition, firstPosition + static_cast<std::ptrdiff_t>(count), positions.begin())) {
            // 只复用本次调用完全相同的位置行，不按shape缓存不同请求的rotary
            std::copy_n(rotations.begin(), rowRotationSize,
                rotations.begin() + static_cast<std::ptrdiff_t>(row * rowRotationSize));
            continue;
        }
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t token = row * count + local;
            for (std::size_t j = 0; j < pairs; ++j) {
                const float angle = static_cast<float>(positions[token]) * inverseFrequencies[j];
                rotations[(token * pairs + j) * 2] = std::cos(angle);
                rotations[(token * pairs + j) * 2 + 1] = std::sin(angle);
            }
        }
    }
    executor->begin();
    try {
        std::size_t state = executor->upload(x.values);
        const std::size_t meta = executor->upload(metadata);
        const std::size_t rotary = executor->upload(rotations);
        std::size_t timeLayer = 0;
        std::vector<std::pair<std::size_t, std::size_t>> pageBuffers;
        bool cacheComplete = false;
        for (std::size_t layer = 0; layer < config.blockSchedule.size(); ++layer) {
            const std::string prefix = "transformer.layers." + std::to_string(layer);
            const auto axes = attentionAxes(config.blockSchedule[layer]);
            for (std::size_t axis = 0; axis < axes.size(); ++axis) {
                const bool isTime = axes[axis];
                const std::string attn = prefix + ".attns." + std::to_string(axis);
                const bool gated = (isTime ? config.timeAttnGate : config.variateAttnGate) == "sdpa_sigmoid";
                const auto normalized = deviceNormalize(state, tokens, prefix + ".attn_norms." + std::to_string(axis) + ".scale", config.normEps);
                const auto fused = deviceProject(normalized, tokens, attn + ".wQKV", config.attnBias);
                const auto q = executor->allocate(elements);
                const auto k = executor->allocate(elements);
                const auto v = executor->allocate(elements);
                executor->dispatch(
                    "prepareQkv", {fused, rotary, q, k, v},
                    {gpuCount(tokens), gpuCount(config.headCount), gpuCount(config.headDim()), gpuCount(config.ropeDim()), isTime ? 1u : 0u},
                    gpuGrid(gpuGroups(elements))
                );
                if (isTime) {
                    pageBuffers.emplace_back(k, v);
                    // KV-only prefill不消费最终hidden，最后time的K/V就绪后停止
                    if (prefill && !recordTrace && pageBuffers.size() == timeLayerCount(config)) {
                        cacheComplete = true;
                        break;
                    }
                }
                std::size_t keys = k;
                std::size_t values = v;
                const bool packedPrefix = isTime && base > 0 && sharedPages && executor->supportsPrefixAttention();
                if (isTime && base > 0) {
                    const bool transfer = executor->supportsCopy();
                    const std::size_t prefixRows = transfer && sharedPages ? pool.geometry().rowCount : rows;
                    std::vector<float> prefixKeys(elementCount({prefixRows, base, dim}));
                    std::vector<float> prefixValues(prefixKeys.size());
                    for (std::size_t row = 0; row < prefixRows; ++row) {
                        const std::size_t pageRow = sharedPages ? row % pool.geometry().rowCount : row;
                        for (std::size_t slot = 0; slot < base; ++slot) {
                            const auto& entry = pool.page(table[slot]);
                            const auto target = static_cast<std::ptrdiff_t>((row * base + slot) * dim);
                            std::copy_n(entry.keys[timeLayer].begin() + static_cast<std::ptrdiff_t>(pageRow * dim), dim, prefixKeys.begin() + target);
                            std::copy_n(entry.values[timeLayer].begin() + static_cast<std::ptrdiff_t>(pageRow * dim), dim, prefixValues.begin() + target);
                        }
                    }
                    const auto oldKeys = executor->upload(prefixKeys);
                    const auto oldValues = executor->upload(prefixValues);
                    const std::size_t packedCount = packedPrefix
                        ? elementCount({prefixRows, base, dim}) + elements : elementCount({rows, slots, dim});
                    keys = executor->allocate(packedCount);
                    values = executor->allocate(packedCount);
                    if (packedPrefix) {
                        const std::size_t prefixCount = elementCount({prefixRows, base, dim});
                        executor->copy(oldKeys, keys, 0, 0, prefixCount);
                        executor->copy(oldValues, values, 0, 0, prefixCount);
                        executor->copy(k, keys, 0, prefixCount, elements);
                        executor->copy(v, values, 0, prefixCount, elements);
                    } else if (transfer) {
                        // 共享组前缀只上传一次，设备内按对应通道复制，不消耗shader窗口
                        for (std::size_t row = 0; row < rows; ++row) {
                            const std::size_t source = (row % prefixRows) * base * dim;
                            const std::size_t target = row * slots * dim;
                            executor->copy(oldKeys, keys, source, target, base * dim);
                            executor->copy(oldValues, values, source, target, base * dim);
                            executor->copy(k, keys, row * count * dim, target + base * dim, count * dim);
                            executor->copy(v, values, row * count * dim, target + base * dim, count * dim);
                        }
                    } else {
                        const std::vector<std::uint32_t> push = {gpuCount(rows), gpuCount(count), gpuCount(base), gpuCount(dim)};
                        const std::array<std::uint32_t, 3> dispatch = gpuGrid(gpuGroups(rows * slots * dim));
                        executor->dispatch("mergeKv", {oldKeys, k, keys}, push, dispatch);
                        executor->dispatch("mergeKv", {oldValues, v, values}, push, dispatch);
                    }
                }
                const std::size_t scoreRows = elementCount({tokens, config.headCount});
                std::vector<std::uint32_t> geometry = {
                    gpuCount(rows), gpuCount(count), gpuCount(base), gpuCount(encoded), gpuCount(contextPatches),
                    gpuCount(config.headCount), gpuCount(config.headDim()), covariates ? 1u : 0u, isTime ? 1u : 0u
                };
                std::size_t attended = executor->allocate(elements);
                const bool ordered = executor->supportsOrderedAttention() && (isTime ? slots : rows) <= 2048;
                if (packedPrefix || ordered) {
                    geometry.push_back(packedPrefix ? gpuCount(pool.geometry().rowCount) : 0u);
                }
                const std::string attentionKernel = ordered ? "attentionOrdered"
                    : (packedPrefix ? "attentionPrefix"
                        : (!isTime && executor->supportsGroupedAttention() ? "attentionGroup" : "attentionFused"));
                executor->dispatch(attentionKernel,
                    {q, keys, values, meta, attended}, geometry, gpuGrid(scoreRows));
                if (gated) {
                    const auto gate = deviceProject(normalized, tokens, attn + "._gate", config.attnBias);
                    attended = elementwise(attended, gate, elements, dim, 1);
                }
                const auto projected = deviceProject(attended, tokens, attn + ".wO", config.attnBias);
                state = elementwise(state, projected, elements, dim, 0);
                if (isTime) {
                    ++timeLayer;
                }
            }
            if (cacheComplete) {
                break;
            }
            const auto normalized = deviceNormalize(state, tokens, prefix + ".mlp_norm.scale", config.normEps);
            const auto fused = deviceProject(normalized, tokens, prefix + ".mlp.w12", config.mlpBias);
            const auto hidden = elementwise(fused, zero, tokens * config.mlpHiddenDim, config.mlpHiddenDim, 2);
            const auto projected = deviceProject(hidden, tokens, prefix + ".mlp.w3", config.mlpBias);
            state = elementwise(state, projected, elements, dim, 0);
            if (recordTrace) {
                Tensor snapshot(x.shape);
                snapshot.values = executor->download(state);
                trace.layers.push_back(std::move(snapshot));
            }
        }
        if (!cacheComplete) {
            state = deviceNormalize(state, tokens, "transformer.out_norm.scale", config.normEps);
            x.values = executor->download(state);
        }
        // 落页在整图执行后统一回读，避免每个time子层中断计算提交
        if (!sharedPages) {
            const bool writePages = std::any_of(newPages.begin(), newPages.end(), [&](std::size_t page) {
                return page != pool.capacity();
            });
            if (writePages) {
                for (std::size_t layer = 0; layer < pageBuffers.size(); ++layer) {
                    const auto hostKeys = executor->download(pageBuffers[layer].first);
                    const auto hostValues = executor->download(pageBuffers[layer].second);
                    for (std::size_t local = 0; local < count; ++local) {
                        if (base + local >= encoded || newPages[local] == pool.capacity()) {
                            continue;
                        }
                        auto& entry = pool.page(newPages[local]);
                        for (std::size_t row = 0; row < rows; ++row) {
                            const auto source = static_cast<std::ptrdiff_t>((row * count + local) * dim);
                            const auto target = static_cast<std::ptrdiff_t>(row * dim);
                            std::copy_n(hostKeys.begin() + source, dim, entry.keys[layer].begin() + target);
                            std::copy_n(hostValues.begin() + source, dim, entry.values[layer].begin() + target);
                        }
                    }
                }
            }
        }
        if (recordTrace) {
            trace.normalized = x;
        }
        executor->end();
    } catch (...) {
        gpuCleanup(*executor);
        throw;
    }
}

} // namespace timestral
