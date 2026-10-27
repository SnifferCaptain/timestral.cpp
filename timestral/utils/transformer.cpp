/***************
 * @file transformer.cpp
 * @brief Timestral公共模型编排、分页增量与共享前缀执行
 * 移植自Mistral AI Timestral layers.py/model.py/masks.py/scaler.py，
 * Apache-2.0，固定基线a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
 ***************/

#include "timestral/utils/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "timestral/utils/patch.hpp"

namespace timestral {

static float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

static float softplus(float value) {
    return value > 20.0f ? value : std::log1p(std::exp(value));
}

static Tensor swiGLU(const Tensor& fused) {
    fused.validate();
    if (fused.shape.empty() || fused.shape.back() == 0 || fused.shape.back() % 2 != 0) {
        throw std::invalid_argument("swiGLU: expected an even nonzero final dimension");
    }
    auto shape = fused.shape;
    const std::size_t width = shape.back() / 2;
    shape.back() = width;
    Tensor output(shape);
    for (std::size_t row = 0; row < output.size() / width; ++row) {
        for (std::size_t j = 0; j < width; ++j) {
            output.values[row * width + j] = fused.values[row * 2 * width + j]
                * silu(fused.values[row * 2 * width + width + j]);
        }
    }
    return output;
}

static Tensor monotoneQuantiles(const Tensor& raw, const std::vector<float>& levels, float widthFactor) {
    raw.validate();
    const auto medianIt = std::find(levels.begin(), levels.end(), 0.5f);
    if (levels.empty() || medianIt == levels.end() || raw.shape.empty() || raw.shape.back() != levels.size()
        || !std::isfinite(widthFactor) || widthFactor < 0.0f) {
        throw std::invalid_argument("monotoneQuantiles: invalid levels or shape");
    }
    for (std::size_t j = 1; j < levels.size(); ++j) {
        if (!(levels[j] > levels[j - 1])) {
            throw std::invalid_argument("monotoneQuantiles: levels must increase");
        }
    }
    const std::size_t median = static_cast<std::size_t>(medianIt - levels.begin());
    Tensor output = raw;
    const float maximum = std::numeric_limits<float>::max();
    for (std::size_t row = 0; row < raw.size() / levels.size(); ++row) {
        const std::size_t base = row * levels.size();
        float sum = 0.0f;
        for (std::size_t j = median; j-- > 0;) {
            sum += widthFactor * (levels[j + 1] - levels[j]) * softplus(raw.values[base + j]);
            output.values[base + j] = raw.values[base + median] - sum;
        }
        sum = 0.0f;
        for (std::size_t j = median + 1; j < levels.size(); ++j) {
            sum += widthFactor * (levels[j] - levels[j - 1]) * softplus(raw.values[base + j]);
            output.values[base + j] = raw.values[base + median] + sum;
        }
    }
    // 上游在拼接后统一夹持，包括自由中位通道，NaN保持不变
    for (float& value : output.values) {
        value = std::clamp(value, -maximum, maximum);
    }
    return output;
}

// 谓词按值内联进classify与fill，块扫描和行展开不经std::function逐元素调用
template <typename Predicate>
static AttentionMask makeAttentionMask(Predicate predicate) {
    AttentionMask mask;
    mask.allowed = predicate;
    mask.fill = [predicate](std::size_t query, std::size_t keyBegin, std::size_t keyCount, std::vector<std::uint8_t>& allowed) {
        for (std::size_t key = 0; key < keyCount; ++key) {
            allowed[key] = predicate(query, keyBegin + key) ? 1 : 0;
        }
    };
    mask.classify = [predicate](
        std::size_t queryBegin,
        std::size_t keyBegin,
        std::size_t queryCount,
        std::size_t keyCount
    ) {
        bool any = false;
        bool all = true;
        for (std::size_t query = 0; query < queryCount; ++query) {
            for (std::size_t key = 0; key < keyCount; ++key) {
                const bool value = predicate(queryBegin + query, keyBegin + key);
                any = any || value;
                all = all && value;
            }
        }
        return !any ? MaskTile::Empty : (all ? MaskTile::Full : MaskTile::Partial);
    };
    return mask;
}

static void addInPlace(Tensor& target, const Tensor& value) {
    if (target.shape != value.shape) {
        throw std::invalid_argument("residual: shape mismatch");
    }
    // size()每次都校验形状，循环外取一次
    const std::size_t count = target.size();
    for (std::size_t i = 0; i < count; ++i) {
        target.values[i] += value.values[i];
    }
}

static void validateRows(
    const Tensor& input,
    const std::vector<std::uint8_t>& observed,
    std::size_t patchSize
) {
    input.validate();
    if (input.shape.size() != 2 || input.shape[0] == 0 || input.shape[1] == 0
        || input.shape[1] % patchSize != 0 || observed.size() != input.size()) {
        throw std::invalid_argument("model: expected nonempty (R,T), patch-aligned T and matching observation mask");
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (observed[i] > 1 || (observed[i] != 0 && !std::isfinite(input.values[i]))) {
            throw std::invalid_argument("model: invalid observation mask or nonfinite observed value");
        }
    }
}

Transformer::Transformer(ModelConfig loadedConfig, utils::Safetensors weights)
    : modelConfig(std::move(loadedConfig)),
      weights(std::move(weights)) {
    modelConfig.validate();
    const auto& config = modelConfig;
    const std::size_t modelDim = config.modelDim;
    std::set<std::string> expected;
    const auto require = [&](const std::string& name, const std::vector<std::size_t>& shape) {
        expected.insert(name);
        const auto& weight = this->weights.at(name);
        if (weight.shape != shape || weight.values.size() != elementCount(shape)) {
            throw std::invalid_argument("weight shape mismatch: " + name);
        }
        for (float value : weight.values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("nonfinite weight: " + name);
            }
        }
    };
    const auto projection = [&](const std::string& name, std::size_t in, std::size_t out, bool bias) {
        require(name + ".weight", {out, in});
        if (bias) {
            require(name + ".bias", {out});
        }
    };
    const auto block = [&](const std::string& name, std::size_t in, std::size_t hidden, std::size_t out, bool bias) {
        projection(name + ".hidden_layer", in, hidden, bias);
        projection(name + ".output_layer", hidden, out, bias);
        projection(name + ".residual_layer", in, out, bias);
    };
    block(
        "representation.patch_proj",
        elementCount({2, config.patchSize}),
        elementCount({4, modelDim}),
        modelDim,
        true
    );
    require("representation.forecast_query", {modelDim});
    block(
        "head.projection",
        modelDim,
        elementCount({4, modelDim}),
        elementCount({config.patchSize, config.quantileKnots.size()}),
        false
    );
    require("transformer.out_norm.scale", {modelDim});
    for (std::size_t layer = 0; layer < config.blockSchedule.size(); ++layer) {
        const std::string prefix = "transformer.layers." + std::to_string(layer);
        const auto axes = attentionAxes(config.blockSchedule[layer]);
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            const std::string index = std::to_string(axis);
            require(prefix + ".attn_norms." + index + ".scale", {modelDim});
            const std::string attn = prefix + ".attns." + index;
            projection(attn + ".wQKV", modelDim, elementCount({3, modelDim}), config.attnBias);
            projection(attn + ".wO", modelDim, modelDim, config.attnBias);
            if ((axes[axis] ? config.timeAttnGate : config.variateAttnGate) == "sdpa_sigmoid") {
                projection(attn + "._gate", modelDim, modelDim, config.attnBias);
            }
        }
        require(prefix + ".mlp_norm.scale", {modelDim});
        projection(prefix + ".mlp.w12", modelDim, elementCount({2, config.mlpHiddenDim}), config.mlpBias);
        projection(prefix + ".mlp.w3", config.mlpHiddenDim, modelDim, config.mlpBias);
    }
    for (const auto& entry : this->weights.tensors()) {
        if (expected.count(entry.first) == 0) {
            throw std::invalid_argument("unexpected weight: " + entry.first);
        }
    }
}

const ModelConfig& Transformer::config() const {
    return modelConfig;
}

PageGeometry Transformer::pageGeometry(std::size_t rowCount) const {
    PageGeometry geometry;
    geometry.rowCount = rowCount;
    geometry.headCount = modelConfig.headCount;
    geometry.headDim = modelConfig.headDim();
    geometry.timeLayers = timeLayerCount(modelConfig);
    return geometry;
}

std::uint64_t Transformer::pageFingerprint() const {
    return modelFingerprint(modelConfig);
}

bool Transformer::supportsIncremental() const {
    return true;
}

Tensor Transformer::projectPhase(
    const Tensor& input,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    (void)phase;
    return project(input, name, bias);
}

Tensor Transformer::projectNormalized(
    const Tensor& input,
    const utils::Weight& scale,
    float epsilon,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    return projectPhase(normalize(input, scale, epsilon), name, bias, phase);
}

void Transformer::projectAttention(
    const Tensor& input,
    const utils::Weight& scale,
    float epsilon,
    const std::string& attn,
    bool bias,
    bool gated,
    Tensor& fused,
    Tensor& gate,
    ProjectionPhase phase
) const {
    const Tensor norm = normalize(input, scale, epsilon);
    fused = projectPhase(norm, attn + ".wQKV", bias, phase);
    gate = gated ? projectPhase(norm, attn + "._gate", bias, phase) : Tensor();
}

void Transformer::prepareCacheKv(
    const Tensor& input,
    const utils::Weight& scale,
    float epsilon,
    const std::string& attn,
    bool bias,
    Tensor& k,
    Tensor& v,
    const std::vector<double>& positions
) const {
    const Tensor norm = normalize(input, scale, epsilon);
    const Tensor fused = project(norm, attn + ".wQKV", bias);
    Tensor q(k.shape);
    prepareQkv(fused, q, k, v, positions, true);
}

void Transformer::applyGate(Tensor& attended, const Tensor& gate) const {
    if (attended.shape != gate.shape) {
        throw std::invalid_argument("gate: shape mismatch");
    }
    const std::size_t count = attended.size();
    for (std::size_t i = 0; i < count; ++i) {
        attended.values[i] *= 1.0f / (1.0f + std::exp(-gate.values[i]));
    }
}

void Transformer::projectResidual(
    Tensor& x,
    const Tensor& input,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    addInPlace(x, projectPhase(input, name, bias, phase));
}

Tensor Transformer::activate(const Tensor& fused) const {
    return swiGLU(fused);
}

bool Transformer::optimizedPrefill() const {
    return false;
}

bool Transformer::broadcastPrefix() const {
    return false;
}

Tensor Transformer::residual(
    const Tensor& input,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    Tensor hidden = projectPhase(input, name + ".hidden_layer", bias, phase);
    for (float& value : hidden.values) {
        value = silu(value);
    }
    Tensor result = projectPhase(hidden, name + ".output_layer", bias, phase);
    addInPlace(result, projectPhase(input, name + ".residual_layer", bias, phase));
    return result;
}

ScaleResult Transformer::scale(const Tensor& context, const std::vector<std::uint8_t>& observed) const {
    const auto& config = modelConfig;
    validateRows(context, observed, config.patchSize);
    const std::size_t time = context.shape[1];
    ScaleResult result{Tensor(context.shape), Tensor(context.shape), Tensor(context.shape)};
    std::vector<float> scaled;
    std::vector<float> loc;
    std::vector<float> scaleValues;
    std::vector<PageScaleState> states;
    normalizePatches(
        context,
        observed,
        std::vector<PageScaleState>{},
        0,
        time / config.patchSize,
        config,
        scaled,
        loc,
        scaleValues,
        states
    );
    result.scaled.values = std::move(scaled);
    result.loc.values = std::move(loc);
    result.scale.values = std::move(scaleValues);
    return result;
}

Tensor Transformer::forward(const ForwardInput& input) const {
    ModelTrace scratch;
    return forwardQuantiles(input, false, false, scratch);
}

Tensor Transformer::forward(const ForwardInput& input, ModelTrace& trace) const {
    return forwardQuantiles(input, false, true, trace);
}

Tensor Transformer::predictBlock(const ForwardInput& input) const {
    ModelTrace scratch;
    return forwardQuantiles(input, true, false, scratch);
}

Tensor Transformer::predictBlock(const ForwardInput& input, ModelTrace& trace) const {
    return forwardQuantiles(input, true, true, trace);
}

Tensor Transformer::headOutput(
    const Tensor& hidden,
    std::size_t patchCount,
    const std::vector<float>& loc,
    const std::vector<float>& scale,
    bool denormalize,
    ProjectionPhase phase
) const {
    const auto& config = modelConfig;
    const std::size_t rowCount = hidden.shape[0];
    const std::size_t horizon = elementCount({patchCount, config.patchSize});
    const std::size_t quantiles = config.quantileKnots.size();
    Tensor raw = residual(hidden, "head.projection", false, phase);
    raw.shape = {rowCount, horizon, quantiles};
    Tensor output = monotoneQuantiles(raw, config.quantileKnots, config.quantileWidthFactor);
    if (!denormalize) {
        return output;
    }
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t t = 0; t < horizon; ++t) {
            const std::size_t base = (row * horizon + t) * quantiles;
            float minimum = std::numeric_limits<float>::infinity();
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < quantiles; ++j) {
                float value = output.values[base + j];
                if (config.useAsinh) {
                    value = std::sinh(value);
                }
                value = value * scale[row] + loc[row];
                output.values[base + j] = value;
                if (std::isfinite(value)) {
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
            }
            for (std::size_t j = 0; j < quantiles; ++j) {
                float& value = output.values[base + j];
                if (std::isinf(value)) {
                    value = value > 0.0f ? maximum : minimum;
                    if (std::isinf(value)) {
                        value = 0.0f;
                    }
                }
            }
        }
    }
    return output;
}

Tensor Transformer::predictFrozen(
    const ForwardInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    double positionOffset
) const {
    ModelTrace trace;
    return forwardQuantiles(input, true, false, trace, pool, table, positionOffset);
}

Tensor Transformer::forwardQuantiles(
    const ForwardInput& input,
    bool denormalize,
    bool recordTrace,
    ModelTrace& trace
) const {
    PagePool pool;
    const std::vector<std::size_t> table;
    return forwardQuantiles(input, denormalize, recordTrace, trace, pool, table, 0.0);
}

Tensor Transformer::forwardQuantiles(
    const ForwardInput& input,
    bool denormalize,
    bool recordTrace,
    ModelTrace& trace,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    double positionOffset
) const {
    const auto& config = modelConfig;
    const Tensor& context = input.context;
    const std::vector<std::uint8_t>& observed = input.observed;
    const std::vector<std::size_t>& groupIds = input.groupIds;
    const std::size_t horizonPatches = input.horizonPatches;
    const std::vector<int>& roles = input.roles;
    const std::vector<std::uint8_t>& indicators = input.indicators;
    validateRows(context, observed, config.patchSize);
    const std::size_t rowCount = context.shape[0];
    const std::size_t time = context.shape[1];
    const std::size_t contextPatches = time / config.patchSize;
    const std::size_t horizon = elementCount({horizonPatches, config.patchSize});
    if (horizonPatches == 0 || groupIds.size() != rowCount || (!roles.empty() && roles.size() != rowCount)
        || (!indicators.empty() && indicators.size() != rowCount)) {
        throw std::invalid_argument("model: invalid horizon, groupIds, roles or indicators");
    }
    bool covariates = false;
    for (int role : roles) {
        if (role < 0 || role > 2) {
            throw std::invalid_argument("model: role must be Target=0, PastCov=1 or KnownCov=2");
        }
        covariates = covariates || role != 0;
    }
    covariates = covariates && config.covariateEnabled;
    if (covariates && horizonPatches > config.blockSize) {
        throw std::invalid_argument("model: covariate horizon exceeds blockSize");
    }
    if (input.hasFuture) {
        validateRows(input.future, input.futureObserved, config.patchSize);
        if (input.future.shape != std::vector<std::size_t>{rowCount, horizon}) {
            throw std::invalid_argument("model: future must have shape (R,horizonPatches*P)");
        }
    } else if (!input.futureObserved.empty()) {
        throw std::invalid_argument("model: futureObserved requires future");
    }
    if (contextPatches > std::numeric_limits<std::size_t>::max() - horizonPatches) {
        throw std::overflow_error("model: patch count overflow");
    }
    const std::size_t encodedCount = covariates ? contextPatches + horizonPatches : contextPatches;
    const std::size_t decodedCount = covariates ? encodedCount : horizonPatches;
    if (encodedCount > std::numeric_limits<std::size_t>::max() - decodedCount) {
        throw std::overflow_error("model: token count overflow");
    }
    const std::size_t slots = encodedCount + decodedCount;
    if (slots > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("model: positions exceed int64 range");
    }
    const std::size_t width = elementCount({encodedCount, config.patchSize});
    Tensor inputs({rowCount, width});
    std::vector<std::uint8_t> fullObserved(inputs.size(), std::uint8_t{0});
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t t = 0; t < time; ++t) {
            inputs.values[row * width + t] = observed[row * time + t] != 0 ? context.values[row * time + t] : 0.0f;
            fullObserved[row * width + t] = observed[row * time + t];
        }
        if (covariates && input.hasFuture) {
            for (std::size_t t = 0; t < horizon; ++t) {
                const std::size_t index = row * horizon + t;
                inputs.values[row * width + time + t] = input.futureObserved[index] != 0
                    ? input.future.values[index] : 0.0f;
                fullObserved[row * width + time + t] = input.futureObserved[index];
            }
        }
    }
    ScaleResult normalization = scale(inputs, fullObserved);
    // known-future行按完整窗口总体方差归一化，二值行固定(0,1)，不做asinh
    if (covariates) {
        for (std::size_t row = 0; row < rowCount; ++row) {
            if (roles[row] != 2) {
                continue;
            }
            float count = 0.0f;
            float sum = 0.0f;
            bool binary = true;
            for (std::size_t t = 0; t < width; ++t) {
                const std::size_t index = row * width + t;
                if (fullObserved[index] != 0) {
                    const float value = inputs.values[index];
                    sum += value;
                    count += 1.0f;
                    binary = binary && (value == 0.0f || value == 1.0f);
                }
            }
            binary = binary && count > 0.0f;
            float loc = sum / std::max(count, 1.0f);
            float variance = 0.0f;
            for (std::size_t t = 0; t < width; ++t) {
                const std::size_t index = row * width + t;
                if (fullObserved[index] != 0) {
                    const float difference = inputs.values[index] - loc;
                    variance += difference * difference;
                }
            }
            float scale = std::max(std::sqrt(variance / std::max(count, 1.0f)), config.minimumScale);
            if (binary) {
                loc = 0.0f;
                scale = 1.0f;
            }
            for (std::size_t t = 0; t < width; ++t) {
                const std::size_t index = row * width + t;
                normalization.loc.values[index] = loc;
                normalization.scale.values[index] = scale;
                normalization.scaled.values[index] = fullObserved[index] != 0
                    ? scaledValue(inputs.values[index], loc, scale, config.useAsinh && !binary)
                    : 0.0f;
            }
        }
    }
    const std::size_t modelDim = config.modelDim;
    const std::size_t base = table.size();
    if (base > contextPatches || !std::isfinite(positionOffset) || positionOffset < 0.0) {
        throw std::invalid_argument("invalid frozen prefix or position offset");
    }
    if (base > 0) {
        verifyPoolGeometry(pool, config, rowCount);
    }
    const std::size_t activeSlots = slots - base;
    const std::size_t tailWidth = (encodedCount - base) * config.patchSize;
    std::vector<float> scaledTail(rowCount * tailWidth);
    for (std::size_t row = 0; row < rowCount; ++row) {
        std::copy_n(
            normalization.scaled.values.begin() + row * width + base * config.patchSize,
            tailWidth,
            scaledTail.begin() + row * tailWidth
        );
    }
    std::vector<float> patchValues;
    assemblePatchInput(
        scaledTail,
        fullObserved,
        rowCount,
        width,
        base,
        encodedCount - base,
        config.patchSize,
        patchValues
    );
    Tensor patchInput({rowCount, encodedCount - base, elementCount({2, config.patchSize})}, std::move(patchValues));
    std::vector<std::uint8_t> keyValid;
    buildKeyValid(fullObserved, rowCount, width, config.patchSize, config.attendMissingPatches, keyValid);
    std::map<std::size_t, std::size_t> first = firstObservedPatches(
        fullObserved,
        groupIds,
        rowCount,
        width,
        config.patchSize
    );
    if (positionOffset > 0.0) {
        for (auto& entry : first) {
            entry.second = 0;
        }
        for (std::size_t slot = 0; slot < base; ++slot) {
            for (std::size_t row = 0; row < rowCount; ++row) {
                keyValid[row * encodedCount + slot] = pool.page(table[slot]).maskBits[row];
            }
        }
    }
    Tensor encoded = residual(patchInput, "representation.patch_proj", true);
    Tensor x({rowCount, activeSlots, modelDim});
    const auto& query = weights.at("representation.forecast_query").values;
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t slot = base; slot < slots; ++slot) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                x.values[(row * activeSlots + slot - base) * modelDim + j] = slot < encodedCount
                    ? encoded.values[(row * (encodedCount - base) + slot - base) * modelDim + j] : query[j];
            }
        }
    }
    if (recordTrace) {
        trace.embedding = x;
        trace.layers.clear();
        trace.normalized = Tensor();
    }
    std::vector<double> positions;
    buildPositions(rowCount, activeSlots, base, encodedCount, covariates, groupIds, first, positions, input.coordinates);
    for (double& position : positions) {
        position += positionOffset;
    }
    if (base > 0) {
        verifyTableGrid(
            pool,
            table,
            rowCount,
            keyValid,
            groupIds,
            first,
            input.coordinates,
            positionOffset
        );
    }
    const std::vector<std::size_t> newPages(activeSlots, pool.capacity());
    const std::vector<int> targetRoles(rowCount, 0);
    runLayers(
        x,
        base,
        encodedCount,
        contextPatches,
        positions,
        keyValid,
        covariates,
        roles.empty() ? targetRoles : roles,
        groupIds,
        pool,
        table,
        newPages,
        false,
        false,
        recordTrace,
        trace
    );
    Tensor decoder({rowCount, horizonPatches, modelDim});
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t patch = 0; patch < horizonPatches; ++patch) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                const std::size_t slot = encodedCount + (covariates ? contextPatches : 0) + patch;
                const std::size_t source = (row * activeSlots + slot - base) * modelDim + j;
                decoder.values[(row * horizonPatches + patch) * modelDim + j] = x.values[source];
            }
        }
    }
    std::vector<float> loc(rowCount, 0.0f);
    std::vector<float> scaleValues(rowCount, 1.0f);
    for (std::size_t row = 0; row < rowCount; ++row) {
        loc[row] = normalization.loc.values[row * width + time - 1];
        scaleValues[row] = normalization.scale.values[row * width + time - 1];
    }
    return headOutput(decoder, horizonPatches, loc, scaleValues, denormalize);
}

void Transformer::attentionWork(std::size_t tasks, const std::function<void(std::size_t)>& function) const {
    for (std::size_t task = 0; task < tasks; ++task) {
        function(task);
    }
}

void Transformer::runLayers(
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
    const auto& config = modelConfig;
    const std::size_t rowCount = x.shape[0];
    const std::size_t count = x.shape[1];
    const std::size_t slots = base + count;
    const std::size_t modelDim = config.modelDim;
    if (table.size() != base || newPages.size() != count
        || positions.size() != rowCount * count || keyValid.size() != rowCount * encoded
        || roles.size() != rowCount || groupIds.size() != rowCount) {
        throw std::invalid_argument("model: inconsistent incremental layer input");
    }
    if (phase == ProjectionPhase::Decode
        && (prefill || recordTrace || covariates || base == 0 || encoded != base)) {
        throw std::invalid_argument("model: decode projection phase requires fully cached FP32 history");
    }
    const bool prefillOnly = prefill && optimizedPrefill() && !recordTrace && !sharedPages && !covariates
        && count > 0 && slots <= encoded
        && std::all_of(newPages.begin(), newPages.end(), [&](std::size_t page) {
            return page < pool.capacity();
        });
    // reference保留连续KV布局，广播后端只物化整组前缀，新尾部直接借用当前K/V
    const bool prefixView = broadcastPrefix();
    const std::size_t prefixRows = sharedPages ? pool.geometry().rowCount : rowCount;
    std::vector<float> keyBuffer(prefixView ? 0 : rowCount * slots * modelDim, 0.0f);
    std::vector<float> valueBuffer(keyBuffer.size(), 0.0f);
    std::vector<float> prefixKeys(prefixView ? prefixRows * base * modelDim : 0, 0.0f);
    std::vector<float> prefixValues(prefixKeys.size(), 0.0f);
    const auto timeAllowed = [&](std::size_t row, std::size_t q, std::size_t k) {
        if (q < encoded) {
            return k < encoded && (q == k || (keyValid[row * encoded + k] != 0
                && ((covariates && roles[row] == 2) || k <= q)));
        }
        if (k < encoded) {
            return (!covariates || q - encoded >= contextPatches) && k < contextPatches
                && keyValid[row * encoded + k] != 0;
        }
        const std::size_t qDec = q - encoded;
        const std::size_t kDec = k - encoded;
        return q == k || (kDec <= qDec && (!covariates || (qDec >= contextPatches && kDec >= contextPatches)));
    };
    // 每行keyValid前缀和，几何分类据此O(1)判定一段编码键是否全部有效
    std::vector<std::size_t> validPrefix(rowCount * (encoded + 1), 0);
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t k = 0; k < encoded; ++k) {
            validPrefix[row * (encoded + 1) + k + 1] = validPrefix[row * (encoded + 1) + k]
                + (keyValid[row * encoded + k] != 0 ? 1 : 0);
        }
    }
    const auto allValid = [&](std::size_t row, std::size_t first, std::size_t last) {
        const std::size_t offset = row * (encoded + 1);
        return validPrefix[offset + last + 1] - validPrefix[offset + first] == last + 1 - first;
    };
    const auto variateAllowed = [&](std::size_t slot, std::size_t q, std::size_t k) {
        if (q == k) {
            return true;
        }
        if (groupIds[q] != groupIds[k] || (slot < encoded && keyValid[k * encoded + slot] == 0)) {
            return false;
        }
        return !covariates || !((roles[q] != 0 && roles[k] == 0) || (roles[q] == 2 && roles[k] != 2));
    };
    std::size_t timeLayer = 0;
    for (std::size_t layer = 0; layer < config.blockSchedule.size(); ++layer) {
        const std::string prefix = "transformer.layers." + std::to_string(layer);
        const auto axes = attentionAxes(config.blockSchedule[layer]);
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            const bool isTime = axes[axis];
            const std::string attn = prefix + ".attns." + std::to_string(axis);
            const bool gated = (isTime ? config.timeAttnGate : config.variateAttnGate) == "sdpa_sigmoid";
            // 头内连续布局与展平的(rowCount,count,modelDim)相同
            Tensor q({rowCount, count, config.headCount, config.headDim()});
            Tensor k(q.shape);
            Tensor v(q.shape);
            Tensor gate;
            const auto& scale = weights.at(prefix + ".attn_norms." + std::to_string(axis) + ".scale");
            if (prefillOnly && isTime && timeLayer + 1 == timeLayerCount(config)) {
                prepareCacheKv(x, scale, config.normEps, attn, config.attnBias, k, v, positions);
            } else {
                Tensor fused;
                projectAttention(x, scale, config.normEps, attn, config.attnBias, gated, fused, gate, phase);
                prepareQkv(fused, q, k, v, positions, isTime);
            }
            Tensor attended({rowCount, count, modelDim});
            if (isTime) {
                for (std::size_t row = 0; row < rowCount; ++row) {
                    for (std::size_t local = 0; local < count; ++local) {
                        const std::size_t slot = base + local;
                        const std::size_t offset = (row * count + local) * modelDim;
                        if (!prefixView) {
                            std::copy_n(
                                k.values.begin() + static_cast<std::ptrdiff_t>(offset),
                                modelDim,
                                keyBuffer.begin() + static_cast<std::ptrdiff_t>((row * slots + slot) * modelDim)
                            );
                            std::copy_n(
                                v.values.begin() + static_cast<std::ptrdiff_t>(offset),
                                modelDim,
                                valueBuffer.begin() + static_cast<std::ptrdiff_t>((row * slots + slot) * modelDim)
                            );
                        }
                        if (!sharedPages && slot < encoded && newPages[local] != pool.capacity()) {
                            CachePage& entry = pool.page(newPages[local]);
                            std::copy_n(
                                k.values.begin() + static_cast<std::ptrdiff_t>(offset),
                                modelDim,
                                entry.keys[timeLayer].begin() + static_cast<std::ptrdiff_t>(row * modelDim)
                            );
                            std::copy_n(
                                v.values.begin() + static_cast<std::ptrdiff_t>(offset),
                                modelDim,
                                entry.values[timeLayer].begin() + static_cast<std::ptrdiff_t>(row * modelDim)
                            );
                        }
                    }
                }
                // 纯prefill的持久输出到最后time轴写页即齐备，不影响forward与trace
                if (prefillOnly && timeLayer + 1 == timeLayerCount(config)) {
                    return;
                }
                for (std::size_t slot = 0; slot < base; ++slot) {
                    const CachePage& entry = pool.page(table[slot]);
                    const std::size_t copiedRows = prefixView ? prefixRows : rowCount;
                    for (std::size_t row = 0; row < copiedRows; ++row) {
                        // 广播路径每通道只复制一次前缀，连续路径维持reference原布局
                        const std::size_t pageRow = sharedPages ? row % pool.geometry().rowCount : row;
                        const std::size_t target = prefixView
                            ? (row * base + slot) * modelDim
                            : (row * slots + slot) * modelDim;
                        auto& keys = prefixView ? prefixKeys : keyBuffer;
                        auto& values = prefixView ? prefixValues : valueBuffer;
                        std::copy_n(
                            entry.keys[timeLayer].begin() + static_cast<std::ptrdiff_t>(pageRow * modelDim),
                            modelDim,
                            keys.begin() + static_cast<std::ptrdiff_t>(target)
                        );
                        std::copy_n(
                            entry.values[timeLayer].begin() + static_cast<std::ptrdiff_t>(pageRow * modelDim),
                            modelDim,
                            values.begin() + static_cast<std::ptrdiff_t>(target)
                        );
                    }
                }
                attentionWork(elementCount({rowCount, config.headCount}), [&](std::size_t task) {
                    const std::size_t row = task / config.headCount;
                    const std::size_t head = task % config.headCount;
                    AttentionMask mask = makeAttentionMask([&](std::size_t query, std::size_t key) {
                        return timeAllowed(row, base + query, key);
                    });
                    const auto classifyExact = mask.classify;
                    mask.classify = [&, row, classifyExact](
                        std::size_t queryBegin,
                        std::size_t keyBegin,
                        std::size_t queryCount,
                        std::size_t keyCount
                    ) {
                        if (queryCount == 0 || keyCount == 0) {
                            return MaskTile::Empty;
                        }
                        const std::size_t queryFirst = base + queryBegin;
                        const std::size_t queryLast = queryFirst + queryCount - 1;
                        const std::size_t keyLast = keyBegin + keyCount - 1;
                        if (queryLast < encoded) {
                            // 编码query不能读解码流，非Known行还受时间因果界限约束
                            if (keyBegin >= encoded
                                || (!(covariates && roles[row] == 2) && keyBegin > queryLast)) {
                                return MaskTile::Empty;
                            }
                        } else if (queryFirst >= encoded) {
                            if (keyLast < encoded) {
                                // 双流历史query不读编码流，未来query只读原历史编码槽
                                if ((covariates && queryLast - encoded < contextPatches)
                                    || keyBegin >= contextPatches) {
                                    return MaskTile::Empty;
                                }
                            } else if (keyBegin >= encoded) {
                                if (keyBegin > queryLast) {
                                    return MaskTile::Empty;
                                }
                                if (covariates && queryFirst - encoded >= contextPatches
                                    && keyLast - encoded < contextPatches) {
                                    return MaskTile::Empty;
                                }
                                if (covariates && queryLast - encoded < contextPatches && keyLast < queryFirst) {
                                    return MaskTile::Empty;
                                }
                            }
                        }
                        // 无协变量时按timeAllowed逐支证明Full，其余情形逐元素核对
                        // 编码query读不晚于自身的有效编码键，解码query读有效历史编码键与不晚于自身的解码键
                        if (!covariates) {
                            if (queryLast < encoded && keyLast < encoded && keyLast <= queryFirst
                                && allValid(row, keyBegin, keyLast)) {
                                return MaskTile::Full;
                            }
                            if (queryFirst >= encoded && keyLast < std::min(encoded, contextPatches)
                                && allValid(row, keyBegin, keyLast)) {
                                return MaskTile::Full;
                            }
                            if (queryFirst >= encoded && keyBegin >= encoded && keyLast <= queryFirst) {
                                return MaskTile::Full;
                            }
                        }
                        return classifyExact(queryBegin, keyBegin, queryCount, keyCount);
                    };
                    {
                        const std::size_t queryOffset = row * count * modelDim + head * config.headDim();
                        const std::size_t keyOffset = prefixView
                            ? queryOffset
                            : row * slots * modelDim + head * config.headDim();
                        const std::size_t prefixOffset = (row % prefixRows) * base * modelDim
                            + head * config.headDim();
                        const AttentionView view{
                            q.values,
                            prefixView ? k.values : keyBuffer,
                            prefixView ? v.values : valueBuffer,
                            attended.values,
                            count,
                            slots,
                            config.headDim(),
                            queryOffset,
                            keyOffset,
                            keyOffset,
                            queryOffset,
                            modelDim,
                            modelDim,
                            modelDim,
                            modelDim,
                            prefixKeys,
                            prefixValues,
                            prefixView ? base : 0,
                            prefixOffset,
                            modelDim,
                            true
                        };
                        attention(view, mask);
                    }
                });
                ++timeLayer;
            } else {
                const auto ranges = attentionGroups(groupIds);
                attentionWork(elementCount({ranges.size(), count, config.headCount}), [&](std::size_t task) {
                    const auto& range = ranges[task / (count * config.headCount)];
                    const std::size_t begin = range.first;
                    const std::size_t width = range.second;
                    const std::size_t local = task / config.headCount % count;
                    const std::size_t head = task % config.headCount;
                    {
                        const std::size_t slot = base + local;
                        const AttentionMask mask = makeAttentionMask([&](std::size_t query, std::size_t key) {
                            return variateAllowed(slot, begin + query, begin + key);
                        });
                        {
                            const std::size_t offset = (begin * count + local) * modelDim + head * config.headDim();
                            const std::size_t stride = count * modelDim;
                            const AttentionView view{
                                q.values, k.values, v.values, attended.values,
                                width, width, config.headDim(), offset, offset, offset, offset,
                                stride, stride, stride, stride
                            };
                            attention(view, mask);
                        }
                    }
                });
            }
            if (gated) {
                applyGate(attended, gate);
            }
            projectResidual(x, attended, attn + ".wO", config.attnBias, phase);
        }
        Tensor hidden = activate(projectNormalized(
            x,
            weights.at(prefix + ".mlp_norm.scale"),
            config.normEps,
            prefix + ".mlp.w12",
            config.mlpBias,
            phase
        ));
        projectResidual(x, hidden, prefix + ".mlp.w3", config.mlpBias, phase);
        if (recordTrace) {
            trace.layers.push_back(x);
        }
    }
    x = normalize(x, weights.at("transformer.out_norm.scale"), config.normEps);
    if (recordTrace) {
        trace.normalized = x;
    }
}

void Transformer::appendPages(
    const CacheAppendInput& input,
    PagePool& pool,
    std::vector<std::size_t>& table
) const {
    const auto& config = modelConfig;
    if (!std::isfinite(input.positionOffset) || input.positionOffset < 0.0) {
        throw std::invalid_argument("invalid rolling position offset");
    }
    validateRows(input.context, input.observed, config.patchSize);
    const std::size_t rowCount = input.context.shape[0];
    const std::size_t time = input.context.shape[1];
    const std::size_t patchCount = time / config.patchSize;
    if (input.groupIds.size() != rowCount) {
        throw std::invalid_argument("model: appendPages requires one group id per row");
    }
    if (table.size() > patchCount) {
        throw std::invalid_argument("model: page table exceeds context patches");
    }
    verifyPoolGeometry(pool, config, rowCount);
    std::map<std::size_t, std::size_t> first = firstObservedPatches(
        input.observed,
        input.groupIds,
        rowCount,
        time,
        config.patchSize
    );
    std::vector<std::uint8_t> keyValid;
    buildKeyValid(input.observed, rowCount, time, config.patchSize, config.attendMissingPatches, keyValid);
    if (input.positionOffset > 0.0) {
        for (auto& entry : first) {
            entry.second = 0;
        }
        for (std::size_t row = 0; row < rowCount; ++row) {
            if (config.attendMissingPatches && !table.empty()
                && pool.page(table.front()).scaleStates[row].count > 0.0) {
                std::fill_n(keyValid.begin() + row * patchCount, patchCount, std::uint8_t{1});
            }
        }
    }
    verifyTableGrid(pool, table, rowCount, keyValid, input.groupIds, first, input.coordinates, input.positionOffset);
    std::vector<PageScaleState> running(rowCount, PageScaleState{});
    if (!table.empty()) {
        running = pool.page(table.back()).scaleStates;
    }
    const std::vector<int> targetRoles(rowCount, 0);
    const std::size_t modelDim = config.modelDim;
    ModelTrace scratch;
    std::vector<float> scaled;
    std::vector<float> loc;
    std::vector<float> scaleValues;
    const std::size_t pendingCount = patchCount - table.size();
    // 先行按父键链生成各待补页的页键，用peek判定命中，定位尾部连续未命中段
    std::vector<std::uint64_t> patchParentKeys(pendingCount);
    std::vector<std::uint64_t> patchKeys(pendingCount);
    std::vector<std::uint64_t> patchFingerprints(pendingCount);
    std::vector<std::size_t> patchFound(pendingCount);
    std::vector<std::vector<std::uint8_t>> patchMaskBits(pendingCount);
    std::vector<std::vector<double>> patchPositions(pendingCount);
    std::size_t runStart = 0;
    {
        std::uint64_t parentKey = table.empty() ? 0 : pool.page(table.back()).key;
        for (std::size_t i = 0; i < pendingCount; ++i) {
            const std::size_t patch = table.size() + i;
            patchMaskBits[i].assign(rowCount, std::uint8_t{0});
            for (std::size_t row = 0; row < rowCount; ++row) {
                patchMaskBits[i][row] = keyValid[row * patchCount + patch];
            }
            buildPositions(rowCount, 1, patch, patchCount, false, input.groupIds, first, patchPositions[i], input.coordinates);
            for (double& position : patchPositions[i]) {
                position += input.positionOffset;
            }
            pageKeys(
                input.context,
                input.observed,
                rowCount,
                time,
                patch,
                config.patchSize,
                patchMaskBits[i],
                patchPositions[i],
                parentKey,
                pool.fingerprint(),
                patchKeys[i],
                patchFingerprints[i]
            );
            patchParentKeys[i] = parentKey;
            patchFound[i] = pool.peek(patchKeys[i], parentKey, patchFingerprints[i]);
            if (patchFound[i] != pool.capacity()) {
                runStart = i + 1;
            }
            parentKey = patchKeys[i];
        }
    }
    for (std::size_t i = 0; i < runStart; ++i) {
        const std::size_t patch = table.size();
        std::vector<PageScaleState> states;
        normalizePatches(
            input.context,
            input.observed,
            running,
            patch,
            patch + 1,
            config,
            scaled,
            loc,
            scaleValues,
            states
        );
        const std::vector<PageScaleState> pageStates(
            states.begin(),
            states.begin() + static_cast<std::ptrdiff_t>(rowCount)
        );
        const std::size_t found = pool.find(patchKeys[i], patchParentKeys[i], patchFingerprints[i]);
        if (found != pool.capacity()) {
            const std::vector<PageScaleState>& stored = pool.page(found).scaleStates;
            for (std::size_t row = 0; row < rowCount; ++row) {
                if (stored[row].count != pageStates[row].count || stored[row].sum != pageStates[row].sum
                    || stored[row].m2 != pageStates[row].m2) {
                    throw std::runtime_error("model: cached page normalization state does not match prefix");
                }
            }
            table.push_back(found);
            running = stored;
            continue;
        }
        const std::size_t index = pool.acquire();
        if (index == pool.capacity()) {
            throw std::runtime_error("model: page pool is full; capacity must cover the cached prefix");
        }
        CachePage& entry = pool.page(index);
        entry.parentKey = patchParentKeys[i];
        entry.key = patchKeys[i];
        entry.inputFingerprint = patchFingerprints[i];
        entry.maskBits = patchMaskBits[i];
        entry.positions = patchPositions[i];
        entry.scaleStates = pageStates;
        std::vector<float> patchValues;
        assemblePatchInput(scaled, input.observed, rowCount, time, patch, 1, config.patchSize, patchValues);
        Tensor patchInput({rowCount, 1, elementCount({2, config.patchSize})}, std::move(patchValues));
        Tensor encoded = residual(patchInput, "representation.patch_proj", true);
        Tensor x({rowCount, 1, modelDim});
        x.values = encoded.values;
        const std::vector<std::size_t> newPages(1, index);
        runLayers(
            x,
            table.size(),
            patchCount,
            patchCount,
            patchPositions[i],
            keyValid,
            false,
            targetRoles,
            input.groupIds,
            pool,
            table,
            newPages,
            false,
            true,
            false,
            scratch
        );
        table.push_back(index);
        running = pageStates;
    }
    if (runStart < pendingCount) {
        // 尾部连续未命中段一次性归一化、投影与写页，等价于整段prefill
        const std::size_t runBegin = table.size();
        const std::size_t runLength = pendingCount - runStart;
        std::vector<PageScaleState> states;
        normalizePatches(
            input.context,
            input.observed,
            running,
            runBegin,
            patchCount,
            config,
            scaled,
            loc,
            scaleValues,
            states
        );
        std::vector<double> runPositions;
        buildPositions(rowCount, runLength, runBegin, patchCount, false, input.groupIds, first, runPositions, input.coordinates);
        for (double& position : runPositions) {
            position += input.positionOffset;
        }
        std::vector<std::size_t> newPages(runLength);
        for (std::size_t j = 0; j < runLength; ++j) {
            const std::size_t i = runStart + j;
            const std::size_t index = pool.acquire();
            if (index == pool.capacity()) {
                throw std::runtime_error("model: page pool is full; capacity must cover the cached prefix");
            }
            newPages[j] = index;
            CachePage& entry = pool.page(index);
            entry.parentKey = patchParentKeys[i];
            entry.key = patchKeys[i];
            entry.inputFingerprint = patchFingerprints[i];
            entry.maskBits = patchMaskBits[i];
            entry.positions = patchPositions[i];
            std::vector<PageScaleState> pageStates(rowCount);
            for (std::size_t row = 0; row < rowCount; ++row) {
                pageStates[row] = states[row * runLength + j];
            }
            entry.scaleStates = pageStates;
        }
        std::vector<float> patchValues;
        assemblePatchInput(scaled, input.observed, rowCount, time, runBegin, runLength, config.patchSize, patchValues);
        Tensor patchInput({rowCount, runLength, elementCount({2, config.patchSize})}, std::move(patchValues));
        Tensor encoded = residual(patchInput, "representation.patch_proj", true);
        Tensor x({rowCount, runLength, modelDim});
        x.values = encoded.values;
        runLayers(
            x,
            runBegin,
            patchCount,
            patchCount,
            runPositions,
            keyValid,
            false,
            targetRoles,
            input.groupIds,
            pool,
            table,
            newPages,
            false,
            true,
            false,
            scratch
        );
        for (std::size_t j = 0; j < runLength; ++j) {
            table.push_back(newPages[j]);
        }
        for (std::size_t row = 0; row < rowCount; ++row) {
            running[row] = states[row * runLength + runLength - 1];
        }
    }

}

Tensor Transformer::decodePages(
    const CacheAppendInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table
) const {
    return decodePagesImpl(input, pool, table, false);
}

Tensor Transformer::decodePagesShared(
    const CacheAppendInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table
) const {
    return decodePagesImpl(input, pool, table, true);
}

Tensor Transformer::decodePagesImpl(
    const CacheAppendInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    bool shared
) const {
    const auto& config = modelConfig;
    if (!std::isfinite(input.positionOffset) || input.positionOffset < 0.0) {
        throw std::invalid_argument("invalid rolling position offset");
    }
    validateRows(input.context, input.observed, config.patchSize);
    const std::size_t rowCount = input.context.shape[0];
    const std::size_t time = input.context.shape[1];
    const std::size_t patchCount = time / config.patchSize;
    const std::size_t patches = input.horizonPatches;
    if (input.groupIds.size() != rowCount) {
        throw std::invalid_argument("model: decodePages requires one group id per row");
    }
    if (patches == 0) {
        throw std::invalid_argument("model: decodePages requires a positive horizon");
    }
    if (table.size() > patchCount) {
        throw std::invalid_argument("model: page table exceeds context patches");
    }
    if (shared && input.positionOffset != 0.0) {
        throw std::invalid_argument("rolling offsets cannot use shared request pages");
    }
    if (shared && table.empty()) {
        throw std::invalid_argument("model: decodePagesShared requires a nonempty page table");
    }
    // 共享页保存一组通道，批内请求按相同通道顺序读取
    verifyPoolGeometry(pool, config, shared ? pool.geometry().rowCount : rowCount);
    std::map<std::size_t, std::size_t> first = firstObservedPatches(
        input.observed,
        input.groupIds,
        rowCount,
        time,
        config.patchSize
    );
    std::vector<std::uint8_t> keyValid;
    buildKeyValid(input.observed, rowCount, time, config.patchSize, config.attendMissingPatches, keyValid);
    if (shared) {
        verifySharedTableGrid(pool, table, rowCount, keyValid, input.groupIds, first, input.coordinates);
        // 页内K/V只由观测到的前缀值与掩码决定，未观测点不参与归一化
        const std::size_t prefixLength = table.size() * config.patchSize;
        for (std::size_t row = pool.geometry().rowCount; row < rowCount; ++row) {
            const std::size_t sourceRow = row % pool.geometry().rowCount;
            for (std::size_t t = 0; t < prefixLength; ++t) {
                const std::size_t index = row * time + t;
                if (input.observed[index] != input.observed[sourceRow * time + t]) {
                    throw std::invalid_argument("model: shared prefix rows must be identical");
                }
                if (input.observed[index] != 0
                    && input.context.values[index] != input.context.values[sourceRow * time + t]) {
                    throw std::invalid_argument("model: shared prefix rows must be identical");
                }
            }
        }
    } else {
        verifyTableGrid(pool, table, rowCount, keyValid, input.groupIds, first, input.coordinates, input.positionOffset);
    }
    const std::size_t base = table.size();
    std::vector<PageScaleState> running(rowCount, PageScaleState{});
    if (base > 0) {
        if (shared) {
            // 每条请求从对应通道的累计量继续归一化
            for (std::size_t row = 0; row < rowCount; ++row) {
                running[row] = pool.page(table.back()).scaleStates[row % pool.geometry().rowCount];
            }
        } else {
            running = pool.page(table.back()).scaleStates;
        }
    }
    const std::size_t freshPatches = patchCount - base;
    std::vector<float> scaled;
    std::vector<float> loc;
    std::vector<float> scaleValues;
    std::vector<PageScaleState> states;
    if (freshPatches > 0) {
        normalizePatches(
            input.context,
            input.observed,
            running,
            base,
            patchCount,
            config,
            scaled,
            loc,
            scaleValues,
            states
        );
    }
    const std::size_t modelDim = config.modelDim;
    const std::size_t count = freshPatches + patches;
    const auto& query = weights.at("representation.forecast_query").values;
    Tensor x({rowCount, count, modelDim});
    if (freshPatches > 0) {
        std::vector<float> patchValues;
        assemblePatchInput(scaled, input.observed, rowCount, time, base, freshPatches, config.patchSize, patchValues);
        Tensor patchInput({rowCount, freshPatches, elementCount({2, config.patchSize})}, std::move(patchValues));
        Tensor encoded = residual(patchInput, "representation.patch_proj", true);
        for (std::size_t row = 0; row < rowCount; ++row) {
            for (std::size_t patch = 0; patch < freshPatches; ++patch) {
                for (std::size_t j = 0; j < modelDim; ++j) {
                    const std::size_t source = (row * freshPatches + patch) * modelDim + j;
                    x.values[(row * count + patch) * modelDim + j] = encoded.values[source];
                }
            }
        }
    }
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t patch = 0; patch < patches; ++patch) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                x.values[(row * count + freshPatches + patch) * modelDim + j] = query[j];
            }
        }
    }
    std::vector<double> positions;
    buildPositions(rowCount, count, base, patchCount, false, input.groupIds, first, positions, input.coordinates);
    for (double& position : positions) {
        position += input.positionOffset;
    }
    const std::vector<std::size_t> newPages(count, pool.capacity());
    const std::vector<int> targetRoles(rowCount, 0);
    ModelTrace scratch;
    runLayers(
        x,
        base,
        patchCount,
        patchCount,
        positions,
        keyValid,
        false,
        targetRoles,
        input.groupIds,
        pool,
        table,
        newPages,
        shared,
        false,
        false,
        scratch,
        freshPatches == 0 && base > 0 ? ProjectionPhase::Decode : ProjectionPhase::Float32
    );
    Tensor decoder({rowCount, patches, modelDim});
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t patch = 0; patch < patches; ++patch) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                const std::size_t source = (row * count + freshPatches + patch) * modelDim + j;
                decoder.values[(row * patches + patch) * modelDim + j] = x.values[source];
            }
        }
    }
    std::vector<float> frameLoc(rowCount, 0.0f);
    std::vector<float> frameScale(rowCount, config.minimumScale);
    for (std::size_t row = 0; row < rowCount; ++row) {
        const PageScaleState& state = freshPatches > 0
            ? states[row * freshPatches + (freshPatches - 1)]
            : pool.page(table.back()).scaleStates[shared ? row % pool.geometry().rowCount : row];
        const double mean = state.count > 0.0 ? state.sum / state.count : 0.0;
        frameLoc[row] = static_cast<float>(mean);
        frameScale[row] = static_cast<float>(std::max(
            std::sqrt(state.m2 / std::max(state.count - 1.0, 1.0)),
            static_cast<double>(config.minimumScale)
        ));
    }
    return headOutput(
        decoder,
        patches,
        frameLoc,
        frameScale,
        true,
        freshPatches == 0 && base > 0 ? ProjectionPhase::Decode : ProjectionPhase::Float32
    );
}

} // namespace timestral
