/***************
 * @file config.cpp
 * @brief 上游config.json字段解析与推理约束校验。
 ***************/

#include "timestral/utils/config.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>

#include "timestral/utils/json.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

static float readFloat(const utils::Json& json, const std::string& field) {
    const double value = json.asDouble();
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
        throw std::invalid_argument("config field " + field + ": not a finite FP32 value");
    }
    return static_cast<float>(value);
}

static void requirePositive(float value, const std::string& field) {
    if (!std::isfinite(value) || value <= 0.0f) {
        throw std::invalid_argument("config field " + field + ": must be finite and positive");
    }
}

ModelConfig ModelConfig::load(std::filesystem::path path) {
    try {
        return fromJson(utils::Json::load(path));
    } catch (const std::exception& error) {
        throw std::runtime_error("config " + path.string() + ": " + error.what());
    }
}

ModelConfig ModelConfig::fromJson(const utils::Json& json) {
    const auto& object = json.asObject();
    const std::set<std::string> known = {
        "patch_size",
        "d_model",
        "num_heads",
        "mlp_hidden_dim",
        "block_size",
        "context_length",
        "rope_theta",
        "rope_partial_factor",
        "norm_eps",
        "minimum_scale",
        "quantile_width_factor",
        "attn_bias",
        "mlp_bias",
        "use_asinh",
        "stabilize_with_global",
        "covariate_enabled",
        "split_context_horizon_attn",
        "attend_missing_patches",
        "time_attn_gate",
        "variate_attn_gate",
        "block_schedule",
        "quantile_knots"
    };
    for (const auto& entry : object) {
        if (known.find(entry.first) == known.end()) {
            throw std::invalid_argument("config: unknown field " + entry.first);
        }
    }
    for (const char* field : {
        "patch_size", "d_model", "num_heads", "mlp_hidden_dim",
        "block_schedule", "quantile_knots", "block_size", "context_length"
    }) {
        if (object.find(field) == object.end()) {
            throw std::invalid_argument(std::string("config: missing required field ") + field);
        }
    }

    ModelConfig config;
    for (const auto& entry : object) {
        const std::string& field = entry.first;
        const utils::Json& value = entry.second;
        try {
            if (field == "patch_size") {
                config.patchSize = value.asSize();
            } else if (field == "d_model") {
                config.modelDim = value.asSize();
            } else if (field == "num_heads") {
                config.headCount = value.asSize();
            } else if (field == "mlp_hidden_dim") {
                config.mlpHiddenDim = value.asSize();
            } else if (field == "block_size") {
                config.blockSize = value.asSize();
            } else if (field == "context_length") {
                config.contextLength = value.asSize();
            } else if (field == "rope_theta") {
                config.ropeTheta = readFloat(value, field);
            } else if (field == "rope_partial_factor") {
                config.ropePartialFactor = readFloat(value, field);
            } else if (field == "norm_eps") {
                config.normEps = readFloat(value, field);
            } else if (field == "minimum_scale") {
                config.minimumScale = readFloat(value, field);
            } else if (field == "quantile_width_factor") {
                config.quantileWidthFactor = readFloat(value, field);
            } else if (field == "attn_bias") {
                config.attnBias = value.asBool();
            } else if (field == "mlp_bias") {
                config.mlpBias = value.asBool();
            } else if (field == "use_asinh") {
                config.useAsinh = value.asBool();
            } else if (field == "stabilize_with_global") {
                config.stabilizeWithGlobal = value.asBool();
            } else if (field == "covariate_enabled") {
                config.covariateEnabled = value.asBool();
            } else if (field == "split_context_horizon_attn") {
                config.splitContextHorizonAttn = value.asBool();
            } else if (field == "attend_missing_patches") {
                config.attendMissingPatches = value.asBool();
            } else if (field == "time_attn_gate") {
                config.timeAttnGate = value.asString();
            } else if (field == "variate_attn_gate") {
                config.variateAttnGate = value.asString();
            } else if (field == "block_schedule") {
                for (const auto& item : value.asArray()) {
                    config.blockSchedule.push_back(item.asString());
                }
            } else if (field == "quantile_knots") {
                for (const auto& item : value.asArray()) {
                    config.quantileKnots.push_back(readFloat(item, field));
                }
            }
        } catch (const std::exception& error) {
            throw std::invalid_argument("config field " + field + ": " + error.what());
        }
    }
    config.validate();
    return config;
}

std::size_t ModelConfig::headDim() const {
    if (headCount == 0 || modelDim == 0 || modelDim % headCount != 0) {
        throw std::invalid_argument("config d_model/num_heads: positive divisible dimensions required");
    }
    return modelDim / headCount;
}

std::size_t ModelConfig::ropeDim() const {
    if (!std::isfinite(ropePartialFactor) || ropePartialFactor <= 0.0f || ropePartialFactor > 1.0f) {
        throw std::invalid_argument("config rope_partial_factor: expected finite value in (0, 1]");
    }
    const std::size_t width = headDim();
    // 先用扩展精度检查转换边界，避免超大维度转浮点后舍入到size_t上界之外。
    const long double rotated = static_cast<long double>(width) * ropePartialFactor;
    if (rotated >= static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("config rope_dim: dimension overflow");
    }
    const std::size_t dimension = static_cast<std::size_t>(rotated);
    if (dimension == 0 || dimension % 2 != 0 || dimension > width) {
        throw std::invalid_argument("config rope_partial_factor: rope_dim must be positive, even and within head_dim");
    }
    return dimension;
}

std::size_t ModelConfig::medianIndex() const {
    const auto median = std::find(quantileKnots.begin(), quantileKnots.end(), 0.5f);
    if (median == quantileKnots.end()) {
        throw std::invalid_argument("config quantile_knots: missing median 0.5");
    }
    return static_cast<std::size_t>(median - quantileKnots.begin());
}

void ModelConfig::validate() const {
    if (patchSize == 0) {
        throw std::invalid_argument("config patch_size: must be positive");
    }
    if (modelDim == 0) {
        throw std::invalid_argument("config d_model: must be positive");
    }
    if (headCount == 0) {
        throw std::invalid_argument("config num_heads: must be positive");
    }
    if (mlpHiddenDim == 0) {
        throw std::invalid_argument("config mlp_hidden_dim: must be positive");
    }
    if (blockSize == 0) {
        throw std::invalid_argument("config block_size: must be positive");
    }
    if (contextLength == 0) {
        throw std::invalid_argument("config context_length: must be positive");
    }
    headDim();
    ropeDim();
    requirePositive(ropeTheta, "rope_theta");
    requirePositive(normEps, "norm_eps");
    requirePositive(minimumScale, "minimum_scale");
    if (!std::isfinite(quantileWidthFactor) || quantileWidthFactor < 0.0f) {
        throw std::invalid_argument("config quantile_width_factor: must be finite and nonnegative");
    }
    if (stabilizeWithGlobal) {
        throw std::invalid_argument("config stabilize_with_global: advancing refine frame requires false");
    }
    if (blockSchedule.empty()) {
        throw std::invalid_argument("config block_schedule: at least one layer required");
    }
    for (const std::string& kind : blockSchedule) {
        if (kind != "time" && kind != "variate" && kind != "time_variate" && kind != "variate_time") {
            throw std::invalid_argument("config block_schedule: unknown layer kind " + kind);
        }
    }
    for (const std::string& gate : {timeAttnGate, variateAttnGate}) {
        if (gate != "none" && gate != "sdpa_sigmoid") {
            throw std::invalid_argument("config attention gate: unknown value " + gate);
        }
    }
    medianIndex();
    for (std::size_t i = 0; i < quantileKnots.size(); ++i) {
        const float knot = quantileKnots[i];
        if (!std::isfinite(knot) || knot <= 0.0f || knot >= 1.0f) {
            throw std::invalid_argument("config quantile_knots: expected finite probabilities in (0, 1)");
        }
        if (i > 0 && !(quantileKnots[i - 1] < knot)) {
            throw std::invalid_argument("config quantile_knots: must be strictly increasing");
        }
    }
    // 对称性由预测选项决定；配置允许非对称分位数及仅中位数的头。
    elementCount({modelDim, modelDim});
    elementCount({modelDim, mlpHiddenDim});
    elementCount({patchSize, quantileKnots.size()});
    elementCount({patchSize, blockSize});
}

bool ModelConfig::supportsKnownCovariates() const {
    if (!covariateEnabled) {
        return false;
    }
    // 对应固定上游a534bdb的LAYER_KIND_AXES及supports_known_covariates。
    std::vector<bool> axes;
    for (const std::string& kind : blockSchedule) {
        if (kind == "time") {
            axes.push_back(true);
        } else if (kind == "variate") {
            axes.push_back(false);
        } else if (kind == "time_variate") {
            axes.push_back(true);
            axes.push_back(false);
        } else if (kind == "variate_time") {
            axes.push_back(false);
            axes.push_back(true);
        } else {
            throw std::invalid_argument("config block_schedule: unknown layer kind " + kind);
        }
    }
    bool laterTime = false;
    bool laterVariate = false;
    for (std::size_t i = axes.size(); i > 0; --i) {
        if (axes[i - 1] && laterTime && laterVariate) {
            return true;
        }
        laterTime = laterTime || axes[i - 1];
        laterVariate = laterVariate || !axes[i - 1];
    }
    return false;
}

} // namespace timestral
