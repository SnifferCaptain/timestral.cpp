/***************
 * @file kernels.cpp
 * @brief FP32线性、归一化、旋转和分位数算子
 * 移植自Mistral AI Timestral layers.py/model.py，Apache-2.0，基线a534bdbd
 ***************/

#include "timestral/reference/kernels.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace timestral::reference {

Tensor matmul(const Tensor& left, const Tensor& right) {
    left.validate();
    right.validate();
    if (left.shape.size() != 2 || right.shape.size() != 2 || left.shape[1] != right.shape[0]) {
        throw std::invalid_argument("matmul: incompatible matrix shapes");
    }
    Tensor output({left.shape[0], right.shape[1]});
    for (std::size_t i = 0; i < left.shape[0]; ++i) {
        for (std::size_t k = 0; k < left.shape[1]; ++k) {
            for (std::size_t j = 0; j < right.shape[1]; ++j) {
                output.values[i * right.shape[1] + j] += left.values[i * left.shape[1] + k]
                    * right.values[k * right.shape[1] + j];
            }
        }
    }
    return output;
}

// 共享内核，hasBias为false时不读取bias，累加式中加0保持原表达式
static Tensor linearKernel(
    const Tensor& input,
    const utils::Weight& weight,
    const utils::Weight& bias,
    bool hasBias
) {
    input.validate();
    if (input.shape.empty() || weight.shape.size() != 2
        || input.shape.back() != weight.shape[1]
        || elementCount(weight.shape) != weight.values.size()) {
        throw std::invalid_argument("linear: incompatible input or weight shape");
    }
    const std::size_t inDim = weight.shape[1];
    const std::size_t outDim = weight.shape[0];
    if (inDim == 0 || (hasBias && (bias.shape != std::vector<std::size_t>{outDim}
        || bias.values.size() != outDim))) {
        throw std::invalid_argument("linear: invalid input width or bias");
    }
    auto shape = input.shape;
    shape.back() = outDim;
    Tensor output(shape);
    for (std::size_t row = 0; row < input.size() / inDim; ++row) {
        for (std::size_t out = 0; out < outDim; ++out) {
            float sum = 0.0f;
            for (std::size_t in = 0; in < inDim; ++in) {
                sum += input.values[row * inDim + in] * weight.values[out * inDim + in];
            }
            output.values[row * outDim + out] = sum + (hasBias ? bias.values[out] : 0.0f);
        }
    }
    return output;
}

Tensor linear(const Tensor& input, const utils::Weight& weight) {
    return linearKernel(input, weight, utils::Weight{}, false);
}

Tensor linear(const Tensor& input, const utils::Weight& weight, const utils::Weight& bias) {
    return linearKernel(input, weight, bias, true);
}

Tensor rmsNorm(const Tensor& input, const utils::Weight& scale, float epsilon) {
    input.validate();
    if (input.shape.empty() || input.shape.back() == 0 || epsilon <= 0.0f
        || scale.shape != std::vector<std::size_t>{input.shape.back()}
        || scale.values.size() != input.shape.back()) {
        throw std::invalid_argument("rmsNorm: invalid shape or epsilon");
    }
    Tensor output(input.shape);
    const std::size_t width = input.shape.back();
    for (std::size_t row = 0; row < input.size() / width; ++row) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < width; ++j) {
            const float value = input.values[row * width + j];
            sum += value * value;
        }
        const float inverse = 1.0f / std::sqrt(sum / static_cast<float>(width) + epsilon);
        for (std::size_t j = 0; j < width; ++j) {
            output.values[row * width + j] = input.values[row * width + j] * inverse * scale.values[j];
        }
    }
    return output;
}

void attention(const AttentionView& view, const AttentionMask& mask) {
    const float factor = 1.0f / std::sqrt(static_cast<float>(view.headDim));
    std::vector<float> scores(view.keyCount, 0.0f);
    for (std::size_t query = 0; query < view.queryCount; ++query) {
        const std::size_t queryOffset = view.queryOffset + query * view.queryStride;
        for (std::size_t key = 0; key < view.keyCount; ++key) {
            if (!mask.allowed(query, key)) {
                scores[key] = -std::numeric_limits<float>::infinity();
                continue;
            }
            float sum = 0.0f;
            const bool prefix = key < view.prefixCount;
            const auto& keys = prefix ? view.prefixKeys : view.keys;
            const std::size_t keyOffset = prefix
                ? view.prefixOffset + key * view.prefixStride
                : view.keyOffset + (key - view.prefixCount) * view.keyStride;
            for (std::size_t j = 0; j < view.headDim; ++j) {
                sum += view.query[queryOffset + j] * keys[keyOffset + j];
            }
            scores[key] = sum * factor;
        }
        softmax(scores);
        const std::size_t outputOffset = view.outputOffset + query * view.outputStride;
        std::fill_n(view.output.begin() + outputOffset, view.headDim, 0.0f);
        // 每维仍按key顺序累加，改为连续读取一行V以复用缓存
        for (std::size_t key = 0; key < view.keyCount; ++key) {
            const bool prefix = key < view.prefixCount;
            const auto& values = prefix ? view.prefixValues : view.values;
            const std::size_t offset = prefix
                ? view.prefixOffset + key * view.prefixStride
                : view.valueOffset + (key - view.prefixCount) * view.valueStride;
            const float probability = scores[key];
            for (std::size_t j = 0; j < view.headDim; ++j) {
                view.output[outputOffset + j] += probability * values[offset + j];
            }
        }
    }
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

float softplus(float value) {
    return value > 20.0f ? value : std::log1p(std::exp(value));
}

void softmax(std::vector<float>& values) {
    softmax(values.begin(), values.end());
}

Tensor swiGLU(const Tensor& fused) {
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

void rotary(
    Tensor& input,
    const std::vector<double>& positionIds,
    std::size_t rotaryDim,
    float theta
) {
    input.validate();
    if (input.shape.size() != 4 || input.shape[3] % 2 != 0 || rotaryDim == 0
        || rotaryDim % 2 != 0 || rotaryDim > input.shape[3] || !std::isfinite(theta) || theta <= 0.0f
        || positionIds.size() != elementCount({input.shape[0], input.shape[1]})) {
        throw std::invalid_argument("rotary: invalid shape, positions or rotary parameters");
    }
    for (double position : positionIds) {
        if (!std::isfinite(static_cast<float>(position))) {
            throw std::invalid_argument("rotary: positions must be finite FP32 coordinates");
        }
    }
    const std::size_t heads = input.shape[2];
    const std::size_t dim = input.shape[3];
    for (std::size_t token = 0; token < positionIds.size(); ++token) {
        for (std::size_t j = 0; j < rotaryDim / 2; ++j) {
            const float inverse = 1.0f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(rotaryDim));
            const float angle = static_cast<float>(positionIds[token]) * inverse;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            for (std::size_t h = 0; h < heads; ++h) {
                const std::size_t index = (token * heads + h) * dim + j;
                const float first = input.values[index];
                const float second = input.values[index + dim / 2];
                input.values[index] = first * cosine - second * sine;
                input.values[index + dim / 2] = second * cosine + first * sine;
            }
        }
    }
}

Tensor monotoneQuantiles(const Tensor& raw, const std::vector<float>& levels, float widthFactor) {
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

} // namespace timestral::reference
