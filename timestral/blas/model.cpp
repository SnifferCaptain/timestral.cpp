/***************
 * @file model.cpp
 * @brief Timestral专用预打包投影、布局融合与在线注意力
 ***************/

#include "timestral/blas/model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "timestral/blas/attention.hpp"
#include "timestral/blas/elementwise.hpp"
#include "timestral/reference/kernels.hpp"
#include "timestral/utils/patch.hpp"

namespace timestral::blas {

Model::Model(ModelConfig config, utils::Safetensors loadedWeights, DecodeFormat format, std::size_t threads)
    : Transformer(std::move(config), std::move(loadedWeights)),
      decodeFormat(format),
      parallel(threads) {
    if (format != DecodeFormat::Float32 && format != DecodeFormat::Float16
        && format != DecodeFormat::Bfloat16 && format != DecodeFormat::Int8) {
        throw std::invalid_argument("blas model: invalid decode format");
    }
    std::string lastTime;
    for (std::size_t layer = 0; layer < modelConfig.blockSchedule.size(); ++layer) {
        const auto axes = attentionAxes(modelConfig.blockSchedule[layer]);
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            if (axes[axis]) {
                lastTime = "transformer.layers." + std::to_string(layer) + ".attns." + std::to_string(axis);
            }
        }
    }
    for (const auto& entry : weights.tensors()) {
        const auto& weight = entry.second;
        if (weight.shape.size() != 2) {
            continue;
        }
        const std::string suffix = ".wQKV.weight";
        if (entry.first.size() >= suffix.size()
            && entry.first.compare(entry.first.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const std::string attn = entry.first.substr(0, entry.first.size() - suffix.size());
            if (attn == lastTime) {
                const std::size_t width = modelConfig.modelDim;
                packedWeights.emplace(
                    attn + ".wKV.weight",
                    PackedWeight(MatrixView(weight.values, width, 2 * width, width * width, 1, width))
                );
            }
            if (weights.tensors().count(attn + "._gate.weight") != 0) {
                const auto& gate = weights.at(attn + "._gate.weight");
                std::vector<float> combined(weight.values);
                combined.insert(combined.end(), gate.values.begin(), gate.values.end());
                packedWeights.emplace(
                    attn + ".wQKVGate.weight",
                    PackedWeight(MatrixView(
                        combined,
                        weight.shape[1],
                        weight.shape[0] + gate.shape[0],
                        0,
                        1,
                        weight.shape[1]
                    ))
                );
                continue;
            }
        }
        const std::string gateSuffix = "._gate.weight";
        if (entry.first.size() >= gateSuffix.size()
            && entry.first.compare(entry.first.size() - gateSuffix.size(), gateSuffix.size(), gateSuffix) == 0) {
            continue;
        }
        packedWeights.emplace(
            entry.first,
            PackedWeight(MatrixView(
                weight.values,
                weight.shape[1],
                weight.shape[0],
                0,
                1,
                weight.shape[1]
            ))
        );
    }
    if (decodeFormat != DecodeFormat::Float32) {
        for (const auto& entry : packedWeights) {
            // 历史patch嵌入与prefill专用KV投影始终使用FP32，不分配无用压缩副本
            if (entry.first.compare(0, 15, "representation.") != 0
                && entry.first.find(".wKV.weight") == std::string::npos) {
                decodeWeights.emplace(entry.first, DecodeWeight(entry.second, decodeFormat));
            }
        }
    }
    const auto& configValues = modelConfig;
    rotaryInverse.resize(configValues.ropeDim() / 2);
    for (std::size_t j = 0; j < rotaryInverse.size(); ++j) {
        rotaryInverse[j] = 1.0f / std::pow(
            configValues.ropeTheta,
            static_cast<float>(2 * j) / static_cast<float>(configValues.ropeDim())
        );
    }
}

std::size_t Model::threadCount() const {
    return parallel.count();
}

void Model::attentionWork(std::size_t tasks, const std::function<void(std::size_t)>& function) const {
    parallel.run(tasks, function);
}

Tensor Model::project(const Tensor& input, const std::string& name, bool bias) const {
    input.validate();
    const auto& weight = packedWeights.at(name + ".weight");
    if (input.shape.empty() || input.shape.back() != weight.rows() || weight.rows() == 0) {
        throw std::invalid_argument("blas linear: incompatible input or weight shape: " + name);
    }
    auto shape = input.shape;
    shape.back() = weight.columns();
    Tensor result(shape);
    const std::size_t rows = input.size() / weight.rows();
    gemmPacked(
        MatrixView(input.values, rows, weight.rows()),
        weight,
        MutableMatrixView(result.values, rows, weight.columns()),
        parallel, 0.0f, GemmMode::Ordered
    );
    if (bias) {
        const auto& values = weights.at(name + ".bias").values;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t j = 0; j < values.size(); ++j) {
                result.values[row * values.size() + j] += values[j];
            }
        }
    }
    return result;
}

Tensor Model::projectPhase(
    const Tensor& input,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    input.validate();
    const auto& weight = packedWeights.at(name + ".weight");
    if (input.shape.empty() || weight.rows() == 0 || input.shape.back() != weight.rows()) {
        throw std::invalid_argument("blas decode: incompatible input or weight shape: " + name);
    }
    const std::size_t rows = input.size() / weight.rows();
    const auto compressed = decodeWeights.find(name + ".weight");
    if (phase != ProjectionPhase::Decode || decodeFormat == DecodeFormat::Float32
        || rows > 4 || compressed == decodeWeights.end()) {
        return project(input, name, bias);
    }
    auto shape = input.shape;
    shape.back() = weight.columns();
    Tensor output(shape);
    decodeProject(
        MatrixView(input.values, rows, weight.rows()),
        compressed->second,
        MutableMatrixView(output.values, rows, weight.columns()),
        parallel
    );
    if (bias) {
        const auto& values = weights.at(name + ".bias").values;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t j = 0; j < values.size(); ++j) {
                output.values[row * values.size() + j] += values[j];
            }
        }
    }
    return output;
}

static std::vector<float> inverseRms(const Tensor& input, const utils::Weight& scale, float epsilon) {
    input.validate();
    if (input.shape.empty() || input.shape.back() == 0 || epsilon <= 0.0f
        || scale.shape != std::vector<std::size_t>{input.shape.back()} || scale.values.size() != input.shape.back()) {
        throw std::invalid_argument("blas RMS packing: invalid shape or epsilon");
    }
    const std::size_t width = input.shape.back();
    std::vector<float> inverse(input.size() / width);
    for (std::size_t row = 0; row < inverse.size(); ++row) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < width; ++j) {
            const float value = input.values[row * width + j];
            sum += value * value;
        }
        inverse[row] = 1.0f / std::sqrt(sum / static_cast<float>(width) + epsilon);
    }
    return inverse;
}

Tensor Model::projectNormalized(
    const Tensor& input,
    const utils::Weight& scale,
    float epsilon,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    input.validate();
    if (phase == ProjectionPhase::Decode && decodeFormat != DecodeFormat::Float32
        && !input.shape.empty() && input.shape.back() != 0
        && input.size() / input.shape.back() <= 4) {
        return projectPhase(normalize(input, scale, epsilon), name, bias, phase);
    }
    const auto inverse = inverseRms(input, scale, epsilon);
    const auto& packed = packedWeights.at(name + ".weight");
    auto shape = input.shape;
    shape.back() = packed.columns();
    Tensor output(shape);
    gemmNormalized(
        MatrixView(input.values, inverse.size(), input.shape.back()),
        packed,
        MutableMatrixView(output.values, inverse.size(), packed.columns()),
        NormPacking{inverse, scale.values},
        parallel,
        GemmMode::Ordered
    );
    if (bias) {
        const auto& values = weights.at(name + ".bias").values;
        for (std::size_t row = 0; row < inverse.size(); ++row) {
            for (std::size_t j = 0; j < values.size(); ++j) {
                output.values[row * values.size() + j] += values[j];
            }
        }
    }
    return output;
}

void Model::projectAttention(
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
    if (!gated) {
        fused = projectNormalized(input, scale, epsilon, attn + ".wQKV", bias, phase);
        gate = Tensor();
        return;
    }
    // fused保留(…,4D)行宽供prepareQkv跨行读取，只把gate的D列拷出
    fused = projectNormalized(input, scale, epsilon, attn + ".wQKVGate", false, phase);
    const std::size_t width = modelConfig.modelDim;
    const std::size_t rows = input.size() / width;
    gate = Tensor(input.shape);
    for (std::size_t row = 0; row < rows; ++row) {
        std::copy_n(fused.values.begin() + row * 4 * width + 3 * width, width, gate.values.begin() + row * width);
    }
    if (bias) {
        const auto& qkvBias = weights.at(attn + ".wQKV.bias").values;
        const auto& gateBias = weights.at(attn + "._gate.bias").values;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t j = 0; j < 3 * width; ++j) {
                fused.values[row * 4 * width + j] += qkvBias[j];
            }
            for (std::size_t j = 0; j < width; ++j) {
                gate.values[row * width + j] += gateBias[j];
            }
        }
    }
}

Tensor Model::normalize(const Tensor& input, const utils::Weight& scale, float epsilon) const {
    return reference::rmsNorm(input, scale, epsilon);
}

void Model::prepareQkv(
    const Tensor& fused,
    Tensor& q,
    Tensor& k,
    Tensor& v,
    const std::vector<double>& positions,
    bool isTime
) const {
    const std::size_t dim = modelConfig.headDim();
    const std::size_t heads = modelConfig.headCount;
    const std::size_t width = modelConfig.modelDim;
    const std::size_t tokens = positions.size();
    const std::size_t fusedWidth = fused.shape.empty() ? 0 : fused.shape.back();
    if ((fusedWidth != 3 * width && fusedWidth != 4 * width) || fused.values.size() != tokens * fusedWidth) {
        throw std::invalid_argument("blas QKV layout: fused width must be 3D or 4D per token");
    }
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t j = 0; j < dim; ++j) {
                const std::size_t source = token * fusedWidth + j * heads + head;
                const std::size_t dest = token * width + head * dim + j;
                q.values[dest] = fused.values[source];
                k.values[dest] = fused.values[source + width];
                v.values[dest] = fused.values[source + 2 * width];
            }
        }
        if (!isTime) {
            continue;
        }
        if (!std::isfinite(static_cast<float>(positions[token]))) {
            throw std::invalid_argument("blas rotary: positions must be finite FP32 coordinates");
        }
        for (std::size_t j = 0; j < rotaryInverse.size(); ++j) {
            const float angle = static_cast<float>(positions[token]) * rotaryInverse[j];
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            // partial RoPE配对headDim两半，不是ropeDim两半或交错配对
            for (std::size_t head = 0; head < heads; ++head) {
                const std::size_t index = token * width + head * dim + j;
                const float qFirst = q.values[index];
                const float qSecond = q.values[index + dim / 2];
                q.values[index] = qFirst * cosine - qSecond * sine;
                q.values[index + dim / 2] = qSecond * cosine + qFirst * sine;
                const float kFirst = k.values[index];
                const float kSecond = k.values[index + dim / 2];
                k.values[index] = kFirst * cosine - kSecond * sine;
                k.values[index + dim / 2] = kSecond * cosine + kFirst * sine;
            }
        }
    }
}

void Model::prepareCacheKv(
    const Tensor& input,
    const utils::Weight& scale,
    float epsilon,
    const std::string& attn,
    bool bias,
    Tensor& k,
    Tensor& v,
    const std::vector<double>& positions
) const {
    Tensor fused = projectNormalized(input, scale, epsilon, attn + ".wKV", false);
    const std::size_t width = modelConfig.modelDim;
    const std::size_t dim = modelConfig.headDim();
    const std::size_t heads = modelConfig.headCount;
    if (bias) {
        const auto& values = weights.at(attn + ".wQKV.bias").values;
        for (std::size_t token = 0; token < positions.size(); ++token) {
            for (std::size_t j = 0; j < 2 * width; ++j) {
                fused.values[token * 2 * width + j] += values[width + j];
            }
        }
    }
    for (std::size_t token = 0; token < positions.size(); ++token) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t j = 0; j < dim; ++j) {
                const std::size_t source = token * 2 * width + j * heads + head;
                const std::size_t dest = token * width + head * dim + j;
                k.values[dest] = fused.values[source];
                v.values[dest] = fused.values[source + width];
            }
        }
        if (!std::isfinite(static_cast<float>(positions[token]))) {
            throw std::invalid_argument("blas KV rotary: positions must be finite FP32 coordinates");
        }
        for (std::size_t j = 0; j < rotaryInverse.size(); ++j) {
            const float angle = static_cast<float>(positions[token]) * rotaryInverse[j];
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            for (std::size_t head = 0; head < heads; ++head) {
                const std::size_t index = token * width + head * dim + j;
                const float first = k.values[index];
                const float second = k.values[index + dim / 2];
                k.values[index] = first * cosine - second * sine;
                k.values[index + dim / 2] = second * cosine + first * sine;
            }
        }
    }
}

void Model::attention(const AttentionView& view, const AttentionMask& mask) const {
    // 单条查询直接读分段KV，避免为GEMV重复打包整份历史
    if (view.queryCount == 1) {
        reference::attention(view, mask);
    } else {
        orderedAttention(view, mask);
    }
}

void Model::applyGate(Tensor& attended, const Tensor& gate) const {
    if (attended.shape != gate.shape) {
        throw std::invalid_argument("blas gate: shape mismatch");
    }
    const std::size_t count = attended.size();
    if (parallel.count() == 1) {
        sigmoidGateScalar(attended.values.data(), gate.values.data(), count);
        return;
    }
    parallel.run(count / 1024 + (count % 1024 != 0), [&](std::size_t task) {
        const auto begin = task * 1024;
        sigmoidGateScalar(attended.values.data() + begin, gate.values.data() + begin, std::min(std::size_t{1024}, count - begin));
    });
}

void Model::projectResidual(
    Tensor& x,
    const Tensor& input,
    const std::string& name,
    bool bias,
    ProjectionPhase phase
) const {
    Transformer::projectResidual(x, input, name, bias, phase);
}

Tensor Model::activate(const Tensor& fused) const {
    fused.validate();
    if (fused.shape.empty() || fused.shape.back() == 0 || fused.shape.back() % 2 != 0) {
        throw std::invalid_argument("blas swiGLU: expected an even nonzero final dimension");
    }
    auto shape = fused.shape;
    const std::size_t width = shape.back() / 2;
    shape.back() = width;
    Tensor output(shape);
    const auto rows = output.values.size() / width;
    if (parallel.count() == 1) {
        swiGLUScalar(fused.values.data(), output.values.data(), rows, width);
    } else {
        parallel.run(rows, [&](std::size_t row) {
            swiGLUScalar(fused.values.data() + row * 2 * width, output.values.data() + row * width, 1, width);
        });
    }
    return output;
}

bool Model::optimizedPrefill() const {
    return true;
}

bool Model::broadcastPrefix() const {
    return true;
}

} // namespace timestral::blas
