/***************
 * @file model.cpp
 * @brief 公共Transformer的标量投影、RMS、QKV布局与注意力
 * 移植自Mistral AI Timestral layers.py，Apache-2.0，基线a534bdbd
 ***************/

#include "timestral/reference/model.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "timestral/reference/kernels.hpp"

namespace timestral::reference {

Model::Model(ModelConfig config, utils::Safetensors weights)
    : Transformer(std::move(config), std::move(weights)) {
}

Tensor Model::project(const Tensor& input, const std::string& name, bool bias) const {
    if (bias) {
        return linear(input, weights.at(name + ".weight"), weights.at(name + ".bias"));
    }
    return linear(input, weights.at(name + ".weight"));
}

Tensor Model::normalize(const Tensor& input, const utils::Weight& scale, float epsilon) const {
    return rmsNorm(input, scale, epsilon);
}

void Model::prepareQkv(
    const Tensor& fused,
    Tensor& q,
    Tensor& k,
    Tensor& v,
    const std::vector<double>& positions,
    bool isTime
) const {
    const std::size_t modelDim = modelConfig.modelDim;
    // 发布权重按(qkv,headDim,headCount)排列，注意力按(headCount,headDim)展平
    for (std::size_t token = 0; token < q.size() / modelDim; ++token) {
        for (std::size_t head = 0; head < modelConfig.headCount; ++head) {
            for (std::size_t j = 0; j < modelConfig.headDim(); ++j) {
                const std::size_t source = token * 3 * modelDim + j * modelConfig.headCount + head;
                const std::size_t dest = token * modelDim + head * modelConfig.headDim() + j;
                q.values[dest] = fused.values[source];
                k.values[dest] = fused.values[source + modelDim];
                v.values[dest] = fused.values[source + 2 * modelDim];
            }
        }
    }
    if (isTime) {
        rotary(q, positions, modelConfig.ropeDim(), modelConfig.ropeTheta);
        rotary(k, positions, modelConfig.ropeDim(), modelConfig.ropeTheta);
    }
}

void Model::attention(const AttentionView& view, const AttentionMask& mask) const {
    reference::attention(view, mask);
}

bool Model::broadcastPrefix() const {
    return true;
}

std::unique_ptr<timestral::Model> makeModel(ModelConfig config, utils::Safetensors weights) {
    return std::make_unique<Model>(std::move(config), std::move(weights));
}

} // namespace timestral::reference
