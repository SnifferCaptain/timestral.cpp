/***************
 * @file model.hpp
 * @brief 公共Transformer的FP32参考数值后端
 * 移植自Mistral AI Timestral，Apache-2.0，基线a534bdbd
 ***************/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "timestral/utils/transformer.hpp"

namespace timestral::reference {

using ScaleResult = timestral::ScaleResult;
using ModelTrace = timestral::ModelTrace;

class Model final : public Transformer {
public:
    // 拥有配置与权重，公共执行器负责校验权重名称、形状与有限值
    Model(ModelConfig config, utils::Safetensors weights);

protected:
    Tensor project(const Tensor& input, const std::string& name, bool bias) const override;

    Tensor normalize(const Tensor& input, const utils::Weight& scale, float epsilon) const override;

    void prepareQkv(
        const Tensor& fused,
        Tensor& q,
        Tensor& k,
        Tensor& v,
        const std::vector<double>& positions,
        bool isTime
    ) const override;

    void attention(const AttentionView& view, const AttentionMask& mask) const override;
    bool broadcastPrefix() const override;
};

// 创建只读reference模型，配置与权重的所有权移交给模型
std::unique_ptr<timestral::Model> makeModel(ModelConfig config, utils::Safetensors weights);

} // namespace timestral::reference
