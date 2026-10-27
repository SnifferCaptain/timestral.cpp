/***************
 * @file model.hpp
 * @brief 专用BLAS数值后端与只读预打包权重
 ***************/

#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "timestral/blas/decode.hpp"
#include "timestral/blas/gemm.hpp"
#include "timestral/blas/parallel.hpp"
#include "timestral/utils/transformer.hpp"

namespace timestral::blas {

class Model final : public Transformer {
public:
    Model(
        ModelConfig config,
        utils::Safetensors loadedWeights,
        DecodeFormat format = defaultDecodeFormat(),
        std::size_t threads = 1
    );
    std::size_t threadCount() const override;

protected:
    void attentionWork(std::size_t tasks, const std::function<void(std::size_t)>& function) const override;
    Tensor project(const Tensor& input, const std::string& name, bool bias) const override;
    Tensor projectPhase(
        const Tensor& input,
        const std::string& name,
        bool bias,
        ProjectionPhase phase
    ) const override;
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
    void applyGate(Tensor& attended, const Tensor& gate) const override;
    Tensor activate(const Tensor& fused) const override;
    void projectResidual(
        Tensor& x,
        const Tensor& input,
        const std::string& name,
        bool bias,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const override;
    void prepareCacheKv(
        const Tensor& input,
        const utils::Weight& scale,
        float epsilon,
        const std::string& attn,
        bool bias,
        Tensor& k,
        Tensor& v,
        const std::vector<double>& positions
    ) const override;
    bool optimizedPrefill() const override;
    bool broadcastPrefix() const override;
    Tensor projectNormalized(
        const Tensor& input,
        const utils::Weight& scale,
        float epsilon,
        const std::string& name,
        bool bias,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const override;
    void projectAttention(
        const Tensor& input,
        const utils::Weight& scale,
        float epsilon,
        const std::string& attn,
        bool bias,
        bool gated,
        Tensor& fused,
        Tensor& gate,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const override;

private:
    const DecodeFormat decodeFormat;
    mutable Parallel parallel;
    std::map<std::string, PackedWeight> packedWeights;
    std::map<std::string, DecodeWeight> decodeWeights;
    std::vector<float> rotaryInverse;
};

} // namespace timestral::blas
