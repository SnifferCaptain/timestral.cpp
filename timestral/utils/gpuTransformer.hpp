/***************
 * @file gpuTransformer.hpp
 * @brief GPU后端共用的设备常驻Transformer层编排
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "timestral/utils/gpuExecutor.hpp"
#include "timestral/utils/transformer.hpp"

namespace timestral {

// 归一化、页键、条件与预测流程沿用Transformer，仅数值层使用GPU
class GpuTransformer : public Transformer {
public:
    GpuTransformer(
        ModelConfig config,
        utils::Safetensors weights,
        std::shared_ptr<GpuExecutor> executor,
        bool residentResidual = false
    );

protected:
    Tensor residual(
        const Tensor& input,
        const std::string& name,
        bool bias,
        ProjectionPhase phase
    ) const override;
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
    void runLayers(
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
    ) const override;

private:
    std::size_t deviceProject(std::size_t input, std::size_t rows, const std::string& name, bool bias) const;
    std::size_t deviceNormalize(std::size_t input, std::size_t rows, const std::string& scale, float epsilon) const;
    std::size_t elementwise(std::size_t first, std::size_t second, std::size_t count, std::size_t width, std::uint32_t mode) const;

    std::shared_ptr<GpuExecutor> executor;
    std::map<std::string, std::size_t> deviceWeights;
    std::size_t zero = 0;
    bool residentResidual = false;
    mutable std::recursive_mutex executionMutex;
};

} // namespace timestral
