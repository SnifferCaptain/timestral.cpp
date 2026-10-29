/***************
 * @file backend.cpp
 * @brief 可选Vulkan模型工厂及统一GPU运行期配置
 ***************/

#include "timestral/utils/backend.hpp"

#include <memory>
#include <utility>

#include "timestral/reference/model.hpp"
#include "timestral/utils/gpuSettings.hpp"
#include "timestral/vulkan/model.hpp"

namespace timestral {

std::shared_ptr<const Model> loadBackendModel(
    const ModelConfig& config,
    utils::Safetensors&& weights,
    std::size_t,
    std::size_t gpuThreads,
    std::function<bool(std::size_t, std::size_t)> timeMask
) {
    // 附加谓词只在共用Transformer前向中实现，本后端改用reference计算，速度明显下降
    if (timeMask) {
        auto model = std::make_shared<reference::Model>(config, std::move(weights));
        model->setTimeMask(std::move(timeMask));
        return model;
    }
    const auto settings = loadGpuSettings("vulkan", vulkan::defaultShaderDirectory());
    return std::shared_ptr<const Model>(vulkan::makeModel(
        config, std::move(weights), settings.shaderDirectory, settings.deviceIndex, settings.validation, gpuThreads
    ));
}

} // namespace timestral
