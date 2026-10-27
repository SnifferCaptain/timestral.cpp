/***************
 * @file backend.cpp
 * @brief 可选Vulkan模型工厂及统一GPU运行期配置
 ***************/

#include "timestral/utils/backend.hpp"

#include <memory>
#include <utility>

#include "timestral/utils/gpuSettings.hpp"
#include "timestral/vulkan/model.hpp"

namespace timestral {

std::shared_ptr<const Model> loadBackendModel(
    const ModelConfig& config,
    utils::Safetensors&& weights,
    std::size_t,
    std::size_t gpuThreads
) {
    const auto settings = loadGpuSettings("vulkan", vulkan::defaultShaderDirectory());
    return std::shared_ptr<const Model>(vulkan::makeModel(
        config, std::move(weights), settings.shaderDirectory, settings.deviceIndex, settings.validation, gpuThreads
    ));
}

} // namespace timestral
