/***************
 * @file model.cpp
 * @brief Vulkan模型构造与设备执行器所有权
 ***************/

#include "timestral/vulkan/model.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "timestral/vulkan/executor.hpp"

namespace timestral::vulkan {

std::unique_ptr<timestral::Model> makeModel(
    ModelConfig config,
    utils::Safetensors weights,
    const std::filesystem::path& shaders,
    std::size_t deviceIndex,
    bool validation,
    std::size_t gpuThreads
) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4996)
#endif
    const char* setting = std::getenv("TIMESTRAL_VULKAN_OPTIMIZATIONS");
    const char* residualSetting = std::getenv("TIMESTRAL_VULKAN_RESIDENT_RESIDUAL");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    const std::string mode = setting != nullptr ? setting : "1";
    if (mode != "0" && mode != "1") {
        throw std::invalid_argument("vulkan: TIMESTRAL_VULKAN_OPTIMIZATIONS must be 0 or 1");
    }
    const std::string residualMode = residualSetting != nullptr ? residualSetting : "0";
    if (residualMode != "0" && residualMode != "1") {
        throw std::invalid_argument("vulkan: TIMESTRAL_VULKAN_RESIDENT_RESIDUAL must be 0 or 1");
    }
    auto executor = std::make_shared<Executor>(shaders, deviceIndex, validation, mode == "1", gpuThreads);
    return std::make_unique<GpuTransformer>(
        std::move(config),
        std::move(weights),
        std::move(executor),
        mode == "1" && residualMode == "1"
    );
}

} // namespace timestral::vulkan
