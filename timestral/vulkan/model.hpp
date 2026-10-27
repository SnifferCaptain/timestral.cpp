/***************
 * @file model.hpp
 * @brief 使用Vulkan常驻执行器的共享GPU模型入口
 ***************/

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "timestral/utils/gpuTransformer.hpp"

namespace timestral::vulkan {

// 构建生成的默认路径，发布目录与运行期环境变量优先
std::filesystem::path defaultShaderDirectory();

std::unique_ptr<timestral::Model> makeModel(
    ModelConfig config,
    utils::Safetensors weights,
    const std::filesystem::path& shaders,
    std::size_t deviceIndex = 0,
    bool validation = false,
    std::size_t gpuThreads = 0
);

} // namespace timestral::vulkan
