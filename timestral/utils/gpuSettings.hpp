/***************
 * @file gpuSettings.hpp
 * @brief 可选GPU后端共用的运行期配置解析
 ***************/

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace timestral {

struct GpuSettings {
    std::filesystem::path shaderDirectory; // 显式目录优先，其次发布目录与构建默认值
    std::size_t deviceIndex = 0;           // 所选API的设备枚举号
    bool validation = false;              // 显式启用驱动校验
};

// 环境变量统一为TIMESTRAL_GPU_SHADER_DIR、TIMESTRAL_GPU_DEVICE和TIMESTRAL_GPU_VALIDATION
// 非法设备号或布尔值直接拒绝，不静默回退，设备顺序由各API定义
GpuSettings loadGpuSettings(
    const std::string& api,
    const std::filesystem::path& buildShaderDirectory,
    std::size_t defaultDevice = 0
);

} // namespace timestral
