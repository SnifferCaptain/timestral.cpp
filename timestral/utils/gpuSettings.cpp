/***************
 * @file gpuSettings.cpp
 * @brief GPU设备号、shader目录与校验开关的严格解析
 ***************/

#include "timestral/utils/gpuSettings.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace timestral {

static std::string environment(const char* name, bool& present) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4996)
#endif
    const auto value = std::getenv(name);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    present = value != nullptr;
    return present ? std::string(value) : std::string{};
}

GpuSettings loadGpuSettings(
    const std::string& api,
    const std::filesystem::path& buildShaderDirectory,
    std::size_t defaultDevice
) {
    GpuSettings settings;
    settings.deviceIndex = defaultDevice;
    bool hasShader = false;
    const auto shader = environment("TIMESTRAL_GPU_SHADER_DIR", hasShader);
    if (hasShader) {
        if (shader.empty()) {
            throw std::invalid_argument("gpu: TIMESTRAL_GPU_SHADER_DIR is empty");
        }
        settings.shaderDirectory = std::filesystem::u8path(shader);
    } else {
        const auto installed = std::filesystem::path("share") / "timestral" / api;
        settings.shaderDirectory = std::filesystem::is_directory(installed) ? installed : buildShaderDirectory;
    }
    bool hasDevice = false;
    const auto device = environment("TIMESTRAL_GPU_DEVICE", hasDevice);
    if (hasDevice) {
        const std::string text(device);
        if (text.empty()) {
            throw std::invalid_argument("gpu: TIMESTRAL_GPU_DEVICE is empty");
        }
        settings.deviceIndex = 0;
        for (const char digit : text) {
            if (digit < '0' || digit > '9') {
                throw std::invalid_argument("gpu: TIMESTRAL_GPU_DEVICE must contain decimal digits only");
            }
            const auto value = static_cast<std::size_t>(digit - '0');
            if (settings.deviceIndex > ((std::numeric_limits<std::size_t>::max)() - value) / 10) {
                throw std::invalid_argument("gpu: TIMESTRAL_GPU_DEVICE overflows size_t");
            }
            settings.deviceIndex = settings.deviceIndex * 10 + value;
        }
    }
    bool hasValidation = false;
    const auto validation = environment("TIMESTRAL_GPU_VALIDATION", hasValidation);
    if (hasValidation) {
        const std::string text(validation);
        if (text != "0" && text != "1") {
            throw std::invalid_argument("gpu: TIMESTRAL_GPU_VALIDATION must be 0 or 1");
        }
        settings.validation = text == "1";
    }
    return settings;
}

} // namespace timestral
