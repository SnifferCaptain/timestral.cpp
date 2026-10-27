/***************
 * @file kernels.hpp
 * @brief Vulkan FP32算子验证入口与显式资源所有权
 ***************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace timestral::vulkan {

struct Buffer {
    vk::UniqueDeviceMemory memory; // 先声明内存，确保buffer先销毁
    vk::UniqueBuffer handle;       // 拥有设备buffer
    vk::DeviceSize bytes = 0;      // 描述符可见字节数
    vk::DeviceSize capacity = 0;   // 可复用分配的实际容量
    vk::MemoryPropertyFlags memoryFlags; // 实际绑定内存类型的属性
};

struct Kernel {
    vk::UniqueDescriptorSetLayout descriptors; // binding按输入顺序加末尾输出
    vk::UniquePipelineLayout layout;           // 包含push常量布局
    vk::UniquePipeline pipeline;               // 先于布局销毁
    std::uint32_t bindingCount = 0;            // storage buffer总数
    std::uint32_t pushBytes = 0;               // push常量字节数
    std::uint32_t localInvocations = 0;         // SPIR-V固定工作组线程数
    std::array<std::uint32_t, 3> localSize = {0, 0, 0}; // 固定工作组各轴大小
    bool flatDispatch = false;                // NumWorkgroups仅作展平步距
};

// 仅供内部算子验收，当前每次调用上传与回读，不作为整模型GPU加速接口
class Kernels {
public:
    // @param shaderDirectory 包含四个Vulkan1.1 SPIR-V文件的目录
    // @param deviceIndex Vulkan枚举顺序中的设备号，无设备或越界直接拒绝
    // @param validation 显式要求Khronos校验层，缺失时构造失败
    explicit Kernels(
        const std::filesystem::path& shaderDirectory,
        std::size_t deviceIndex = 0,
        bool validation = false
    );

    std::string deviceName() const;

    // @brief C=input*weights转置+bias，三矩阵均为连续行主序，空bias表示不加偏置
    std::vector<float> linear(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& bias,
        std::size_t rows,
        std::size_t columns,
        std::size_t inner
    );

    std::vector<float> rmsNorm(
        const std::vector<float>& input,
        const std::vector<float>& scale,
        std::size_t rows,
        std::size_t width,
        float epsilon
    );

    // @brief 稳定行softmax，允许负无穷掩码，全屏蔽行输出0，NaN与正无穷拒绝
    std::vector<float> softmax(const std::vector<float>& input, std::size_t rows, std::size_t width);
    std::vector<float> silu(const std::vector<float>& input);

protected:
    Buffer createBuffer(const std::vector<float>& values, bool preferCached = false) const;
    Kernel createKernel(
        const std::filesystem::path& path,
        std::uint32_t bindingCount,
        std::uint32_t pushBytes,
        std::uint32_t logicalGridWidth = 0
    ) const;
    void checkCount(std::size_t count) const;
    void checkGroups(const std::array<std::uint32_t, 3>& groups) const;
    std::vector<float> run(
        const Kernel& kernel,
        const std::vector<std::reference_wrapper<const std::vector<float>>>& inputs,
        std::size_t outputCount,
        const std::vector<std::uint32_t>& push,
        const std::array<std::uint32_t, 3>& groups
    );

    vk::UniqueInstance instance;
    vk::PhysicalDevice physical;
    vk::PhysicalDeviceProperties properties;
    vk::UniqueDevice device;
    vk::Queue queue;
    vk::UniqueCommandPool commandPool;
    std::uint32_t queueTimestampBits = 0;
    Kernel linearKernel;
    Kernel normKernel;
    Kernel softmaxKernel;
    Kernel siluKernel;
    std::mutex submissionMutex;
};

} // namespace timestral::vulkan
