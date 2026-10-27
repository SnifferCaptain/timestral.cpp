/***************
 * @file executor.hpp
 * @brief Vulkan设备常驻权重、工作区及批命令执行
 ***************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "timestral/utils/gpuExecutor.hpp"
#include "timestral/vulkan/kernels.hpp"
#include "timestral/vulkan/profiling.hpp"

namespace timestral::vulkan {

class Executor final : public GpuExecutor, private Kernels {
public:
    explicit Executor(
        const std::filesystem::path& shaders,
        std::size_t deviceIndex = 0,
        bool validation = false,
        bool optimized = false,
        std::size_t gpuThreads = 0
    );
    ~Executor() override;

    // 可选kernel的设备上限不满足时不创建管线，模型选核保留v1
    bool supportsKernel(const std::string& kernel) const;

    // 仅允许在非batch状态切换/清零，统计默认关闭，不改变提交边界
    bool supportsProfiling() const;
    void setProfiling(bool enabled);
    void resetProfile();
    const GpuProfile& profile() const;
    std::uint32_t gpuThreadBudget() const;
    std::string profileDeviceName() const;
    std::uint32_t profileSubgroupSize() const;
    std::string profileMemoryTypes() const;

    void begin() override;
    std::size_t upload(const std::vector<float>& values, bool persistent = false) override;
    std::size_t allocate(std::size_t count) override;
    void dispatch(
        const std::string& kernel,
        const std::vector<std::size_t>& bindings,
        const std::vector<std::uint32_t>& push,
        const std::array<std::uint32_t, 3>& groups
    ) override;
    bool supportsCopy() const override;
    bool supportsPrefixAttention() const override;
    bool supportsGroupedAttention() const override;
    bool supportsOrderedAttention() const override;
    void copy(
        std::size_t source,
        std::size_t destination,
        std::size_t sourceOffset,
        std::size_t destinationOffset,
        std::size_t count
    ) override;
    std::vector<float> download(std::size_t buffer) override;
    void end() override;

private:
    Buffer createDeviceBuffer(std::size_t count) const;
    Buffer stagingBuffer(const std::vector<float>& values);
    Buffer readbackBuffer(const std::vector<float>& values);
    vk::CommandBuffer command();
    void submit();
    void barrier(vk::CommandBuffer command) const;

    std::uint32_t startTimestamp(
        vk::CommandBuffer recorded,
        const std::string& name,
        vk::PipelineStageFlagBits stage,
        std::uint64_t bytes = 0,
        double gemmFlops = 0.0
    );
    void endTimestamp(vk::CommandBuffer recorded, std::uint32_t first, vk::PipelineStageFlagBits stage) const;
    void collectTimestamps();

    static constexpr std::uint32_t timestampCapacity = 4096;
    vk::UniqueQueryPool timestampPool;
    std::vector<TimestampEvent> timestampEvents;
    std::uint32_t nextTimestamp = 2;
    std::uint32_t timestampBits = 0;
    std::uint32_t subgroupSize = 0;
    bool profiling = false;
    GpuProfile measured;
    std::map<std::string, Kernel> kernels;
    std::map<std::size_t, Buffer> buffers;
    std::vector<std::size_t> scratch;
    std::vector<Buffer> staging;
    std::vector<Buffer> reusableDevice;
    std::vector<Buffer> reusableStaging;
    std::vector<Buffer> reusableReadback;
    std::vector<vk::UniqueDescriptorPool> descriptorPools;
    std::vector<vk::UniqueCommandBuffer> commands;
    vk::UniqueFence submissionFence;
    std::size_t nextHandle = 1;
    std::filesystem::path shaderDirectory;
    std::uint32_t gpuThreads = 0;
    std::map<std::string, Kernel> budgetKernels;
    bool optimized = false;
    bool active = false;
    bool recording = false;
    bool submissionPending = false;
    bool poisoned = false;
};

} // namespace timestral::vulkan
