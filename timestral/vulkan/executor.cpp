/***************
 * @file executor.cpp
 * @brief device-local缓冲、staging传输及整图批提交
 ***************/

#include "timestral/vulkan/executor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace timestral::vulkan {

// 预算在基类创建Vulkan实例/设备之前校验，不用非法local_size试驱动
static const std::filesystem::path& checkedBudget(const std::filesystem::path& shaders, std::size_t gpuThreads) {
    if (gpuThreads != 0 && (gpuThreads % 256 != 0 || gpuThreads > (std::numeric_limits<std::uint32_t>::max)())) {
        throw std::invalid_argument("vulkan: gpuThreads must be 0 or a positive multiple of 256 within uint32 range");
    }
    return shaders;
}

// 按容量级复用，避免递增前缀为每种长度保留一份Vulkan分配
static std::size_t reserveCount(std::size_t count, std::size_t limit) {
    if (count == 0 || count > limit) {
        throw std::invalid_argument("vulkan: buffer size exceeds storage range");
    }
    std::size_t capacity = std::min<std::size_t>(256, limit);
    while (capacity < count) {
        if (capacity > limit / 2) {
            return count;
        }
        capacity *= 2;
    }
    return capacity;
}

// 只处理fence完成后的空闲缓冲，swap/pop保证buffer先于绑定内存销毁
static void trimBuffers(std::vector<Buffer>& buffers, std::size_t limit) {
    while (buffers.size() > limit) {
        std::size_t smallest = 0;
        for (std::size_t i = 1; i < buffers.size(); ++i) {
            if (buffers[i].capacity < buffers[smallest].capacity) {
                smallest = i;
            }
        }
        if (smallest + 1 != buffers.size()) {
            std::swap(buffers[smallest], buffers.back());
        }
        buffers.pop_back();
    }
}

Executor::Executor(
    const std::filesystem::path& shaders,
    std::size_t deviceIndex,
    bool validation,
    bool optimized,
    std::size_t gpuThreads
)
    : Kernels(checkedBudget(shaders, gpuThreads), deviceIndex, validation),
      shaderDirectory(shaders),
      gpuThreads(static_cast<std::uint32_t>(gpuThreads)),
      optimized(optimized) {
    if (properties.limits.maxPushConstantsSize < 36
        || properties.limits.maxPerStageDescriptorStorageBuffers < 5
        || properties.limits.maxDescriptorSetStorageBuffers < 5) {
        throw std::runtime_error("vulkan: device does not meet resident model shader limits");
    }
    const auto info = physical.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
    const auto& subgroup = info.get<vk::PhysicalDeviceSubgroupProperties>();
    subgroupSize = subgroup.subgroupSize;
    kernels.emplace("gemm", std::move(linearKernel));
    kernels.emplace("gemmTile", createKernel(shaders / "gemmTile.spv", 4, 16));
    kernels.emplace("gemv", createKernel(shaders / "gemv.spv", 4, 16));
    if (properties.limits.maxComputeSharedMemorySize >= 8256) {
        kernels.emplace("gemmWide", createKernel(shaders / "gemmWide.spv", 4, 16));
    }
    kernels.emplace("rmsNorm", std::move(normKernel));
    kernels.emplace("softmax", std::move(softmaxKernel));
    kernels.emplace("silu", std::move(siluKernel));
    kernels.emplace("siluGrid", createKernel(shaders / "siluGrid.spv", 2, 4));
    kernels.emplace("prepareQkv", createKernel(shaders / "prepareQkv.spv", 5, 20));
    kernels.emplace("mergeKv", createKernel(shaders / "mergeKv.spv", 3, 16));
    kernels.emplace("elementwise", createKernel(shaders / "elementwise.spv", 3, 12));
    kernels.emplace("attentionScores", createKernel(shaders / "attentionScores.spv", 4, 36));
    kernels.emplace("attentionValues", createKernel(shaders / "attentionValues.spv", 3, 36));
    kernels.emplace("attentionFused", createKernel(shaders / "attentionFused.spv", 5, 36));
    kernels.emplace("attentionGroup", createKernel(shaders / "attentionGroup.spv", 5, 36));
    if (properties.limits.maxPushConstantsSize >= 40 && properties.limits.maxComputeSharedMemorySize >= 8200) {
        kernels.emplace("attentionOrdered", createKernel(shaders / "attentionOrdered.spv", 5, 40));
        kernels.emplace("rmsNormOrdered", createKernel(shaders / "rmsNormOrdered.spv", 3, 12));
        kernels.emplace("elementwiseOrdered", createKernel(shaders / "elementwiseOrdered.spv", 3, 12));
    }
    kernels.emplace("attentionRegister", createKernel(shaders / "attentionRegister.spv", 5, 36));
    if (properties.limits.maxPushConstantsSize >= 40) {
        kernels.emplace("attentionPrefix", createKernel(shaders / "attentionPrefix.spv", 5, 40));
    }
    if (properties.limits.maxPushConstantsSize >= 40 && properties.limits.maxComputeSharedMemorySize >= 17408) {
        kernels.emplace("attentionDecode", createKernel(shaders / "attentionDecode.spv", 5, 40));
        kernels.emplace("attentionDecodeMerge", createKernel(shaders / "attentionDecodeMerge.spv", 2, 12));
    }
    if ((gpuThreads == 0 || gpuThreads >= 512) && properties.limits.maxComputeWorkGroupInvocations >= 512
        && properties.limits.maxComputeWorkGroupSize[1] >= 8
        && properties.limits.maxComputeSharedMemorySize >= 39168) {
        kernels.emplace("attentionTile", createKernel(shaders / "attentionTile.spv", 5, 36));
        kernels.emplace("attentionCausal", createKernel(shaders / "attentionCausal.spv", 5, 36));
        kernels.emplace("attentionPacked", createKernel(shaders / "attentionPacked.spv", 5, 36));
        const auto operations = vk::SubgroupFeatureFlagBits::eBasic | vk::SubgroupFeatureFlagBits::eVote
            | vk::SubgroupFeatureFlagBits::eArithmetic;
        if (properties.limits.maxComputeSharedMemorySize >= 39300
            && (subgroupSize == 32 || subgroupSize == 64)
            && (subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute)
            && (subgroup.supportedOperations & operations) == operations) {
            kernels.emplace("attentionSubgroup", createKernel(shaders / "attentionSubgroup.spv", 5, 36));
        }
    }
}

Executor::~Executor() {
    // 析构不能传播异常，正常路径的end仍完整报告提交错误
    try {
        end();
    } catch (...) {
        try {
            device->waitIdle();
        } catch (...) {
        }
    }
}

bool Executor::supportsKernel(const std::string& kernel) const {
    return kernels.count(kernel) != 0;
}

bool Executor::supportsProfiling() const {
    return queueTimestampBits != 0;
}

void Executor::setProfiling(bool enabled) {
    if (active) {
        throw std::logic_error("vulkan: cannot switch profiling inside batch");
    }
    if (enabled && !timestampPool) {
        timestampBits = queueTimestampBits;
        if (timestampBits == 0) {
            throw std::runtime_error("vulkan: selected compute queue has no timestamps");
        }
        vk::QueryPoolCreateInfo info;
        info.setQueryType(vk::QueryType::eTimestamp).setQueryCount(timestampCapacity);
        timestampPool = device->createQueryPoolUnique(info);
    }
    profiling = enabled;
}

void Executor::resetProfile() {
    if (active) {
        throw std::logic_error("vulkan: cannot reset profile inside batch");
    }
    measured = GpuProfile{};
}

const GpuProfile& Executor::profile() const {
    return measured;
}

std::uint32_t Executor::gpuThreadBudget() const {
    return gpuThreads;
}

std::string Executor::profileDeviceName() const {
    return deviceName();
}

std::uint32_t Executor::profileSubgroupSize() const {
    return subgroupSize;
}

std::string Executor::profileMemoryTypes() const {
    const auto memory = physical.getMemoryProperties();
    std::string result;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        result += " type" + std::to_string(i) + "=flags"
            + std::to_string(static_cast<std::uint32_t>(memory.memoryTypes[i].propertyFlags))
            + "/heap" + std::to_string(memory.memoryTypes[i].heapIndex);
    }
    return result;
}

std::uint32_t Executor::startTimestamp(
    vk::CommandBuffer recorded,
    const std::string& name,
    vk::PipelineStageFlagBits stage,
    std::uint64_t bytes,
    double gemmFlops
) {
    if (!profiling) {
        return timestampCapacity;
    }
    if (nextTimestamp + 1 >= timestampCapacity) {
        ++measured.droppedEvents;
        return timestampCapacity;
    }
    const std::uint32_t first = nextTimestamp;
    nextTimestamp += 2;
    timestampEvents.push_back({name, first, bytes, gemmFlops});
    recorded.writeTimestamp(stage, *timestampPool, first);
    return first;
}

void Executor::endTimestamp(vk::CommandBuffer recorded, std::uint32_t first, vk::PipelineStageFlagBits stage) const {
    if (first < timestampCapacity) {
        recorded.writeTimestamp(stage, *timestampPool, first + 1);
    }
}

void Executor::collectTimestamps() {
    ProfileScope scope(measured.queryMilliseconds, profiling);
    if (!profiling) {
        return;
    }
    std::vector<std::uint64_t> ticks(nextTimestamp, 0);
    const auto status = device->getQueryPoolResults(
        *timestampPool,
        0,
        nextTimestamp,
        ticks.size() * sizeof(std::uint64_t),
        ticks.data(),
        sizeof(std::uint64_t),
        vk::QueryResultFlagBits::e64
    );
    if (status != vk::Result::eSuccess) {
        throw std::runtime_error("vulkan: completed timestamp results unavailable");
    }
    const std::uint64_t mask = timestampBits == 64
        ? (std::numeric_limits<std::uint64_t>::max)() : (std::uint64_t{1} << timestampBits) - 1;
    const double millisecondsPerTick = static_cast<double>(properties.limits.timestampPeriod) / 1e6;
    measured.deviceMilliseconds += static_cast<double>((ticks[1] - ticks[0]) & mask) * millisecondsPerTick;
    ++measured.submissions;
    for (const auto& event : timestampEvents) {
        auto& row = measured.rows[event.name];
        ++row.calls;
        row.milliseconds += static_cast<double>((ticks[event.first + 1] - ticks[event.first]) & mask) * millisecondsPerTick;
        row.bytes += event.bytes;
        row.gemmFlops += event.gemmFlops;
    }
}

void Executor::begin() {
    if (poisoned) {
        throw std::runtime_error("vulkan: executor is poisoned after a failed submission");
    }
    if (active) {
        throw std::logic_error("vulkan: nested GPU batch");
    }
    active = true;
}

Buffer Executor::createDeviceBuffer(std::size_t count) const {
    checkCount(count);
    Buffer result;
    result.bytes = static_cast<vk::DeviceSize>(count) * sizeof(float);
    result.capacity = reserveCount(count, properties.limits.maxStorageBufferRange / sizeof(float)) * sizeof(float);
    vk::BufferCreateInfo info;
    info.setSize(result.capacity).setUsage(vk::BufferUsageFlagBits::eStorageBuffer
        | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst)
        .setSharingMode(vk::SharingMode::eExclusive);
    result.handle = device->createBufferUnique(info);
    const auto requirements = device->getBufferMemoryRequirements(*result.handle);
    const auto memory = physical.getMemoryProperties();
    std::uint32_t type = memory.memoryTypeCount;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) != 0
            && (memory.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            type = i;
            break;
        }
    }
    if (type == memory.memoryTypeCount) {
        throw std::runtime_error("vulkan: no device-local memory for resident model");
    }
    vk::MemoryAllocateInfo allocation;
    allocation.setAllocationSize(requirements.size).setMemoryTypeIndex(type);
    result.memoryFlags = memory.memoryTypes[type].propertyFlags;
    result.memory = device->allocateMemoryUnique(allocation);
    device->bindBufferMemory(*result.handle, *result.memory, 0);
    return result;
}

Buffer Executor::stagingBuffer(const std::vector<float>& values) {
    const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(values.size()) * sizeof(float);
    for (std::size_t i = 0; i < reusableStaging.size(); ++i) {
        if (reusableStaging[i].capacity >= bytes) {
            std::swap(reusableStaging[i], reusableStaging.back());
            Buffer result = std::move(reusableStaging.back());
            reusableStaging.pop_back();
            result.bytes = bytes;
            std::memcpy(device->mapMemory(*result.memory, 0, bytes), values.data(), static_cast<std::size_t>(bytes));
            device->unmapMemory(*result.memory);
            return result;
        }
    }
    if (profiling) {
        ++measured.stagingAllocations;
    }
    std::vector<float> reserved(reserveCount(values.size(), properties.limits.maxStorageBufferRange / sizeof(float)), 0.0f);
    std::copy(values.begin(), values.end(), reserved.begin());
    auto result = createBuffer(reserved);
    result.bytes = bytes;
    return result;
}

Buffer Executor::readbackBuffer(const std::vector<float>& values) {
    const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(values.size()) * sizeof(float);
    for (std::size_t i = 0; i < reusableReadback.size(); ++i) {
        if (reusableReadback[i].capacity >= bytes) {
            std::swap(reusableReadback[i], reusableReadback.back());
            Buffer result = std::move(reusableReadback.back());
            reusableReadback.pop_back();
            result.bytes = bytes;
            return result;
        }
    }
    if (profiling) {
        ++measured.stagingAllocations;
    }
    // 回读优先HOST_CACHED，不能和偏向CPU写入的upload staging混用
    std::vector<float> reserved(reserveCount(values.size(), properties.limits.maxStorageBufferRange / sizeof(float)), 0.0f);
    auto result = createBuffer(reserved, true);
    result.bytes = bytes;
    return result;
}

vk::CommandBuffer Executor::command() {
    if (!active) {
        throw std::logic_error("vulkan: GPU operation outside batch");
    }
    if (commands.empty()) {
        vk::CommandBufferAllocateInfo allocation;
        allocation.setCommandPool(*commandPool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1);
        commands = device->allocateCommandBuffersUnique(allocation);
        vk::CommandBufferBeginInfo begin;
        begin.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        commands.front()->begin(begin);
        if (profiling) {
            timestampEvents.clear();
            nextTimestamp = 2;
            commands.front()->resetQueryPool(*timestampPool, 0, timestampCapacity);
            commands.front()->writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestampPool, 0);
        }
        recording = true;
    }
    return *commands.front();
}

void Executor::barrier(vk::CommandBuffer recorded) const {
    vk::MemoryBarrier visibility;
    visibility.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
            | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
    recorded.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
        {}, visibility, {}, {}
    );
}

std::size_t Executor::allocate(std::size_t count) {
    if (!active) {
        throw std::logic_error("vulkan: allocate outside batch");
    }
    const std::size_t handle = nextHandle++;
    checkCount(count);
    const vk::DeviceSize bytes = static_cast<vk::DeviceSize>(count) * sizeof(float);
    std::size_t best = reusableDevice.size();
    for (std::size_t i = 0; i < reusableDevice.size(); ++i) {
        if (reusableDevice[i].capacity >= bytes
            && (best == reusableDevice.size() || reusableDevice[i].capacity < reusableDevice[best].capacity)) {
            best = i;
        }
    }
    if (best == reusableDevice.size()) {
        buffers.emplace(handle, createDeviceBuffer(count));
        if (profiling) {
            ++measured.deviceAllocations;
        }
    } else {
        std::swap(reusableDevice[best], reusableDevice.back());
        Buffer buffer = std::move(reusableDevice.back());
        reusableDevice.pop_back();
        buffer.bytes = bytes;
        buffers.emplace(handle, std::move(buffer));
    }
    scratch.push_back(handle);
    return handle;
}

std::size_t Executor::upload(const std::vector<float>& values, bool persistent) {
    const std::size_t handle = allocate(values.size());
    if (persistent) {
        scratch.pop_back();
    }
    staging.push_back(stagingBuffer(values));
    const auto recorded = command();
    vk::MemoryBarrier host;
    host.setSrcAccessMask(vk::AccessFlagBits::eHostWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    recorded.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer, {}, host, {}, {});
    const vk::BufferCopy copy(0, 0, buffers.at(handle).bytes);
    const auto timestamp = startTimestamp(recorded, "copy/upload", vk::PipelineStageFlagBits::eTransfer, copy.size);
    recorded.copyBuffer(*staging.back().handle, *buffers.at(handle).handle, copy);
    endTimestamp(recorded, timestamp, vk::PipelineStageFlagBits::eTransfer);
    return handle;
}

bool Executor::supportsOrderedAttention() const {
    return supportsKernel("attentionOrdered");
}

bool Executor::supportsGroupedAttention() const {
    return supportsKernel("attentionGroup");
}

bool Executor::supportsPrefixAttention() const {
    return supportsKernel("attentionPrefix");
}

bool Executor::supportsCopy() const {
    return true;
}

void Executor::copy(
    std::size_t source,
    std::size_t destination,
    std::size_t sourceOffset,
    std::size_t destinationOffset,
    std::size_t count
) {
    if (!active || poisoned) {
        throw std::logic_error("vulkan: copy requires a healthy active batch");
    }
    const auto& input = buffers.at(source);
    const auto& output = buffers.at(destination);
    const auto inputCount = static_cast<std::size_t>(input.bytes / sizeof(float));
    const auto outputCount = static_cast<std::size_t>(output.bytes / sizeof(float));
    if (count == 0 || sourceOffset > inputCount || count > inputCount - sourceOffset
        || destinationOffset > outputCount || count > outputCount - destinationOffset) {
        throw std::invalid_argument("vulkan: copy range exceeds logical buffer size");
    }
    if (source == destination && sourceOffset < destinationOffset + count
        && destinationOffset < sourceOffset + count) {
        throw std::invalid_argument("vulkan: overlapping device copy ranges");
    }
    const auto recorded = command();
    barrier(recorded);
    const vk::BufferCopy region(
        static_cast<vk::DeviceSize>(sourceOffset) * sizeof(float),
        static_cast<vk::DeviceSize>(destinationOffset) * sizeof(float),
        static_cast<vk::DeviceSize>(count) * sizeof(float)
    );
    const auto timestamp = startTimestamp(recorded, "copy/device", vk::PipelineStageFlagBits::eTransfer, region.size);
    recorded.copyBuffer(*input.handle, *output.handle, region);
    endTimestamp(recorded, timestamp, vk::PipelineStageFlagBits::eTransfer);
}

void Executor::dispatch(
    const std::string& name,
    const std::vector<std::size_t>& bindings,
    const std::vector<std::uint32_t>& push,
    const std::array<std::uint32_t, 3>& groups
) {
    const auto& requested = kernels.at(name);
    checkGroups(groups);
    if (bindings.size() != requested.bindingCount || push.size() * sizeof(std::uint32_t) != requested.pushBytes) {
        throw std::invalid_argument("vulkan: resident dispatch ABI mismatch: " + name);
    }
    if (optimized && name == "attentionFused" && push[1] == 1 && push[6] == 64 && push[8] != 0
        && push[2] >= 511 && supportsKernel("attentionDecode")) {
        const std::uint64_t headRows = static_cast<std::uint64_t>(push[0]) * push[5];
        const std::uint64_t keyCount = static_cast<std::uint64_t>(push[2]) + 1;
        const std::uint64_t splits = (keyCount + 127) / 128;
        const std::uint64_t partialCount = headRows * splits * 66;
        if (headRows > 0 && headRows <= 65535 && splits <= 65535
            && partialCount <= (std::numeric_limits<std::uint32_t>::max)()
            && groups[2] == 1 && static_cast<std::uint64_t>(groups[0]) * groups[1] >= headRows) {
            const auto partial = allocate(static_cast<std::size_t>(partialCount));
            auto splitPush = push;
            splitPush.push_back(128);
            const auto count = static_cast<std::uint32_t>(splits);
            const auto heads = static_cast<std::uint32_t>(headRows);
            dispatch("attentionDecode", {bindings[0], bindings[1], bindings[2], bindings[3], partial}, splitPush, {heads, count, 1});
            dispatch("attentionDecodeMerge", {partial, bindings[4]}, {push[0], push[5], count}, {heads, 1, 1});
            return;
        }
    }
    std::string selected = name;
    auto dispatchGroups = groups;

    if (optimized && name == "gemm") {
        const std::uint32_t rows = push[0];
        const std::uint32_t columns = push[1];
        const std::uint32_t inner = push[2];
        const bool knownProjection = (inner == 192 && (columns == 192 || columns == 576 || columns == 1024))
            || ((inner == 512 || inner == 768 || inner == 64) && columns == 192)
            || (inner == 64 && columns == 768);
        const std::array<std::uint32_t, 3> native = {
            columns / 16 + (columns % 16 != 0 ? 1u : 0u),
            rows / 16 + (rows % 16 != 0 ? 1u : 0u), 1
        };
        if (rows <= 8 && rows > 0 && columns >= 64 && inner >= 64 && groups == native) {
            const std::uint32_t tile = 16;
            selected = "gemv";
            dispatchGroups = {columns / tile + (columns % tile != 0 ? 1u : 0u), rows, 1};
        }
        // 只改完整原生grid，诊断用的部分派发仍保持调用方的覆盖范围
        if (rows >= 128 && knownProjection && groups == native) {
            selected = "gemmTile";
            dispatchGroups = {
                columns / 32 + (columns % 32 != 0 ? 1u : 0u),
                rows / 32 + (rows % 32 != 0 ? 1u : 0u), 1
            };
        }
    }
    if (optimized && name == "attentionFused" && push[6] == 64 && push[8] != 0
        && static_cast<std::uint64_t>(push[2]) + push[1] >= 128 && supportsKernel("attentionTile")) {
        const std::uint64_t nativeGroups = static_cast<std::uint64_t>(push[0]) * push[1] * push[5];
        if (groups[2] == 1 && static_cast<std::uint64_t>(groups[0]) * groups[1] >= nativeGroups) {
            selected = push[2] == 0 && push[3] >= push[1] && push[1] >= 128 ? "attentionCausal" : "attentionTile";
            if (push[1] >= 128 && supportsKernel("attentionSubgroup")) {
                selected = "attentionSubgroup";
            }
            const std::uint64_t tiledGroups = static_cast<std::uint64_t>(push[0])
                * (push[1] / 8 + (push[1] % 8 != 0 ? 1u : 0u)) * push[5];
            const auto x = static_cast<std::uint32_t>(std::min<std::uint64_t>(tiledGroups, 65535));
            if (x != 0) {
                dispatchGroups = {x, static_cast<std::uint32_t>((tiledGroups + x - 1) / x), 1};
                checkGroups(dispatchGroups);
            }
        }
    }
    const Kernel* selectedKernel = &kernels.at(selected);
    const bool flatDispatch = selectedKernel->flatDispatch;
    const std::uint32_t gridWidth = std::min(properties.limits.maxComputeWorkGroupCount[0], 65535u);
    std::uint64_t flatGroups = 0;
    if (gpuThreads != 0 && flatDispatch) {
        if (gridWidth == 0) {
            throw std::runtime_error("vulkan: invalid device workgroup limit");
        }
        flatGroups = static_cast<std::uint64_t>(dispatchGroups[0]) * dispatchGroups[1];
        const std::uint64_t rows = flatGroups / gridWidth + (flatGroups % gridWidth != 0 ? 1u : 0u);
        if (rows > (std::numeric_limits<std::uint32_t>::max)()) {
            throw std::invalid_argument("vulkan: flattened budget grid exceeds index range");
        }
        dispatchGroups[0] = static_cast<std::uint32_t>(std::min<std::uint64_t>(flatGroups, gridWidth));
        dispatchGroups[1] = static_cast<std::uint32_t>(rows);
    }
    if (gpuThreads != 0) {
        if (selectedKernel->localInvocations > gpuThreads) {
            throw std::invalid_argument("vulkan: kernel workgroup exceeds gpuThreads budget");
        }
        auto found = budgetKernels.find(selected);
        if (found == budgetKernels.end()) {
            found = budgetKernels.emplace(selected, createKernel(
                shaderDirectory / (selected + "Budget.spv"),
                selectedKernel->bindingCount,
                selectedKernel->pushBytes,
                gridWidth
            )).first;
        }
        if (found->second.localSize != selectedKernel->localSize || found->second.localInvocations > gpuThreads) {
            throw std::runtime_error("vulkan: budget variant LocalSize mismatch");
        }
        selectedKernel = &found->second;
    }
    const auto& kernel = *selectedKernel;
    if (bindings.size() != kernel.bindingCount || push.size() * sizeof(std::uint32_t) != kernel.pushBytes) {
        throw std::invalid_argument("vulkan: resident dispatch ABI mismatch: " + name);
    }
    vk::DescriptorPoolSize size;
    size.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(kernel.bindingCount);
    vk::DescriptorPoolCreateInfo pool;
    pool.setMaxSets(1).setPoolSizes(size);
    descriptorPools.push_back(device->createDescriptorPoolUnique(pool));
    const std::array<vk::DescriptorSetLayout, 1> layouts = {*kernel.descriptors};
    vk::DescriptorSetAllocateInfo allocation;
    allocation.setDescriptorPool(*descriptorPools.back()).setSetLayouts(layouts);
    const auto sets = device->allocateDescriptorSets(allocation);
    std::vector<vk::DescriptorBufferInfo> infos;
    infos.reserve(bindings.size());
    for (const auto handle : bindings) {
        const auto& buffer = buffers.at(handle);
        infos.emplace_back(*buffer.handle, 0, buffer.bytes);
    }
    std::vector<vk::WriteDescriptorSet> writes;
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        vk::WriteDescriptorSet write;
        write.setDstSet(sets.front()).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(infos[i]);
        writes.push_back(write);
    }
    device->updateDescriptorSets(writes, {});
    const auto recorded = command();
    barrier(recorded);
    std::string label;
    double gemmFlops = 0.0;
    if (profiling) {
        label = selected;
        if (name == "gemm") {
            label += "/M" + std::to_string(push[0]) + "N" + std::to_string(push[1]) + "K" + std::to_string(push[2]);
            gemmFlops = 2.0 * push[0] * push[1] * push[2];
        } else if (name == "attentionFused") {
            label += push[8] != 0 ? "/time" : "/variate";
        }
    }
    const auto timestamp = startTimestamp(recorded, label, vk::PipelineStageFlagBits::eComputeShader, 0, gemmFlops);
    recorded.bindPipeline(vk::PipelineBindPoint::eCompute, *kernel.pipeline);
    recorded.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *kernel.layout, 0, sets, {});
    recorded.pushConstants<std::uint32_t>(*kernel.layout, vk::ShaderStageFlagBits::eCompute, 0, push);
    if (gpuThreads == 0) {
        recorded.dispatch(dispatchGroups[0], dispatchGroups[1], dispatchGroups[2]);
        if (profiling) {
            ++measured.dispatchWindows;
        }
    } else {
        const std::uint32_t windowGroups = gpuThreads / kernel.localInvocations;
        bool firstWindow = true;
        for (std::uint32_t z = 0; z < dispatchGroups[2]; ++z) {
            for (std::uint32_t y = 0; y < dispatchGroups[1];) {
                const auto rowGroups = flatDispatch
                    ? static_cast<std::uint32_t>(std::min<std::uint64_t>(gridWidth, flatGroups - static_cast<std::uint64_t>(y) * gridWidth))
                    : dispatchGroups[0];
                const auto firstWidth = std::min(windowGroups, rowGroups);
                const auto completeRows = flatDispatch
                    ? static_cast<std::uint32_t>((flatGroups - static_cast<std::uint64_t>(y) * gridWidth) / gridWidth)
                    : dispatchGroups[1] - y;
                const auto windowRows = completeRows != 0
                    ? std::min(completeRows, windowGroups / firstWidth) : 1u;
                for (std::uint32_t x = 0; x < rowGroups;) {
                    const auto count = std::min(windowGroups, rowGroups - x);
                    if (!firstWindow) {
                        barrier(recorded);
                    }
                    firstWindow = false;
                    recorded.dispatchBase(x, y, z, count, windowRows, 1);
                    if (profiling) {
                        ++measured.dispatchWindows;
                        measured.maximumWindowInvocations = std::max(measured.maximumWindowInvocations,
                            static_cast<std::uint64_t>(count) * windowRows * kernel.localInvocations);
                    }
                    x += count;
                }
                y += windowRows;
            }
        }
    }
    endTimestamp(recorded, timestamp, vk::PipelineStageFlagBits::eComputeShader);
}

void Executor::submit() {
    if (poisoned) {
        throw std::runtime_error("vulkan: cannot resubmit a poisoned command batch");
    }
    if (commands.empty()) {
        return;
    }
    try {
        if (recording) {
            // 先撤销recording标记，vkEnd失败也不能在清理时再次结束同一buffer
            recording = false;
            if (profiling) {
                commands.front()->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *timestampPool, 1);
            }
            commands.front()->end();
        }
        {
            ProfileScope scope(measured.submitMilliseconds, profiling);
            submissionFence = device->createFenceUnique({});
            vk::SubmitInfo submission;
            submission.setCommandBuffers(*commands.front());
            // queue.submit抛错时保守视为可能已进入设备，排空前保留全部资源
            submissionPending = true;
            queue.submit(submission, *submissionFence);
        }
        {
            ProfileScope scope(measured.waitMilliseconds, profiling);
            const auto status = device->waitForFences(*submissionFence, VK_TRUE, (std::numeric_limits<std::uint64_t>::max)());
            if (status != vk::Result::eSuccess) {
                throw std::runtime_error("vulkan: resident model fence failed");
            }
        }
        submissionPending = false;
        collectTimestamps();
    } catch (...) {
        poisoned = true;
        throw;
    }
    ProfileScope retire(measured.retireMilliseconds, profiling);
    submissionFence.reset();
    commands.clear();
    descriptorPools.clear();
    for (auto& buffer : staging) {
        reusableStaging.push_back(std::move(buffer));
    }
    staging.clear();
}

std::vector<float> Executor::download(std::size_t handle) {
    const auto& source = buffers.at(handle);
    std::vector<float> values;
    Buffer target;
    {
        ProfileScope scope(measured.downloadPrepareMilliseconds, profiling);
        values.resize(static_cast<std::size_t>(source.bytes / sizeof(float)), 0.0f);
        target = readbackBuffer(values);
    }
    if (profiling) {
        measured.downloadMemoryFlags |= static_cast<std::uint32_t>(target.memoryFlags);
    }
    const auto recorded = command();
    barrier(recorded);
    const vk::BufferCopy copy(0, 0, source.bytes);
    const auto timestamp = startTimestamp(recorded, "copy/download", vk::PipelineStageFlagBits::eTransfer, copy.size);
    recorded.copyBuffer(*source.handle, *target.handle, copy);
    endTimestamp(recorded, timestamp, vk::PipelineStageFlagBits::eTransfer);
    vk::MemoryBarrier host;
    host.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eHostRead);
    recorded.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, host, {}, {});
    submit();
    {
        ProfileScope scope(measured.downloadReadMilliseconds, profiling);
        std::memcpy(values.data(), device->mapMemory(*target.memory, 0, target.bytes), static_cast<std::size_t>(target.bytes));
        device->unmapMemory(*target.memory);
    }
    reusableReadback.push_back(std::move(target));
    return values;
}

void Executor::end() {
    if (!active) {
        return;
    }
    if (poisoned) {
        if (submissionPending) {
            // 若排空仍抛错，保持commands、descriptor、staging和scratch存活供析构重试
            device->waitIdle();
            submissionPending = false;
        }
        submissionFence.reset();
        commands.clear();
        descriptorPools.clear();
        staging.clear();
        recording = false;
    } else {
        submit();
    }
    for (const auto handle : scratch) {
        reusableDevice.push_back(std::move(buffers.at(handle)));
        buffers.erase(handle);
    }
    scratch.clear();
    const auto cacheLimit = std::max<std::size_t>(8, properties.limits.maxMemoryAllocationCount / 16);
    trimBuffers(reusableDevice, cacheLimit);
    trimBuffers(reusableStaging, cacheLimit);
    trimBuffers(reusableReadback, cacheLimit);
    active = false;
}

} // namespace timestral::vulkan
