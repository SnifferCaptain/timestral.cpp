/***************
 * @file kernels.cpp
 * @brief Vulkan设备选择、资源回收、同步提交与FP32算子边界校验
 ***************/

#include "timestral/vulkan/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace timestral::vulkan {

static std::size_t checkedProduct(std::size_t a, std::size_t b) {
    const auto maximum = static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)());
    if (a == 0 || b == 0 || a > maximum || b > maximum || a > maximum / b) {
        throw std::invalid_argument("vulkan: zero dimension or FP32 shader index overflow");
    }
    return a * b;
}

static void checkValues(const std::vector<float>& values, bool masked = false) {
    for (const float value : values) {
        if (!std::isfinite(value) && !(masked && value == -std::numeric_limits<float>::infinity())) {
            throw std::invalid_argument("vulkan: non-finite input");
        }
    }
}

static std::uint32_t groupsFor(std::size_t count, std::uint32_t width) {
    return static_cast<std::uint32_t>((count - 1) / width + 1);
}

static std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "FP32 push constant size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

Kernels::Kernels(const std::filesystem::path& shaderDirectory, std::size_t deviceIndex, bool validation) {
    vk::ApplicationInfo application;
    application.setPApplicationName("timestral-vulkan-kernels").setApiVersion(VK_API_VERSION_1_1);
    vk::InstanceCreateInfo instanceInfo;
    instanceInfo.setPApplicationInfo(&application);
    const std::array<const char*, 1> layers = {"VK_LAYER_KHRONOS_validation"};
    if (validation) {
        const auto available = vk::enumerateInstanceLayerProperties();
        const auto found = std::any_of(available.begin(), available.end(), [](const auto& layer) {
            return std::string(layer.layerName.data()) == "VK_LAYER_KHRONOS_validation";
        });
        if (!found) {
            throw std::runtime_error("vulkan: requested validation layer is unavailable");
        }
        instanceInfo.setPEnabledLayerNames(layers);
    }
    instance = vk::createInstanceUnique(instanceInfo);
    const auto devices = instance->enumeratePhysicalDevices();
    if (deviceIndex >= devices.size()) {
        throw std::invalid_argument("vulkan: requested device index is unavailable");
    }
    physical = devices[deviceIndex];
    properties = physical.getProperties();
    const auto& limits = properties.limits;
    if (properties.apiVersion < VK_API_VERSION_1_1 || limits.maxComputeWorkGroupInvocations < 256
        || limits.maxComputeWorkGroupSize[0] < 64 || limits.maxComputeWorkGroupSize[1] < 16
        || limits.maxComputeSharedMemorySize < 2048 || limits.maxPushConstantsSize < 16
        || limits.maxPerStageDescriptorStorageBuffers < 4 || limits.maxDescriptorSetStorageBuffers < 4) {
        throw std::runtime_error("vulkan: device does not meet FP32 kernel limits");
    }
    const auto families = physical.getQueueFamilyProperties();
    std::uint32_t familyIndex = static_cast<std::uint32_t>(families.size());
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueCount > 0 && (families[i].queueFlags & vk::QueueFlagBits::eCompute)) {
            familyIndex = i;
            break;
        }
    }
    if (familyIndex == families.size()) {
        throw std::runtime_error("vulkan: device has no compute queue");
    }
    queueTimestampBits = families[familyIndex].timestampValidBits;
    const std::array<float, 1> priorities = {1.0f};
    vk::DeviceQueueCreateInfo queueInfo;
    queueInfo.setQueueFamilyIndex(familyIndex).setQueuePriorities(priorities);
    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfos(queueInfo);
    device = physical.createDeviceUnique(deviceInfo);
    queue = device->getQueue(familyIndex, 0);
    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.setQueueFamilyIndex(familyIndex).setFlags(vk::CommandPoolCreateFlagBits::eTransient);
    commandPool = device->createCommandPoolUnique(poolInfo);
    linearKernel = createKernel(shaderDirectory / "gemm.spv", 4, 16);
    normKernel = createKernel(shaderDirectory / "rmsNorm.spv", 3, 12);
    softmaxKernel = createKernel(shaderDirectory / "softmax.spv", 2, 8);
    siluKernel = createKernel(shaderDirectory / "silu.spv", 2, 4);
}

std::string Kernels::deviceName() const {
    return properties.deviceName.data();
}

void Kernels::checkCount(std::size_t count) const {
    if (count == 0 || count > (std::numeric_limits<std::uint32_t>::max)()
        || count > properties.limits.maxStorageBufferRange / sizeof(float)) {
        throw std::invalid_argument("vulkan: buffer exceeds shader index or device storage limit, count=" + std::to_string(count));
    }
}

void Kernels::checkGroups(const std::array<std::uint32_t, 3>& groups) const {
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (groups[i] == 0 || groups[i] > properties.limits.maxComputeWorkGroupCount[i]) {
            throw std::invalid_argument("vulkan: dispatch exceeds device workgroup limit");
        }
    }
}

Buffer Kernels::createBuffer(const std::vector<float>& values, bool preferCached) const {
    checkCount(values.size());
    Buffer result;
    result.bytes = static_cast<vk::DeviceSize>(values.size()) * sizeof(float);
    result.capacity = result.bytes;
    vk::BufferCreateInfo info;
    info.setSize(result.bytes).setUsage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst)
        .setSharingMode(vk::SharingMode::eExclusive);
    result.handle = device->createBufferUnique(info);
    const auto requirements = device->getBufferMemoryRequirements(*result.handle);
    const auto memoryProperties = physical.getMemoryProperties();
    const auto flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    std::uint32_t typeIndex = memoryProperties.memoryTypeCount;
    if (preferCached) {
        const auto cached = flags | vk::MemoryPropertyFlagBits::eHostCached;
        for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) != 0
                && (memoryProperties.memoryTypes[i].propertyFlags & cached) == cached) {
                typeIndex = i;
                break;
            }
        }
    }
    // 无cached coherent类型时仍走既有coherent路径，不引入未处理的invalidate语义
    for (std::uint32_t i = 0; typeIndex == memoryProperties.memoryTypeCount && i < memoryProperties.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) != 0
            && (memoryProperties.memoryTypes[i].propertyFlags & flags) == flags) {
            typeIndex = i;
            break;
        }
    }
    if (typeIndex == memoryProperties.memoryTypeCount) {
        throw std::runtime_error("vulkan: no host-visible coherent memory type for kernel verification");
    }
    vk::MemoryAllocateInfo allocation;
    allocation.setAllocationSize(requirements.size).setMemoryTypeIndex(typeIndex);
    result.memoryFlags = memoryProperties.memoryTypes[typeIndex].propertyFlags;
    result.memory = device->allocateMemoryUnique(allocation);
    device->bindBufferMemory(*result.handle, *result.memory, 0);
    // 映射地址仅用于Vulkan要求的主机拷贝，不保留非拥有指针
    std::memcpy(device->mapMemory(*result.memory, 0, result.bytes), values.data(), static_cast<std::size_t>(result.bytes));
    device->unmapMemory(*result.memory);
    return result;
}

Kernel Kernels::createKernel(
    const std::filesystem::path& path,
    std::uint32_t bindingCount,
    std::uint32_t pushBytes,
    std::uint32_t logicalGridWidth
) const {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("vulkan: cannot open shader " + path.filename().string());
    }
    const auto bytes = input.tellg();
    if (bytes < 20 || bytes % 4 != 0 || bytes > 16 * 1024 * 1024) {
        throw std::runtime_error("vulkan: invalid SPIR-V file length");
    }
    std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / sizeof(std::uint32_t));
    input.seekg(0);
    // 二进制流接口要求char指针，存储始终由对齐的uint32_t容器拥有
    input.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(bytes));
    if (!input || code[0] != 0x07230203u || code[1] > 0x00010300u || code[4] != 0) {
        throw std::runtime_error("vulkan: invalid or unsupported SPIR-V header");
    }
    Kernel result;
    result.bindingCount = bindingCount;
    result.pushBytes = pushBytes;
    std::array<std::uint32_t, 3> localSize = {0, 0, 0};
    std::uint32_t entryId = 0;
    std::uint32_t localEntry = 0;
    std::size_t entryCount = 0;
    bool gridSpecialization = false;
    for (std::size_t offset = 5; offset < code.size();) {
        const auto count = code[offset] >> 16;
        const auto opcode = code[offset] & 0xffffu;
        if (count == 0 || count > code.size() - offset) {
            throw std::runtime_error("vulkan: malformed SPIR-V instruction");
        }
        if (opcode == 15 && count >= 4) {
            entryId = code[offset + 2];
            ++entryCount;
        }
        if (opcode == 16 && count == 6 && code[offset + 2] == 17) {
            if (localEntry != 0) {
                throw std::runtime_error("vulkan: multiple LocalSize declarations");
            }
            localEntry = code[offset + 1];
            localSize = {code[offset + 3], code[offset + 4], code[offset + 5]};
        }
        if (opcode == 71 && count == 4 && code[offset + 2] == 11 && code[offset + 3] == 24) {
            result.flatDispatch = true;
        }
        if (opcode == 71 && count == 4 && code[offset + 2] == 1 && code[offset + 3] == 255) {
            gridSpecialization = true;
        }
        offset += count;
    }
    if (entryCount != 1 || localEntry == 0 || localEntry != entryId || gridSpecialization != (logicalGridWidth != 0)) {
        throw std::runtime_error("vulkan: fixed LocalSize or budget specialization contract mismatch");
    }
    std::uint32_t invocations = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (localSize[axis] == 0 || localSize[axis] > properties.limits.maxComputeWorkGroupSize[axis]
            || invocations > properties.limits.maxComputeWorkGroupInvocations / localSize[axis]) {
            throw std::runtime_error("vulkan: shader LocalSize exceeds device limits");
        }
        invocations *= localSize[axis];
    }
    result.localInvocations = invocations;
    result.localSize = localSize;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (std::uint32_t i = 0; i < bindingCount; ++i) {
        vk::DescriptorSetLayoutBinding binding;
        binding.setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
        bindings.push_back(binding);
    }
    vk::DescriptorSetLayoutCreateInfo descriptorInfo;
    descriptorInfo.setBindings(bindings);
    result.descriptors = device->createDescriptorSetLayoutUnique(descriptorInfo);
    vk::PushConstantRange range;
    range.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(pushBytes);
    vk::PipelineLayoutCreateInfo layoutInfo;
    const std::array<vk::DescriptorSetLayout, 1> layouts = {*result.descriptors};
    layoutInfo.setSetLayouts(layouts).setPushConstantRanges(range);
    result.layout = device->createPipelineLayoutUnique(layoutInfo);
    vk::ShaderModuleCreateInfo moduleInfo;
    moduleInfo.setCode(code);
    const auto shader = device->createShaderModuleUnique(moduleInfo);
    vk::PipelineShaderStageCreateInfo stage;
    stage.setStage(vk::ShaderStageFlagBits::eCompute).setModule(*shader).setPName("main");
    vk::SpecializationMapEntry widthEntry(255, 0, sizeof(logicalGridWidth));
    vk::SpecializationInfo specialization(1, &widthEntry, sizeof(logicalGridWidth), &logicalGridWidth);
    vk::ComputePipelineCreateInfo pipelineInfo;
    if (logicalGridWidth != 0) {
        stage.setPSpecializationInfo(&specialization);
        pipelineInfo.setFlags(vk::PipelineCreateFlagBits::eDispatchBase);
    }
    pipelineInfo.setStage(stage).setLayout(*result.layout);
    result.pipeline = device->createComputePipelineUnique({}, pipelineInfo).value;
    return result;
}

std::vector<float> Kernels::run(
    const Kernel& kernel,
    const std::vector<std::reference_wrapper<const std::vector<float>>>& inputs,
    std::size_t outputCount,
    const std::vector<std::uint32_t>& push,
    const std::array<std::uint32_t, 3>& groups
) {
    checkCount(outputCount);
    checkGroups(groups);
    if (inputs.size() + 1 != kernel.bindingCount || push.size() * sizeof(std::uint32_t) != kernel.pushBytes) {
        throw std::invalid_argument("vulkan: kernel binding or push constant mismatch");
    }
    // queue与commandPool需要外部同步，同一个验证器允许并发调用但串行提交
    const std::lock_guard<std::mutex> lock(submissionMutex);
    std::vector<Buffer> buffers;
    buffers.reserve(kernel.bindingCount);
    for (const auto& input : inputs) {
        buffers.push_back(createBuffer(input.get()));
    }
    std::vector<float> output(outputCount, std::numeric_limits<float>::quiet_NaN());
    buffers.push_back(createBuffer(output));
    vk::DescriptorPoolSize poolSize;
    poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(kernel.bindingCount);
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(1).setPoolSizes(poolSize);
    const auto descriptorPool = device->createDescriptorPoolUnique(poolInfo);
    const std::array<vk::DescriptorSetLayout, 1> layouts = {*kernel.descriptors};
    vk::DescriptorSetAllocateInfo allocation;
    allocation.setDescriptorPool(*descriptorPool).setSetLayouts(layouts);
    const auto descriptorSets = device->allocateDescriptorSets(allocation);
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        bufferInfos.emplace_back(*buffer.handle, 0, buffer.bytes);
    }
    std::vector<vk::WriteDescriptorSet> writes;
    for (std::uint32_t i = 0; i < buffers.size(); ++i) {
        vk::WriteDescriptorSet write;
        write.setDstSet(descriptorSets[0]).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(bufferInfos[i]);
        writes.push_back(write);
    }
    device->updateDescriptorSets(writes, {});
    vk::CommandBufferAllocateInfo commandInfo;
    commandInfo.setCommandPool(*commandPool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1);
    auto commands = device->allocateCommandBuffersUnique(commandInfo);
    const auto command = *commands[0];
    vk::CommandBufferBeginInfo begin;
    begin.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    command.begin(begin);
    vk::MemoryBarrier upload;
    upload.setSrcAccessMask(vk::AccessFlagBits::eHostWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    command.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eComputeShader, {}, upload, {}, {});
    command.bindPipeline(vk::PipelineBindPoint::eCompute, *kernel.pipeline);
    command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *kernel.layout, 0, descriptorSets, {});
    command.pushConstants<std::uint32_t>(*kernel.layout, vk::ShaderStageFlagBits::eCompute, 0, push);
    command.dispatch(groups[0], groups[1], groups[2]);
    vk::MemoryBarrier download;
    download.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eHostRead);
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, download, {}, {});
    command.end();
    const auto fence = device->createFenceUnique({});
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command);
    queue.submit(submit, *fence);
    const auto status = device->waitForFences(*fence, VK_TRUE, (std::numeric_limits<std::uint64_t>::max)());
    if (status != vk::Result::eSuccess) {
        throw std::runtime_error("vulkan: compute fence wait failed");
    }
    const auto& resultBuffer = buffers.back();
    std::memcpy(output.data(), device->mapMemory(*resultBuffer.memory, 0, resultBuffer.bytes), static_cast<std::size_t>(resultBuffer.bytes));
    device->unmapMemory(*resultBuffer.memory);
    return output;
}

std::vector<float> Kernels::linear(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    std::size_t rows,
    std::size_t columns,
    std::size_t inner
) {
    const auto inputCount = checkedProduct(rows, inner);
    const auto weightCount = checkedProduct(columns, inner);
    const auto outputCount = checkedProduct(rows, columns);
    checkCount(inputCount);
    checkCount(weightCount);
    checkCount(outputCount);
    if (input.size() != inputCount || weights.size() != weightCount || (!bias.empty() && bias.size() != columns)) {
        throw std::invalid_argument("vulkan: linear shape mismatch");
    }
    checkValues(input);
    checkValues(weights);
    checkValues(bias);
    const std::vector<float> zeroBias(bias.empty() ? columns : 0, 0.0f);
    const auto& selectedBias = bias.empty() ? zeroBias : bias;
    const std::vector<std::uint32_t> push = {
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(columns),
        static_cast<std::uint32_t>(inner),
        bias.empty() ? 0u : 1u
    };
    return run(
        linearKernel,
        {std::cref(input), std::cref(weights), std::cref(selectedBias)},
        outputCount,
        push,
        {groupsFor(columns, 16), groupsFor(rows, 16), 1}
    );
}

std::vector<float> Kernels::rmsNorm(
    const std::vector<float>& input,
    const std::vector<float>& scale,
    std::size_t rows,
    std::size_t width,
    float epsilon
) {
    const auto count = checkedProduct(rows, width);
    checkCount(count);
    if (input.size() != count || scale.size() != width || !std::isfinite(epsilon) || epsilon <= 0.0f) {
        throw std::invalid_argument("vulkan: rmsNorm shape or epsilon is invalid");
    }
    checkValues(input);
    checkValues(scale);
    return run(
        normKernel,
        {std::cref(input), std::cref(scale)},
        count,
        {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(width), floatBits(epsilon)},
        {static_cast<std::uint32_t>(rows), 1, 1}
    );
}

std::vector<float> Kernels::softmax(const std::vector<float>& input, std::size_t rows, std::size_t width) {
    const auto count = checkedProduct(rows, width);
    checkCount(count);
    if (input.size() != count) {
        throw std::invalid_argument("vulkan: softmax shape mismatch");
    }
    checkValues(input, true);
    return run(
        softmaxKernel,
        {std::cref(input)},
        count,
        {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(width)},
        {static_cast<std::uint32_t>(rows), 1, 1}
    );
}

std::vector<float> Kernels::silu(const std::vector<float>& input) {
    checkCount(input.size());
    checkValues(input);
    return run(
        siluKernel,
        {std::cref(input)},
        input.size(),
        {static_cast<std::uint32_t>(input.size())},
        {groupsFor(input.size(), 64), 1, 1}
    );
}

} // namespace timestral::vulkan
