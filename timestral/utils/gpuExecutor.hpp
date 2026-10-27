/***************
 * @file gpuExecutor.hpp
 * @brief GPU后端共用的常驻缓冲与批命令执行契约
 ***************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace timestral {

// 句柄仅在所属executor内有效，所有buffer均为连续FP32或等宽uint元信息
// 模型持有只读权重，调用方在整个batch期间串行访问executor
class GpuExecutor {
public:
    virtual ~GpuExecutor() = default;

    // 开始记录一个batch，调用方必须保护整个begin/end区间而非单次dispatch
    virtual void begin() = 0;
    virtual std::size_t upload(const std::vector<float>& values, bool persistent = false) = 0;
    virtual std::size_t allocate(std::size_t count) = 0;

    // bindings按shader binding顺序排列，push按32位原样传递
    virtual void dispatch(
        const std::string& kernel,
        const std::vector<std::size_t>& bindings,
        const std::vector<std::uint32_t>& push,
        const std::array<std::uint32_t, 3>& groups
    ) = 0;

    // 可选的设备内精确复制，偏移和长度以FP32元素计，禁止重叠区间
    virtual bool supportsCopy() const {
        return false;
    }

    virtual void copy(std::size_t, std::size_t, std::size_t, std::size_t, std::size_t) {
        throw std::runtime_error("gpu: device copy is not supported");
    }

    // 可直接读取一组前缀，不为批内每条查询物化完整历史
    virtual bool supportsPrefixAttention() const {
        return false;
    }

    virtual bool supportsGroupedAttention() const {
        return false;
    }

    virtual bool supportsOrderedAttention() const {
        return false;
    }

    // download提交此前命令并等待，buffer继续有效以便trace后继续记录
    virtual std::vector<float> download(std::size_t buffer) = 0;

    // 完成尚未提交的命令，释放本batch的工作区，异常路径亦可调用
    virtual void end() = 0;
};

} // namespace timestral
