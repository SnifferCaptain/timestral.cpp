/***************
 * @file profiling.hpp
 * @brief 可选Vulkan时间戳与主机耗时归因数据
 ***************/

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace timestral::vulkan {

struct GpuProfileRow {
    std::size_t calls = 0;       // 完成的事件数
    double milliseconds = 0.0;   // 同阶段GPU时间戳差值
    std::uint64_t bytes = 0;     // 显式copy字节，不是实际DRAM流量
    double gemmFlops = 0.0;      // GEMM有效乘加数，FMA按2 FLOPs
};

struct GpuProfile {
    std::map<std::string, GpuProfileRow> rows;
    std::size_t submissions = 0;       // 已完成提交
    std::uint64_t dispatchWindows = 0;
    std::uint64_t maximumWindowInvocations = 0;
    std::size_t droppedEvents = 0;     // 查询容量不足时未记录的事件
    std::size_t deviceAllocations = 0; // 新建device缓冲数
    std::size_t stagingAllocations = 0; // 新建staging缓冲数
    double deviceMilliseconds = 0.0;   // 每提交top到bottom跨度之和
    double submitMilliseconds = 0.0;   // fence创建与queue.submit，不含等待
    double waitMilliseconds = 0.0;     // host fence等待，与GPU时间重叠
    double retireMilliseconds = 0.0;   // 命令/描述符回收与staging归还
    double queryMilliseconds = 0.0;    // 查询读取与汇总诊断成本
    double downloadPrepareMilliseconds = 0.0; // CPU结果初始化与staging准备
    double downloadReadMilliseconds = 0.0;    // map、CPU读取与unmap
    std::uint32_t downloadMemoryFlags = 0;    // 回读实际内存属性并集
};

struct TimestampEvent {
    std::string name;
    std::uint32_t first = 0;  // 一对连续query的起始下标
    std::uint64_t bytes = 0;  // 传输统计
    double gemmFlops = 0.0;   // 投影计算量
};

// 只用于诊断，关闭时不读取时钟，不把嵌套范围当作可相加的独占时间
class ProfileScope {
public:
    ProfileScope(double& destination, bool enabled)
        : destination(destination),
          enabled(enabled),
          start(enabled ? Clock::now() : Clock::time_point{}) {
    }

    ~ProfileScope() {
        if (enabled) {
            destination += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
    }

    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    using Clock = std::chrono::steady_clock;
    double& destination;
    bool enabled = false;
    Clock::time_point start;
};

} // namespace timestral::vulkan
