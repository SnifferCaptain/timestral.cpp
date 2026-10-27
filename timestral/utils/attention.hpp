/***************
 * @file attention.hpp
 * @brief 后端共用的单头注意力视图与结构掩码
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace timestral {

// 一次调用处理一个请求行与一个头，offset和stride均以FP32元素计
// 各矩阵内层headDim连续，query/output为queryCount行，键值总计keyCount行
// 前prefixCount行读取prefixKeys/prefixValues，后续行按key-prefixCount读取keys/values
struct AttentionView {
    const std::vector<float>& query;                    // 查询存储
    const std::vector<float>& keys;                     // 键存储
    const std::vector<float>& values;                   // 值存储
    std::vector<float>& output;                         // 结果存储
    std::size_t queryCount = 0;                         // 查询行数
    std::size_t keyCount = 0;                           // 键行数
    std::size_t headDim = 0;                            // 单头宽度
    std::size_t queryOffset = 0;                        // 查询首元素
    std::size_t keyOffset = 0;                          // 键首元素
    std::size_t valueOffset = 0;                        // 值首元素
    std::size_t outputOffset = 0;                       // 结果首元素
    std::size_t queryStride = 0;                        // 查询行距
    std::size_t keyStride = 0;                          // 键行距
    std::size_t valueStride = 0;                        // 值行距
    std::size_t outputStride = 0;                       // 结果行距
    const std::vector<float>& prefixKeys = keys;        // 共享前缀键
    const std::vector<float>& prefixValues = values;    // 共享前缀值
    std::size_t prefixCount = 0;                        // 前缀键行数
    std::size_t prefixOffset = 0;                       // 前缀首元素
    std::size_t prefixStride = 0;                       // 前缀行距
    bool stableKeyBlocks = false;                      // time轴短前缀也保持相同分块归约
};

enum class MaskTile {
    Empty,
    Full,
    Partial
};

// allowed接收视图内从0开始的query/key下标，classify接收起点与行列数量
// classify可为空，此时后端须按allowed精确扫描，不假设纯因果掩码
// fill(query,keyBegin,keyCount,bits)为一行连续keyCount个键写0/1到bits前keyCount项，可为空
// 非空时必须与allowed逐元素一致，bits长度由调用方保证不小于keyCount
struct AttentionMask {
    std::function<bool(std::size_t, std::size_t)> allowed;
    std::function<MaskTile(std::size_t, std::size_t, std::size_t, std::size_t)> classify;
    std::function<void(std::size_t, std::size_t, std::size_t, std::vector<std::uint8_t>&)> fill;
};

} // namespace timestral
