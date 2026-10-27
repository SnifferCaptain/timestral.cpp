/***************
 * @file attention.hpp
 * @brief 块掩码、有序归约与分块在线注意力
 ***************/

#pragma once

#include <cstddef>

#include "timestral/blas/gemm.hpp"
#include "timestral/utils/attention.hpp"

namespace timestral::blas {

struct AttentionStats {
    std::size_t emptyTiles = 0;
    std::size_t fullTiles = 0;
    std::size_t partialTiles = 0;
    std::size_t scoreElements = 0;
};

// 两遍softmax与有序SIMD QK/PV，分块查询限制工作区，不重排每个dot的K顺序
void orderedAttention(const AttentionView& view, const AttentionMask& mask);

// 保留原生1/sqrt(headDim)缩放，全屏蔽行输出零，允许键的NaN分数传播到输出，不改变视图外的元素
// stats每次重置，mask.classify与mask.fill必须与allowed一致
// keyCount超过16时按6×64块执行，mode可强制标量微内核，stats记录块分类与参与计算的分数元素
// keyCount不超过16且未要求stableKeyBlocks时直接两遍softmax，不受mode影响
// time轴要求稳定分块，避免短prefill与长查询全量路径使用不同归约
void onlineAttention(
    const AttentionView& view,
    const AttentionMask& mask,
    AttentionStats& stats,
    GemmMode mode = GemmMode::Auto,
    bool stableKeyBlocks = false
);

} // namespace timestral::blas
