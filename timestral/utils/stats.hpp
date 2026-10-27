/***************
 * @file stats.hpp
 * @brief Welford加权均值/方差状态的合并。
 ***************/

#pragma once

namespace timestral {

// 加权Welford状态，方差按m2 / (count - 1)还原，与上游advancing frame的样本方差口径一致。
struct Welford {
    double count = 0.0; // 权重和
    double mean = 0.0; // 加权均值
    double m2 = 0.0; // 加权平方差和
};

// @brief 把suffix的观测接在prefix之后合并两组统计。
// @param prefix 前一组状态。
// @param suffix 后一组状态，按观测顺序接在prefix之后。
// @return 合并后的状态；两者都为空时返回零状态。
// @note 不修改入参；调用方固定合并顺序，以免改变浮点累加顺序。
Welford combine(const Welford& prefix, const Welford& suffix);

} // namespace timestral
