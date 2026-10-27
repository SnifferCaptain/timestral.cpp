/***************
 * @file postprocess.hpp
 * @brief 与后端无关的分位数后处理：帧折叠、范围cap、对称平均与非负截断。
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace timestral {

// @brief 把一个归一化中位数折叠回目标空间。
// @param normalizedMedian 归一化中位数，(中位数-loc0)/scale0。
// @param scale 该块advancing尺度。
// @param mean 已合并块的均值。
// @param offset 初始目标窗的偏移。
// @param clip 截断上界，防止sinh溢出。
// @return 截断到[-clip,clip]后再加回offset的数值。
// @note 与上游advancing refine frame一致；clip通常取sqrt(FLT_MAX)/1e3。
double foldAdvancingFrame(double normalizedMedian, double scale, double mean, double offset, double clip);

// @brief 把一条轨迹的预测值限制在初始观测窗的扩展范围内。
// @param values 待原地修改的连续存储。
// @param begin 该轨迹在存储中的起始下标。
// @param count 该轨迹的元素数。
// @param low 初始目标窗的观测下界。
// @param high 初始目标窗的观测上界。
// @param width cap宽度，通常为该轨迹归一化尺度乘以cap系数。
// @note 调用方对每条轨迹各调用一次，结果不参与自回归反馈。
void capQuantileRange(
    std::vector<float>& values,
    std::size_t begin,
    std::size_t count,
    float low,
    float high,
    float width
);

// @brief 按分位点对称配对做平均。
// @param quantiles 主结果，形状(B,V,H,Q)连续行主序，原地改写。
// @param mirrored 镜像结果，布局与quantiles相同。
// @param levels 分位点数Q。
// @note 写入0.5 * (quantiles - mirrored的镜像分量)，即q与(Q-1-q)配对。
// @throw std::invalid_argument 长度不匹配或levels为0时抛出。
void averageMirrored(
    std::vector<float>& quantiles,
    const std::vector<float>& mirrored,
    std::size_t levels
);

// @brief 判断每条轨迹的历史是否非负。
// @param values 形状(R,T)连续行主序的读数。
// @param observed 与values同形，非0表示观测。
// @param rowCount 轨迹数。
// @param time 时间轴长度。
// @return 每轨迹一个标志：存在观测且所有观测值非负时为1，否则为0。
// @note NaN不参与判断，按缺失处理。
std::vector<std::uint8_t> nonnegativeTrajectories(
    const std::vector<float>& values,
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time
);

} // namespace timestral
