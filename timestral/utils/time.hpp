/***************
 * @file time.hpp
 * @brief 实际采样时间到无量纲patch旋转坐标
 ***************/

#pragma once

#include <cstddef>
#include <vector>

namespace timestral {

// @brief 用patch内采样时间的中心构造实数坐标，同组通道共用时间轴
// @param times 按(B,T)排列的有限且递增时间戳，T为patchSize倍数
// @param batch 请求数
// @param channels 每组通道数，包含本次模型输入的协变量
// @param time 每组上下文采样点数
// @param horizonPatches 未来patch数，按interval外推采样时间
// @param patchSize 每个patch的采样点数
// @param interval 预测采样间隔，单位秒
// @param referenceInterval 旋转的参考采样间隔，0采用interval
// @return 按(B*channels,T/patchSize+horizonPatches)排列的坐标
// @note 只改变patch间位置，不能表示patch投影内部的所有采样间隔
// @throw std::invalid_argument 形状、时间戳或间隔非法时抛出
std::vector<double> rotaryCoordinates(
    const std::vector<double>& times,
    std::size_t batch,
    std::size_t channels,
    std::size_t time,
    std::size_t horizonPatches,
    std::size_t patchSize,
    double interval,
    double referenceInterval
);

} // namespace timestral
