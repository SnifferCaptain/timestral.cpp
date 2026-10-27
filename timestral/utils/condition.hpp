/***************
 * @file condition.hpp
 * @brief 按观测可用性分段的多通道条件补全
 ***************/

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "timestral/timestral.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

class Engine;

// @brief 把已知读数放到公共网格，不插值
// @param request 各通道的原始读数和可选间隔
// @param patch 模型patch长度，用于最低上下文校验
// @param ends 输出各通道末次有效观测之后的网格下标
// @param interval 输出公共网格间隔，单位秒
// @param origin 输出网格起点，单位秒
// @return 形状(V,T)的观测网格，未观测处为NaN
// @throw std::invalid_argument 时间无法对齐、没有有效读数或样本不合法时抛出
Tensor alignObservations(
    const Timestral::Request& request,
    std::size_t patch,
    std::vector<std::size_t>& ends,
    double& interval,
    double& origin
);

// @brief 按共同观测布局分段拼批，原始读数保持各请求独立
// @param engine 已加载的内部引擎
// @param requests 请求的只读借用，不复制采样序列
// @param observations 已校验的(B,V,T)公共网格，预测窗口内的观测布局一致
// @param ends 各通道末次有效观测之后的共享网格下标
// @param interval 公共网格间隔，单位秒
// @param origins 各请求的网格起点，单位秒
// @return 与requests同序的结果，计时为共同拼批执行的耗时
// @note 不跨角色阶段复用KV，不回填首段之前的历史缺口
std::vector<Timestral::Result> predictConditionedBatch(
    const Engine& engine,
    const std::vector<std::reference_wrapper<const Timestral::Request>>& requests,
    const Tensor& observations,
    const std::vector<std::size_t>& ends,
    double interval,
    const std::vector<double>& origins
);

} // namespace timestral
