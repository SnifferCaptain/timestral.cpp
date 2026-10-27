/***************
 * @file forecast.hpp
 * @brief 公开入口与流式会话共用的历史校验、步数推导与分位数结果提取
 ***************/

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "timestral/timestral.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

// @brief 校验历史序列的时间戳与读数
// @param history 待校验的采样点
// @param patch 模型patch长度，样本数至少为其两倍
// @param name 错误消息中的调用名
// @note 时间戳必须有限且严格递增，读数须有限或NaN，NaN表示缺测，无穷非法
// @throw std::invalid_argument 样本不足、时间戳非递增或读数非法时抛出
void validateHistory(
    const std::vector<Sample>& history,
    std::size_t patch,
    const std::string& name
);

// @brief 解析生效的采样间隔
// @param history 采样点，间隔为0时用于推断
// @param sampleInterval 调用方给定的间隔，0表示从history相邻时间戳推断
// @param name 错误消息中的调用名
// @return 最终的采样间隔，单位秒
// @throw std::invalid_argument 间隔为负或样本不足无法推断时抛出
double resolveInterval(
    const std::vector<Sample>& history,
    double sampleInterval,
    const std::string& name
);

// @brief 由预测跨度推导步数
// @param horizonSeconds 预测跨度，必须为正且有限
// @param interval 采样间隔，必须为正
// @param name 错误消息中的调用名
// @return round(horizonSeconds/interval)，至少为1
// @throw std::invalid_argument 跨度非正或超出可表示范围时抛出
std::size_t stepCount(double horizonSeconds, double interval, const std::string& name);

// @brief 选出参与预测的历史区间
// @param history 全部采样点
// @param limit 上下文上限，按样本数计
// @param patch 模型patch长度
// @param name 错误消息中的调用名
// @return 起始下标与保留样本数，超出上限时保留最近的整patch
// @throw std::invalid_argument 保留长度不足2*patch时抛出
std::pair<std::size_t, std::size_t> contextRange(
    const std::vector<Sample>& history,
    std::size_t limit,
    std::size_t patch,
    const std::string& name
);

// @brief 把单个读数额度转成FP32
// @param history 采样点
// @param index 待转换的下标
// @param name 错误消息中的调用名
// @return FP32读数，NaN保持缺失
// @throw std::invalid_argument 读数超出FP32范围时抛出，消息含样本下标
float fp32Sample(
    const std::vector<Sample>& history,
    std::size_t index,
    const std::string& name
);

// @brief 取模型配置的分位点水平
// @param config 模型配置
// @return 升序分位点，已转成double
std::vector<double> quantileLevels(const ModelConfig& config);

// @brief 取一条轨迹的预测分位数
// @param quantiles 形状(B,V,H,Q)的预测结果
// @param batchIndex 批下标
// @param steps 期望的步数H
// @param levels 期望的分位点数Q
// @return 按(V,H,Q)行主序展开的该请求结果
// @throw std::runtime_error shape与期望不符时抛出
std::vector<double> quantileRow(
    const Tensor& quantiles,
    std::size_t batchIndex,
    std::size_t steps,
    std::size_t levels
);

} // namespace timestral
