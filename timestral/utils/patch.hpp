/***************
 * @file patch.hpp
 * @brief 与后端无关的patch级计算：因果归一化、patch输入装配、位置/掩码、页键与页表校验
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "timestral/utils/cache.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

// @brief 把block_schedule的一项展开成注意力轴序列
// @param kind 上游层类型：time、variate、time_variate或variate_time
// @return 按子层顺序排列的轴，true为time轴，false为variate轴
// @throw std::invalid_argument 取值未知时抛出
std::vector<bool> attentionAxes(const std::string& kind);

// 连续同组区间，组号不连续重现时回退全行区间以保留任意groupIds语义
std::vector<std::pair<std::size_t, std::size_t>> attentionGroups(const std::vector<std::size_t>& groupIds);

// @brief 统计time轴注意力子层数
// @param config 模型配置
// @return time轴子层总数，页的层编号按blockSchedule中的出现顺序
std::size_t timeLayerCount(const ModelConfig& config);

// @brief 计算单个观测点的归一化值
// @param value 原始读数
// @param loc 该patch的均值
// @param scale 该patch的尺度，调用方已保证不低于minimumScale
// @param useAsinh 为true时在缩放后再取asinh
// @return (value-loc)/max(scale,FP32 epsilon)，可选asinh
float scaledValue(float value, float loc, float scale, bool useAsinh);

// @brief 从既有状态继续做因果归一化
// @param context 形状(rowCount,T)的连续行主序输入，T为正patch倍数
// @param observed 与context同形，非0表示观测
// @param startStates 每个patch起点前已累计的状态，共rowCount项，为空表示从零开始
// @param startPatch 起始patch下标
// @param endPatch 结束patch下标，不含
// @param config 模型配置
// @param scaled 输出，形状(rowCount,count*P)
// @param loc 输出，形状(rowCount,count*P)
// @param scale 输出，形状(rowCount,count*P)
// @param states 输出，rowCount*count，保存每个patch末尾的累计量
// @note count为endPatch-startPatch，P为patchSize，累加顺序、asinh与minimumScale与上游causal_patch_loc_scale一致
void normalizePatches(
    const Tensor& context,
    const std::vector<std::uint8_t>& observed,
    const std::vector<PageScaleState>& startStates,
    std::size_t startPatch,
    std::size_t endPatch,
    const ModelConfig& config,
    std::vector<float>& scaled,
    std::vector<float>& loc,
    std::vector<float>& scale,
    std::vector<PageScaleState>& states
);

// @brief 装配patch输入张量
// @param scaled 归一化值，按(rowCount,count,P)局部布局
// @param observed 与context同形，非0表示观测
// @param rowCount 轨迹数
// @param time 原始时间轴长度
// @param startPatch 起始patch下标
// @param count patch数
// @param patchSize 每个patch的采样点数P
// @param patchInput 输出(rowCount,count,2P)：前P段为归一化值，后P段为缺失指示（缺失为1）
void assemblePatchInput(
    const std::vector<float>& scaled,
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time,
    std::size_t startPatch,
    std::size_t count,
    std::size_t patchSize,
    std::vector<float>& patchInput
);

// @brief 求每个组的首个观测patch
// @param observed 与context同形，非0表示观测
// @param groupIds 每行一个组号
// @param rowCount 轨迹数
// @param time 时间轴长度
// @param patchSize 每个patch的采样点数
// @return 组号到首个观测patch的映射，整组没有观测时取0，与上游first fallback一致
std::map<std::size_t, std::size_t> firstObservedPatches(
    const std::vector<std::uint8_t>& observed,
    const std::vector<std::size_t>& groupIds,
    std::size_t rowCount,
    std::size_t time,
    std::size_t patchSize
);

// @brief 计算每个patch能否作为后续槽的key
// @param observed 与context同形，非0表示观测
// @param rowCount 轨迹数
// @param time 时间轴长度
// @param patchSize 每个patch的采样点数
// @param attendMissingPatches 缺失patch是否可被关注
// @param keyValid 输出(rowCount,time/patchSize)，非0表示有效
// @note 有效条件：该patch有观测，或允许关注缺失且该行此前已出现观测
void buildKeyValid(
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time,
    std::size_t patchSize,
    bool attendMissingPatches,
    std::vector<std::uint8_t>& keyValid
);

// @brief 计算槽的rotary位置
// @param rowCount 轨迹数
// @param count 槽数
// @param firstSlot 首个槽的全局下标
// @param encoded 上下文编码槽数，带协变量时目标槽跳过该偏移
// @param covariates 是否带协变量行
// @param groupIds 每行一个组号
// @param first 组号到首个观测patch的映射
// @param positions 输出(rowCount*count)，相对该组首个观测patch
// @param coordinates 可选(R,C+H)实数坐标，空时使用槽序号
void buildPositions(
    std::size_t rowCount,
    std::size_t count,
    std::size_t firstSlot,
    std::size_t encoded,
    bool covariates,
    const std::vector<std::size_t>& groupIds,
    const std::map<std::size_t, std::size_t>& first,
    std::vector<double>& positions,
    const std::vector<double>& coordinates = {}
);

// @brief 校验页池几何与模型一致
// @param pool 页池
// @param config 模型配置
// @param rowCount 本次预测的行数
// @note 几何不一致时页内K/V布局无法解释
// @throw std::invalid_argument 行数、头数、头维或time轴层数不匹配时抛出
void verifyPoolGeometry(const PagePool& pool, const ModelConfig& config, std::size_t rowCount);

// @brief 校验页表保存的网格与本次context一致
// @param pool 页池
// @param table 页表，按patch顺序
// @param rowCount 轨迹数
// @param keyValid 每个patch的有效位
// @param groupIds 每行一个组号
// @param first 组号到首个观测patch的映射
// @param coordinates 可选实数patch坐标，空则按离散位置校验
// @throw std::invalid_argument rotary位置或掩码位不一致时抛出
void verifyTableGrid(
    const PagePool& pool,
    const std::vector<std::size_t>& table,
    std::size_t rowCount,
    const std::vector<std::uint8_t>& keyValid,
    const std::vector<std::size_t>& groupIds,
    const std::map<std::size_t, std::size_t>& first,
    const std::vector<double>& coordinates = {},
    double positionOffset = 0.0
);

// @brief 校验整组页表与批内各请求的对应通道网格一致
// @param pool 页池，几何为一组通道
// @param table 共享前缀页表
// @param rowCount 本次解码的行数
// @param keyValid 每行每patch的有效位
// @param groupIds 每行一个组号
// @param first 组号到首个观测patch的映射
// @param coordinates 可选实数patch坐标，空则按离散位置校验
// @note 共享K/V按row%V取页内通道，组号必须按完整通道组隔离
// @throw std::invalid_argument 组布局、对应通道的位置或掩码位不一致时抛出
void verifySharedTableGrid(
    const PagePool& pool,
    const std::vector<std::size_t>& table,
    std::size_t rowCount,
    const std::vector<std::uint8_t>& keyValid,
    const std::vector<std::size_t>& groupIds,
    const std::map<std::size_t, std::size_t>& first,
    const std::vector<double>& coordinates = {}
);

// @brief 计算一个patch的页键与输入指纹
// @param context 形状(rowCount,T)的连续行主序输入
// @param observed 与context同形，非0表示观测
// @param rowCount 轨迹数
// @param time 时间轴长度
// @param patch 目标patch下标
// @param patchSize 每个patch的采样点数
// @param maskBits 该patch每行的掩码位
// @param positions 该patch每行的rotary位置
// @param parentKey 父页键，把整条前缀串进键里
// @param configFingerprint 配置指纹
// @param key 输出页键
// @param inputFingerprint 输出输入指纹，用于二次校验
void pageKeys(
    const Tensor& context,
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time,
    std::size_t patch,
    std::size_t patchSize,
    const std::vector<std::uint8_t>& maskBits,
    const std::vector<double>& positions,
    std::uint64_t parentKey,
    std::uint64_t configFingerprint,
    std::uint64_t& key,
    std::uint64_t& inputFingerprint
);

} // namespace timestral
