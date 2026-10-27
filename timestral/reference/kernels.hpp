/***************
 * @file kernels.hpp
 * @brief 连续FP32参考算子
 * 移植自Mistral AI Timestral，Apache-2.0，基线a534bdbd
 ***************/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "timestral/utils/attention.hpp"
#include "timestral/utils/safetensors.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral::reference {

// 有序FP32归约，支持共享前缀分段视图
void attention(const AttentionView& view, const AttentionMask& mask);

// @brief 二维矩阵相乘
// @param left 形状(M,K)
// @param right 形状(K,N)
// @return 形状(M,N)的输出
// @throw std::invalid_argument 形状不兼容时抛出
Tensor matmul(const Tensor& left, const Tensor& right);

// @brief 最后一维线性投影，权重布局为(out,in)
// @param input 形状(...,in)
// @param weight 权重，形状(out,in)
// @return 形状(...,out)
// @throw std::invalid_argument 输入宽度或权重形状不符时抛出
Tensor linear(const Tensor& input, const utils::Weight& weight);

// @brief 带偏置的线性投影
// @param input 形状(...,in)
// @param weight 权重，形状(out,in)
// @param bias 偏置，形状(out)
// @return 形状(...,out)
// @throw std::invalid_argument 输入宽度、权重或偏置形状不符时抛出
Tensor linear(const Tensor& input, const utils::Weight& weight, const utils::Weight& bias);

// @brief RMS归一化
// @param input 形状(...,D)
// @param scale 缩放权重，形状(D)
// @param epsilon 数值稳定项，必须为正
// @return 与input同形
// @throw std::invalid_argument 形状或epsilon非法时抛出
Tensor rmsNorm(const Tensor& input, const utils::Weight& scale, float epsilon);

// @brief SiLU激活
// @param value 输入
// @return value/(1+exp(-value))
float silu(float value);

// @brief Softplus激活
// @param value 输入
// @return value>20时原样返回，否则log1p(exp(value))
float softplus(float value);

// @brief 随机访问迭代器区间上的原地稳定softmax
// @param first 区间起点
// @param last 区间终点
// @note 全被屏蔽（负无穷）的区间置零，累加顺序与向量重载一致
template <class It>
void softmax(It first, It last) {
    if (first == last) {
        return;
    }
    const float maximum = *std::max_element(first, last);
    if (maximum == -std::numeric_limits<float>::infinity()) {
        std::fill(first, last, 0.0f);
        return;
    }
    float sum = 0.0f;
    for (It it = first; it != last; ++it) {
        *it = std::exp(*it - maximum);
        sum += *it;
    }
    for (It it = first; it != last; ++it) {
        *it /= sum;
    }
}

// @brief 原地稳定softmax
// @param values 待归一化的分数，原地修改
// @note 全被屏蔽（负无穷）的行返回零
void softmax(std::vector<float>& values);

// @brief 按上游顺序计算门控激活
// @param fused 形状(...,2H)，前H段为gate，后H段为up
// @return 形状(...,H)，即gate * silu(up)
// @throw std::invalid_argument 末维为零或奇数时抛出
Tensor swiGLU(const Tensor& fused);

// @brief 部分维度RoPE旋转
// @param input 形状(R,N,H,D)，原地修改
// @param positionIds 每行每token的位置，长度R*N
// @param rotaryDim 参与旋转的维数，必须为正偶数且不超过D
// @param theta RoPE基频，必须为正
// @note 配对D的两半，非交错配对
// @throw std::invalid_argument 形状、位置数或旋转参数非法时抛出
void rotary(
    Tensor& input,
    const std::vector<double>& positionIds,
    std::size_t rotaryDim,
    float theta
);

// @brief 把原始输出变换成单调分位数
// @param raw 形状(...,Q)的原始输出
// @param levels 升序分位点，必须含0.5
// @param widthFactor 宽度系数，必须有限且非负
// @return 与raw同形，以0.5分位为锚点向两侧累加softplus宽度并夹持到FP32范围
// @throw std::invalid_argument 分位点或形状非法时抛出
Tensor monotoneQuantiles(const Tensor& raw, const std::vector<float>& levels, float widthFactor);

} // namespace timestral::reference
