/***************
 * @file elementwise.hpp
 * @brief 在线softmax行更新、sigmoid门控与SwiGLU的标量及AVX2实现
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>

namespace timestral::blas {

// 一行至多64个键的在线softmax分块更新，scores须可读到count向上取8的倍数
// allowed须指向至少64字节的有效数组，非0表示允许，probabilities按stride写count个概率，屏蔽键写0
// accumulator的width个元素按alpha缩放，整行屏蔽时返回false且不改任何状态，允许键出现NaN时状态变为NaN
bool softmaxRowScalar(
    const float* scores,
    const std::uint8_t* allowed,
    std::size_t count,
    float factor,
    float& maximum,
    float& denominator,
    float* accumulator,
    std::size_t width,
    float* probabilities,
    std::size_t stride
);
bool softmaxRowNative(
    const float* scores,
    const std::uint8_t* allowed,
    std::size_t count,
    float factor,
    float& maximum,
    float& denominator,
    float* accumulator,
    std::size_t width,
    float* probabilities,
    std::size_t stride
);

// values[i] *= 1 / (1 + exp(-gate[i]))
void sigmoidGateScalar(float* values, const float* gate, std::size_t count);
void sigmoidGateNative(float* values, const float* gate, std::size_t count);

// 每行output[j] = fused[j] * SiLU(fused[width + j])，fused行宽2*width
void swiGLUScalar(const float* fused, float* output, std::size_t rows, std::size_t width);
void swiGLUNative(const float* fused, float* output, std::size_t rows, std::size_t width);

} // namespace timestral::blas
