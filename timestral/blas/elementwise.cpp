/***************
 * @file elementwise.cpp
 * @brief 逐元素算子的标量实现，与reference的运算顺序一致
 ***************/

#include "timestral/blas/elementwise.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace timestral::blas {

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
) {
    constexpr float negative = -std::numeric_limits<float>::infinity();
    float scaled[64] = {};
    float tileMaximum = negative;
    bool hasNaN = false;
    for (std::size_t key = 0; key < count; ++key) {
        scaled[key] = allowed[key] != 0 ? scores[key] * factor : negative;
        tileMaximum = std::max(tileMaximum, scaled[key]);
        hasNaN = hasNaN || std::isnan(scaled[key]);
    }
    // 允许键中出现NaN时整行最大值取NaN并向O与l传播，不当作整行屏蔽
    if (hasNaN) {
        tileMaximum = std::numeric_limits<float>::quiet_NaN();
    }
    if (tileMaximum == negative) {
        return false;
    }
    const float nextMaximum = hasNaN || std::isnan(maximum) ? tileMaximum + maximum : std::max(maximum, tileMaximum);
    const float alpha = denominator == 0.0f ? 0.0f : std::exp(maximum - nextMaximum);
    denominator *= alpha;
    for (std::size_t d = 0; d < width; ++d) {
        accumulator[d] *= alpha;
    }
    for (std::size_t key = 0; key < count; ++key) {
        const float probability = scaled[key] == negative ? 0.0f : std::exp(scaled[key] - nextMaximum);
        probabilities[key * stride] = probability;
        denominator += probability;
    }
    maximum = nextMaximum;
    return true;
}

void sigmoidGateScalar(float* values, const float* gate, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        values[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
    }
}

void swiGLUScalar(const float* fused, float* output, std::size_t rows, std::size_t width) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t j = 0; j < width; ++j) {
            const float value = fused[row * 2 * width + width + j];
            output[row * width + j] = fused[row * 2 * width + j] * (value / (1.0f + std::exp(-value)));
        }
    }
}

} // namespace timestral::blas
