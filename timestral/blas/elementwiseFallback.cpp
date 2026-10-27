/***************
 * @file elementwiseFallback.cpp
 * @brief 未编译AVX2时逐元素原生入口的链接替代
 ***************/

#include "timestral/blas/elementwise.hpp"

namespace timestral::blas {

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
) {
    return softmaxRowScalar(
        scores,
        allowed,
        count,
        factor,
        maximum,
        denominator,
        accumulator,
        width,
        probabilities,
        stride
    );
}

void sigmoidGateNative(float* values, const float* gate, std::size_t count) {
    sigmoidGateScalar(values, gate, count);
}

void swiGLUNative(const float* fused, float* output, std::size_t rows, std::size_t width) {
    swiGLUScalar(fused, output, rows, width);
}

} // namespace timestral::blas
