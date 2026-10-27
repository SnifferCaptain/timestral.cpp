/***************
 * @file decodeFallback.cpp
 * @brief 未编译AVX2时的decode热点链接回退
 ***************/

#include "timestral/blas/decodeKernel.hpp"

#include "timestral/blas/lowPrecision.hpp"

namespace timestral::blas {

void decodeHalfNative(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const std::uint16_t* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t j = 0; j < 16; ++j) {
            float value = beta == 0.0f ? 0.0f : beta * c[row * rowStride + j];
            for (std::size_t p = 0; p < k; ++p) {
                const float weight = halfToFloat(b[p * 16 + j]);
                value += a[row * k + p] * weight;
            }
            c[row * rowStride + j] = value;
        }
    }
}

void decodeBfloatNative(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const std::uint16_t* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t j = 0; j < 16; ++j) {
            float value = beta == 0.0f ? 0.0f : beta * c[row * rowStride + j];
            for (std::size_t p = 0; p < k; ++p) {
                const float weight = bfloat16ToFloat(b[p * 16 + j]);
                value += a[row * k + p] * weight;
            }
            c[row * rowStride + j] = value;
        }
    }
}

void decodeInt8Native(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const std::int8_t* b,
    const float* scales,
    float* c,
    std::size_t rowStride,
    float beta
) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t j = 0; j < 16; ++j) {
            float value = beta == 0.0f ? 0.0f : beta * c[row * rowStride + j];
            for (std::size_t p = 0; p < k; ++p) {
                const float weight = static_cast<float>(b[p * 16 + j]) * scales[p / 32 * 16 + j];
                value += a[row * k + p] * weight;
            }
            c[row * rowStride + j] = value;
        }
    }
}

} // namespace timestral::blas
