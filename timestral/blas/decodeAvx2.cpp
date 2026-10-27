/***************
 * @file decodeAvx2.cpp
 * @brief AVX2低精度面板直接GEMV，FP16入口另要求F16C
 ***************/

#include "timestral/blas/decodeKernel.hpp"

#include <immintrin.h>

namespace timestral::blas {

template <std::size_t Rows, typename Load>
static void projectRows(
    std::size_t k,
    const float* a,
    float* c,
    std::size_t rowStride,
    float beta,
    Load load
) {
    __m256 accumulators[Rows][2] = {};
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t lane = 0; lane < 2; ++lane) {
            accumulators[row][lane] = beta == 0.0f ? _mm256_setzero_ps()
                : _mm256_mul_ps(_mm256_loadu_ps(c + row * rowStride + lane * 8), _mm256_set1_ps(beta));
        }
    }
    for (std::size_t p = 0; p < k; ++p) {
        const __m256 first = load(p, 0);
        const __m256 second = load(p, 8);
        for (std::size_t row = 0; row < Rows; ++row) {
            const __m256 value = _mm256_broadcast_ss(a + row * k + p);
            accumulators[row][0] = _mm256_fmadd_ps(value, first, accumulators[row][0]);
            accumulators[row][1] = _mm256_fmadd_ps(value, second, accumulators[row][1]);
        }
    }
    for (std::size_t row = 0; row < Rows; ++row) {
        _mm256_storeu_ps(c + row * rowStride, accumulators[row][0]);
        _mm256_storeu_ps(c + row * rowStride + 8, accumulators[row][1]);
    }
}

template <typename Load>
static void projectSmall(
    std::size_t rows,
    std::size_t k,
    const float* a,
    float* c,
    std::size_t rowStride,
    float beta,
    Load load
) {
    switch (rows) {
    case 1:
        projectRows<1>(k, a, c, rowStride, beta, load);
        break;
    case 2:
        projectRows<2>(k, a, c, rowStride, beta, load);
        break;
    case 3:
        projectRows<3>(k, a, c, rowStride, beta, load);
        break;
    case 4:
        projectRows<4>(k, a, c, rowStride, beta, load);
        break;
    }
}

void decodeHalfNative(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const std::uint16_t* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    projectSmall(rows, k, a, c, rowStride, beta, [&](std::size_t p, std::size_t lane) {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + p * 16 + lane)));
    });
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
    projectSmall(rows, k, a, c, rowStride, beta, [&](std::size_t p, std::size_t lane) {
        const __m256i bits = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + p * 16 + lane)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(bits, 16));
    });
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
    projectSmall(rows, k, a, c, rowStride, beta, [&](std::size_t p, std::size_t lane) {
        const __m256i values = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + p * 16 + lane)));
        return _mm256_mul_ps(_mm256_cvtepi32_ps(values), _mm256_loadu_ps(scales + p / 32 * 16 + lane));
    });
}

} // namespace timestral::blas
