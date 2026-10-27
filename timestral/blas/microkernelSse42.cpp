/***************
 * @file microkernelSse42.cpp
 * @brief 保持6乘16打包的独立SSE微内核
 * 只用intrinsic和手写循环，局部模板保持内链，避免跨ISA的可执行COMDAT
 ***************/

#include "timestral/blas/microkernel.hpp"

#include <immintrin.h>
#include <utility>

namespace timestral::blas {

template <std::size_t Row>
static inline void accumulateSseRow(const float* a, __m128 b0, __m128 b1, __m128* sums) {
    const __m128 value = _mm_set1_ps(a[Row]);
    sums[Row * 2] = _mm_add_ps(sums[Row * 2], _mm_mul_ps(value, b0));
    sums[Row * 2 + 1] = _mm_add_ps(sums[Row * 2 + 1], _mm_mul_ps(value, b1));
}

template <std::size_t... Rows>
static inline void accumulateSseStep(
    const float* a,
    const float* b,
    __m128* sums,
    std::index_sequence<Rows...>
) {
    const __m128 b0 = _mm_loadu_ps(b);
    const __m128 b1 = _mm_loadu_ps(b + 4);
    (accumulateSseRow<Rows>(a, b0, b1, sums), ...);
}

template <std::size_t Rows>
static void computeSseHalf(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    __m128 sums[Rows * 2];
    for (auto& value : sums) {
        value = _mm_setzero_ps();
    }
    if (beta != 0.0f) {
        const __m128 scale = _mm_set1_ps(beta);
        for (std::size_t row = 0; row < Rows; ++row) {
            for (std::size_t half = 0; half < 2; ++half) {
                sums[row * 2 + half] = _mm_mul_ps(_mm_loadu_ps(c + row * rowStride + half * 4), scale);
            }
        }
    }
    // 每个输出只有一条按K递增的mul加add链，B保持16列步幅
    for (std::size_t p = 0; p < k; ++p) {
        accumulateSseStep(a + p * 6, b + p * 16, sums, std::make_index_sequence<Rows>{});
    }
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t half = 0; half < 2; ++half) {
            _mm_storeu_ps(c + row * rowStride + half * 4, sums[row * 2 + half]);
        }
    }
}

template <std::size_t Rows>
static void computeSseRows(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    // 6乘8只需12个累加XMM，两个列块不改变各输出的K累加顺序
    computeSseHalf<Rows>(k, a, b, c, rowStride, beta);
    computeSseHalf<Rows>(k, a, k == 0 ? b : b + 8, c + 8, rowStride, beta);
}

void computeSse42Kernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    computeSseRows<6>(k, a, b, c, rowStride, beta);
}

void computeSse42NarrowRows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    switch (rows) {
    case 1:
        computeSseRows<1>(k, a, b, c, rowStride, beta);
        break;
    case 2:
        computeSseRows<2>(k, a, b, c, rowStride, beta);
        break;
    case 3:
        computeSseRows<3>(k, a, b, c, rowStride, beta);
        break;
    case 4:
        computeSseRows<4>(k, a, b, c, rowStride, beta);
        break;
    case 5:
        computeSseRows<5>(k, a, b, c, rowStride, beta);
        break;
    default:
        computeSseRows<6>(k, a, b, c, rowStride, beta);
        break;
    }
}

} // namespace timestral::blas
