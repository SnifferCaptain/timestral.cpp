/***************
 * @file microkernelNarrowAvx2.cpp
 * @brief 小M矩阵的独立AVX2微内核
 * 行数由基线单元校验，本单元不构造异常或实例化std内联函数
 ***************/

#include "timestral/blas/microkernel.hpp"

#include <immintrin.h>
#include <utility>

namespace timestral::blas {

template <std::size_t Row>
static inline void accumulateRow(const float* a, __m256 b0, __m256 b1, __m256* sums) {
    const __m256 value = _mm256_broadcast_ss(a + Row);
    sums[Row * 2] = _mm256_fmadd_ps(value, b0, sums[Row * 2]);
    sums[Row * 2 + 1] = _mm256_fmadd_ps(value, b1, sums[Row * 2 + 1]);
}

template <std::size_t... Rows>
static inline void accumulateStep(
    const float* a,
    const float* b,
    __m256* sums,
    std::index_sequence<Rows...>
) {
    const __m256 b0 = _mm256_loadu_ps(b);
    const __m256 b1 = _mm256_loadu_ps(b + 8);
    (accumulateRow<Rows>(a, b0, b1, sums), ...);
}

template <std::size_t Rows>
static void computeNarrow(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    __m256 sums[Rows * 2];
    for (auto& value : sums) {
        value = _mm256_setzero_ps();
    }
    if (beta != 0.0f) {
        const __m256 scale = _mm256_set1_ps(beta);
        for (std::size_t row = 0; row < Rows; ++row) {
            for (std::size_t half = 0; half < 2; ++half) {
                sums[row * 2 + half] = _mm256_mul_ps(_mm256_loadu_ps(c + row * rowStride + half * 8), scale);
            }
        }
    }
    // 行以编译期索引展开，K按顺序单链累加，不按运行期rows迭代
    for (std::size_t p = 0; p < k; ++p) {
        accumulateStep(a + p * 6, b + p * 16, sums, std::make_index_sequence<Rows>{});
    }
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t half = 0; half < 2; ++half) {
            _mm256_storeu_ps(c + row * rowStride + half * 8, sums[row * 2 + half]);
        }
    }
}

void computeNarrowRows(
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
        computeNarrow<1>(k, a, b, c, rowStride, beta);
        break;
    case 2:
        computeNarrow<2>(k, a, b, c, rowStride, beta);
        break;
    case 3:
        computeNarrow<3>(k, a, b, c, rowStride, beta);
        break;
    case 4:
        computeNarrow<4>(k, a, b, c, rowStride, beta);
        break;
    case 5:
        computeNarrow<5>(k, a, b, c, rowStride, beta);
        break;
    default:
        computeNativeKernel(k, a, b, c, rowStride, beta);
        break;
    }
}

} // namespace timestral::blas
