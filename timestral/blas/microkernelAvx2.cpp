/***************
 * @file microkernelAvx2.cpp
 * @brief AVX2及FMA3的独立6乘16微内核
 ***************/

#include "timestral/blas/microkernel.hpp"

#include <immintrin.h>

namespace timestral::blas {

bool nativeKernelAvailable() {
    return true;
}

void computeNativeKernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();
    if (beta != 0.0f) {
        const __m256 scale = _mm256_set1_ps(beta);
        c00 = _mm256_mul_ps(_mm256_loadu_ps(c + 0 * rowStride + 0), scale);
        c01 = _mm256_mul_ps(_mm256_loadu_ps(c + 0 * rowStride + 8), scale);
        c10 = _mm256_mul_ps(_mm256_loadu_ps(c + 1 * rowStride + 0), scale);
        c11 = _mm256_mul_ps(_mm256_loadu_ps(c + 1 * rowStride + 8), scale);
        c20 = _mm256_mul_ps(_mm256_loadu_ps(c + 2 * rowStride + 0), scale);
        c21 = _mm256_mul_ps(_mm256_loadu_ps(c + 2 * rowStride + 8), scale);
        c30 = _mm256_mul_ps(_mm256_loadu_ps(c + 3 * rowStride + 0), scale);
        c31 = _mm256_mul_ps(_mm256_loadu_ps(c + 3 * rowStride + 8), scale);
        c40 = _mm256_mul_ps(_mm256_loadu_ps(c + 4 * rowStride + 0), scale);
        c41 = _mm256_mul_ps(_mm256_loadu_ps(c + 4 * rowStride + 8), scale);
        c50 = _mm256_mul_ps(_mm256_loadu_ps(c + 5 * rowStride + 0), scale);
        c51 = _mm256_mul_ps(_mm256_loadu_ps(c + 5 * rowStride + 8), scale);
    }

    // C的6行在K循环前预取，收尾写回时已在L1
    for (std::size_t row = 0; row < 6; ++row) {
        _mm_prefetch(reinterpret_cast<const char*>(c + row * rowStride), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(c + row * rowStride + 15), _MM_HINT_T0);
    }

    // 12个累加寄存器、2个B向量和1个A广播，每k步192FLOPs
    for (std::size_t p = 0; p < k; ++p) {
        const __m256 b0 = _mm256_loadu_ps(b + p * 16);
        const __m256 b1 = _mm256_loadu_ps(b + p * 16 + 8);
        __m256 av = _mm256_broadcast_ss(a + p * 6);
        c00 = _mm256_fmadd_ps(av, b0, c00);
        c01 = _mm256_fmadd_ps(av, b1, c01);
        av = _mm256_broadcast_ss(a + p * 6 + 1);
        c10 = _mm256_fmadd_ps(av, b0, c10);
        c11 = _mm256_fmadd_ps(av, b1, c11);
        av = _mm256_broadcast_ss(a + p * 6 + 2);
        c20 = _mm256_fmadd_ps(av, b0, c20);
        c21 = _mm256_fmadd_ps(av, b1, c21);
        av = _mm256_broadcast_ss(a + p * 6 + 3);
        c30 = _mm256_fmadd_ps(av, b0, c30);
        c31 = _mm256_fmadd_ps(av, b1, c31);
        av = _mm256_broadcast_ss(a + p * 6 + 4);
        c40 = _mm256_fmadd_ps(av, b0, c40);
        c41 = _mm256_fmadd_ps(av, b1, c41);
        av = _mm256_broadcast_ss(a + p * 6 + 5);
        c50 = _mm256_fmadd_ps(av, b0, c50);
        c51 = _mm256_fmadd_ps(av, b1, c51);
    }

    _mm256_storeu_ps(c + 0 * rowStride + 0, c00);
    _mm256_storeu_ps(c + 0 * rowStride + 8, c01);
    _mm256_storeu_ps(c + 1 * rowStride + 0, c10);
    _mm256_storeu_ps(c + 1 * rowStride + 8, c11);
    _mm256_storeu_ps(c + 2 * rowStride + 0, c20);
    _mm256_storeu_ps(c + 2 * rowStride + 8, c21);
    _mm256_storeu_ps(c + 3 * rowStride + 0, c30);
    _mm256_storeu_ps(c + 3 * rowStride + 8, c31);
    _mm256_storeu_ps(c + 4 * rowStride + 0, c40);
    _mm256_storeu_ps(c + 4 * rowStride + 8, c41);
    _mm256_storeu_ps(c + 5 * rowStride + 0, c50);
    _mm256_storeu_ps(c + 5 * rowStride + 8, c51);
}

} // namespace timestral::blas
