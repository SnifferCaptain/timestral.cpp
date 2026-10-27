/***************
 * @file microkernelNarrowFallback.cpp
 * @brief 小M微内核的无AVX链接替代
 ***************/

#include "timestral/blas/microkernel.hpp"

namespace timestral::blas {

void computeNarrowRows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < 16; ++j) {
            float sum = beta == 0.0f ? 0.0f : beta * c[i * rowStride + j];
            for (std::size_t p = 0; p < k; ++p) {
                sum += a[p * 6 + i] * b[p * 16 + j];
            }
            c[i * rowStride + j] = sum;
        }
    }
}

} // namespace timestral::blas
