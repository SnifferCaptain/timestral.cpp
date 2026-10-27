/***************
 * @file microkernelSse42Fallback.cpp
 * @brief 未编译SSE时的链接替代
 ***************/

#include "timestral/blas/microkernel.hpp"

namespace timestral::blas {

bool sse42KernelAvailable() {
    return false;
}

void computeSse42Kernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    computeScalarKernel(k, a, b, c, rowStride, beta);
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
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < 16; ++column) {
            float sum = beta == 0.0f ? 0.0f : beta * c[row * rowStride + column];
            for (std::size_t p = 0; p < k; ++p) {
                sum += a[p * 6 + row] * b[p * 16 + column];
            }
            c[row * rowStride + column] = sum;
        }
    }
}

} // namespace timestral::blas
