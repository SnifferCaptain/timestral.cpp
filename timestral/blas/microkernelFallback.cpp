/***************
 * @file microkernelFallback.cpp
 * @brief 未编译AVX2时的微内核链接替代
 ***************/

#include "timestral/blas/microkernel.hpp"

namespace timestral::blas {

bool nativeKernelAvailable() {
    return false;
}

void computeOrderedKernel(
    std::size_t k, const float* a, const float* b, float* c, std::size_t rowStride, float beta
) {
    computeScalarKernel(k, a, b, c, rowStride, beta);
}

void computeNativeKernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    computeScalarKernel(k, a, b, c, rowStride, beta);
}

} // namespace timestral::blas
