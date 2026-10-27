/***************
 * @file decodeKernel.hpp
 * @brief 压缩16列面板的decode热点接口
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>

namespace timestral::blas {

// rows为1至4，A按行主序，B按完整K乘16面板，C每行有效16列
// K方向逐项累加，bias由调用方处理，不做水平或多链归约
void decodeHalfNative(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const std::uint16_t* b,
    float* c,
    std::size_t rowStride,
    float beta
);
void decodeBfloatNative(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const std::uint16_t* b,
    float* c,
    std::size_t rowStride,
    float beta
);
void decodeInt8Native(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const std::int8_t* b,
    const float* scales,
    float* c,
    std::size_t rowStride,
    float beta
);

} // namespace timestral::blas
