/***************
 * @file microkernel.hpp
 * @brief FP32微内核与基线指令集检测
 ***************/

#pragma once

#include <cstddef>

namespace timestral::blas {

// 输入按每k步6个A及16个B打包，输出完整6行16列
void computeScalarKernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);
void computeNativeKernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);
void computeOrderedKernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);
bool nativeKernelAvailable();
bool cpuSupportsAvx2Fma();
bool cpuSupportsF16c();
bool cpuSupportsSse42();
bool sse42KernelAvailable();

// SSE保持6乘16打包，内部按两个8列块顺序累加，不使用FMA
void computeSse42Kernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);

// 只写rows行，rows必须为1至6，校验与不可用错误由基线单元处理
void computeSse42Rows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);

// SSE指令集实现，调用方已校验rows和运行期可用性
void computeSse42NarrowRows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);

enum class GemmMode;
using Microkernel = decltype(&computeScalarKernel);
Microkernel selectKernel(GemmMode mode);

// A仍以6行步幅打包，只读写rows行输出，rows必须为1至6，否则抛invalid_argument
void computeNativeRows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);

// computeNativeRows的指令集实现，不校验rows
void computeNarrowRows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
);

} // namespace timestral::blas
