/***************
 * @file decode.hpp
 * @brief decode专用压缩权重与少行FP32累加投影
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "timestral/blas/gemm.hpp"

namespace timestral::blas {

enum class DecodeFormat {
    Float32,
    Float16,
    Bfloat16,
    Int8
};

// 权重仍按[K][16列]面板保存，INT8每32个K值各输出列独立对称缩放
// 只读副本供decode共享，FP32 prefill副本由调用方另行保留
class DecodeWeight {
public:
    DecodeWeight(const PackedWeight& source, DecodeFormat format);
    DecodeFormat format() const;
    std::size_t rows() const;
    std::size_t columns() const;
    std::size_t bytes() const;
    const std::vector<std::uint16_t>& words() const;
    const std::vector<std::int8_t>& integers() const;
    const std::vector<float>& scales() const;

private:
    DecodeFormat storageFormat = DecodeFormat::Float32;
    std::size_t rowCount = 0;
    std::size_t columnCount = 0;
    std::vector<std::uint16_t> packedWords;
    std::vector<std::int8_t> packedIntegers;
    std::vector<float> groupScales;
};

// 构建系统只链接一种默认策略，公开采样接口不暴露后端格式
DecodeFormat defaultDecodeFormat();

// 最多4行的小投影，A与累计均FP32，不提供量化Level3路径
// 允许stride视图，beta为0不读旧C，不允许输入输出/权重别名
void decodeProject(
    const MatrixView& a,
    const DecodeWeight& b,
    const MutableMatrixView& c,
    float beta = 0.0f,
    GemmMode mode = GemmMode::Auto
);

// 压缩权重按16列面板并行，不在K或INT8量化组内做跨线程归约
void decodeProject(const MatrixView& a, const DecodeWeight& b, const MutableMatrixView& c,
    Parallel& parallel, float beta = 0.0f, GemmMode mode = GemmMode::Auto);

} // namespace timestral::blas
