/***************
 * @file gemm.hpp
 * @brief 无外部依赖的FP32矩阵乘法及权重打包
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace timestral::blas {

class Parallel;

// Sse42强制选择已编译且CPU支持的SSE内核，否则抛invalid_argument
enum class GemmMode {
    Auto,
    Scalar,
    Sse42,
    Ordered
};

// 非拥有的FP32矩阵视图，storage须在视图使用期间存活，offset与stride以元素计，构造时校验不越界
struct MatrixView {
    const std::vector<float>& storage;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t offset = 0;
    std::size_t rowStride = 0;
    std::size_t columnStride = 0;

    MatrixView(const std::vector<float>& values, std::size_t rows, std::size_t columns);
    MatrixView(
        const std::vector<float>& values,
        std::size_t rows,
        std::size_t columns,
        std::size_t offset,
        std::size_t rowStride,
        std::size_t columnStride
    );
};

// 可写的矩阵视图，构造时另校验各输出元素互不重叠
struct MutableMatrixView {
    std::vector<float>& storage;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t offset = 0;
    std::size_t rowStride = 0;
    std::size_t columnStride = 0;

    MutableMatrixView(std::vector<float>& values, std::size_t rows, std::size_t columns);
    MutableMatrixView(
        std::vector<float>& values,
        std::size_t rows,
        std::size_t columns,
        std::size_t offset,
        std::size_t rowStride,
        std::size_t columnStride
    );
};

// 工作区由调用方独占，可跨顺序调用复用，不可同时供多个调用写入
struct Workspace {
    std::vector<float> packedA;
    std::vector<float> packedB;
};

// B按16列面板打包的只读副本，不足16列补0，构造后可并发共享
class PackedWeight {
public:
    explicit PackedWeight(const MatrixView& source);
    std::size_t rows() const;
    std::size_t columns() const;
    const std::vector<float>& data() const;

private:
    std::size_t rowCount = 0;
    std::size_t columnCount = 0;
    std::vector<float> packed;
};

// C = A * B + beta * C，beta为零时不读取旧C，输入与输出不得共享容器
void gemm(
    const MatrixView& a,
    const MatrixView& b,
    const MutableMatrixView& c,
    float beta = 0.0f,
    GemmMode mode = GemmMode::Auto
);
void gemm(
    const MatrixView& a,
    const MatrixView& b,
    const MutableMatrixView& c,
    Workspace& workspace,
    float beta = 0.0f,
    GemmMode mode = GemmMode::Auto
);
void gemmPacked(
    const MatrixView& a,
    const PackedWeight& b,
    const MutableMatrixView& c,
    float beta = 0.0f,
    GemmMode mode = GemmMode::Auto
);
void gemmPacked(
    const MatrixView& a,
    const PackedWeight& b,
    const MutableMatrixView& c,
    Workspace& workspace,
    float beta = 0.0f,
    GemmMode mode = GemmMode::Auto
);

// 在A微面板内按(input*inverse[row])*scale[column]归一化，不物化完整norm
struct NormPacking {
    const std::vector<float>& inverse;
    const std::vector<float>& scale;
};

void gemmNormalized(
    const MatrixView& a,
    const PackedWeight& b,
    const MutableMatrixView& c,
    const NormPacking& normalization,
    Workspace& workspace,
    GemmMode mode = GemmMode::Auto
);

bool avx2Available();
bool sse42Available();

// 只转换当前最多16列面板，输出rowCount乘16元素，不缓存全量FP32副本
void packInt8Panel(
    const std::vector<std::int8_t>& source,
    std::size_t rows,
    std::size_t columns,
    float scale,
    std::size_t row0,
    std::size_t rowCount,
    std::size_t column0,
    std::size_t columnCount,
    std::vector<float>& output
);

// 按完整输出块分派，各线程独占工作区，不拆分K维归约
void gemmPacked(const MatrixView& a, const PackedWeight& b, const MutableMatrixView& c,
    Parallel& parallel, float beta = 0.0f, GemmMode mode = GemmMode::Auto);
void gemmNormalized(const MatrixView& a, const PackedWeight& b, const MutableMatrixView& c,
    const NormPacking& normalization, Parallel& parallel, GemmMode mode = GemmMode::Auto);

} // namespace timestral::blas
