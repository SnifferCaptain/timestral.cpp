/***************
 * @file gemm.cpp
 * @brief 分块矩阵乘法、视图校验及局部权重转换
 ***************/

#include "timestral/blas/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "timestral/blas/microkernel.hpp"
#include "timestral/blas/parallel.hpp"

namespace timestral::blas {

static constexpr std::size_t blockK = 256;
static constexpr std::size_t blockM = 120;
static constexpr std::size_t blockN = 256;

static std::size_t checkedProduct(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::invalid_argument("gemm: size overflow");
    }
    return a * b;
}

static std::size_t checkedSum(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::invalid_argument("gemm: offset overflow");
    }
    return a + b;
}

static void validateRange(
    std::size_t size,
    std::size_t rows,
    std::size_t columns,
    std::size_t offset,
    std::size_t rowStride,
    std::size_t columnStride
) {
    if (offset > size) {
        throw std::invalid_argument("gemm: offset outside storage");
    }
    if (rows == 0 || columns == 0) {
        return;
    }
    const auto last = checkedSum(
        checkedSum(offset, checkedProduct(rows - 1, rowStride)),
        checkedProduct(columns - 1, columnStride)
    );
    if (last >= size) {
        throw std::invalid_argument("gemm: view outside storage");
    }
}

static void validate(const MatrixView& view) {
    validateRange(view.storage.size(), view.rows, view.columns, view.offset, view.rowStride, view.columnStride);
}

static void validate(const MutableMatrixView& view) {
    validateRange(view.storage.size(), view.rows, view.columns, view.offset, view.rowStride, view.columnStride);
    if (view.rows == 0 || view.columns == 0) {
        return;
    }
    if ((view.rows > 1 && view.rowStride == 0) || (view.columns > 1 && view.columnStride == 0)) {
        throw std::invalid_argument("gemm: output elements overlap");
    }
    if (view.rows > 1 && view.columns > 1) {
        const auto divisor = std::gcd(view.rowStride, view.columnStride);
        if (view.columns > view.rowStride / divisor && view.rows > view.columnStride / divisor) {
            throw std::invalid_argument("gemm: output elements overlap");
        }
    }
}

MatrixView::MatrixView(const std::vector<float>& values, std::size_t rows, std::size_t columns)
    : MatrixView(values, rows, columns, 0, columns, 1) {
}

MatrixView::MatrixView(
    const std::vector<float>& values,
    std::size_t rows,
    std::size_t columns,
    std::size_t offset,
    std::size_t rowStride,
    std::size_t columnStride
)
    : storage(values),
      rows(rows),
      columns(columns),
      offset(offset),
      rowStride(rowStride),
      columnStride(columnStride) {
    validate(*this);
}

MutableMatrixView::MutableMatrixView(std::vector<float>& values, std::size_t rows, std::size_t columns)
    : MutableMatrixView(values, rows, columns, 0, columns, 1) {
}

MutableMatrixView::MutableMatrixView(
    std::vector<float>& values,
    std::size_t rows,
    std::size_t columns,
    std::size_t offset,
    std::size_t rowStride,
    std::size_t columnStride
)
    : storage(values),
      rows(rows),
      columns(columns),
      offset(offset),
      rowStride(rowStride),
      columnStride(columnStride) {
    validate(*this);
}

bool avx2Available() {
    static const bool available = cpuSupportsAvx2Fma() && nativeKernelAvailable();
    return available;
}

bool sse42Available() {
    static const bool available = sse42KernelAvailable() && cpuSupportsSse42();
    return available;
}

Microkernel selectKernel(GemmMode mode) {
    if (mode == GemmMode::Ordered) {
        return avx2Available() ? computeOrderedKernel
            : (sse42Available() ? computeSse42Kernel : computeScalarKernel);
    }
    switch (mode) {
    case GemmMode::Auto:
        if (avx2Available()) {
            return computeNativeKernel;
        }
        return sse42Available() ? computeSse42Kernel : computeScalarKernel;
    case GemmMode::Scalar:
        return computeScalarKernel;
    case GemmMode::Sse42:
        if (!sse42Available()) {
            throw std::invalid_argument("gemm: SSE4.2 kernel is unavailable");
        }
        return computeSse42Kernel;
    default:
        throw std::invalid_argument("gemm: invalid dispatch mode");
    }
}

void computeScalarKernel(
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    float accumulators[6][16] = {};
    if (beta != 0.0f) {
        for (std::size_t i = 0; i < 6; ++i) {
            for (std::size_t j = 0; j < 16; ++j) {
                accumulators[i][j] = beta * c[i * rowStride + j];
            }
        }
    }
    for (std::size_t p = 0; p < k; ++p) {
        for (std::size_t i = 0; i < 6; ++i) {
            for (std::size_t j = 0; j < 16; ++j) {
                accumulators[i][j] += a[p * 6 + i] * b[p * 16 + j];
            }
        }
    }
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 16; ++j) {
            c[i * rowStride + j] = accumulators[i][j];
        }
    }
}

// 校验放在基线单元，AVX2单元不构造异常对象
void computeNativeRows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    if (rows == 0 || rows > 6) {
        throw std::invalid_argument("microkernel: rows must be in [1,6]");
    }
    computeNarrowRows(rows, k, a, b, c, rowStride, beta);
}

void computeSse42Rows(
    std::size_t rows,
    std::size_t k,
    const float* a,
    const float* b,
    float* c,
    std::size_t rowStride,
    float beta
) {
    if (rows == 0 || rows > 6) {
        throw std::invalid_argument("SSE microkernel: rows must be in [1,6]");
    }
    if (!sse42Available()) {
        throw std::invalid_argument("SSE microkernel: SSE4.2 kernel is unavailable");
    }
    computeSse42NarrowRows(rows, k, a, b, c, rowStride, beta);
}

PackedWeight::PackedWeight(const MatrixView& source)
    : rowCount(source.rows),
      columnCount(source.columns) {
    validate(source);
    if (rowCount == 0 || columnCount == 0) {
        return;
    }
    const auto panels = columnCount / 16 + (columnCount % 16 != 0 ? 1 : 0);
    packed.resize(checkedProduct(checkedProduct(panels, rowCount), 16), 0.0f);
    for (std::size_t panel = 0; panel < panels; ++panel) {
        for (std::size_t p = 0; p < rowCount; ++p) {
            for (std::size_t j = 0; j < std::min(std::size_t{16}, columnCount - panel * 16); ++j) {
                const std::size_t index = source.offset + p * source.rowStride + (panel * 16 + j) * source.columnStride;
                packed[(panel * rowCount + p) * 16 + j] = source.storage[index];
            }
        }
    }
}

std::size_t PackedWeight::rows() const {
    return rowCount;
}

std::size_t PackedWeight::columns() const {
    return columnCount;
}

const std::vector<float>& PackedWeight::data() const {
    return packed;
}

// 行主序A按6行微面板交错，不足6行的面板补0
static void packRowMajorA(
    const MatrixView& a,
    std::size_t row0,
    std::size_t p0,
    std::size_t rows,
    std::size_t k,
    std::vector<float>& packed
) {
    packed.resize(((rows + 5) / 6) * k * 6);
    const float* base = a.storage.data() + a.offset + row0 * a.rowStride + p0;
    float* target = packed.data();
    for (std::size_t tile = 0; tile < rows; tile += 6, target += k * 6) {
        const std::size_t count = std::min(std::size_t{6}, rows - tile);
        const float* source[6] = {};
        for (std::size_t i = 0; i < 6; ++i) {
            // 不足6行时以首行占位读取，写入0，避免越界
            source[i] = base + (tile + (i < count ? i : 0)) * a.rowStride;
        }
        if (count == 6) {
            for (std::size_t p = 0; p < k; ++p) {
                for (std::size_t i = 0; i < 6; ++i) {
                    target[p * 6 + i] = source[i][p];
                }
            }
        } else {
            for (std::size_t p = 0; p < k; ++p) {
                for (std::size_t i = 0; i < 6; ++i) {
                    target[p * 6 + i] = i < count ? source[i][p] : 0.0f;
                }
            }
        }
    }
}

// 打包同packRowMajorA，逐行乘RMS倒数再逐列乘scale，与reference的(x*inv)*scale舍入一致
static void packRowMajorNormalizedA(
    const MatrixView& a,
    std::size_t row0,
    std::size_t p0,
    std::size_t rows,
    std::size_t k,
    const NormPacking& normalization,
    std::vector<float>& packed
) {
    packed.resize(((rows + 5) / 6) * k * 6);
    const float* base = a.storage.data() + a.offset + row0 * a.rowStride + p0;
    float* target = packed.data();
    for (std::size_t tile = 0; tile < rows; tile += 6, target += k * 6) {
        const std::size_t count = std::min(std::size_t{6}, rows - tile);
        const float* source[6] = {};
        float factor[6] = {};
        for (std::size_t i = 0; i < 6; ++i) {
            // 不足6行时以首行占位读取，写入0，避免越界
            source[i] = base + (tile + (i < count ? i : 0)) * a.rowStride;
            factor[i] = i < count ? normalization.inverse[row0 + tile + i] : 0.0f;
        }
        for (std::size_t p = 0; p < k; ++p) {
            const float column = normalization.scale[p0 + p];
            for (std::size_t i = 0; i < 6; ++i) {
                target[p * 6 + i] = i < count ? (source[i][p] * factor[i]) * column : 0.0f;
            }
        }
    }
}

static void packA(
    const MatrixView& a,
    std::size_t row0,
    std::size_t p0,
    std::size_t rows,
    std::size_t k,
    std::vector<float>& packed
) {
    if (a.columnStride == 1) {
        packRowMajorA(a, row0, p0, rows, k, packed);
        return;
    }
    packed.resize(((rows + 5) / 6) * k * 6);
    for (std::size_t tile = 0; tile < rows; tile += 6) {
        for (std::size_t p = 0; p < k; ++p) {
            for (std::size_t i = 0; i < 6; ++i) {
                packed[(tile / 6 * k + p) * 6 + i] = tile + i < rows
                    ? a.storage[a.offset + (row0 + tile + i) * a.rowStride + (p0 + p) * a.columnStride]
                    : 0.0f;
            }
        }
    }
}

static void computeTile(
    const MutableMatrixView& c,
    std::size_t row0,
    std::size_t column0,
    std::size_t rows,
    std::size_t columns,
    std::size_t k,
    const float* a,
    const float* b,
    float beta,
    Microkernel kernel
) {
    const bool narrowAvx2 = rows < 6 && kernel == computeNativeKernel;
    const bool narrowSse42 = rows < 6 && kernel == computeSse42Kernel;
    const bool narrow = narrowAvx2 || narrowSse42;
    if (columns == 16 && c.columnStride == 1 && (rows == 6 || narrow)) {
        float* target = c.storage.data() + c.offset + row0 * c.rowStride + column0;
        if (narrowSse42) {
            computeSse42Rows(rows, k, a, b, target, c.rowStride, beta);
        } else if (narrowAvx2) {
            computeNativeRows(rows, k, a, b, target, c.rowStride, beta);
        } else {
            kernel(k, a, b, target, c.rowStride, beta);
        }
        return;
    }
    alignas(32) float tile[6 * 16] = {};
    if (beta != 0.0f) {
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < columns; ++j) {
                tile[i * 16 + j] = c.storage[c.offset + (row0 + i) * c.rowStride + (column0 + j) * c.columnStride];
            }
        }
    }
    if (narrowSse42) {
        computeSse42Rows(rows, k, a, b, tile, 16, beta);
    } else if (narrowAvx2) {
        computeNativeRows(rows, k, a, b, tile, 16, beta);
    } else {
        kernel(k, a, b, tile, 16, beta);
    }
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            c.storage[c.offset + (row0 + i) * c.rowStride + (column0 + j) * c.columnStride] = tile[i * 16 + j];
        }
    }
}

static void validateOperation(
    const MatrixView& a,
    std::size_t bRows,
    std::size_t bColumns,
    const MutableMatrixView& c,
    float beta
) {
    validate(a);
    validate(c);
    if (a.columns != bRows || a.rows != c.rows || bColumns != c.columns || !std::isfinite(beta)) {
        throw std::invalid_argument("gemm: incompatible shape or non-finite beta");
    }
    if (&a.storage == &c.storage) {
        throw std::invalid_argument("gemm: input and output share storage");
    }
}

template <typename PrepareB, typename PrepareA>
static void executeGemm(
    const MatrixView& a,
    std::size_t columns,
    const MutableMatrixView& c,
    Workspace& workspace,
    float beta,
    GemmMode mode,
    PrepareB prepareB,
    PrepareA prepareA,
    std::size_t columnBlock = blockN,
    std::size_t columnBegin = 0,
    std::size_t columnEnd = (std::numeric_limits<std::size_t>::max)()
) {
    const auto kernel = selectKernel(mode);
    const auto endColumn = std::min(columns, columnEnd);
    if (a.rows == 0 || columns == 0) {
        return;
    }
    if (a.columns == 0) {
        for (std::size_t i = 0; i < c.rows; ++i) {
            for (std::size_t j = columnBegin; j < endColumn; ++j) {
                auto& value = c.storage[c.offset + i * c.rowStride + j * c.columnStride];
                value = beta == 0.0f ? 0.0f : beta * value;
            }
        }
        return;
    }
    for (std::size_t jc = columnBegin; jc < endColumn; jc += columnBlock) {
        const auto nc = std::min(columnBlock, endColumn - jc);
        for (std::size_t pc = 0; pc < a.columns; pc += blockK) {
            const auto kc = std::min(blockK, a.columns - pc);
            std::size_t panelStride = 0;
            const float* packedB = prepareB(jc, pc, nc, kc, panelStride);
            for (std::size_t ic = 0; ic < a.rows; ic += blockM) {
                const auto mc = std::min(blockM, a.rows - ic);
                prepareA(ic, pc, mc, kc);
                for (std::size_t jr = 0; jr < nc; jr += 16) {
                    for (std::size_t ir = 0; ir < mc; ir += 6) {
                        computeTile(
                            c,
                            ic + ir,
                            jc + jr,
                            std::min(std::size_t{6}, mc - ir),
                            std::min(std::size_t{16}, nc - jr),
                            kc,
                            workspace.packedA.data() + ir / 6 * kc * 6,
                            packedB + jr / 16 * panelStride,
                            pc == 0 ? beta : 1.0f,
                            kernel
                        );
                    }
                }
            }
        }
    }
}

void gemm(
    const MatrixView& a,
    const MatrixView& b,
    const MutableMatrixView& c,
    Workspace& workspace,
    float beta,
    GemmMode mode
) {
    validate(b);
    validateOperation(a, b.rows, b.columns, c, beta);
    if (&b.storage == &c.storage
        || &workspace.packedA == &a.storage
        || &workspace.packedB == &a.storage
        || &workspace.packedA == &b.storage
        || &workspace.packedB == &b.storage
        || &workspace.packedA == &c.storage
        || &workspace.packedB == &c.storage) {
        throw std::invalid_argument("gemm: workspace or output aliases input");
    }
    const auto prepareB = [&](std::size_t jc, std::size_t pc, std::size_t nc, std::size_t kc, std::size_t& panelStride) {
        panelStride = kc * 16;
        workspace.packedB.resize(((nc + 15) / 16) * panelStride);
        for (std::size_t tile = 0; tile < nc; tile += 16) {
            for (std::size_t p = 0; p < kc; ++p) {
                for (std::size_t j = 0; j < 16; ++j) {
                    workspace.packedB[tile / 16 * panelStride + p * 16 + j] = tile + j < nc
                        ? b.storage[b.offset + (pc + p) * b.rowStride + (jc + tile + j) * b.columnStride]
                        : 0.0f;
                }
            }
        }
        return workspace.packedB.data();
    };
    const auto prepareA = [&](std::size_t ic, std::size_t pc, std::size_t mc, std::size_t kc) {
        packA(a, ic, pc, mc, kc, workspace.packedA);
    };
    executeGemm(a, b.columns, c, workspace, beta, mode, prepareB, prepareA);
}

void gemm(const MatrixView& a, const MatrixView& b, const MutableMatrixView& c, float beta, GemmMode mode) {
    Workspace workspace;
    gemm(a, b, c, workspace, beta, mode);
}

static void gemmPackedRange(
    const MatrixView& a,
    const PackedWeight& b,
    const MutableMatrixView& c,
    Workspace& workspace,
    float beta,
    GemmMode mode,
    std::size_t columnBegin,
    std::size_t columnEnd
) {
    validateOperation(a, b.rows(), b.columns(), c, beta);
    const auto panels = b.columns() / 16 + (b.columns() % 16 != 0 ? 1 : 0);
    if (b.data().size() != checkedProduct(checkedProduct(panels, b.rows()), 16)) {
        throw std::invalid_argument("gemm: invalid or moved-from packed weight");
    }
    if (&workspace.packedA == &a.storage
        || &workspace.packedB == &a.storage
        || &workspace.packedA == &c.storage
        || &workspace.packedB == &c.storage
        || &b.data() == &c.storage
        || &b.data() == &workspace.packedA
        || &b.data() == &workspace.packedB) {
        throw std::invalid_argument("gemm: workspace or output aliases input");
    }
    const auto prepareB = [&](std::size_t jc, std::size_t pc, std::size_t, std::size_t, std::size_t& panelStride) {
        panelStride = b.rows() * 16;
        return b.data().data() + (jc / 16 * b.rows() + pc) * 16;
    };
    const auto prepareA = [&](std::size_t ic, std::size_t pc, std::size_t mc, std::size_t kc) {
        packA(a, ic, pc, mc, kc, workspace.packedA);
    };
    executeGemm(a, b.columns(), c, workspace, beta, mode, prepareB, prepareA, b.columns(), columnBegin, columnEnd);
}

void gemmPacked(const MatrixView& a, const PackedWeight& b, const MutableMatrixView& c,
    Workspace& workspace, float beta, GemmMode mode) {
    gemmPackedRange(a, b, c, workspace, beta, mode, 0, b.columns());
}

void gemmPacked(const MatrixView& a, const PackedWeight& b, const MutableMatrixView& c,
    Parallel& parallel, float beta, GemmMode mode) {
    validateOperation(a, b.rows(), b.columns(), c, beta);
    if (parallel.count() == 1 || a.rows == 0 || b.columns() == 0) {
        Workspace workspace;
        gemmPackedRange(a, b, c, workspace, beta, mode, 0, b.columns());
        return;
    }
    const auto panels = b.columns() / 16 + (b.columns() % 16 != 0);
    if (b.data().size() != checkedProduct(checkedProduct(panels, b.rows()), 16) || &b.data() == &c.storage) {
        throw std::invalid_argument("parallel gemm: invalid or aliased packed weight");
    }
    parallel.tiles(a.rows, b.columns(), [&](std::size_t row, std::size_t count, std::size_t column, std::size_t width) {
        Workspace workspace;
        gemmPackedRange(MatrixView(a.storage, count, a.columns, a.offset + row * a.rowStride, a.rowStride, a.columnStride),
            b, MutableMatrixView(c.storage, count, c.columns, c.offset + row * c.rowStride, c.rowStride, c.columnStride),
            workspace, beta, mode, column, column + width);
    });
}

void gemmPacked(const MatrixView& a, const PackedWeight& b, const MutableMatrixView& c, float beta, GemmMode mode) {
    Workspace workspace;
    gemmPacked(a, b, c, workspace, beta, mode);
}

static void gemmNormalizedRange(
    const MatrixView& a,
    const PackedWeight& b,
    const MutableMatrixView& c,
    const NormPacking& normalization,
    Workspace& workspace,
    GemmMode mode,
    std::size_t columnBegin,
    std::size_t columnEnd
) {
    validateOperation(a, b.rows(), b.columns(), c, 0.0f);
    const auto panels = b.columns() / 16 + (b.columns() % 16 != 0 ? 1 : 0);
    if (normalization.inverse.size() != a.rows || normalization.scale.size() != a.columns
        || b.data().size() != checkedProduct(checkedProduct(panels, b.rows()), 16)) {
        throw std::invalid_argument("gemm normalized: incompatible scales or packed weight");
    }
    for (const auto& source : {
        std::cref(a.storage),
        std::cref(b.data()),
        std::cref(normalization.inverse),
        std::cref(normalization.scale),
        std::cref(c.storage)
    }) {
        if (&source.get() == &workspace.packedA || &source.get() == &workspace.packedB) {
            throw std::invalid_argument("gemm normalized: workspace aliases source or output");
        }
    }
    if (&b.data() == &c.storage || &normalization.inverse == &c.storage || &normalization.scale == &c.storage) {
        throw std::invalid_argument("gemm normalized: output aliases weight or scales");
    }
    const auto prepareB = [&](std::size_t jc, std::size_t pc, std::size_t, std::size_t, std::size_t& panelStride) {
        panelStride = b.rows() * 16;
        return b.data().data() + (jc / 16 * b.rows() + pc) * 16;
    };
    const auto prepareA = [&](std::size_t ic, std::size_t pc, std::size_t mc, std::size_t kc) {
        if (a.columnStride == 1) {
            packRowMajorNormalizedA(a, ic, pc, mc, kc, normalization, workspace.packedA);
            return;
        }
        workspace.packedA.resize(((mc + 5) / 6) * kc * 6);
        for (std::size_t tile = 0; tile < mc; tile += 6) {
            for (std::size_t p = 0; p < kc; ++p) {
                for (std::size_t i = 0; i < 6; ++i) {
                    const float value = tile + i < mc
                        ? (a.storage[a.offset + (ic + tile + i) * a.rowStride + (pc + p) * a.columnStride]
                            * normalization.inverse[ic + tile + i]) * normalization.scale[pc + p]
                        : 0.0f;
                    workspace.packedA[(tile / 6 * kc + p) * 6 + i] = value;
                }
            }
        }
    };
    executeGemm(a, b.columns(), c, workspace, 0.0f, mode, prepareB, prepareA, b.columns(), columnBegin, columnEnd);
}

void gemmNormalized(const MatrixView& a, const PackedWeight& b, const MutableMatrixView& c,
    const NormPacking& normalization, Workspace& workspace, GemmMode mode) {
    gemmNormalizedRange(a, b, c, normalization, workspace, mode, 0, b.columns());
}

void gemmNormalized(const MatrixView& a, const PackedWeight& b, const MutableMatrixView& c,
    const NormPacking& normalization, Parallel& parallel, GemmMode mode) {
    validateOperation(a, b.rows(), b.columns(), c, 0.0f);
    if (normalization.inverse.size() != a.rows || normalization.scale.size() != a.columns
        || &normalization.inverse == &c.storage || &normalization.scale == &c.storage || &b.data() == &c.storage) {
        throw std::invalid_argument("parallel normalized gemm: invalid or aliased scales/weights");
    }
    if (parallel.count() == 1 || a.rows == 0 || b.columns() == 0) {
        Workspace workspace;
        gemmNormalizedRange(a, b, c, normalization, workspace, mode, 0, b.columns());
        return;
    }
    parallel.tiles(a.rows, b.columns(), [&](std::size_t row, std::size_t count, std::size_t column, std::size_t width) {
        const std::vector<float> inverse(normalization.inverse.begin() + row, normalization.inverse.begin() + row + count);
        Workspace workspace;
        gemmNormalizedRange(MatrixView(a.storage, count, a.columns, a.offset + row * a.rowStride, a.rowStride, a.columnStride),
            b, MutableMatrixView(c.storage, count, c.columns, c.offset + row * c.rowStride, c.rowStride, c.columnStride),
            NormPacking{inverse, normalization.scale}, workspace, mode, column, column + width);
    });
}

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
) {
    if (checkedProduct(rows, columns) != source.size()
        || !std::isfinite(scale)
        || scale <= 0.0f
        || columnCount > 16
        || row0 > rows
        || rowCount > rows - row0
        || column0 > columns
        || columnCount > columns - column0) {
        throw std::invalid_argument("gemm: invalid INT8 panel");
    }
    output.assign(checkedProduct(rowCount, 16), 0.0f);
    for (std::size_t p = 0; p < rowCount; ++p) {
        for (std::size_t j = 0; j < columnCount; ++j) {
            output[p * 16 + j] = static_cast<float>(source[(row0 + p) * columns + column0 + j]) * scale;
        }
    }
}

} // namespace timestral::blas
