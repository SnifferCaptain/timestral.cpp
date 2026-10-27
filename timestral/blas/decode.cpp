/***************
 * @file decode.cpp
 * @brief decode权重量化、视图校验与Level2分派
 ***************/

#include "timestral/blas/decode.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "timestral/blas/decodeKernel.hpp"
#include "timestral/blas/lowPrecision.hpp"
#include "timestral/blas/microkernel.hpp"
#include "timestral/blas/parallel.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral::blas {

static std::int8_t nearestInt8(double value) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    int rounded = static_cast<int>(lower);
    if (fraction > 0.5 || (fraction == 0.5 && rounded % 2 != 0)) {
        ++rounded;
    }
    rounded = std::max(-127, std::min(127, rounded));
    return static_cast<std::int8_t>(rounded);
}

DecodeWeight::DecodeWeight(const PackedWeight& source, DecodeFormat format)
    : storageFormat(format),
      rowCount(source.rows()),
      columnCount(source.columns()) {
    if (format != DecodeFormat::Float16 && format != DecodeFormat::Bfloat16 && format != DecodeFormat::Int8) {
        throw std::invalid_argument("decode weight: expected Float16, Bfloat16 or Int8");
    }
    const std::size_t panels = columnCount / 16 + (columnCount % 16 != 0 ? 1 : 0);
    const std::size_t size = elementCount({panels, rowCount, 16});
    if (source.data().size() != size) {
        throw std::invalid_argument("decode weight: invalid or moved-from packed source");
    }
    for (float value : source.data()) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("decode weight: source must be finite");
        }
    }
    if (format == DecodeFormat::Float16 || format == DecodeFormat::Bfloat16) {
        packedWords.resize(size);
        for (std::size_t i = 0; i < size; ++i) {
            packedWords[i] = format == DecodeFormat::Float16
                ? floatToHalf(source.data()[i]) : floatToBfloat16(source.data()[i]);
            const float converted = format == DecodeFormat::Float16
                ? halfToFloat(packedWords[i]) : bfloat16ToFloat(packedWords[i]);
            if (!std::isfinite(converted)) {
                throw std::invalid_argument("decode weight: low precision conversion overflow");
            }
        }
        return;
    }
    const std::size_t groups = rowCount / 32 + (rowCount % 32 != 0 ? 1 : 0);
    packedIntegers.resize(size);
    groupScales.resize(elementCount({panels, groups, 16}), 1.0f);
    for (std::size_t panel = 0; panel < panels; ++panel) {
        for (std::size_t group = 0; group < groups; ++group) {
            const std::size_t begin = group * 32;
            const std::size_t end = std::min(rowCount, begin + std::min(std::size_t{32}, rowCount - begin));
            for (std::size_t j = 0; j < 16; ++j) {
                float maximum = 0.0f;
                for (std::size_t p = begin; p < end; ++p) {
                    maximum = std::max(maximum, std::abs(source.data()[(panel * rowCount + p) * 16 + j]));
                }
                const float scale = maximum == 0.0f ? 1.0f
                    : std::max(maximum / 127.0f, std::numeric_limits<float>::denorm_min());
                groupScales[(panel * groups + group) * 16 + j] = scale;
                for (std::size_t p = begin; p < end; ++p) {
                    const std::size_t index = (panel * rowCount + p) * 16 + j;
                    packedIntegers[index] = nearestInt8(static_cast<double>(source.data()[index]) / scale);
                }
            }
        }
    }
}

DecodeFormat DecodeWeight::format() const {
    return storageFormat;
}

std::size_t DecodeWeight::rows() const {
    return rowCount;
}

std::size_t DecodeWeight::columns() const {
    return columnCount;
}

std::size_t DecodeWeight::bytes() const {
    return packedWords.size() * sizeof(std::uint16_t)
        + packedIntegers.size() * sizeof(std::int8_t) + groupScales.size() * sizeof(float);
}

const std::vector<std::uint16_t>& DecodeWeight::words() const {
    return packedWords;
}

const std::vector<std::int8_t>& DecodeWeight::integers() const {
    return packedIntegers;
}

const std::vector<float>& DecodeWeight::scales() const {
    return groupScales;
}

static float decodedValue(const DecodeWeight& weight, std::size_t p, std::size_t j) {
    const std::size_t panel = j / 16;
    const std::size_t lane = j % 16;
    const std::size_t index = (panel * weight.rows() + p) * 16 + lane;
    if (weight.format() == DecodeFormat::Float16) {
        return halfToFloat(weight.words()[index]);
    }
    if (weight.format() == DecodeFormat::Bfloat16) {
        return bfloat16ToFloat(weight.words()[index]);
    }
    const std::size_t groups = weight.rows() / 32 + (weight.rows() % 32 != 0 ? 1 : 0);
    return static_cast<float>(weight.integers()[index]) * weight.scales()[(panel * groups + p / 32) * 16 + lane];
}

static void decodeProjectRange(
    const MatrixView& a,
    const DecodeWeight& b,
    const MutableMatrixView& c,
    float beta,
    GemmMode mode,
    std::size_t columnBegin,
    std::size_t columnEnd
) {
    // 重新构造视图以校验调用期间容器是否缩小，形状限制不转成量化Level3
    const MatrixView checkedA(a.storage, a.rows, a.columns, a.offset, a.rowStride, a.columnStride);
    const MutableMatrixView checkedC(c.storage, c.rows, c.columns, c.offset, c.rowStride, c.columnStride);
    (void)checkedA;
    (void)checkedC;
    // 权重实际数据随格式不同在words或integers/scales里，输出别名要按格式对应的存储检查
    const void* outputObject = static_cast<const void*>(&c.storage);
    const bool aliased = b.format() == DecodeFormat::Int8
        ? static_cast<const void*>(&b.integers()) == outputObject
            || static_cast<const void*>(&b.scales()) == outputObject
        : static_cast<const void*>(&b.words()) == outputObject;
    if (a.rows > 4 || a.columns != b.rows() || a.rows != c.rows || b.columns() != c.columns
        || !std::isfinite(beta) || &a.storage == &c.storage || aliased) {
        throw std::invalid_argument("decode project: incompatible shape, beta or aliased output");
    }
    const auto selected = selectKernel(mode);
    const std::size_t panels = b.columns() / 16 + (b.columns() % 16 != 0 ? 1 : 0);
    const std::size_t elements = elementCount({panels, b.rows(), 16});
    const std::size_t groups = b.rows() / 32 + (b.rows() % 32 != 0 ? 1 : 0);
    if ((b.format() == DecodeFormat::Int8
            && (b.integers().size() != elements || b.scales().size() != elementCount({panels, groups, 16})))
        || (b.format() != DecodeFormat::Int8 && b.words().size() != elements)) {
        throw std::invalid_argument("decode project: invalid or moved-from weight");
    }
    if (a.rows == 0 || c.columns == 0) {
        return;
    }
    if (a.columns == 0) {
        for (std::size_t row = 0; row < c.rows; ++row) {
            for (std::size_t j = columnBegin; j < columnEnd; ++j) {
                auto& value = c.storage[c.offset + row * c.rowStride + j * c.columnStride];
                value = beta == 0.0f ? 0.0f : beta * value;
            }
        }
        return;
    }
    // FP16 native依赖AVX2单元顺带编译了F16C路径：当前AVX2 CPU（IvyBridge起）都带F16C，
    // 且MSVC /arch:AVX2允许F16C；再单独查F16C是为防御未来出现AVX2但无F16C的组合
    const bool native = selected == computeNativeKernel
        && (b.format() != DecodeFormat::Float16 || cpuSupportsF16c());
    if (!native) {
        for (std::size_t row = 0; row < a.rows; ++row) {
            for (std::size_t j = columnBegin; j < columnEnd; ++j) {
                const std::size_t output = c.offset + row * c.rowStride + j * c.columnStride;
                float value = beta == 0.0f ? 0.0f : beta * c.storage[output];
                for (std::size_t p = 0; p < a.columns; ++p) {
                    value += a.storage[a.offset + row * a.rowStride + p * a.columnStride] * decodedValue(b, p, j);
                }
                c.storage[output] = value;
            }
        }
        return;
    }
    // 只暂存最多4行激活与4乘16输出，权重不展开为完整FP32矩阵
    // 逐次堆分配是有意简化：rows<=4已上限，暂存量远小于权重，换取实现清晰
    std::vector<float> input(elementCount({a.rows, a.columns}));
    for (std::size_t row = 0; row < a.rows; ++row) {
        for (std::size_t p = 0; p < a.columns; ++p) {
            input[row * a.columns + p] = a.storage[a.offset + row * a.rowStride + p * a.columnStride];
        }
    }
    std::vector<float> output(elementCount({a.rows, std::size_t{16}}), 0.0f);
    for (std::size_t panel = columnBegin / 16; panel < columnEnd / 16 + (columnEnd % 16 != 0); ++panel) {
        const std::size_t width = std::min(std::size_t{16}, columnEnd - panel * 16);
        if (beta != 0.0f) {
            for (std::size_t row = 0; row < a.rows; ++row) {
                for (std::size_t j = 0; j < width; ++j) {
                    output[row * 16 + j] = c.storage[c.offset + row * c.rowStride + (panel * 16 + j) * c.columnStride];
                }
            }
        }
        if (b.format() == DecodeFormat::Float16) {
            decodeHalfNative(
                a.rows,
                a.columns,
                input.data(),
                b.words().data() + panel * a.columns * 16,
                output.data(),
                16,
                beta
            );
        } else if (b.format() == DecodeFormat::Bfloat16) {
            decodeBfloatNative(
                a.rows,
                a.columns,
                input.data(),
                b.words().data() + panel * a.columns * 16,
                output.data(),
                16,
                beta
            );
        } else {
            decodeInt8Native(
                a.rows,
                a.columns,
                input.data(),
                b.integers().data() + panel * a.columns * 16,
                b.scales().data() + panel * groups * 16,
                output.data(),
                16,
                beta
            );
        }
        for (std::size_t row = 0; row < a.rows; ++row) {
            for (std::size_t j = 0; j < width; ++j) {
                c.storage[c.offset + row * c.rowStride + (panel * 16 + j) * c.columnStride] = output[row * 16 + j];
            }
        }
    }
}

void decodeProject(const MatrixView& a, const DecodeWeight& b, const MutableMatrixView& c, float beta, GemmMode mode) {
    decodeProjectRange(a, b, c, beta, mode, 0, b.columns());
}

void decodeProject(const MatrixView& a, const DecodeWeight& b, const MutableMatrixView& c,
    Parallel& parallel, float beta, GemmMode mode) {
    if (parallel.count() == 1 || a.rows == 0 || b.columns() == 0) {
        decodeProjectRange(a, b, c, beta, mode, 0, b.columns());
        return;
    }
    parallel.tiles(1, b.columns(), [&](std::size_t, std::size_t, std::size_t column, std::size_t width) {
        decodeProjectRange(a, b, c, beta, mode, column, column + width);
    });
}

} // namespace timestral::blas
