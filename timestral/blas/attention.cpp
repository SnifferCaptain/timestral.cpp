/***************
 * @file attention.cpp
 * @brief 有序与在线注意力，复用打包SGEMM微内核
 ***************/

#include "timestral/blas/attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "timestral/blas/elementwise.hpp"
#include "timestral/blas/microkernel.hpp"
#include "timestral/reference/kernels.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral::blas {

static void validateSpan(
    const std::vector<float>& values,
    std::size_t offset,
    std::size_t rows,
    std::size_t stride,
    std::size_t width
) {
    if (rows == 0) {
        return;
    }
    if (offset > values.size()) {
        throw std::invalid_argument("attention: offset exceeds storage");
    }
    const std::size_t available = values.size() - offset;
    if (width > available || (rows > 1 && stride > (available - width) / (rows - 1))) {
        throw std::invalid_argument("attention: view exceeds storage");
    }
}

static MaskTile classifyTile(
    const AttentionMask& mask,
    std::size_t qBegin,
    std::size_t kBegin,
    std::size_t qCount,
    std::size_t kCount
) {
    if (mask.classify) {
        return mask.classify(qBegin, kBegin, qCount, kCount);
    }
    bool any = false;
    bool all = true;
    for (std::size_t q = qBegin; q < qBegin + qCount; ++q) {
        for (std::size_t k = kBegin; k < kBegin + kCount; ++k) {
            const bool allowed = mask.allowed(q, k);
            any = any || allowed;
            all = all && allowed;
        }
    }
    return !any ? MaskTile::Empty : (all ? MaskTile::Full : MaskTile::Partial);
}

// 键数不超过一个16键面板时打包与分块开销远超计算，直接两遍softmax，variate轴R很小走此路径
// 此路径与GemmMode无关，stats只累计允许的分数元素，不计块数
static void smallAttention(const AttentionView& view, const AttentionMask& mask, AttentionStats& stats) {
    constexpr float negative = -std::numeric_limits<float>::infinity();
    const float factor = 1.0f / std::sqrt(static_cast<float>(view.headDim));
    std::array<const float*, 16> keyRows{};
    std::array<const float*, 16> valueRows{};
    for (std::size_t key = 0; key < view.keyCount; ++key) {
        const bool prefix = key < view.prefixCount;
        keyRows[key] = prefix
            ? view.prefixKeys.data() + view.prefixOffset + key * view.prefixStride
            : view.keys.data() + view.keyOffset + (key - view.prefixCount) * view.keyStride;
        valueRows[key] = prefix
            ? view.prefixValues.data() + view.prefixOffset + key * view.prefixStride
            : view.values.data() + view.valueOffset + (key - view.prefixCount) * view.valueStride;
    }
    std::array<float, 16> scores{};
    for (std::size_t q = 0; q < view.queryCount; ++q) {
        const float* query = view.query.data() + view.queryOffset + q * view.queryStride;
        float* output = view.output.data() + view.outputOffset + q * view.outputStride;
        float maximum = negative;
        bool hasNaN = false;
        for (std::size_t key = 0; key < view.keyCount; ++key) {
            scores[key] = negative;
            if (!mask.allowed(q, key)) {
                continue;
            }
            float sum = 0.0f;
            for (std::size_t d = 0; d < view.headDim; ++d) {
                sum += query[d] * keyRows[key][d];
            }
            scores[key] = sum * factor;
            maximum = std::max(maximum, scores[key]);
            hasNaN = hasNaN || std::isnan(scores[key]);
            ++stats.scoreElements;
        }
        // 允许键出现NaN时向输出传播，不当作整行屏蔽
        if (hasNaN) {
            maximum = std::numeric_limits<float>::quiet_NaN();
        }
        std::fill_n(output, view.headDim, 0.0f);
        if (maximum == negative) {
            continue;
        }
        float denominator = 0.0f;
        for (std::size_t key = 0; key < view.keyCount; ++key) {
            if (scores[key] == negative) {
                continue;
            }
            const float probability = std::exp(scores[key] - maximum);
            denominator += probability;
            for (std::size_t d = 0; d < view.headDim; ++d) {
                output[d] += probability * valueRows[key][d];
            }
        }
        const float inverse = 1.0f / denominator;
        for (std::size_t d = 0; d < view.headDim; ++d) {
            output[d] *= inverse;
        }
    }
}

void orderedAttention(const AttentionView& view, const AttentionMask& mask) {
    if (view.headDim == 0 || view.prefixCount > view.keyCount || !mask.allowed
        || &view.output == &view.query || &view.output == &view.keys || &view.output == &view.values
        || &view.output == &view.prefixKeys || &view.output == &view.prefixValues
        || (view.queryCount > 1 && view.outputStride < view.headDim)) {
        throw std::invalid_argument("ordered attention: invalid width, prefix or mask");
    }
    validateSpan(view.query, view.queryOffset, view.queryCount, view.queryStride, view.headDim);
    validateSpan(view.output, view.outputOffset, view.queryCount, view.outputStride, view.headDim);
    validateSpan(view.keys, view.keyOffset, view.keyCount - view.prefixCount, view.keyStride, view.headDim);
    validateSpan(view.values, view.valueOffset, view.keyCount - view.prefixCount, view.valueStride, view.headDim);
    validateSpan(view.prefixKeys, view.prefixOffset, view.prefixCount, view.prefixStride, view.headDim);
    validateSpan(view.prefixValues, view.prefixOffset, view.prefixCount, view.prefixStride, view.headDim);
    std::vector<float> keys(elementCount({view.keyCount, view.headDim}));
    std::vector<float> values(keys.size());
    for (std::size_t key = 0; key < view.keyCount; ++key) {
        const bool prefix = key < view.prefixCount;
        const auto& keyData = prefix ? view.prefixKeys : view.keys;
        const auto& valueData = prefix ? view.prefixValues : view.values;
        const auto keyOffset = prefix ? view.prefixOffset + key * view.prefixStride
            : view.keyOffset + (key - view.prefixCount) * view.keyStride;
        const auto valueOffset = prefix ? view.prefixOffset + key * view.prefixStride
            : view.valueOffset + (key - view.prefixCount) * view.valueStride;
        std::copy_n(keyData.begin() + keyOffset, view.headDim, keys.begin() + key * view.headDim);
        std::copy_n(valueData.begin() + valueOffset, view.headDim, values.begin() + key * view.headDim);
    }
    const PackedWeight packedKeys(MatrixView(keys, view.headDim, view.keyCount, 0, 1, view.headDim));
    const PackedWeight packedValues(MatrixView(values, view.keyCount, view.headDim));
    const float factor = 1.0f / std::sqrt(static_cast<float>(view.headDim));
    constexpr std::size_t queryBlock = 32;
    Workspace workspace;
    for (std::size_t begin = 0; begin < view.queryCount; begin += queryBlock) {
        const auto count = std::min(queryBlock, view.queryCount - begin);
        std::vector<float> scores(elementCount({count, view.keyCount}));
        gemmPacked(MatrixView(view.query, count, view.headDim,
            view.queryOffset + begin * view.queryStride, view.queryStride, 1),
            packedKeys, MutableMatrixView(scores, count, view.keyCount), workspace, 0.0f, GemmMode::Ordered);
        for (std::size_t query = 0; query < count; ++query) {
            bool hasNaN = false;
            for (std::size_t key = 0; key < view.keyCount; ++key) {
                auto& score = scores[query * view.keyCount + key];
                score = mask.allowed(begin + query, key) ? score * factor : -std::numeric_limits<float>::infinity();
                hasNaN = hasNaN || std::isnan(score);
            }
            const auto first = scores.begin() + query * view.keyCount;
            const auto last = scores.begin() + (query + 1) * view.keyCount;
            if (hasNaN) {
                std::fill(first, last, std::numeric_limits<float>::quiet_NaN());
            } else {
                reference::softmax(first, last);
            }
        }
        gemmPacked(MatrixView(scores, count, view.keyCount), packedValues,
            MutableMatrixView(view.output, count, view.headDim,
                view.outputOffset + begin * view.outputStride, view.outputStride, 1),
            workspace, 0.0f, GemmMode::Ordered);
    }
}

void onlineAttention(
    const AttentionView& view,
    const AttentionMask& mask,
    AttentionStats& stats,
    GemmMode mode,
    bool stableKeyBlocks
) {
    constexpr std::size_t queryBlock = 6;
    constexpr std::size_t keyBlock = 64;
    constexpr std::size_t lanes = 16;
    stats = AttentionStats{};
    if (view.headDim == 0 || view.prefixCount > view.keyCount || !mask.allowed
        || &view.output == &view.query || &view.output == &view.keys || &view.output == &view.values
        || &view.output == &view.prefixKeys || &view.output == &view.prefixValues
        || (view.queryCount > 1 && view.outputStride < view.headDim)) {
        throw std::invalid_argument("attention: invalid width, mask or aliased output");
    }
    validateSpan(view.query, view.queryOffset, view.queryCount, view.queryStride, view.headDim);
    validateSpan(view.keys, view.keyOffset, view.keyCount - view.prefixCount, view.keyStride, view.headDim);
    validateSpan(view.values, view.valueOffset, view.keyCount - view.prefixCount, view.valueStride, view.headDim);
    validateSpan(view.prefixKeys, view.prefixOffset, view.prefixCount, view.prefixStride, view.headDim);
    validateSpan(view.prefixValues, view.prefixOffset, view.prefixCount, view.prefixStride, view.headDim);
    validateSpan(view.output, view.outputOffset, view.queryCount, view.outputStride, view.headDim);
    const auto kernel = selectKernel(mode);
    if (view.queryCount == 0) {
        return;
    }
    if (view.keyCount <= 16 && !stableKeyBlocks) {
        smallAttention(view, mask, stats);
        return;
    }
    const std::size_t keyBlocks = view.keyCount / keyBlock + (view.keyCount % keyBlock != 0);
    const std::size_t dimBlocks = view.headDim / lanes + (view.headDim % lanes != 0);
    const std::size_t paddedDim = elementCount({dimBlocks, lanes});
    const std::size_t paddedKeys = elementCount({keyBlocks, keyBlock});
    std::vector<float> packedKeys(elementCount({paddedKeys, view.headDim}), 0.0f);
    std::vector<float> packedValues(elementCount({paddedKeys, paddedDim}), 0.0f);
    // K按16键一面板[d][16]，V按(键块,16维块)[64键][16]，逐行读一次源，不在内层做除法
    for (std::size_t key = 0; key < view.keyCount; ++key) {
        const bool prefix = key < view.prefixCount;
        const float* keyRow = prefix
            ? view.prefixKeys.data() + view.prefixOffset + key * view.prefixStride
            : view.keys.data() + view.keyOffset + (key - view.prefixCount) * view.keyStride;
        const float* valueRow = prefix
            ? view.prefixValues.data() + view.prefixOffset + key * view.prefixStride
            : view.values.data() + view.valueOffset + (key - view.prefixCount) * view.valueStride;
        float* keyTarget = packedKeys.data() + key / lanes * view.headDim * lanes + key % lanes;
        for (std::size_t d = 0; d < view.headDim; ++d) {
            keyTarget[d * lanes] = keyRow[d];
        }
        float* valueTarget = packedValues.data() + (key / keyBlock * dimBlocks * keyBlock + key % keyBlock) * lanes;
        for (std::size_t block = 0; block < dimBlocks; ++block) {
            const std::size_t width = std::min(lanes, view.headDim - block * lanes);
            std::copy_n(valueRow + block * lanes, width, valueTarget + block * keyBlock * lanes);
        }
    }
    const auto softmaxRow = kernel == computeNativeKernel ? softmaxRowNative : softmaxRowScalar;
    const float factor = 1.0f / std::sqrt(static_cast<float>(view.headDim));
    std::vector<float> packedQuery(elementCount({view.headDim, queryBlock}), 0.0f);
    std::vector<float> accumulator(elementCount({queryBlock, paddedDim}), 0.0f);
    std::array<float, queryBlock * keyBlock> scores{};
    std::array<float, queryBlock * keyBlock> probabilities{};
    std::array<float, queryBlock> maximum{};
    std::array<float, queryBlock> denominator{};
    std::vector<std::uint8_t> allowed(keyBlock, 0);
    // Full块传全1掩码，与Partial缓冲同为一个键块长度，满足按8字节读取
    static const std::vector<std::uint8_t> allAllowed(keyBlock, 1);
    for (std::size_t qBegin = 0; qBegin < view.queryCount; qBegin += queryBlock) {
        const std::size_t qCount = std::min(queryBlock, view.queryCount - qBegin);
        const auto compute = [&](std::size_t depth, const float* a, const float* b, float* c, std::size_t stride, float beta) {
            if (qCount < queryBlock && kernel == computeSse42Kernel) {
                computeSse42Rows(qCount, depth, a, b, c, stride, beta);
            } else if (qCount < queryBlock && kernel == computeNativeKernel) {
                computeNativeRows(qCount, depth, a, b, c, stride, beta);
            } else {
                kernel(depth, a, b, c, stride, beta);
            }
        };
        std::fill(packedQuery.begin(), packedQuery.end(), 0.0f);
        std::fill(accumulator.begin(), accumulator.end(), 0.0f);
        maximum.fill(-std::numeric_limits<float>::infinity());
        denominator.fill(0.0f);
        for (std::size_t q = 0; q < qCount; ++q) {
            for (std::size_t d = 0; d < view.headDim; ++d) {
                packedQuery[d * queryBlock + q] = view.query[view.queryOffset + (qBegin + q) * view.queryStride + d];
            }
        }
        for (std::size_t kBegin = 0; kBegin < view.keyCount; kBegin += keyBlock) {
            const std::size_t kCount = std::min(keyBlock, view.keyCount - kBegin);
            const MaskTile state = classifyTile(mask, qBegin, kBegin, qCount, kCount);
            if (state == MaskTile::Empty) {
                ++stats.emptyTiles;
                continue;
            }
            if (state == MaskTile::Full) {
                ++stats.fullTiles;
            } else {
                ++stats.partialTiles;
            }
            stats.scoreElements += qCount * kCount;
            for (std::size_t lane = 0; lane < kCount; lane += lanes) {
                compute(
                    view.headDim,
                    packedQuery.data(),
                    packedKeys.data() + ((kBegin + lane) / lanes * view.headDim) * lanes,
                    scores.data() + lane,
                    keyBlock,
                    0.0f
                );
            }
            probabilities.fill(0.0f);
            for (std::size_t q = 0; q < qCount; ++q) {
                // Full块不逐元素调用谓词，Partial块先把谓词展开成字节掩码
                if (state == MaskTile::Partial && mask.fill) {
                    mask.fill(qBegin + q, kBegin, kCount, allowed);
                } else if (state == MaskTile::Partial) {
                    for (std::size_t key = 0; key < kCount; ++key) {
                        allowed[key] = mask.allowed(qBegin + q, kBegin + key) ? 1 : 0;
                    }
                }
                softmaxRow(
                    scores.data() + q * keyBlock,
                    state == MaskTile::Full ? allAllowed.data() : allowed.data(),
                    kCount,
                    factor,
                    maximum[q],
                    denominator[q],
                    accumulator.data() + q * paddedDim,
                    paddedDim,
                    probabilities.data() + q,
                    queryBlock
                );
            }
            for (std::size_t dim = 0; dim < dimBlocks; ++dim) {
                compute(
                    keyBlock,
                    probabilities.data(),
                    packedValues.data() + (kBegin / keyBlock * dimBlocks + dim) * keyBlock * lanes,
                    accumulator.data() + dim * lanes,
                    paddedDim,
                    1.0f
                );
            }
        }
        for (std::size_t q = 0; q < qCount; ++q) {
            const float inverse = denominator[q] == 0.0f ? 0.0f : 1.0f / denominator[q];
            for (std::size_t d = 0; d < view.headDim; ++d) {
                view.output[view.outputOffset + (qBegin + q) * view.outputStride + d]
                    = accumulator[q * paddedDim + d] * inverse;
            }
        }
    }
}

} // namespace timestral::blas
