/***************
 * @file postprocess.cpp
 * @brief 分位数后处理：帧折叠、范围cap、对称平均与非负扫描。
 ***************/

#include "timestral/utils/postprocess.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "timestral/utils/tensor.hpp"

namespace timestral {

double foldAdvancingFrame(double normalizedMedian, double scale, double mean, double offset, double clip) {
    return std::clamp(normalizedMedian * scale + mean - offset, -clip, clip) + offset;
}

void capQuantileRange(
    std::vector<float>& values,
    std::size_t begin,
    std::size_t count,
    float low,
    float high,
    float width
) {
    clampRange(values, begin, count, low - width, high + width);
}

void averageMirrored(
    std::vector<float>& quantiles,
    const std::vector<float>& mirrored,
    std::size_t levels
) {
    if (levels == 0 || quantiles.size() != mirrored.size()
        || quantiles.size() % levels != 0) {
        throw std::invalid_argument("postprocess: mirrored quantiles require matching (B,V,H,Q)");
    }
    for (std::size_t i = 0; i < quantiles.size(); i += levels) {
        for (std::size_t q = 0; q < levels; ++q) {
            quantiles[i + q] = 0.5f * (quantiles[i + q] - mirrored[i + levels - q - 1]);
        }
    }
}

std::vector<std::uint8_t> nonnegativeTrajectories(
    const std::vector<float>& values,
    const std::vector<std::uint8_t>& observed,
    std::size_t rowCount,
    std::size_t time
) {
    if (observed.size() != values.size() || values.size() != rowCount * time) {
        throw std::invalid_argument("postprocess: nonnegative scan requires matching (R,T)");
    }
    std::vector<std::uint8_t> result(rowCount, 0);
    for (std::size_t r = 0; r < rowCount; ++r) {
        bool any = false;
        bool negative = false;
        for (std::size_t t = 0; t < time; ++t) {
            const std::size_t index = r * time + t;
            if (observed[index] != 0) {
                any = true;
                negative = negative || values[index] < 0.0f;
            }
        }
        result[r] = any && !negative ? 1 : 0;
    }
    return result;
}

} // namespace timestral
