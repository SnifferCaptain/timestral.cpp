/***************
 * @file forecast.cpp
 * @brief 公开入口与调度服务共用的历史校验、步数推导与分位数结果提取
 ***************/

#include "timestral/utils/forecast.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace timestral {

// 相邻dt的中位数，至少两个样本
static double inferInterval(const std::vector<Sample>& history, const std::string& name) {
    if (history.size() < 2) {
        throw std::invalid_argument(name + ": at least 2 samples are required to infer sampleInterval");
    }
    std::vector<double> intervals(history.size() - 1);
    for (std::size_t i = 1; i < history.size(); ++i) {
        intervals[i - 1] = history[i].time - history[i - 1].time;
    }
    const std::size_t count = intervals.size();
    const auto middle = intervals.begin() + static_cast<std::ptrdiff_t>(count / 2);
    std::nth_element(intervals.begin(), middle, intervals.end());
    if (count % 2 == 1) {
        return intervals[count / 2];
    }
    const double upper = intervals[count / 2];
    const double lower = *std::max_element(intervals.begin(), middle);
    return 0.5 * (lower + upper);
}

double resolveInterval(
    const std::vector<Sample>& history,
    double sampleInterval,
    const std::string& name
) {
    if (!std::isfinite(sampleInterval) || sampleInterval < 0.0) {
        throw std::invalid_argument(name + ": sampleInterval must be finite and nonnegative");
    }
    if (sampleInterval > 0.0) {
        return sampleInterval;
    }
    return inferInterval(history, name);
}

std::size_t stepCount(double horizonSeconds, double interval, const std::string& name) {
    if (!std::isfinite(horizonSeconds) || horizonSeconds <= 0.0) {
        throw std::invalid_argument(name + ": horizonSeconds must be finite and positive");
    }
    const double ratio = horizonSeconds / interval;
    if (!std::isfinite(ratio)
        || ratio >= static_cast<double>(std::numeric_limits<long long>::max())) {
        throw std::invalid_argument(name + ": horizonSeconds is outside the supported range");
    }
    const long long rounded = std::llround(ratio);
    return static_cast<std::size_t>(std::max<long long>(rounded, 1));
}

// 时间戳有限、严格递增，读数有限或NaN（NaN表示缺测），无穷非法
void validateHistory(
    const std::vector<Sample>& history,
    std::size_t patch,
    const std::string& name
) {
    const std::size_t minimum = 2 * patch;
    if (history.size() < minimum) {
        throw std::invalid_argument(
            name + ": history requires at least " + std::to_string(minimum)
            + " samples (2 * patchSize), got " + std::to_string(history.size())
        );
    }
    for (std::size_t i = 0; i < history.size(); ++i) {
        const Sample& sample = history[i];
        const std::string label = name + ": sample " + std::to_string(i);
        if (!std::isfinite(sample.time)) {
            throw std::invalid_argument(label + " time must be finite");
        }
        if (std::isinf(sample.value)) {
            throw std::invalid_argument(label + " value must be finite or NaN");
        }
        if (i > 0 && !(sample.time > history[i - 1].time)) {
            throw std::invalid_argument(label + " time must be strictly increasing");
        }
    }
}

std::pair<std::size_t, std::size_t> contextRange(
    const std::vector<Sample>& history,
    std::size_t limit,
    std::size_t patch,
    const std::string& name
) {
    const std::size_t minimum = 2 * patch;
    if (history.size() <= limit) {
        return {0, history.size()};
    }
    const std::size_t keep = limit / patch * patch;
    if (keep < minimum) {
        throw std::invalid_argument(
            name + ": maxContext keeps " + std::to_string(keep)
            + " samples, requires at least " + std::to_string(minimum)
        );
    }
    return {history.size() - keep, keep};
}

float fp32Sample(
    const std::vector<Sample>& history,
    std::size_t index,
    const std::string& name
) {
    const double value = history[index].value;
    if (std::isfinite(value)
        && std::abs(value) > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument(
            name + ": sample " + std::to_string(index) + " exceeds the FP32 range"
        );
    }
    return static_cast<float>(value);
}

std::vector<double> quantileLevels(const ModelConfig& config) {
    std::vector<double> levels;
    levels.reserve(config.quantileKnots.size());
    for (float knot : config.quantileKnots) {
        levels.push_back(static_cast<double>(knot));
    }
    return levels;
}

std::vector<double> quantileRow(
    const Tensor& quantiles,
    std::size_t batchIndex,
    std::size_t steps,
    std::size_t levels
) {
    quantiles.validate();
    if (quantiles.shape.size() != 4
        || quantiles.shape[0] <= batchIndex
        || quantiles.shape[1] == 0
        || quantiles.shape[2] != steps
        || quantiles.shape[3] != levels
        || levels == 0) {
        throw std::runtime_error("engine returned an unexpected quantile shape");
    }
    std::vector<double> row(quantiles.shape[1] * steps * levels);
    const std::size_t offset = batchIndex * quantiles.shape[1] * steps * levels;
    for (std::size_t i = 0; i < row.size(); ++i) {
        row[i] = static_cast<double>(quantiles.values[offset + i]);
    }
    return row;
}

} // namespace timestral
