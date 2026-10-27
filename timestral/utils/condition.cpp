/***************
 * @file condition.cpp
 * @brief 公共时间网格与分段条件推理
 ***************/

#include "timestral/utils/condition.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "timestral/utils/engine.hpp"
#include "timestral/utils/forecast.hpp"

namespace timestral {

// 先减时间原点再换算下标，容差覆盖绝对时间戳的FP64舍入
static std::size_t gridIndex(double time, double origin, double interval) {
    const long double position = (static_cast<long double>(time) - origin) / interval;
    const long double rounded = std::round(position);
    const double tolerance = 1e-6 + 16.0 * std::numeric_limits<double>::epsilon()
        * std::max(std::abs(time), std::abs(origin)) / interval;
    if (!std::isfinite(position) || rounded < 0.0L
        || rounded >= static_cast<long double>(std::numeric_limits<std::size_t>::max())
        || tolerance >= 0.25 || std::abs(position - rounded) > tolerance) {
        throw std::invalid_argument("submit: observations must align to a common time grid");
    }
    return static_cast<std::size_t>(rounded);
}

static std::size_t addLength(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::invalid_argument("conditional prediction length overflow");
    }
    return left + right;
}

Tensor alignObservations(
    const Timestral::Request& request,
    std::size_t patch,
    std::vector<std::size_t>& ends,
    double& interval,
    double& origin
) {
    if (request.variables.empty()) {
        throw std::invalid_argument("submit: variables must not be empty");
    }
    origin = std::numeric_limits<double>::infinity();
    interval = request.sampleInterval;
    if (!std::isfinite(interval) || interval < 0.0) {
        throw std::invalid_argument("submit: sampleInterval must be finite and nonnegative");
    }
    double inferred = std::numeric_limits<double>::infinity();
    for (const auto& channel : request.variables) {
        validateHistory(channel, patch, "submit");
        origin = std::min(origin, channel.front().time);
        inferred = std::min(inferred, resolveInterval(channel, 0.0, "submit"));
    }
    if (interval == 0.0) {
        interval = inferred;
    }
    if (!std::isfinite(interval) || interval <= 0.0) {
        throw std::invalid_argument("submit: inferred interval must be finite and positive");
    }
    ends.assign(request.variables.size(), 0);
    for (std::size_t channel = 0; channel < request.variables.size(); ++channel) {
        std::size_t previous = 0;
        for (std::size_t i = 0; i < request.variables[channel].size(); ++i) {
            const Sample& sample = request.variables[channel][i];
            const std::size_t index = gridIndex(sample.time, origin, interval);
            if (i > 0 && index <= previous) {
                throw std::invalid_argument("submit: multiple samples occupy the same grid position");
            }
            previous = index;
            fp32Sample(request.variables[channel], i, "submit");
            if (std::isfinite(sample.value)) {
                ends[channel] = index + 1;
            }
        }
        if (ends[channel] == 0) {
            throw std::invalid_argument("submit: every variable needs a finite observation");
        }
    }
    const std::size_t time = *std::max_element(ends.begin(), ends.end());
    Tensor observations({request.variables.size(), time}, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t channel = 0; channel < request.variables.size(); ++channel) {
        for (const Sample& sample : request.variables[channel]) {
            if (std::isfinite(sample.value)) {
                const std::size_t index = gridIndex(sample.time, origin, interval);
                observations.values[channel * time + index] = static_cast<float>(sample.value);
            }
        }
    }
    return observations;
}

std::vector<Timestral::Result> predictConditionedBatch(
    const Engine& engine,
    const std::vector<std::reference_wrapper<const Timestral::Request>>& requests,
    const Tensor& observations,
    const std::vector<std::size_t>& ends,
    double interval,
    const std::vector<double>& origins
) {
    const auto started = std::chrono::steady_clock::now();
    observations.validate();
    const std::size_t batch = requests.size();
    if (batch == 0 || origins.size() != batch || observations.shape.size() != 3
        || observations.shape[0] != batch || !std::isfinite(interval) || interval <= 0.0) {
        throw std::invalid_argument("conditional batch observation shape mismatch");
    }
    const std::size_t channels = observations.shape[1];
    const std::size_t knownTime = observations.shape[2];
    if (channels == 0 || ends.size() != channels || knownTime == 0) {
        throw std::invalid_argument("conditional batch variable shape mismatch");
    }
    const std::size_t first = *std::min_element(ends.begin(), ends.end());
    if (first == 0 || *std::max_element(ends.begin(), ends.end()) != knownTime) {
        throw std::invalid_argument("conditional observation boundaries do not match the grid");
    }
    const auto& controls = requests.front().get();
    const std::size_t future = stepCount(controls.horizonSeconds, interval, "submit");
    const std::size_t finish = addLength(knownTime, future);
    const std::size_t steps = finish - first;
    const auto levels = quantileLevels(engine.config());
    const std::size_t levelCount = levels.size();
    const auto knownAt = [&](std::size_t b, std::size_t channel, std::size_t time) {
        return time < knownTime && std::isfinite(observations.values[(b * channels + channel) * knownTime + time]);
    };
    for (std::size_t b = 0; b < batch; ++b) {
        const auto& request = requests[b].get();
        if (request.variables.size() != channels || !std::isfinite(origins[b])
            || stepCount(request.horizonSeconds, interval, "submit") != future
            || request.symmetricAveraging != controls.symmetricAveraging
            || request.clipNonnegative != controls.clipNonnegative) {
            throw std::invalid_argument("conditional batch requests must have compatible controls");
        }
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (std::size_t time = first; time < knownTime; ++time) {
                if (knownAt(b, channel, time) != knownAt(0, channel, time)) {
                    throw std::invalid_argument("conditional batch requests must have the same observed layout");
                }
            }
        }
    }
    Tensor feedback({batch, channels, finish}, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t row = 0; row < batch * channels; ++row) {
        std::copy_n(
            observations.values.begin() + row * knownTime,
            knownTime,
            feedback.values.begin() + row * finish
        );
    }
    std::vector<Timestral::Result> results(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        auto& result = results[b];
        result.channels = channels;
        result.sampleInterval = interval;
        result.origin = origins[b] + static_cast<double>(first - 1) * interval;
        result.levels = levels;
        result.quantiles.assign(elementCount({channels, steps, levelCount}), std::numeric_limits<double>::quiet_NaN());
        result.observed.assign(elementCount({channels, steps}), 0);
    }
    std::size_t cursor = first;
    while (cursor < finish) {
        std::vector<std::size_t> targets;
        std::vector<std::size_t> known;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            if (knownAt(0, channel, cursor)) {
                known.push_back(channel);
            } else {
                targets.push_back(channel);
            }
        }
        std::size_t stop = cursor + 1;
        while (stop < finish) {
            bool unchanged = true;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                if (knownAt(0, channel, stop) != knownAt(0, channel, cursor)) {
                    unchanged = false;
                    break;
                }
            }
            if (!unchanged) {
                break;
            }
            ++stop;
        }
        const std::size_t horizon = stop - cursor;
        // 同一阶段按(B,V,T)原生前向，groupIds隔离各请求的目标与已知通道
        const std::size_t kept = std::min(cursor, engine.maxContext());
        const std::size_t begin = cursor - kept;
        Tensor input({batch, targets.size(), kept});
        // 已知通道作为模型known行，长度覆盖保留上下文与当前阶段horizon
        Tensor knownValues;
        if (!known.empty()) {
            knownValues = Tensor({batch, known.size(), kept + horizon});
        }
        Timestral::Request stage;
        stage.symmetricAveraging = controls.symmetricAveraging;
        stage.clipNonnegative = controls.clipNonnegative;
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t row = 0; row < targets.size(); ++row) {
                std::copy_n(
                    feedback.values.begin() + (b * channels + targets[row]) * finish + begin,
                    kept,
                    input.values.begin() + (b * targets.size() + row) * kept
                );
            }
            for (std::size_t row = 0; row < known.size(); ++row) {
                for (std::size_t time = 0; time < kept + horizon; ++time) {
                    knownValues.values[(b * known.size() + row) * (kept + horizon) + time] =
                        feedback.values[(b * channels + known[row]) * finish + begin + time];
                }
            }
        }
        const Tensor noCovariates;
        if (!targets.empty()) {
            std::vector<double> times;
            if (engine.continuousRope()) {
                times.resize(elementCount({batch, kept}));
                for (std::size_t b = 0; b < batch; ++b) {
                    for (std::size_t i = 0; i < kept; ++i) {
                        times[b * kept + i] = origins[b] + static_cast<double>(begin + i) * interval;
                    }
                }
            }
            const auto prediction = engine.predict(
                input,
                horizon,
                stage,
                noCovariates,
                knownValues,
                times,
                interval
            );
            for (std::size_t b = 0; b < batch; ++b) {
                for (std::size_t row = 0; row < targets.size(); ++row) {
                    const std::size_t channel = targets[row];
                    for (std::size_t time = 0; time < horizon; ++time) {
                        const std::size_t source = ((b * targets.size() + row) * horizon + time) * levelCount;
                        const std::size_t destination = (channel * steps + cursor - first + time) * levelCount;
                        std::copy_n(
                            prediction.values.begin() + source,
                            levelCount,
                            results[b].quantiles.begin() + destination
                        );
                        feedback.values[(b * channels + channel) * finish + cursor + time] =
                            prediction.values[source + engine.config().medianIndex()];
                    }
                }
            }
        }
        cursor = stop;
    }
    // 每条请求分别恢复原始double读数，不从其他批行复制条件或观测来源
    for (std::size_t b = 0; b < batch; ++b) {
        auto& result = results[b];
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (const Sample& sample : requests[b].get().variables[channel]) {
                if (!std::isfinite(sample.value)) {
                    continue;
                }
                const std::size_t time = gridIndex(sample.time, origins[b], interval);
                if (time >= first && time < finish) {
                    const std::size_t cell = channel * steps + time - first;
                    result.observed[cell] = 1;
                    std::fill_n(result.quantiles.begin() + cell * levelCount, levelCount, sample.value);
                }
            }
        }
        for (double value : result.quantiles) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("conditional prediction returned a nonfinite value");
            }
        }
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (auto& result : results) {
        result.totalSeconds = elapsed;
        result.decodeSeconds = elapsed;
    }
    return results;
}

} // namespace timestral
