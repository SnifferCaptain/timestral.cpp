/***************
 * @file time.cpp
 * @brief 连续patch坐标与规则网格退化
 ***************/

#include "timestral/utils/time.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "timestral/utils/tensor.hpp"

namespace timestral {

std::vector<double> rotaryCoordinates(
    const std::vector<double>& times,
    std::size_t batch,
    std::size_t channels,
    std::size_t time,
    std::size_t horizonPatches,
    std::size_t patchSize,
    double interval,
    double referenceInterval
) {
    if (batch == 0 || channels == 0 || time == 0 || patchSize == 0 || time % patchSize != 0
        || times.size() != elementCount({batch, time}) || !std::isfinite(interval) || interval <= 0.0
        || !std::isfinite(referenceInterval) || referenceInterval < 0.0) {
        throw std::invalid_argument("continuous RoPE: invalid time grid or reference interval");
    }
    const double reference = referenceInterval == 0.0 ? interval : referenceInterval;
    const std::size_t contextPatches = time / patchSize;
    if (horizonPatches > std::numeric_limits<std::size_t>::max() - contextPatches) {
        throw std::invalid_argument("continuous RoPE: patch count overflow");
    }
    const std::size_t count = contextPatches + horizonPatches;
    elementCount({count, patchSize});
    std::vector<double> result(elementCount({batch, channels, count}), 0.0);
    const long double unit = static_cast<long double>(patchSize) * reference;
    if (!std::isfinite(unit) || unit <= 0.0L) {
        throw std::invalid_argument("continuous RoPE: reference patch duration is not representable");
    }
    for (std::size_t b = 0; b < batch; ++b) {
        const double origin = times[b * time];
        double magnitude = std::abs(origin);
        for (std::size_t i = 0; i < time; ++i) {
            const double value = times[b * time + i];
            if (!std::isfinite(value) || (i > 0 && value <= times[b * time + i - 1])) {
                throw std::invalid_argument("continuous RoPE: timestamps must be finite and strictly increasing");
            }
            magnitude = std::max(magnitude, std::abs(value));
        }
        const double tolerance = 1e-10 + 32.0 * std::numeric_limits<double>::epsilon() * magnitude / static_cast<double>(unit);
        if (!std::isfinite(tolerance) || tolerance >= 0.001) {
            throw std::invalid_argument("continuous RoPE: timestamp precision is insufficient for the reference interval");
        }
        long double anchor = 0.0L;
        for (std::size_t p = 0; p < count; ++p) {
            long double center = 0.0L;
            for (std::size_t i = 0; i < patchSize; ++i) {
                if (p < contextPatches) {
                    center += static_cast<long double>(times[b * time + p * patchSize + i]) - origin;
                } else {
                    const std::size_t future = (p - contextPatches) * patchSize + i + 1;
                    center += static_cast<long double>(times[(b + 1) * time - 1]) - origin
                        + static_cast<long double>(interval) * future;
                }
            }
            center /= patchSize;
            if (p == 0) {
                anchor = center;
            }
            double position = static_cast<double>((center - anchor) / unit);
            if (!std::isfinite(position) || !std::isfinite(static_cast<float>(position))) {
                throw std::invalid_argument("continuous RoPE: position exceeds FP32 rotation range");
            }
            const double integer = std::round(position);
            if (std::abs(position - integer) <= tolerance) {
                position = integer;
            }
            for (std::size_t channel = 0; channel < channels; ++channel) {
                result[(b * channels + channel) * count + p] = position;
            }
        }
    }
    return result;
}

} // namespace timestral
