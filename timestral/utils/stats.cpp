/***************
 * @file stats.cpp
 * @brief 加权Welford状态的合并实现。
 ***************/

#include "timestral/utils/stats.hpp"

#include <algorithm>
#include <limits>

namespace timestral {

Welford combine(const Welford& prefix, const Welford& suffix) {
    Welford result;
    result.count = prefix.count + suffix.count;
    // 两组都为空时分母取最小正数，避免0/0产生NaN；随后显式归零。
    const double denominator = std::max(result.count, std::numeric_limits<double>::min());
    const double delta = suffix.mean - prefix.mean;
    result.mean = prefix.mean + delta * suffix.count / denominator;
    result.m2 = prefix.m2 + suffix.m2 + delta * delta * prefix.count * suffix.count / denominator;
    if (result.count == 0.0) {
        result.mean = 0.0;
        result.m2 = 0.0;
    }
    return result;
}

} // namespace timestral
