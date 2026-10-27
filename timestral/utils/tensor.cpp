/***************
 * @file tensor.cpp
 * @brief 张量形状校验与逐元素范围原语。
 ***************/

#include "timestral/utils/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace timestral {

std::size_t elementCount(const std::vector<std::size_t>& shape) {
    // 零维度优先，避免空张量因其余维度的中间乘积被误报溢出。
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
        return 0;
    }
    std::size_t count = 1;
    for (std::size_t dimension : shape) {
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor shape: element count overflow");
        }
        count *= dimension;
    }
    return count;
}

Tensor::Tensor(std::vector<std::size_t> shape, float value)
    : shape(std::move(shape)),
      values(elementCount(this->shape), value) {
}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> values)
    : shape(std::move(shape)),
      values(std::move(values)) {
    validate();
}

void Tensor::validate() const {
    const std::size_t expected = elementCount(shape);
    if (expected != values.size()) {
        throw std::invalid_argument(
            "tensor shape: expected " + std::to_string(expected)
                + " values, got " + std::to_string(values.size())
        );
    }
}

std::size_t Tensor::size() const {
    validate();
    return values.size();
}

// 越界属于调用方的布局错误，直接报错而不是截断区间。
static void requireRange(const std::vector<float>& values, std::size_t begin, std::size_t count) {
    if (begin > values.size() || count > values.size() - begin) {
        throw std::invalid_argument("tensor: range exceeds storage");
    }
}

// 两遍扫描：先取区间内有限值的边界，再只替换无穷；NaN与有限值保持原样。
void clampInfiniteRange(std::vector<float>& values, std::size_t begin, std::size_t count) {
    requireRange(values, begin, count);
    float low = std::numeric_limits<float>::infinity();
    float high = -std::numeric_limits<float>::infinity();
    for (std::size_t q = 0; q < count; ++q) {
        const float value = values[begin + q];
        if (std::isfinite(value)) {
            low = std::min(low, value);
            high = std::max(high, value);
        }
    }
    for (std::size_t q = 0; q < count; ++q) {
        const float value = values[begin + q];
        if (std::isinf(value)) {
            const float replacement = value > 0.0f ? high : low;
            values[begin + q] = std::isfinite(replacement) ? replacement : 0.0f;
        }
    }
}

void clampMinRange(std::vector<float>& values, std::size_t begin, std::size_t count, float minimum) {
    requireRange(values, begin, count);
    for (std::size_t j = 0; j < count; ++j) {
        values[begin + j] = std::max(values[begin + j], minimum);
    }
}

void clampRange(
    std::vector<float>& values,
    std::size_t begin,
    std::size_t count,
    float lower,
    float upper
) {
    requireRange(values, begin, count);
    for (std::size_t j = 0; j < count; ++j) {
        values[begin + j] = std::clamp(values[begin + j], lower, upper);
    }
}

} // namespace timestral
