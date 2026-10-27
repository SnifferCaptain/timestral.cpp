/***************
 * @file tensor.hpp
 * @brief 连续行主序FP32张量、形状校验与逐元素范围原语。
 ***************/

#pragma once

#include <cstddef>
#include <vector>

namespace timestral {

// @brief 计算形状的元素总数。
// @param shape 各维长度；空形状表示标量，任一维为0表示空张量。
// @return 元素总数。
// @throw std::overflow_error 非零维度的乘积超出size_t范围时抛出。
std::size_t elementCount(const std::vector<std::size_t>& shape);

// 拥有连续行主序FP32存储；公开修改后由调用者在计算前执行validate。
struct Tensor {
    std::vector<std::size_t> shape = {0};
    std::vector<float> values;

    Tensor() = default;

    // @brief 按形状分配存储并填充同一初值。
    // @param shape 各维长度，构造后保存在成员中。
    // @param value 所有元素的初值。
    explicit Tensor(std::vector<std::size_t> shape, float value = 0.0f);

    // @brief 用已有存储构造并立即校验长度。
    // @param shape 各维长度。
    // @param values 连续行主序数据，所有权转移给张量。
    // @throw std::invalid_argument 元素数与形状乘积不一致时抛出。
    Tensor(std::vector<std::size_t> shape, std::vector<float> values);

    // @brief 校验形状乘积等于存储长度。
    // @note 不限制数值中的NaN或无穷；修改shape或values后、计算前必须调用。
    // @throw std::invalid_argument 长度不匹配时抛出。
    void validate() const;

    // @brief 返回校验后的元素总数。
    // @return 形状乘积，等于values.size()。
    std::size_t size() const;
};

// @brief 把区间内的正负无穷换成同区间有限值的最大/最小值。
// @param values 待原地修改的连续存储。
// @param begin 区间起始下标。
// @param count 区间元素数。
// @note NaN与有限值保持不变；区间内没有有限值时无穷替换为0。
void clampInfiniteRange(std::vector<float>& values, std::size_t begin, std::size_t count);

// @brief 把区间内的元素截断到不小于minimum。
// @param values 待原地修改的连续存储。
// @param begin 区间起始下标。
// @param count 区间元素数。
// @param minimum 下界。
void clampMinRange(std::vector<float>& values, std::size_t begin, std::size_t count, float minimum);

// @brief 把区间内的元素限制到[lower, upper]。
// @param values 待原地修改的连续存储。
// @param begin 区间起始下标。
// @param count 区间元素数。
// @param lower 下界，不得大于upper。
// @param upper 上界。
// @throw std::invalid_argument 区间超出存储范围时抛出。
void clampRange(
    std::vector<float>& values,
    std::size_t begin,
    std::size_t count,
    float lower,
    float upper
);

} // namespace timestral
