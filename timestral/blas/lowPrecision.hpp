/***************
 * @file lowPrecision.hpp
 * @brief 与浮点环境无关的FP16和BF16最近偶数位转换
 ***************/

#pragma once

#include <cstdint>

namespace timestral::blas {

// FP32转FP16采用最近偶数舍入，溢出为带符号inf，保留零符号与可用NaN载荷并统一quiet
std::uint16_t floatToHalf(float value);

// FP16有限值精确扩展到FP32，保留inf、零符号与NaN载荷，sNaN统一quiet
float halfToFloat(std::uint16_t value);

// FP32转BF16采用最近偶数舍入，溢出为带符号inf，NaN不会被舍入成inf
std::uint16_t floatToBfloat16(float value);

// BF16有限值精确扩展到FP32，保留inf、零符号与NaN载荷，sNaN统一quiet
float bfloat16ToFloat(std::uint16_t value);

} // namespace timestral::blas
