/***************
 * @file cpu.hpp
 * @brief 可选AVX2后端的基线指令集能力检查
 ***************/

#pragma once

#include <cstdint>

namespace timestral::utils {

// @brief 检查CPUID及OS保存XMM/YMM状态的位，不执行任何AVX指令
bool avx2FmaEnabled(std::uint32_t leaf1, std::uint32_t leaf7, std::uint64_t xcr0);

// @brief 仅在CPU与OS均支持时返回true，非x86返回false
bool cpuSupportsAvx2Fma();

} // namespace timestral::utils
