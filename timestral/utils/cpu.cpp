/***************
 * @file cpu.cpp
 * @brief 在普通C++17编译单元检查AVX2/FMA与操作系统YMM状态
 ***************/

#include "timestral/utils/cpu.hpp"

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace timestral::utils {

bool avx2FmaEnabled(std::uint32_t leaf1, std::uint32_t leaf7, std::uint64_t xcr0) {
    constexpr std::uint32_t required = (1u << 12) | (1u << 27) | (1u << 28);
    return (leaf1 & required) == required && (leaf7 & (1u << 5)) != 0 && (xcr0 & 6u) == 6u;
}

bool cpuSupportsAvx2Fma() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
    constexpr std::uint32_t required = (1u << 12) | (1u << 27) | (1u << 28);
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers, 1, 0);
    const auto leaf1 = static_cast<std::uint32_t>(registers[2]);
    if ((leaf1 & required) != required) {
        return false;
    }
    const auto xcr0 = _xgetbv(0);
    __cpuidex(registers, 7, 0);
    return avx2FmaEnabled(leaf1, static_cast<std::uint32_t>(registers[1]), xcr0);
#else
    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    const auto leaf1 = ecx;
    if ((leaf1 & required) != required) {
        return false;
    }
    unsigned int low = 0;
    unsigned int high = 0;
    __asm__ volatile ("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    const std::uint64_t xcr0 = (static_cast<std::uint64_t>(high) << 32) | low;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return avx2FmaEnabled(leaf1, ebx, xcr0);
#endif
#else
    return false;
#endif
}

} // namespace timestral::utils
