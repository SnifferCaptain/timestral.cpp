/***************
 * @file cpuX86.cpp
 * @brief 不依赖AVX编译选项的x86指令集检测
 ***************/

#include "timestral/blas/microkernel.hpp"

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace timestral::blas {

bool cpuSupportsSse42() {
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 0);
    if (registers[0] < 1) {
        return false;
    }
    __cpuidex(registers, 1, 0);
    return (static_cast<std::uint32_t>(registers[2]) & (1u << 20)) != 0;
#else
    if (__get_cpuid_max(0, nullptr) < 1) {
        return false;
    }
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    return (ecx & (1u << 20)) != 0;
#endif
}

bool cpuSupportsF16c() {
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 0);
    if (registers[0] < 1) {
        return false;
    }
    __cpuidex(registers, 1, 0);
    constexpr std::uint32_t required = (1u << 27) | (1u << 28) | (1u << 29);
    return (static_cast<std::uint32_t>(registers[2]) & required) == required && (_xgetbv(0) & 6u) == 6u;
#else
    if (__get_cpuid_max(0, nullptr) < 1) {
        return false;
    }
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    constexpr unsigned int required = (1u << 27) | (1u << 28) | (1u << 29);
    if ((ecx & required) != required) {
        return false;
    }
    unsigned int low = 0;
    unsigned int high = 0;
    __asm__ volatile ("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (low & 6u) == 6u;
#endif
}

bool cpuSupportsAvx2Fma() {
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers, 1, 0);
    const auto features = static_cast<std::uint32_t>(registers[2]);
    constexpr std::uint32_t required = (1u << 12) | (1u << 27) | (1u << 28);
    if ((features & required) != required || (_xgetbv(0) & 6u) != 6u) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (static_cast<std::uint32_t>(registers[1]) & (1u << 5)) != 0;
#else
    if (__get_cpuid_max(0, nullptr) < 7) {
        return false;
    }
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    constexpr unsigned int required = (1u << 12) | (1u << 27) | (1u << 28);
    if ((ecx & required) != required) {
        return false;
    }
    unsigned int xcrLow = 0;
    unsigned int xcrHigh = 0;
    __asm__ volatile ("xgetbv" : "=a"(xcrLow), "=d"(xcrHigh) : "c"(0));
    if ((xcrLow & 6u) != 6u) {
        return false;
    }
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1u << 5)) != 0;
#endif
}

} // namespace timestral::blas
