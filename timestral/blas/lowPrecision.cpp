/***************
 * @file lowPrecision.cpp
 * @brief FP32与低精度格式的确定性位转换及特殊值处理
 ***************/

#include "timestral/blas/lowPrecision.hpp"

#include <cstdint>
#include <cstring>
#include <limits>

namespace timestral::blas {

static_assert(
    sizeof(float) == sizeof(std::uint32_t)
        && std::numeric_limits<float>::is_iec559
        && std::numeric_limits<float>::digits == 24
        && std::numeric_limits<float>::max_exponent == 128,
    "low precision conversion requires IEEE 754 binary32"
);

static std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float floatFromBits(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// shift在[1,24]内，恰中点时只查看保留部分的最低位
static std::uint32_t roundedShift(std::uint32_t value, unsigned int shift) {
    const std::uint32_t retained = value >> shift;
    const std::uint32_t remainder = value & ((std::uint32_t{1} << shift) - 1);
    const std::uint32_t midpoint = std::uint32_t{1} << (shift - 1);
    return retained + (remainder > midpoint || (remainder == midpoint && (retained & 1u) != 0) ? 1u : 0u);
}

std::uint16_t floatToHalf(float value) {
    const std::uint32_t bits = floatBits(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        const std::uint32_t payload = fraction == 0 ? 0u : (fraction >> 13) | 0x200u;
        return static_cast<std::uint16_t>(sign | 0x7c00u | payload);
    }
    const int halfExponent = static_cast<int>(exponent) - 112;
    if (halfExponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        // 次正规舍入可进位到最小正规值，不能提前截掉隐含的首位
        const std::uint32_t rounded = roundedShift(fraction | 0x800000u, static_cast<unsigned int>(14 - halfExponent));
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded = roundedShift(fraction, 13);
    // 尾数进位自然进入下一指数，最大正规值的进位产生inf
    return static_cast<std::uint16_t>(sign | ((static_cast<std::uint32_t>(halfExponent) << 10) + rounded));
}

float halfToFloat(std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value) & 0x8000u) << 16;
    const std::uint32_t exponent = (static_cast<std::uint32_t>(value) >> 10) & 0x1fu;
    std::uint32_t fraction = static_cast<std::uint32_t>(value) & 0x3ffu;
    if (exponent == 0) {
        if (fraction == 0) {
            return floatFromBits(sign);
        }
        int normalExponent = -14;
        while ((fraction & 0x400u) == 0) {
            fraction <<= 1;
            --normalExponent;
        }
        return floatFromBits(
            sign
                | (static_cast<std::uint32_t>(normalExponent + 127) << 23)
                | ((fraction & 0x3ffu) << 13)
        );
    }
    if (exponent == 0x1fu) {
        const std::uint32_t payload = fraction == 0 ? 0u : (fraction << 13) | 0x400000u;
        return floatFromBits(sign | 0x7f800000u | payload);
    }
    return floatFromBits(sign | ((exponent + 112u) << 23) | (fraction << 13));
}

std::uint16_t floatToBfloat16(float value) {
    const std::uint32_t bits = floatBits(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu) != 0) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    }
    const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

float bfloat16ToFloat(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu) != 0) {
        bits |= 0x400000u;
    }
    return floatFromBits(bits);
}

} // namespace timestral::blas
