/***************
 * @file elementwiseAvx2.cpp
 * @brief 基于2^x分解向量exp的softmax、门控与SwiGLU
 * 本单元以AVX2编译，只用intrinsic与手写循环，不实例化std内联模板，
 * 避免链接器把AVX2版本的同名COMDAT选给标量回退路径
 * exp2f多项式系数来自Cephes Math Library Release 2.2（1992年6月）：
 * Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
 * Netlib发布说明：可自由使用，但不提供支持或保证
 ***************/

#include "timestral/blas/elementwise.hpp"

#include <immintrin.h>

namespace timestral::blas {

// e^x = 2^n * 2^f，n = round(x*log2e)由魔数1.5*2^23加法取整，整数位直接取自同一浮点和
// f = x*log2e - n由FMA单次舍入求得，log2e拆高低两段补偿常数误差，2^f用Cephes exp2f 6阶多项式，|f|<=0.5
// 最近偶数舍入下[-110,110]有限输入穷举实测最大1.18ulp，2^n拆两半以覆盖次正规输出
// 两端截断值经运算得到0与+inf，max/min把NaN放在第二操作数以继续传播
static inline __m256 exponent(__m256 x) {
    const __m256 clamped = _mm256_min_ps(
        _mm256_set1_ps(88.7228394f),
        _mm256_max_ps(_mm256_set1_ps(-103.972084f), x)
    );
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 magic = _mm256_set1_ps(12582912.0f);
    const __m256 shifted = _mm256_fmadd_ps(clamped, log2e, magic);
    const __m256 n = _mm256_sub_ps(shifted, magic);
    __m256 f = _mm256_fmsub_ps(clamped, log2e, n);
    f = _mm256_fmadd_ps(clamped, _mm256_set1_ps(1.925963033500011e-8f), f);
    __m256 y = _mm256_set1_ps(1.535336188319500e-4f);
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(1.339887440266574e-3f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(9.618437357674640e-3f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(5.550332471162809e-2f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(2.402264791363012e-1f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(6.931472028550421e-1f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(1.0f));
    // n在[-150,128]，两半各在[-75,64]内，均为正规数的2的幂
    const __m256i whole = _mm256_sub_epi32(_mm256_castps_si256(shifted), _mm256_castps_si256(magic));
    const __m256i first = _mm256_srai_epi32(whole, 1);
    const __m256i second = _mm256_sub_epi32(whole, first);
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256 scaleFirst = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(first, bias), 23));
    const __m256 scaleSecond = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(second, bias), 23));
    return _mm256_mul_ps(_mm256_mul_ps(y, scaleFirst), scaleSecond);
}

static inline float horizontalMaximum(__m256 value) {
    __m128 half = _mm_max_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    half = _mm_max_ps(half, _mm_movehl_ps(half, half));
    half = _mm_max_ss(half, _mm_shuffle_ps(half, half, 1));
    return _mm_cvtss_f32(half);
}

static inline float horizontalSum(__m256 value) {
    __m128 half = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    half = _mm_add_ps(half, _mm_movehl_ps(half, half));
    half = _mm_add_ss(half, _mm_shuffle_ps(half, half, 1));
    return _mm_cvtss_f32(half);
}

bool softmaxRowNative(
    const float* scores,
    const std::uint8_t* allowed,
    std::size_t count,
    float factor,
    float& maximum,
    float& denominator,
    float* accumulator,
    std::size_t width,
    float* probabilities,
    std::size_t stride
) {
    // 魔数约简依赖最近偶数舍入，其余模式交给基线标量函数，不修改调用方的浮点环境
    if ((_mm_getcsr() & _MM_ROUND_MASK) != _MM_ROUND_NEAREST) {
        return softmaxRowScalar(
            scores,
            allowed,
            count,
            factor,
            maximum,
            denominator,
            accumulator,
            width,
            probabilities,
            stride
        );
    }
    const __m256 negatives = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0xff800000u)));
    const float negative = _mm256_cvtss_f32(negatives);
    alignas(32) float scaled[64] = {};
    alignas(32) float exps[64] = {};
    const std::size_t vectors = (count + 7) / 8;
    const __m256 scale = _mm256_set1_ps(factor);
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 tileMaximum = negatives;
    __m256 unordered = _mm256_setzero_ps();
    for (std::size_t v = 0; v < vectors; ++v) {
        // 尾部lane与屏蔽键置为-inf，概率随之为0
        __m256i keep = _mm256_cmpgt_epi32(
            _mm256_set1_epi32(static_cast<int>(count - v * 8)),
            lanes
        );
        const __m256i bytes = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(allowed + v * 8))
        );
        keep = _mm256_andnot_si256(_mm256_cmpeq_epi32(bytes, _mm256_setzero_si256()), keep);
        const __m256 value = _mm256_blendv_ps(
            negatives,
            _mm256_mul_ps(_mm256_loadu_ps(scores + v * 8), scale),
            _mm256_castsi256_ps(keep)
        );
        _mm256_store_ps(scaled + v * 8, value);
        tileMaximum = _mm256_max_ps(tileMaximum, value);
        unordered = _mm256_or_ps(unordered, _mm256_cmp_ps(value, value, _CMP_UNORD_Q));
    }
    // 允许键中出现NaN时整行最大值取NaN并向O与l传播，不当作整行屏蔽
    const bool hasNaN = _mm256_movemask_ps(unordered) != 0;
    const float tileValue = hasNaN ? _mm256_cvtss_f32(_mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000))) : horizontalMaximum(tileMaximum);
    if (tileValue == negative) {
        return false;
    }
    const float nextMaximum = hasNaN || maximum != maximum ? tileValue + maximum : (maximum > tileValue ? maximum : tileValue);
    const float alpha = denominator == 0.0f ? 0.0f : _mm256_cvtss_f32(exponent(_mm256_set1_ps(maximum - nextMaximum)));
    denominator *= alpha;
    const __m256 alphas = _mm256_set1_ps(alpha);
    std::size_t d = 0;
    for (; d + 8 <= width; d += 8) {
        _mm256_storeu_ps(accumulator + d, _mm256_mul_ps(_mm256_loadu_ps(accumulator + d), alphas));
    }
    for (; d < width; ++d) {
        accumulator[d] *= alpha;
    }
    const __m256 shift = _mm256_set1_ps(nextMaximum);
    __m256 sum = _mm256_setzero_ps();
    for (std::size_t v = 0; v < vectors; ++v) {
        const __m256 value = _mm256_load_ps(scaled + v * 8);
        __m256 probability = exponent(_mm256_sub_ps(value, shift));
        // 被屏蔽键的-inf减去有限最大值得-inf，经exp为0，这里显式置0也覆盖最大值为+inf的情形
        probability = _mm256_blendv_ps(probability, _mm256_setzero_ps(), _mm256_cmp_ps(value, negatives, _CMP_EQ_OQ));
        _mm256_store_ps(exps + v * 8, probability);
        sum = _mm256_add_ps(sum, probability);
    }
    denominator += horizontalSum(sum);
    for (std::size_t key = 0; key < count; ++key) {
        probabilities[key * stride] = exps[key];
    }
    maximum = nextMaximum;
    return true;
}

// 尾部经8宽临时缓冲走同一向量exp，结果与元素所在位置无关
template <typename Operation>
static void forEachVector(std::size_t count, Operation operation) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        operation(i, 8);
    }
    if (i < count) {
        operation(i, count - i);
    }
}

void sigmoidGateNative(float* values, const float* gate, std::size_t count) {
    if ((_mm_getcsr() & _MM_ROUND_MASK) != _MM_ROUND_NEAREST) {
        sigmoidGateScalar(values, gate, count);
        return;
    }
    const __m256 one = _mm256_set1_ps(1.0f);
    forEachVector(count, [&](std::size_t i, std::size_t lanes) {
        alignas(32) float valueTail[8] = {};
        alignas(32) float gateTail[8] = {};
        const bool full = lanes == 8;
        if (!full) {
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                valueTail[lane] = values[i + lane];
                gateTail[lane] = gate[i + lane];
            }
        }
        const __m256 value = full ? _mm256_loadu_ps(values + i) : _mm256_load_ps(valueTail);
        const __m256 g = full ? _mm256_loadu_ps(gate + i) : _mm256_load_ps(gateTail);
        const __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exponent(_mm256_sub_ps(_mm256_setzero_ps(), g))));
        const __m256 result = _mm256_mul_ps(value, sigmoid);
        if (full) {
            _mm256_storeu_ps(values + i, result);
        } else {
            _mm256_store_ps(valueTail, result);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                values[i + lane] = valueTail[lane];
            }
        }
    });
}

void swiGLUNative(const float* fused, float* output, std::size_t rows, std::size_t width) {
    if ((_mm_getcsr() & _MM_ROUND_MASK) != _MM_ROUND_NEAREST) {
        swiGLUScalar(fused, output, rows, width);
        return;
    }
    const __m256 one = _mm256_set1_ps(1.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* first = fused + row * 2 * width;
        const float* second = first + width;
        float* target = output + row * width;
        forEachVector(width, [&](std::size_t j, std::size_t lanes) {
            alignas(32) float firstTail[8] = {};
            alignas(32) float secondTail[8] = {};
            const bool full = lanes == 8;
            if (!full) {
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    firstTail[lane] = first[j + lane];
                    secondTail[lane] = second[j + lane];
                }
            }
            const __m256 a = full ? _mm256_loadu_ps(first + j) : _mm256_load_ps(firstTail);
            const __m256 b = full ? _mm256_loadu_ps(second + j) : _mm256_load_ps(secondTail);
            const __m256 silu = _mm256_div_ps(b, _mm256_add_ps(one, exponent(_mm256_sub_ps(_mm256_setzero_ps(), b))));
            const __m256 result = _mm256_mul_ps(a, silu);
            if (full) {
                _mm256_storeu_ps(target + j, result);
            } else {
                _mm256_store_ps(firstTail, result);
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    target[j + lane] = firstTail[lane];
                }
            }
        });
    }
}

} // namespace timestral::blas
