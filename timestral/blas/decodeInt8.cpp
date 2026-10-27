/***************
 * @file decodeInt8.cpp
 * @brief 构建期选择decode的Int8默认策略
 ***************/

#include "timestral/blas/decode.hpp"

namespace timestral::blas {

DecodeFormat defaultDecodeFormat() {
    return DecodeFormat::Int8;
}

} // namespace timestral::blas
