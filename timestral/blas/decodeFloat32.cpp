/***************
 * @file decodeFloat32.cpp
 * @brief 构建期选择decode的Float32默认策略
 ***************/

#include "timestral/blas/decode.hpp"

namespace timestral::blas {

DecodeFormat defaultDecodeFormat() {
    return DecodeFormat::Float32;
}

} // namespace timestral::blas
