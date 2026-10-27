/***************
 * @file decodeFloat16.cpp
 * @brief 构建期选择decode的Float16默认策略
 ***************/

#include "timestral/blas/decode.hpp"

namespace timestral::blas {

DecodeFormat defaultDecodeFormat() {
    return DecodeFormat::Float16;
}

} // namespace timestral::blas
