/***************
 * @file decodeBfloat16.cpp
 * @brief 构建期选择decode的Bfloat16默认策略
 ***************/

#include "timestral/blas/decode.hpp"

namespace timestral::blas {

DecodeFormat defaultDecodeFormat() {
    return DecodeFormat::Bfloat16;
}

} // namespace timestral::blas
