/***************
 * @file microkernelSse42Enabled.cpp
 * @brief SSE内核已链接的基线查询，不带SSE专用编译选项
 ***************/

#include "timestral/blas/microkernel.hpp"

namespace timestral::blas {

bool sse42KernelAvailable() {
    return true;
}

} // namespace timestral::blas
