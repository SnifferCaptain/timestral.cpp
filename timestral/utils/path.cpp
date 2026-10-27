/***************
 * @file path.cpp
 * @brief UTF-8路径转换集中在库的标准版本中，避免混合C++标准的内联定义差异
 ***************/

#include "timestral/utils/path.hpp"

namespace timestral::utils {

std::string pathText(const std::filesystem::path& path) {
    const auto text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

} // namespace timestral::utils
