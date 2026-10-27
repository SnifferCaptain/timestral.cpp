/***************
 * @file path.hpp
 * @brief C++17与C++20共用的文件系统路径UTF-8文本
 ***************/

#pragma once

#include <filesystem>
#include <string>

namespace timestral::utils {

// @brief 保留UTF-8字节，不受char8_t或Windows本地代码页影响
std::string pathText(const std::filesystem::path& path);

} // namespace timestral::utils
