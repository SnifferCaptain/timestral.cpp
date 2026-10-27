cmake_minimum_required(VERSION 3.16)
project(timestralSignal LANGUAGES CXX)

# 示例是独立CMake工程：引入根工程只生成库目标，不构建测试和其它示例
add_subdirectory(../.. timestral)

add_executable(timestralSignal signal.cpp)
target_link_libraries(timestralSignal PRIVATE timestral)
