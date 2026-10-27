cmake_minimum_required(VERSION 3.16)
project(timestralStream LANGUAGES CXX)

# 示例是独立CMake工程：引入根工程只生成库目标，不构建测试和其它示例
add_subdirectory(../.. timestral)

add_executable(timestralStream streamDemo.cpp)
target_link_libraries(timestralStream PRIVATE timestral)
