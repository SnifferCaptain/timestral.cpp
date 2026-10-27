cmake_minimum_required(VERSION 3.16)
project(timestralTui LANGUAGES CXX)

# 示例是独立CMake工程：引入根工程只生成库目标，不构建测试和其它示例
add_subdirectory(../.. timestral)

add_library(demoTerminal STATIC terminal.cpp)
target_compile_features(demoTerminal PUBLIC cxx_std_17)
set_target_properties(demoTerminal PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(demoTerminal PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(WIN32)
    target_compile_definitions(demoTerminal PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if(MSVC)
    target_compile_options(demoTerminal PUBLIC /utf-8 PRIVATE /W4)
endif()

add_executable(timestralTui streamingTui.cpp streaming.cpp)
target_link_libraries(timestralTui PRIVATE timestral demoTerminal)
