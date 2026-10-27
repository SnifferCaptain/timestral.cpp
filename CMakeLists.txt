cmake_minimum_required(VERSION 3.16)
project(timestralCpp LANGUAGES CXX)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

option(TIMESTRAL_BUILD_TESTS "构建无第三方框架的测试（tests为本地工作区，不随仓库分发）" OFF)
option(TIMESTRAL_USE_EIGEN "启用Eigen可选后端" OFF)
option(TIMESTRAL_USE_YTENSOR "启用ytensor可选后端" OFF)
option(TIMESTRAL_USE_BLAS "启用无外部依赖的专用BLAS后端" OFF)
option(TIMESTRAL_BUILD_VULKAN_KERNELS "构建独立Vulkan算子验证组件" OFF)
option(TIMESTRAL_USE_VULKAN "启用Vulkan模型后端" OFF)
option(TIMESTRAL_BLAS_AVX2 "BLAS编译AVX2+FMA3运行期分派内核" ON)
option(TIMESTRAL_BLAS_SSE42 "BLAS编译SSE4.2运行期分派内核" ON)
option(TIMESTRAL_USE_OPENMP "Eigen与ytensor后端启用OpenMP多线程" ON)
# 已移除的API不再作为选项，旧配置显式开启时拒绝，避免误以为选中了后端
foreach(RETIRED_API
    TIMESTRAL_BUILD_OPENGL_KERNELS TIMESTRAL_BUILD_OPENCL_KERNELS
    TIMESTRAL_BUILD_OPENCL_EXECUTOR_EXPERIMENT TIMESTRAL_USE_OPENGL TIMESTRAL_USE_OPENCL)
    if(DEFINED ${RETIRED_API} AND ${RETIRED_API})
        message(FATAL_ERROR "${RETIRED_API}已移除；GPU模型请选择TIMESTRAL_USE_VULKAN，不会静默回退")
    endif()
endforeach()

set(TIMESTRAL_REFERENCE_MANIFEST "" CACHE FILEPATH "可选的上游对照数据清单")
set(TIMESTRAL_BLAS_DECODE "FP32" CACHE STRING "仅完整命中分页decode的小M权重格式")
set_property(CACHE TIMESTRAL_BLAS_DECODE PROPERTY STRINGS FP32 FP16 BF16 INT8)
if(NOT TIMESTRAL_BLAS_DECODE MATCHES "^(FP32|FP16|BF16|INT8)$")
    message(FATAL_ERROR "TIMESTRAL_BLAS_DECODE必须为FP32、FP16、BF16或INT8")
endif()

set(TIMESTRAL_BACKEND_COUNT 0)
foreach(BACKEND TIMESTRAL_USE_EIGEN TIMESTRAL_USE_YTENSOR TIMESTRAL_USE_BLAS
    TIMESTRAL_USE_VULKAN)
    if(${BACKEND})
        math(EXPR TIMESTRAL_BACKEND_COUNT "${TIMESTRAL_BACKEND_COUNT}+1")
    endif()
endforeach()
if(TIMESTRAL_BACKEND_COUNT GREATER 1)
    message(FATAL_ERROR "可选计算后端互斥，至多启用其一")
endif()

# Visual Studio跨目标时SYSTEM_PROCESSOR可能仍是宿主AMD64，优先使用实际生成平台
set(TIMESTRAL_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_GENERATOR_PLATFORM)
    string(REGEX REPLACE ",.*$" "" TIMESTRAL_TARGET_PROCESSOR "${CMAKE_GENERATOR_PLATFORM}")
endif()

# 自写BLAS只在x86上提供并验收；ARM与RISC-V等目标改用Eigen，默认reference不受影响
set(TIMESTRAL_X86_TARGET OFF)
if(TIMESTRAL_TARGET_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64|x64|Win32|i[3-6]86|x86)$")
    set(TIMESTRAL_X86_TARGET ON)
endif()
if(TIMESTRAL_USE_BLAS AND NOT TIMESTRAL_X86_TARGET)
    message(STATUS "非x86目标不提供自写BLAS，实际选择Eigen后端")
    set(TIMESTRAL_USE_BLAS OFF)
    set(TIMESTRAL_USE_EIGEN ON)
endif()
if(NOT TIMESTRAL_USE_BLAS AND NOT TIMESTRAL_BLAS_DECODE STREQUAL "FP32")
    message(FATAL_ERROR "低精度decode选项只适用于x86自写BLAS，当前实际后端不支持该选项")
endif()

# 后端由编译单元选择：loadBackendModel在整个构建中只有本变量指定的一个定义
set(TIMESTRAL_SELECTED_BACKEND timestral/reference/backend.cpp)

if(TIMESTRAL_USE_EIGEN)
    set(TIMESTRAL_SELECTED_BACKEND timestral/eigen/backend.cpp)
endif()

if(TIMESTRAL_USE_YTENSOR)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/timestral/ytensor/model.cpp")
        message(FATAL_ERROR "TIMESTRAL_USE_YTENSOR需要timestral/ytensor/model.cpp")
    endif()
    set(TIMESTRAL_SELECTED_BACKEND timestral/ytensor/backend.cpp)
endif()

if(TIMESTRAL_USE_BLAS)
    set(TIMESTRAL_SELECTED_BACKEND timestral/blas/backend.cpp)
endif()
if(TIMESTRAL_USE_VULKAN)
    set(TIMESTRAL_SELECTED_BACKEND timestral/vulkan/backend.cpp)
endif()

add_library(timestral
    timestral/utils/cache.cpp
    timestral/utils/config.cpp
    timestral/utils/condition.cpp
    timestral/utils/cpu.cpp
    timestral/utils/engine.cpp
    timestral/utils/forecast.cpp
    timestral/utils/json.cpp
    timestral/utils/path.cpp
    timestral/utils/patch.cpp
    timestral/utils/postprocess.cpp
    timestral/utils/safetensors.cpp
    timestral/utils/stats.cpp
    timestral/utils/service.cpp
    timestral/utils/tensor.cpp
    timestral/utils/transformer.cpp
    timestral/utils/time.cpp
    timestral/reference/kernels.cpp
    timestral/reference/model.cpp
    ${TIMESTRAL_SELECTED_BACKEND}
    timestral/timestral.cpp
)

target_compile_features(timestral PUBLIC cxx_std_17)
target_link_libraries(timestral PUBLIC Threads::Threads)
set_target_properties(timestral PROPERTIES CXX_EXTENSIONS OFF)

target_include_directories(timestral
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(timestral PUBLIC /utf-8 PRIVATE /W4 /fp:precise)
else()
    target_compile_options(timestral PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

if(TIMESTRAL_USE_EIGEN)
    target_sources(timestral PRIVATE timestral/eigen/model.cpp)
    if(TIMESTRAL_USE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(timestral PRIVATE OpenMP::OpenMP_CXX)
        else()
            message(WARNING "未找到OpenMP，Eigen后端将保持单线程")
        endif()
    endif()
endif()

if(TIMESTRAL_USE_YTENSOR)
    if(NOT TIMESTRAL_X86_TARGET)
        message(FATAL_ERROR "ytensor需要x86 AVX2+FMA目标；非x86请选择reference或Eigen")
    endif()
    # 仅优化模型单元使用C++20与AVX2，工厂和CPU检查保持普通C++17
    add_library(timestralYtensor OBJECT timestral/ytensor/model.cpp)
    set_target_properties(timestralYtensor PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD OFF
        INTERPROCEDURAL_OPTIMIZATION FALSE
    )
    set_target_properties(timestral PROPERTIES UNITY_BUILD OFF INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_include_directories(timestralYtensor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(timestralYtensor SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/3rd")
    target_sources(timestral PRIVATE $<TARGET_OBJECTS:timestralYtensor>)
    if(MSVC)
        target_compile_options(timestralYtensor PRIVATE /utf-8 /W4 /fp:precise /arch:AVX2 /GL-)
        target_compile_options(timestral PRIVATE /GL-)
    else()
        target_compile_options(timestralYtensor PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off -mavx2 -mfma -fno-lto)
        target_compile_options(timestral PRIVATE -fno-lto)
    endif()
    if(TIMESTRAL_USE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            if(MSVC)
                target_compile_options(timestralYtensor PRIVATE /openmp:experimental)
                target_link_libraries(timestral PRIVATE "${OpenMP_CXX_LIBRARIES}")
            else()
                target_link_libraries(timestralYtensor PRIVATE OpenMP::OpenMP_CXX)
                target_link_libraries(timestral PRIVATE OpenMP::OpenMP_CXX)
            endif()
        else()
            message(WARNING "未找到OpenMP，ytensor后端将保持单线程")
        endif()
    endif()
endif()

if(TIMESTRAL_USE_BLAS)
    # ISA边界不能被unity合并或LTO跨单元内联，默认reference构建不受影响
    set_target_properties(timestral PROPERTIES
        UNITY_BUILD OFF
        INTERPROCEDURAL_OPTIMIZATION FALSE
    )
    if(MSVC)
        target_compile_options(timestral PRIVATE /GL-)
    else()
        target_compile_options(timestral PRIVATE -fno-lto)
    endif()
    target_sources(timestral PRIVATE
        timestral/blas/attention.cpp
        timestral/blas/decode.cpp
        timestral/blas/lowPrecision.cpp
        timestral/blas/elementwise.cpp
        timestral/blas/gemm.cpp
        timestral/blas/model.cpp
        timestral/blas/parallel.cpp
    )
    if(TIMESTRAL_BLAS_DECODE STREQUAL "FP16")
        target_sources(timestral PRIVATE timestral/blas/decodeFloat16.cpp)
    elseif(TIMESTRAL_BLAS_DECODE STREQUAL "BF16")
        target_sources(timestral PRIVATE timestral/blas/decodeBfloat16.cpp)
    elseif(TIMESTRAL_BLAS_DECODE STREQUAL "INT8")
        target_sources(timestral PRIVATE timestral/blas/decodeInt8.cpp)
    else()
        target_sources(timestral PRIVATE timestral/blas/decodeFloat32.cpp)
    endif()
    if(TIMESTRAL_X86_TARGET)
        target_sources(timestral PRIVATE timestral/blas/cpuX86.cpp)
        if(TIMESTRAL_BLAS_AVX2)
            set(TIMESTRAL_BLAS_AVX2_SOURCES
                timestral/blas/decodeAvx2.cpp
                timestral/blas/elementwiseAvx2.cpp
                timestral/blas/microkernelAvx2.cpp
                timestral/blas/microkernelOrderedAvx2.cpp
                timestral/blas/microkernelNarrowAvx2.cpp
            )
            target_sources(timestral PRIVATE ${TIMESTRAL_BLAS_AVX2_SOURCES})
            set_source_files_properties(${TIMESTRAL_BLAS_AVX2_SOURCES}
                PROPERTIES
                    SKIP_UNITY_BUILD_INCLUSION ON
                    SKIP_PRECOMPILE_HEADERS ON
            )
            if(MSVC)
                set_source_files_properties(${TIMESTRAL_BLAS_AVX2_SOURCES}
                    PROPERTIES
                        COMPILE_OPTIONS "/arch:AVX2"
                )
            else()
                set_source_files_properties(${TIMESTRAL_BLAS_AVX2_SOURCES}
                    PROPERTIES
                        COMPILE_OPTIONS "-mavx2;-mfma"
                )
                set_source_files_properties(timestral/blas/decodeAvx2.cpp
                    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
                )
            endif()
        else()
            target_sources(timestral PRIVATE
                timestral/blas/decodeFallback.cpp
                timestral/blas/elementwiseFallback.cpp
                timestral/blas/microkernelFallback.cpp
                timestral/blas/microkernelNarrowFallback.cpp
            )
        endif()
        if(TIMESTRAL_BLAS_SSE42)
            target_sources(timestral PRIVATE
                timestral/blas/microkernelSse42.cpp
                timestral/blas/microkernelSse42Enabled.cpp
            )
            set_source_files_properties(timestral/blas/microkernelSse42.cpp
                PROPERTIES
                    SKIP_UNITY_BUILD_INCLUSION ON
                    SKIP_PRECOMPILE_HEADERS ON
            )
            if(MSVC)
                # x64默认基线已支持SSE2，SSE intrinsic不需要也没有/arch:SSE4.2
                if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                    set_source_files_properties(timestral/blas/microkernelSse42.cpp
                        PROPERTIES
                            COMPILE_OPTIONS "/arch:SSE2"
                    )
                endif()
            else()
                set_source_files_properties(timestral/blas/microkernelSse42.cpp
                    PROPERTIES
                        COMPILE_OPTIONS "-msse4.2;-mno-avx;-mno-avx2;-mno-fma;-ffp-contract=off"
                )
            endif()
        else()
            target_sources(timestral PRIVATE timestral/blas/microkernelSse42Fallback.cpp)
        endif()
    endif()
endif()

# 第三方头按外部头引入：抑制其自身警告，本仓库源码仍按/W4检查
# MSVC的/external:I同时提供搜索路径，其它编译器退回普通include目录
if(TIMESTRAL_USE_EIGEN OR TIMESTRAL_USE_YTENSOR)
    if(MSVC)
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rd" TIMESTRAL_THIRD_PARTY_NATIVE)
        target_compile_options(timestral PRIVATE "/external:I${TIMESTRAL_THIRD_PARTY_NATIVE}" /external:W0)
    else()
        target_include_directories(timestral PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/3rd")
    endif()
endif()

if(TIMESTRAL_BUILD_TESTS)
    enable_testing()

    add_test(NAME retiredGpuBackends COMMAND "${CMAKE_COMMAND}"
        "-DPROJECT_ROOT=${CMAKE_CURRENT_SOURCE_DIR}" "-DCOMPILER=${CMAKE_CXX_COMPILER}"
        "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/retired-api-test"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/retiredGpuTest.cmake")
    add_test(NAME shaderCompilerConfig COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaderCompilerTest.cmake")
    foreach(STANDARD 17 20)
        if(STANDARD EQUAL 20 AND NOT "cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
            message(STATUS "当前编译器不提供C++20，默认C++17部署不受影响")
            continue()
        endif()
        add_executable(portabilityTest${STANDARD} tests/portabilityTest.cpp)
        target_link_libraries(portabilityTest${STANDARD} PRIVATE timestral)
        set_target_properties(portabilityTest${STANDARD} PROPERTIES
            CXX_STANDARD ${STANDARD} CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
        add_test(NAME portability${STANDARD} COMMAND portabilityTest${STANDARD})
    endforeach()
    if(TIMESTRAL_USE_YTENSOR)
        add_executable(ytensorGuardTest tests/ytensorGuardTest.cpp)
        target_link_libraries(ytensorGuardTest PRIVATE timestral)
        add_test(NAME ytensorCpuGuard COMMAND ytensorGuardTest "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny")
    endif()

    if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example/tui/terminal.cpp")
        add_executable(terminalTest tests/terminalTest.cpp example/tui/terminal.cpp)
        target_include_directories(terminalTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example/tui")
        if(WIN32)
            target_compile_definitions(terminalTest PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        endif()
        if(MSVC)
            target_compile_options(terminalTest PRIVATE /utf-8)
        endif()
        add_test(NAME terminal COMMAND terminalTest)
    endif()

    add_executable(jsonTest tests/jsonTest.cpp)
    target_link_libraries(jsonTest PRIVATE timestral)
    add_test(NAME json COMMAND jsonTest "${CMAKE_CURRENT_BINARY_DIR}")

    add_executable(safetensorsTest tests/safetensorsTest.cpp)
    target_link_libraries(safetensorsTest PRIVATE timestral)
    add_test(NAME safetensors COMMAND safetensorsTest "${CMAKE_CURRENT_BINARY_DIR}")

    add_executable(tensorTest tests/tensorTest.cpp)
    target_link_libraries(tensorTest PRIVATE timestral)
    add_test(NAME tensor COMMAND tensorTest)

    add_executable(schedulerTest tests/schedulerTest.cpp)
    target_link_libraries(schedulerTest PRIVATE timestral Threads::Threads)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
        add_test(NAME scheduler
            COMMAND schedulerTest
                "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
        )
    endif()

    add_executable(timestralTest tests/timestralTest.cpp)
    target_link_libraries(timestralTest PRIVATE timestral)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
        add_test(NAME timestral
            COMMAND timestralTest
                "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
        )
    endif()

    add_executable(concurrencyTest tests/concurrencyTest.cpp)
    target_link_libraries(concurrencyTest PRIVATE timestral Threads::Threads)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
        add_test(NAME concurrency
            COMMAND concurrencyTest
                "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
        )
    endif()

    add_executable(configTest tests/configTest.cpp)
    target_link_libraries(configTest PRIVATE timestral)
    add_test(NAME config COMMAND configTest)

    if(TIMESTRAL_USE_BLAS)
        add_executable(lowPrecisionTest tests/lowPrecisionTest.cpp)
        target_link_libraries(lowPrecisionTest PRIVATE timestral)
        add_test(NAME lowPrecision COMMAND lowPrecisionTest)

        add_executable(blasTest tests/blasTest.cpp)
        target_link_libraries(blasTest PRIVATE timestral Threads::Threads)
        add_test(NAME blas COMMAND blasTest)

        add_executable(blasParallelTest tests/blasParallelTest.cpp)
        target_link_libraries(blasParallelTest PRIVATE timestral Threads::Threads)
        add_test(NAME blasParallel COMMAND blasParallelTest)
        set_tests_properties(blasParallel PROPERTIES TIMEOUT 30)

        add_executable(attentionTest tests/attentionTest.cpp)
        target_link_libraries(attentionTest PRIVATE timestral)
        add_test(NAME attention COMMAND attentionTest)

        add_executable(blasModelTest tests/blasModelTest.cpp)
        target_link_libraries(blasModelTest PRIVATE timestral)
        add_test(NAME blasModelTiny
            COMMAND blasModelTest "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
        )
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
            add_test(NAME blasModelReal
                COMMAND blasModelTest "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
            )
        endif()

        add_executable(decodeTest tests/decodeTest.cpp)
        target_link_libraries(decodeTest PRIVATE timestral)
        add_test(NAME decode COMMAND decodeTest)

        add_executable(decodeModelTest tests/decodeModelTest.cpp)
        target_link_libraries(decodeModelTest PRIVATE timestral)
        add_test(NAME decodeModelTiny
            COMMAND decodeModelTest "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
        )
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
            add_test(NAME decodeModelReal
                COMMAND decodeModelTest "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
            )
        endif()
    endif()

    add_executable(kernelsTest tests/kernelsTest.cpp)
    target_link_libraries(kernelsTest PRIVATE timestral)
    add_test(NAME kernels COMMAND kernelsTest)

    add_executable(modelTest tests/modelTest.cpp)
    target_link_libraries(modelTest PRIVATE timestral)
    add_test(NAME model
        COMMAND modelTest
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_executable(streamingTest
        tests/streamingTest.cpp
        example/tui/streaming.cpp
    )
    target_link_libraries(streamingTest PRIVATE timestral)
    add_test(NAME streaming
        COMMAND streamingTest "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
            "${CMAKE_CURRENT_SOURCE_DIR}/example/tui/streaming.json"
    )
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
        add_test(NAME streamingReal
            COMMAND streamingTest "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
                "${CMAKE_CURRENT_SOURCE_DIR}/example/tui/streaming.json"
        )
    endif()

    add_executable(timeTest tests/timeTest.cpp)
    target_link_libraries(timeTest PRIVATE timestral Threads::Threads)
    add_test(NAME time
        COMMAND timeTest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
    )

    add_executable(conditionTest tests/conditionTest.cpp)
    target_link_libraries(conditionTest PRIVATE timestral Threads::Threads)
    add_test(NAME condition
        COMMAND conditionTest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
    )

    add_test(NAME conditionContinuous
        COMMAND conditionTest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny" --continuous
    )

    add_executable(multivariateTest tests/multivariateTest.cpp)
    target_link_libraries(multivariateTest PRIVATE timestral Threads::Threads)
    add_test(NAME multivariate
        COMMAND multivariateTest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
        add_test(NAME timeReal
            COMMAND timeTest
                "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
        )
        add_test(NAME conditionReal
            COMMAND conditionTest
                "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
        )
        add_test(NAME conditionContinuousReal
            COMMAND conditionTest
                "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m" --continuous
        )
        add_test(NAME multivariateReal
            COMMAND multivariateTest
                "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
        )
    endif()

    add_executable(parallelDecodeTest
        tests/parallelDecodeTest.cpp
        example/tui/streaming.cpp
    )
    target_link_libraries(parallelDecodeTest PRIVATE timestral)
    add_test(NAME parallelDecodeTiny
        COMMAND parallelDecodeTest "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
    )
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
        add_test(NAME parallelDecodeReal
            COMMAND parallelDecodeTest "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m"
        )
    endif()

    set_tests_properties(parallelDecodeTiny PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    if(TEST parallelDecodeReal)
        set_tests_properties(parallelDecodeReal PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    add_executable(sharedDecodeTest tests/sharedDecodeTest.cpp)
    target_link_libraries(sharedDecodeTest PRIVATE timestral)
    add_test(NAME sharedDecode
        COMMAND sharedDecodeTest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny"
    )

    add_executable(referenceTest tests/referenceTest.cpp)
    target_link_libraries(referenceTest PRIVATE timestral)

    add_test(NAME referenceTiny
        COMMAND referenceTest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/manifest.json"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    if(TIMESTRAL_REFERENCE_MANIFEST)
        add_test(NAME reference
            COMMAND referenceTest
                "${TIMESTRAL_REFERENCE_MANIFEST}"
                "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()

endif()

if(TIMESTRAL_BUILD_VULKAN_KERNELS OR TIMESTRAL_USE_VULKAN)
    add_subdirectory(timestral/vulkan)
endif()

if(TIMESTRAL_USE_VULKAN)
    target_sources(timestral PRIVATE
        timestral/utils/gpuTransformer.cpp
        timestral/utils/gpuSettings.cpp
        timestral/vulkan/model.cpp
    )
    target_link_libraries(timestral PRIVATE timestralVulkanKernels)
    if(TIMESTRAL_BUILD_TESTS)
        add_executable(gpuModelTest tests/gpuModelTest.cpp)
        target_link_libraries(gpuModelTest PRIVATE timestral)
        add_test(NAME gpuModelTiny COMMAND gpuModelTest "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/reference/tiny")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m/model.safetensors")
            add_test(NAME gpuModelReal COMMAND gpuModelTest "${CMAKE_CURRENT_SOURCE_DIR}/model/timestral-5m")
        endif()
    endif()
endif()
