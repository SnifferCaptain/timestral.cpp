find_package(Vulkan REQUIRED)

set(TIMESTRAL_VULKAN_SPV_DIR "" CACHE PATH "已有Vulkan1.1 SPIR-V目录，空值时离线编译GLSL")
set(TIMESTRAL_VULKAN_TEST_DEVICE "0" CACHE STRING "Vulkan算子测试设备枚举号")
option(TIMESTRAL_VULKAN_TEST_VALIDATION "Vulkan算子测试要求Khronos校验层" OFF)

set(TIMESTRAL_VULKAN_SHADER_NAMES
    gemm gemmTile gemmWide gemv rmsNorm softmax silu siluGrid prepareQkv mergeKv elementwise
    attentionScores attentionValues attentionFused attentionRegister attentionTile attentionCausal
    attentionDecode attentionDecodeMerge attentionSubgroup attentionPacked attentionPrefix attentionGroup attentionOrdered rmsNormOrdered elementwiseOrdered
)
set(TIMESTRAL_VULKAN_ALL_SHADER_NAMES ${TIMESTRAL_VULKAN_SHADER_NAMES})
foreach(NAME IN LISTS TIMESTRAL_VULKAN_SHADER_NAMES)
    list(APPEND TIMESTRAL_VULKAN_ALL_SHADER_NAMES "${NAME}Budget")
endforeach()
set(TIMESTRAL_VULKAN_SHADER_FILES)

find_program(TIMESTRAL_VULKAN_SPIRV_VAL spirv-val
    HINTS "${Vulkan_INCLUDE_DIR}/../Bin" "${Vulkan_INCLUDE_DIR}/../bin" "$ENV{VULKAN_SDK}/Bin")
if(TIMESTRAL_VULKAN_SPV_DIR)
    get_filename_component(TIMESTRAL_VULKAN_SHADER_DIR "${TIMESTRAL_VULKAN_SPV_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    set(MISSING_SHADERS)
    foreach(NAME IN LISTS TIMESTRAL_VULKAN_ALL_SHADER_NAMES)
        set(BINARY "${TIMESTRAL_VULKAN_SHADER_DIR}/${NAME}.spv")
        if(NOT EXISTS "${BINARY}" OR IS_DIRECTORY "${BINARY}")
            list(APPEND MISSING_SHADERS "${NAME}.spv")
        else()
            file(READ "${BINARY}" SPV_MAGIC LIMIT 4 HEX)
            if(NOT SPV_MAGIC STREQUAL "03022307" AND NOT SPV_MAGIC STREQUAL "07230203")
                message(FATAL_ERROR "无效SPIR-V头: ${BINARY}")
            endif()
            if(TIMESTRAL_VULKAN_SPIRV_VAL)
                execute_process(COMMAND "${TIMESTRAL_VULKAN_SPIRV_VAL}" --target-env vulkan1.1 "${BINARY}"
                    RESULT_VARIABLE VALID_RESULT OUTPUT_VARIABLE VALID_OUTPUT ERROR_VARIABLE VALID_ERROR)
                if(NOT VALID_RESULT EQUAL 0)
                    message(FATAL_ERROR "离线SPIR-V验证失败: ${BINARY}
${VALID_OUTPUT}${VALID_ERROR}")
                endif()
            endif()
            list(APPEND TIMESTRAL_VULKAN_SHADER_FILES "${BINARY}")
        endif()
    endforeach()
    if(MISSING_SHADERS)
        message(FATAL_ERROR "Vulkan预编译目录缺少普通/预算shader: ${MISSING_SHADERS}")
    endif()
else()
    include("${CMAKE_CURRENT_SOURCE_DIR}/shaderCompiler.cmake")
    set(TIMESTRAL_VULKAN_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    foreach(NAME IN LISTS TIMESTRAL_VULKAN_SHADER_NAMES)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${NAME}.comp")
        set(BINARY "${TIMESTRAL_VULKAN_SHADER_DIR}/${NAME}.spv")
        set(VALIDATION_COMMAND)
        if(TIMESTRAL_VULKAN_SPIRV_VAL)
            set(VALIDATION_COMMAND COMMAND "${TIMESTRAL_VULKAN_SPIRV_VAL}" --target-env vulkan1.1 "${BINARY}")
        endif()
        add_custom_command(
            OUTPUT "${BINARY}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${TIMESTRAL_VULKAN_SHADER_DIR}"
            COMMAND "${TIMESTRAL_VULKAN_SELECTED_COMPILER}" ${TIMESTRAL_VULKAN_TARGET_FLAGS}
                "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders" "${SOURCE}" -o "${BINARY}"
            ${VALIDATION_COMMAND}
            DEPENDS "${SOURCE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/preciseMath.glsl" "${CMAKE_CURRENT_SOURCE_DIR}/shaderCompiler.cmake"
            VERBATIM
        )
        list(APPEND TIMESTRAL_VULKAN_SHADER_FILES "${BINARY}")
        set(BUDGET_SOURCE "${TIMESTRAL_VULKAN_SHADER_DIR}/${NAME}Budget.comp")
        set(BUDGET_BINARY "${TIMESTRAL_VULKAN_SHADER_DIR}/${NAME}Budget.spv")
        set(BUDGET_VALIDATION_COMMAND)
        if(TIMESTRAL_VULKAN_SPIRV_VAL)
            set(BUDGET_VALIDATION_COMMAND COMMAND "${TIMESTRAL_VULKAN_SPIRV_VAL}" --target-env vulkan1.1 "${BUDGET_BINARY}")
        endif()
        add_custom_command(
            OUTPUT "${BUDGET_BINARY}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${TIMESTRAL_VULKAN_SHADER_DIR}"
            COMMAND ${CMAKE_COMMAND} "-DSOURCE=${SOURCE}" "-DOUTPUT=${BUDGET_SOURCE}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/budgetShader.cmake"
            COMMAND "${TIMESTRAL_VULKAN_SELECTED_COMPILER}" ${TIMESTRAL_VULKAN_TARGET_FLAGS}
                "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders" "${BUDGET_SOURCE}" -o "${BUDGET_BINARY}"
            ${BUDGET_VALIDATION_COMMAND}
            DEPENDS "${SOURCE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/preciseMath.glsl" "${CMAKE_CURRENT_SOURCE_DIR}/budgetShader.cmake" "${CMAKE_CURRENT_SOURCE_DIR}/shaderCompiler.cmake"
            VERBATIM
        )
        list(APPEND TIMESTRAL_VULKAN_SHADER_FILES "${BUDGET_BINARY}")
    endforeach()
endif()

add_custom_target(timestralVulkanShaders DEPENDS ${TIMESTRAL_VULKAN_SHADER_FILES})

file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaderPath.cpp"
    CONTENT "#include <filesystem>\nnamespace timestral::vulkan {\nstd::filesystem::path defaultShaderDirectory() {\n    return std::filesystem::u8path(R\"(${TIMESTRAL_VULKAN_SHADER_DIR})\");\n}\n}\n"
)
add_library(timestralVulkanKernels STATIC
    kernels.cpp
    executor.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/shaderPath.cpp"
)
target_compile_features(timestralVulkanKernels PUBLIC cxx_std_17)
set_target_properties(timestralVulkanKernels PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(timestralVulkanKernels PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(timestralVulkanKernels PUBLIC Vulkan::Vulkan Threads::Threads)
add_dependencies(timestralVulkanKernels timestralVulkanShaders)
if(MSVC)
    target_compile_options(timestralVulkanKernels PUBLIC /utf-8 PRIVATE /W4 /fp:precise)
else()
    target_compile_options(timestralVulkanKernels PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

configure_file(shaderManifest.json "${CMAKE_CURRENT_BINARY_DIR}/shaderManifest.json" COPYONLY)
configure_file("${PROJECT_SOURCE_DIR}/timestral/utils/gpuKernels.md" "${CMAKE_CURRENT_BINARY_DIR}/gpuKernels.md" COPYONLY)
install(FILES ${TIMESTRAL_VULKAN_SHADER_FILES} "${CMAKE_CURRENT_BINARY_DIR}/shaderManifest.json"
    "${CMAKE_CURRENT_BINARY_DIR}/gpuKernels.md"
    DESTINATION share/timestral/vulkan
    COMPONENT VulkanShaders
)

if(TIMESTRAL_BUILD_TESTS)
    add_executable(vulkanShaderAbiTest "${PROJECT_SOURCE_DIR}/tests/vulkanShaderAbiTest.cpp")
    target_link_libraries(vulkanShaderAbiTest PRIVATE timestral)
    add_test(NAME vulkanShaderAbi
        COMMAND vulkanShaderAbiTest "${TIMESTRAL_VULKAN_SHADER_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/shaderManifest.json"
    )
    add_executable(vulkanSmokeTest "${PROJECT_SOURCE_DIR}/tests/vulkanSmokeTest.cpp")
    target_link_libraries(vulkanSmokeTest PRIVATE timestralVulkanKernels)
    add_test(NAME vulkanSmoke COMMAND vulkanSmokeTest "${TIMESTRAL_VULKAN_SHADER_DIR}" "${TIMESTRAL_VULKAN_TEST_DEVICE}")
    add_executable(vulkanExecutorTest "${PROJECT_SOURCE_DIR}/tests/vulkanExecutorTest.cpp")
    target_link_libraries(vulkanExecutorTest PRIVATE timestralVulkanKernels)
    add_executable(vulkanKernelsTest "${PROJECT_SOURCE_DIR}/tests/vulkanKernelsTest.cpp")
    target_link_libraries(vulkanKernelsTest PRIVATE timestralVulkanKernels timestral Threads::Threads)
    set(TIMESTRAL_VULKAN_VALIDATION_ARGUMENT)
    if(TIMESTRAL_VULKAN_TEST_VALIDATION)
        set(TIMESTRAL_VULKAN_VALIDATION_ARGUMENT validation)
    endif()
    add_test(NAME vulkanExecutor
        COMMAND vulkanExecutorTest "${TIMESTRAL_VULKAN_SHADER_DIR}"
            "${TIMESTRAL_VULKAN_TEST_DEVICE}" ${TIMESTRAL_VULKAN_VALIDATION_ARGUMENT}
    )
    add_test(NAME vulkanKernels
        COMMAND vulkanKernelsTest "${TIMESTRAL_VULKAN_SHADER_DIR}"
            "${TIMESTRAL_VULKAN_TEST_DEVICE}" ${TIMESTRAL_VULKAN_VALIDATION_ARGUMENT}
    )
endif()
