/***************
 * @file backend.hpp
 * @brief 后端模型工厂声明，实现由构建系统选择的后端编译单元提供。
 ***************/

#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "timestral/utils/config.hpp"
#include "timestral/utils/model.hpp"
#include "timestral/utils/safetensors.hpp"

namespace timestral {

// @brief 按构建选中的后端构造只读模型。
// @param config 模型配置，须已通过validate()。
// @param weights 已载入的权重，所有权转移给模型。
// @param threads 计算线程数，默认1串行，0采用保守自动值（最多4），N为精确值；reference忽略，Eigen与ytensor按同值设置各自的OpenMP核。
// @param gpuThreads Vulkan逻辑线程窗口预算，0不限制，非0为256倍数，其它后端忽略
// @param timeMask time轴附加可见性谓词，空为原生掩码，reference与BLAS原生执行，Eigen、ytensor与Vulkan收到非空值时改用reference模型
// @return 后端模型句柄，构造完成后可跨线程共享。
// @note 全仓库只有一个定义，位于reference、blas、eigen、ytensor或vulkan目录的backend.cpp之一，由CMake的TIMESTRAL_USE_*选项选择，x86未选时为BLAS。Eigen的线程数由OpenMP进程级控制，须在模型构造前设置。
// @throw std::invalid_argument 权重名称、形状或有限值不满足后端契约时抛出。
std::shared_ptr<const Model> loadBackendModel(
    const ModelConfig& config,
    utils::Safetensors&& weights,
    std::size_t threads = 1,
    std::size_t gpuThreads = 0,
    std::function<bool(std::size_t, std::size_t)> timeMask = {}
);

} // namespace timestral
