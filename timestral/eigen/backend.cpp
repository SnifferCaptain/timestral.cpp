/***************
 * @file backend.cpp
 * @brief Eigen后端的模型工厂实现。
 ***************/

#include "timestral/utils/backend.hpp"

#include <memory>
#include <utility>

#include "timestral/eigen/model.hpp"

namespace timestral {

std::shared_ptr<const Model> loadBackendModel(
    const ModelConfig& config,
    utils::Safetensors&& weights,
    std::size_t threads,
    std::size_t
) {
    eigen::setThreads(threads);
    return std::shared_ptr<const Model>(eigen::makeModel(config, std::move(weights)));
}

} // namespace timestral
