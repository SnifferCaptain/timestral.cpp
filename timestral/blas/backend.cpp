/***************
 * @file backend.cpp
 * @brief 无外部依赖的专用BLAS模型工厂
 ***************/

#include "timestral/utils/backend.hpp"

#include <memory>
#include <utility>

#include "timestral/blas/model.hpp"

namespace timestral {

std::shared_ptr<const Model> loadBackendModel(
    const ModelConfig& config,
    utils::Safetensors&& weights,
    std::size_t threads,
    std::size_t
) {
    return std::make_shared<const blas::Model>(config, std::move(weights), blas::defaultDecodeFormat(), threads);
}

} // namespace timestral
