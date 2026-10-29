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
    std::size_t,
    std::function<bool(std::size_t, std::size_t)> timeMask
) {
    auto model = std::make_shared<blas::Model>(config, std::move(weights), blas::defaultDecodeFormat(), threads);
    model->setTimeMask(std::move(timeMask));
    return model;
}

} // namespace timestral
