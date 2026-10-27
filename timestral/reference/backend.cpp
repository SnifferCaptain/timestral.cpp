/***************
 * @file backend.cpp
 * @brief reference后端的模型工厂实现。
 ***************/

#include "timestral/utils/backend.hpp"

#include <memory>
#include <utility>

#include "timestral/reference/model.hpp"

namespace timestral {

std::shared_ptr<const Model> loadBackendModel(
    const ModelConfig& config,
    utils::Safetensors&& weights,
    std::size_t,
    std::size_t
) {
    return std::shared_ptr<const Model>(reference::makeModel(config, std::move(weights)));
}

} // namespace timestral
