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
    std::size_t,
    std::function<bool(std::size_t, std::size_t)> timeMask
) {
    auto model = std::make_shared<reference::Model>(config, std::move(weights));
    model->setTimeMask(std::move(timeMask));
    return model;
}

} // namespace timestral
