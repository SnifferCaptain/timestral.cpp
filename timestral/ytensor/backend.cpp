/***************
 * @file backend.cpp
 * @brief ytensor后端的模型工厂实现。
 ***************/

#include "timestral/utils/backend.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "timestral/reference/model.hpp"
#include "timestral/utils/cpu.hpp"
#include "timestral/ytensor/model.hpp"

namespace timestral {

std::shared_ptr<const Model> loadBackendModel(
    const ModelConfig& config,
    utils::Safetensors&& weights,
    std::size_t threads,
    std::size_t,
    std::function<bool(std::size_t, std::size_t)> timeMask
) {
    // 附加谓词只在共用Transformer前向中实现，本后端改用reference计算，速度明显下降
    if (timeMask) {
        auto model = std::make_shared<reference::Model>(config, std::move(weights));
        model->setTimeMask(std::move(timeMask));
        return model;
    }
    if (!utils::cpuSupportsAvx2Fma()) {
        throw std::runtime_error("ytensor requires AVX2+FMA and OS YMM support; select reference or Eigen on this CPU");
    }
    ytensor::setThreads(threads);
    return std::shared_ptr<const Model>(ytensor::makeModel(config, std::move(weights)));
}

} // namespace timestral
