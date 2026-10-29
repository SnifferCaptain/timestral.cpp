/***************
 * @file timestral.cpp
 * @brief 公开入口：把加载、提交与结果访问转发给内部调度服务
 ***************/

#include "timestral.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include "timestral/utils/engine.hpp"
#include "timestral/utils/service.hpp"

namespace timestral {

Timestral::RequestSuperseded::RequestSuperseded()
    : std::runtime_error("request superseded by a newer request from the same stream or by abort") {
}

Timestral::Timestral(
    const std::filesystem::path& modelDirectory,
    Settings settings,
    Policy policy
)
    : service(std::make_shared<Service>(loadEngine(modelDirectory, settings), settings, policy)) {
}

Timestral::Timestral(const std::filesystem::path& modelDirectory)
    : Timestral(modelDirectory, Settings{}) {
}

Timestral::~Timestral() = default;

void Timestral::setPolicy(Policy policy) {
    service->setPolicy(policy);
}

Timestral::Policy Timestral::policy() const {
    return service->policy();
}

std::future<Timestral::Result> Timestral::submit(Request request) {
    return service->submit(std::move(request));
}

Timestral::Result Timestral::predict(Request request) {
    return service->predict(std::move(request));
}

void Timestral::abort(const std::string& streamId) {
    service->abort(streamId);
}

void Timestral::close(const std::string& session) {
    service->close(session);
}

std::size_t Timestral::pendingRequests() const {
    return service->pendingRequests();
}

std::size_t Timestral::patchSize() const {
    return service->patchSize();
}

std::size_t Timestral::contextLength() const {
    return service->contextLength();
}

std::size_t Timestral::Result::steps() const {
    return levels.empty() || channels == 0 ? 0 : quantiles.size() / levels.size() / channels;
}

double Timestral::Result::time(std::size_t step) const {
    if (step >= steps()) {
        throw std::invalid_argument("Result::time step is out of range");
    }
    return origin + static_cast<double>(step + 1) * sampleInterval;
}

double Timestral::Result::median(std::size_t step, std::size_t channel) const {
    return quantile(step, 0.5, channel);
}

double Timestral::Result::quantile(std::size_t step, double level, std::size_t channel) const {
    if (step >= steps()) {
        throw std::invalid_argument("Result::quantile step is out of range");
    }
    if (channel >= channels) {
        throw std::invalid_argument("Result::quantile channel is out of range");
    }
    if (!std::isfinite(level) || level <= 0.0 || level >= 1.0) {
        throw std::invalid_argument("Result::quantile level must be in (0,1)");
    }
    const std::size_t count = levels.size();
    const std::size_t begin = (channel * steps() + step) * count;
    if (level <= levels.front()) {
        return quantiles[begin];
    }
    if (level >= levels.back()) {
        return quantiles[begin + count - 1];
    }
    const auto upper = std::upper_bound(levels.begin(), levels.end(), level);
    const std::size_t high = static_cast<std::size_t>(upper - levels.begin());
    const std::size_t low = high - 1;
    const double weight = (level - levels[low]) / (levels[high] - levels[low]);
    return quantiles[begin + low] + weight * (quantiles[begin + high] - quantiles[begin + low]);
}

double Timestral::Result::lower(std::size_t step, double level, std::size_t channel) const {
    if (!std::isfinite(level) || level <= 0.0 || level >= 1.0) {
        throw std::invalid_argument("Result::lower level must be in (0,1)");
    }
    return quantile(step, 0.5 * (1.0 - level), channel);
}

double Timestral::Result::upper(std::size_t step, double level, std::size_t channel) const {
    if (!std::isfinite(level) || level <= 0.0 || level >= 1.0) {
        throw std::invalid_argument("Result::upper level must be in (0,1)");
    }
    return quantile(step, 0.5 * (1.0 + level), channel);
}

bool Timestral::Result::isObserved(std::size_t step, std::size_t channel) const {
    if (step >= steps() || channel >= channels) {
        throw std::invalid_argument("Result::isObserved step or channel is out of range");
    }
    return !observed.empty() && observed.at(channel * steps() + step) != 0;
}

double Timestral::Result::cacheHitRate() const {
    const std::size_t total = reusedPages + computedPages;
    return total == 0 ? 0.0 : static_cast<double>(reusedPages) / static_cast<double>(total);
}

} // namespace timestral
