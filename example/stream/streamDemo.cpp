/***************
 * @file streamDemo.cpp
 * @brief 传感器流式预测演示：Mailbox提交、按来源前缀复用与全量重算对比
 ***************/

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "timestral/timestral.hpp"

static constexpr double sampleInterval = 0.01;
static constexpr std::size_t totalSamples = 600;
static constexpr std::size_t forecastEvery = 30;
static constexpr double horizonSeconds = 0.5;

// 确定性噪声：固定种子线性同余，不引入随机库，只叠加到读数上
static double nextNoise(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (static_cast<double>((state >> 33) & 0xffff) / 65535.0 - 0.5) * 0.2;
}

// 100Hz（10ms）传感器：周期1.28秒的正弦加小噪声，共600点
static std::vector<timestral::Sample> makeSensorSeries() {
    const double pi = 3.14159265358979323846;
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    std::vector<timestral::Sample> samples;
    samples.reserve(totalSamples);
    for (std::size_t i = 0; i < totalSamples; ++i) {
        const double time = static_cast<double>(i) * sampleInterval;
        const double value = 10.0 + 2.0 * std::sin(2.0 * pi * time / 1.28) + nextNoise(state);
        samples.push_back({time, value});
    }
    return samples;
}

// @brief 把秒换算为毫秒，仅用于展示
static double toMilliseconds(double seconds) {
    return seconds * 1000.0;
}

// @brief 把采样前缀提交给Mailbox调度的模型并等待结果
// @param model 调度模型
// @param sensor 完整传感器序列
// @param sampleCount 本次参与预测的样本数
// @return 预测结果
static timestral::Timestral::Result streamForecast(
    timestral::Timestral& model,
    const std::vector<timestral::Sample>& sensor,
    std::size_t sampleCount
) {
    timestral::Timestral::Request request;
    request.streamId = "sensor";
    request.variables.emplace_back(
        sensor.begin(),
        sensor.begin() + static_cast<std::ptrdiff_t>(sampleCount)
    );
    request.horizonSeconds = horizonSeconds;
    return model.submit(std::move(request)).get();
}

// @brief 逐点入流，每30点预测一次，并与同前缀的全量重算对比累计耗时
int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: timestralStream [MODEL_DIRECTORY]\n";
        return 2;
    }
    const std::filesystem::path modelDirectory = argc == 2 ? argv[1] : "model/timestral-5m";
    try {
        const std::vector<timestral::Sample> sensor = makeSensorSeries();
        timestral::Timestral::Settings directSettings;
        directSettings.cachePages = 0;
        timestral::Timestral directModel(modelDirectory, directSettings);
        const std::size_t patchSize = directModel.patchSize();
        timestral::Timestral streamModel(
            modelDirectory,
            {},
            timestral::Timestral::Policy::Mailbox
        );

        std::cout << "streamDemo model=" << modelDirectory.string()
            << " samples=" << totalSamples
            << " dt=" << sampleInterval
            << "s horizon=" << horizonSeconds
            << "s patch=" << patchSize
            << " policy=Mailbox" << '\n';
        std::cout << "note: forecasts start once samples >= 2*patch=" << 2 * patchSize
            << ", so the first row is 90 samples\n";
        std::cout << "samples reused computed hit_rate prefill_ms decode_ms total_ms median0\n";

        double streamSeconds = 0.0;
        double fullSeconds = 0.0;
        double hitRateSum = 0.0;
        std::size_t forecastCount = 0;
        for (std::size_t sampleCount = forecastEvery;
             sampleCount <= totalSamples;
             sampleCount += forecastEvery) {
            if (sampleCount < 2 * patchSize) {
                continue;
            }
            const timestral::Timestral::Result forecast = streamForecast(
                streamModel,
                sensor,
                sampleCount
            );
            streamSeconds += forecast.totalSeconds;
            hitRateSum += forecast.cacheHitRate();
            ++forecastCount;

            // 同一前缀交给无缓存的全量路径，作为加速比基准
            const std::vector<timestral::Sample> prefix(
                sensor.begin(),
                sensor.begin() + static_cast<std::ptrdiff_t>(sampleCount)
            );
            const auto start = std::chrono::steady_clock::now();
            timestral::Timestral::Request directRequest;
            directRequest.variables = {prefix};
            directRequest.horizonSeconds = horizonSeconds;
            const timestral::Timestral::Result direct = directModel.predict(std::move(directRequest));
            fullSeconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start
            ).count();

            std::cout << std::fixed << std::setprecision(6)
                << sampleCount << ' '
                << forecast.reusedPages << ' '
                << forecast.computedPages << ' '
                << forecast.cacheHitRate() << ' '
                << std::setprecision(3)
                << toMilliseconds(forecast.prefillSeconds) << ' '
                << toMilliseconds(forecast.decodeSeconds) << ' '
                << toMilliseconds(forecast.totalSeconds) << ' '
                << std::setprecision(6) << forecast.median(0) << '\n';
        }
        if (forecastCount == 0 || !(streamSeconds > 0.0)) {
            throw std::runtime_error("streamDemo: no forecast was executed");
        }
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "summary forecasts=" << forecastCount
            << " avg_total_ms="
            << toMilliseconds(streamSeconds / static_cast<double>(forecastCount))
            << " avg_hit_rate=" << (hitRateSum / static_cast<double>(forecastCount))
            << " stream_total_ms=" << toMilliseconds(streamSeconds)
            << " full_total_ms=" << toMilliseconds(fullSeconds)
            << " speedup=" << (fullSeconds / streamSeconds) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "streamDemo: " << error.what() << '\n';
        return 1;
    }
}
