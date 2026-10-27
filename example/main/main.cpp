/***************
 * @file main.cpp
 * @brief 读取本地模型，对生成的单变量序列预测并输出CSV
 ***************/

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "timestral/timestral.hpp"

// @brief 解析十进制正整数，空串、非数字和0都按非法参数拒绝
static std::size_t parseLength(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("prediction length must be a positive integer");
    }
    const auto value = std::stoull(text);
    if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("prediction length is out of range");
    }
    return static_cast<std::size_t>(value);
}

// @brief 生成128点正弦历史，按给定步数预测并输出中位数与90%区间的CSV
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: timestralExample MODEL_DIRECTORY [PREDICTION_LENGTH]\n";
        return 2;
    }
    try {
        const std::size_t predictionLength = argc == 3 ? parseLength(argv[2]) : 32;
        const double sampleInterval = 0.01;
        std::vector<timestral::Sample> history;
        history.reserve(128);
        for (std::size_t t = 0; t < 128; ++t) {
            history.push_back({
                static_cast<double>(t) * sampleInterval,
                10.0 + std::sin(static_cast<double>(t) * 0.15)
            });
        }
        const auto start = std::chrono::steady_clock::now();
        timestral::Timestral model{std::filesystem::path(argv[1])};
        const auto loaded = std::chrono::steady_clock::now();
        timestral::Timestral::Request request;
        request.variables = {std::move(history)};
        request.horizonSeconds = static_cast<double>(predictionLength) * sampleInterval;
        const timestral::Timestral::Result result = model.predict(std::move(request));
        const auto finished = std::chrono::steady_clock::now();
        const double loadMs = std::chrono::duration<double, std::milli>(loaded - start).count();
        const double predictMs = std::chrono::duration<double, std::milli>(
            finished - loaded
        ).count();
        std::cerr << "load_ms=" << loadMs << " predict_ms=" << predictMs << '\n';
        std::cout << "time,median,q05,q95\n" << std::setprecision(9);
        for (std::size_t t = 0; t < result.steps(); ++t) {
            std::cout << result.time(t)
                << ',' << result.median(t)
                << ',' << result.quantile(t, 0.05)
                << ',' << result.quantile(t, 0.95) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
