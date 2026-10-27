/***************
 * @file streaming.hpp
 * @brief JSON信号、多通道独立推进与冻结FIFO前缀实验
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "timestral/utils/model.hpp"

struct FourierTerm {
    double amplitude = 1.0;
    double frequency = 1.0; // 弧度每秒
    double phaseNoise = 0.0;
    double phase = 0.0;
};

struct SignalChannel {
    std::vector<FourierTerm> terms;
    double noise = 0.0;
    double bias = 0.0;
    double sampleHz = 100.0;
    double sampleJitter = 0.0; // 采样间隔对数正态扰动标准差
};

struct StreamingConfig {
    std::size_t window = 512; // cached最多保留的采样槽数，整patch
    std::size_t horizon = 512; // aux与predict合计槽数
    std::size_t advance = 32; // 每次按键推进的公共时间槽数
    std::size_t width = 96;
    std::size_t height = 18;
    double interval = 0.01;
    std::uint64_t seed = 42;
    bool color = false; // 显式启用ANSI颜色与重绘，默认重定向安全
    std::vector<SignalChannel> channels;

    static StreamingConfig load(const std::filesystem::path& path);
    void validate(std::size_t patch, std::size_t maxContext) const;
    double truth(std::size_t channel, double time) const;
};

// 规则网格的带噪函数观测，所有通道必须由JSON明确给出，不包含未来真值
timestral::Tensor sampleSignal(const StreamingConfig& config, std::size_t channels, std::size_t length);

struct StreamingPrediction {
    std::vector<double> values; // 仅第一通道，aux及predict中的原始观测也保持原值
    std::vector<double> p10; // 原生分位数头，不由RMSE或高斯假设生成
    std::vector<double> p90;
    double rmse = 0.0;
    double milliseconds = 0.0;
    std::size_t scored = 0; // 只统计真正预测的位置，不把已知读数计入精度
    std::string status = "insufficient history";
};

class StreamingDemo {
public:
    StreamingDemo(std::shared_ptr<const timestral::Model> model, StreamingConfig config);
    void advance(std::size_t channel);
    void extend(std::size_t slots); // 所有通道同步推进，供长流扫描
    void update();
    void render(std::ostream& output) const;
    void setView(bool bands, bool color, std::size_t width, std::size_t height);
    const std::vector<std::size_t>& progress() const;
    const StreamingPrediction& native() const;
    const StreamingPrediction& accelerated() const;
    std::size_t retainedPages() const;
    std::size_t appendedPages() const;
    std::size_t evictedPages() const;
    std::size_t reusedPages() const;
    std::uint64_t cacheDigest() const;
    std::vector<std::uint64_t> pageDigests() const;

private:
    timestral::CacheAppendInput context(std::size_t begin, std::size_t end) const;
    float observation(std::size_t channel, std::size_t slot) const;
    void grow(std::size_t channel, std::size_t end);
    void appendCommon(std::size_t common);
    StreamingPrediction predict(bool frozen, std::size_t common);

    std::shared_ptr<const timestral::Model> model;
    StreamingConfig config;
    timestral::PagePool pool;
    std::vector<std::size_t> table;
    std::vector<std::size_t> ends;
    std::vector<std::deque<float>> data;
    std::vector<std::mt19937_64> generators;
    std::vector<std::normal_distribution<double>> normals;
    std::vector<double> nextSample;
    std::size_t dataBegin = 0;
    std::size_t cachedEnd = 0;
    std::size_t newPages = 0;
    std::size_t droppedPages = 0;
    std::size_t hits = 0;
    double appendMilliseconds = 0.0;
    bool showBands = true;
    StreamingPrediction nativeResult;
    StreamingPrediction fastResult;
};
