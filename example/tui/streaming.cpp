/***************
 * @file streaming.cpp
 * @brief 无噪声真值、可重复采样与持久FIFO缓存对照
 ***************/

#include "streaming.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <set>
#include <stdexcept>
#include <utility>

#include "timestral/utils/engine.hpp"
#include "timestral/utils/json.hpp"

static double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start
    ).count();
}

static void keys(const timestral::utils::Json& value, const std::set<std::string>& allowed) {
    for (const auto& entry : value.asObject()) {
        if (allowed.count(entry.first) == 0) {
            throw std::invalid_argument("unknown config key: " + entry.first);
        }
    }
}

StreamingConfig StreamingConfig::load(const std::filesystem::path& path) {
    const auto root = timestral::utils::Json::load(path);
    keys(root, {"window", "horizon", "advance", "width", "height", "interval", "seed", "color", "channels"});
    StreamingConfig config;
    for (const auto& entry : root.asObject()) {
        const auto& name = entry.first;
        const auto& value = entry.second;
        if (name == "window") {
            config.window = value.asSize();
        } else if (name == "horizon") {
            config.horizon = value.asSize();
        } else if (name == "advance") {
            config.advance = value.asSize();
        } else if (name == "width") {
            config.width = value.asSize();
        } else if (name == "height") {
            config.height = value.asSize();
        } else if (name == "interval") {
            config.interval = value.asDouble();
        } else if (name == "color") {
            config.color = value.asBool();
        } else if (name == "seed") {
            config.seed = value.asSize();
        }
    }
    for (const auto& channel : root.at("channels").asArray()) {
        keys(channel, {"terms", "noise", "bias", "sampleHz", "sampleJitter"});
        SignalChannel signal;
        signal.noise = channel.at("noise").asDouble();
        signal.bias = channel.at("bias").asDouble();
        signal.sampleHz = channel.at("sampleHz").asDouble();
        signal.sampleJitter = channel.at("sampleJitter").asDouble();
        for (const auto& term : channel.at("terms").asArray()) {
            keys(term, {"a", "b", "c", "d"});
            signal.terms.push_back({
                term.at("a").asDouble(),
                term.at("b").asDouble(),
                term.at("c").asDouble(),
                term.at("d").asDouble()
            });
        }
        config.channels.push_back(std::move(signal));
    }
    return config;
}

void StreamingConfig::validate(std::size_t patch, std::size_t maxContext) const {
    if (patch == 0 || window < 2 * patch || window > maxContext || window % patch != 0
        || horizon == 0 || horizon > maxContext - window || horizon % patch != 0
        || advance == 0 || advance > horizon || advance % patch != 0
        || width < 16 || width > 240 || height < 5 || height > 60
        || !std::isfinite(interval) || interval <= 0.0
        || channels.empty() || channels.size() > 9) {
        throw std::invalid_argument("invalid streaming geometry, time unit or channel count");
    }
    for (const auto& signal : channels) {
        if (!std::isfinite(signal.noise) || signal.noise < 0.0 || !std::isfinite(signal.bias)
            || !std::isfinite(signal.sampleHz) || signal.sampleHz <= 0.0
            || signal.sampleHz * interval > 100.0 || signal.terms.empty()
            || !std::isfinite(signal.sampleJitter) || signal.sampleJitter < 0.0
            || signal.sampleJitter > 1.0) {
            throw std::invalid_argument("invalid channel noise, sampling or terms");
        }
        for (const auto& term : signal.terms) {
            if (!std::isfinite(term.amplitude) || !std::isfinite(term.frequency)
                || !std::isfinite(term.phaseNoise) || term.phaseNoise < 0.0
                || !std::isfinite(term.phase)) {
                throw std::invalid_argument("nonfinite Fourier term or negative phase noise");
            }
        }
    }
}

double StreamingConfig::truth(std::size_t channel, double time) const {
    const auto& signal = channels.at(channel);
    double value = signal.bias;
    for (const auto& term : signal.terms) {
        value += term.amplitude * std::sin(term.frequency * time + term.phase);
    }
    return value;
}

timestral::Tensor sampleSignal(const StreamingConfig& signal, std::size_t channels, std::size_t length) {
    if (channels > signal.channels.size()) {
        throw std::invalid_argument("JSON must define every requested channel");
    }
    timestral::Tensor context({channels, length});
    for (std::size_t row = 0; row < channels; ++row) {
        const auto& channel = signal.channels[row];
        std::mt19937_64 generator(signal.seed + row * 10007);
        std::normal_distribution<double> normal(0.0, 1.0);
        for (std::size_t t = 0; t < length; ++t) {
            const double time = t * signal.interval;
            double value = channel.bias;
            for (const auto& term : channel.terms) {
                value += term.amplitude * std::sin(term.frequency * time + term.phaseNoise * normal(generator) + term.phase);
            }
            value += channel.noise * normal(generator);
            if (!std::isfinite(value)) {
                throw std::runtime_error("nonfinite JSON signal");
            }
            context.values[row * length + t] = static_cast<float>(value);
        }
    }
    return context;
}


StreamingDemo::StreamingDemo(
    std::shared_ptr<const timestral::Model> loadedModel,
    StreamingConfig settings
) : model(std::move(loadedModel)), config(std::move(settings)) {
    if (!model || !model->supportsIncremental()) {
        throw std::invalid_argument("streaming experiment needs reference or Eigen pagination");
    }
    config.validate(model->config().patchSize, model->config().contextLength);
    pool = timestral::PagePool(
        model->pageGeometry(config.channels.size()),
        model->pageFingerprint(),
        config.window / model->config().patchSize
    );
    ends.assign(config.channels.size(), 0);
    data.resize(ends.size());
    nextSample.assign(ends.size(), 0.0);
    for (std::size_t channel = 0; channel < ends.size(); ++channel) {
        generators.emplace_back(config.seed + channel * 10007);
        normals.emplace_back(0.0, 1.0);
    }
}

void StreamingDemo::advance(std::size_t channel) {
    if (channel >= ends.size()) {
        throw std::invalid_argument("channel index out of range");
    }
    const auto common = *std::min_element(ends.begin(), ends.end());
    if (ends[channel] > std::numeric_limits<std::size_t>::max() - config.advance
        || ends[channel] + config.advance - common > config.horizon) {
        throw std::invalid_argument("aux would exceed horizon; advance the shorter channels first");
    }
    grow(channel, ends[channel] + config.advance);
}

void StreamingDemo::grow(std::size_t channel, std::size_t end) {
    auto& values = data[channel];
    values.resize(end - dataBegin, std::numeric_limits<float>::quiet_NaN());
    auto& rng = generators[channel];
    // 分布对象跨grow调用保留缓存，保证分块推进不改变随机序列
    auto& normal = normals[channel];
    const auto& signal = config.channels[channel];
    const double stop = static_cast<double>(end) * config.interval;
    while (nextSample[channel] < stop) {
        const double time = nextSample[channel];
        // 分块边界上舍入可能把样本推到end之后，先判断再抽样，不能用掉随机数
        const auto slot = static_cast<std::size_t>(std::floor(time / config.interval + 1e-9));
        if (slot >= end) {
            break;
        }
        double value = signal.bias + signal.noise * normal(rng);
        for (const auto& term : signal.terms) {
            value += term.amplitude * std::sin(
                term.frequency * time + term.phaseNoise * normal(rng) + term.phase
            );
        }
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
            throw std::runtime_error("generated observation exceeds FP32");
        }
        if (slot >= dataBegin) {
            values[slot - dataBegin] = static_cast<float>(value);
        }
        // 对数间隔保证时间严格递增，截断随机极值而非生成非正采样周期
        const double jitter = std::clamp(signal.sampleJitter * normal(rng), -4.0, 4.0);
        nextSample[channel] += std::exp(jitter) / signal.sampleHz;
        if (nextSample[channel] <= time) {
            throw std::runtime_error("sample timestamp precision exhausted");
        }
    }
    ends[channel] = end;
}

void StreamingDemo::extend(std::size_t slots) {
    const auto patch = model->config().patchSize;
    if (slots == 0 || slots % patch != 0) {
        throw std::invalid_argument("extend requires a positive whole-patch slot count");
    }
    const auto common = *std::min_element(ends.begin(), ends.end());
    for (const auto end : ends) {
        if (end != common) {
            throw std::invalid_argument("extend requires equally advanced channels");
        }
    }
    if (common > std::numeric_limits<std::size_t>::max() - slots) {
        throw std::overflow_error("stream timeline overflow");
    }
    const auto end = common + slots;
    for (std::size_t channel = 0; channel < ends.size(); ++channel) {
        grow(channel, end);
    }
    appendCommon(end);
    const auto keep = end - std::min(end, config.window);
    while (dataBegin < keep) {
        for (auto& channel : data) {
            channel.pop_front();
        }
        ++dataBegin;
    }
}

float StreamingDemo::observation(std::size_t channel, std::size_t slot) const {
    if (slot < dataBegin || slot >= ends[channel]) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return data[channel].at(slot - dataBegin);
}

timestral::CacheAppendInput StreamingDemo::context(std::size_t begin, std::size_t end) const {
    timestral::CacheAppendInput input;
    const auto rows = ends.size();
    const auto length = end - begin;
    input.context = timestral::Tensor({rows, length});
    input.observed.assign(rows * length, 0);
    input.groupIds.assign(rows, 0);
    input.positionOffset = static_cast<double>(begin / model->config().patchSize);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t t = 0; t < length; ++t) {
            const auto value = observation(row, begin + t);
            if (std::isfinite(value)) {
                input.context.values[row * length + t] = value;
                input.observed[row * length + t] = 1;
            }
        }
    }
    return input;
}

void StreamingDemo::appendCommon(std::size_t common) {
    const auto start = std::chrono::steady_clock::now();
    newPages = 0;
    droppedPages = 0;
    hits = table.size();
    const auto patch = model->config().patchSize;
    while (cachedEnd < common) {
        if (table.size() == pool.capacity()) {
            const auto oldest = table.front();
            pool.release(oldest);
            pool.discard(oldest);
            table.erase(table.begin());
            ++droppedPages;
            if (hits > 0) {
                --hits;
            }
        }
        const auto begin = cachedEnd - table.size() * patch;
        auto input = context(begin, cachedEnd + patch);
        model->appendPages(input, pool, table);
        cachedEnd += patch;
        ++newPages;
    }
    appendMilliseconds = elapsedMs(start);
}

StreamingPrediction StreamingDemo::predict(bool frozen, std::size_t common) {
    StreamingPrediction result;
    if (common < 2 * model->config().patchSize) {
        return result;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto patch = model->config().patchSize;
    const auto begin = common - std::min(common, config.window);
    if (common > std::numeric_limits<std::size_t>::max() - config.horizon) {
        throw std::overflow_error("stream timeline overflow");
    }
    const auto finish = common + config.horizon;
    const auto rows = ends.size();
    const auto width = finish - begin;
    timestral::Tensor feedback({rows, width});
    std::vector<std::uint8_t> masks(rows * width, 0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t t = begin; t < finish; ++t) {
            const auto value = observation(row, t);
            if (std::isfinite(value)) {
                feedback.values[row * width + t - begin] = value;
                masks[row * width + t - begin] = 1;
            }
        }
    }
    result.values.assign(config.horizon, std::numeric_limits<double>::quiet_NaN());
    result.p10 = result.values;
    result.p90 = result.values;
    const auto& knots = model->config().quantileKnots;
    const auto lower = static_cast<std::size_t>(std::find(knots.begin(), knots.end(), 0.1f) - knots.begin());
    const auto upper = static_cast<std::size_t>(std::find(knots.begin(), knots.end(), 0.9f) - knots.begin());
    if (lower == knots.size() || upper == knots.size()) {
        throw std::runtime_error("model does not contain p10/p90 knots");
    }
    const auto median = static_cast<std::size_t>(
        std::find(knots.begin(), knots.end(), 0.5f) - knots.begin()
    );
    if (median == knots.size()) {
        throw std::runtime_error("model does not contain median knot");
    }
    double squared = 0.0;
    if (!frozen) {
        timestral::Timestral::Settings settings;
        settings.patchBatch = 1;
        settings.cachePages = 0;
        timestral::Engine engine(model, settings);
        timestral::Timestral::Request controls;
        controls.symmetricAveraging = false;
        controls.clipNonnegative = false;
        std::size_t cursor = common;
        while (cursor < finish) {
            std::vector<std::size_t> targets;
            std::vector<std::size_t> known;
            for (std::size_t row = 0; row < rows; ++row) {
                (std::isfinite(observation(row, cursor)) ? known : targets).push_back(row);
            }
            auto stop = cursor + 1;
            while (stop < finish) {
                bool same = true;
                for (std::size_t row = 0; row < rows; ++row) {
                    same = same && (std::isfinite(observation(row, cursor))
                        == std::isfinite(observation(row, stop)));
                }
                if (!same) {
                    break;
                }
                ++stop;
            }
            const auto length = cursor - begin;
            const auto horizon = stop - cursor;
            if (!targets.empty()) {
                timestral::Tensor input({1, targets.size(), length});
                timestral::Tensor covariates;
                if (!known.empty()) {
                    covariates = timestral::Tensor({1, known.size(), length + horizon});
                }
                for (std::size_t row = 0; row < targets.size(); ++row) {
                    for (std::size_t t = 0; t < length; ++t) {
                        const auto index = targets[row] * width + t;
                        input.values[row * length + t] = masks[index] != 0
                            ? feedback.values[index] : std::numeric_limits<float>::quiet_NaN();
                    }
                }
                for (std::size_t row = 0; row < known.size(); ++row) {
                    for (std::size_t t = 0; t < length + horizon; ++t) {
                        const auto index = known[row] * width + t;
                        covariates.values[row * (length + horizon) + t] = masks[index] != 0
                            ? feedback.values[index] : std::numeric_limits<float>::quiet_NaN();
                    }
                }
                const auto prediction = engine.predict(input, horizon, controls, {}, covariates);
                for (std::size_t row = 0; row < targets.size(); ++row) {
                    for (std::size_t t = 0; t < horizon; ++t) {
                        const auto index = targets[row] * width + length + t;
                        feedback.values[index] = prediction.values[(row * horizon + t) * knots.size() + median];
                        masks[index] = 1;
                        if (targets[row] == 0) {
                            const auto base = (row * horizon + t) * knots.size();
                            result.p10[cursor - common + t] = prediction.values[base + lower];
                            result.p90[cursor - common + t] = prediction.values[base + upper];
                        }
                    }
                }
            }
            for (std::size_t t = cursor; t < stop; ++t) {
                const auto value = feedback.values[t - begin];
                if (!std::isfinite(value)) {
                    throw std::runtime_error("nonfinite native prediction");
                }
                result.values[t - common] = value;
                if (std::isfinite(observation(0, t))) {
                    result.p10[t - common] = value;
                    result.p90[t - common] = value;
                }
                if (!std::isfinite(observation(0, t))) {
                    const double error = value - config.truth(0, static_cast<double>(t) * config.interval);
                    squared += error * error;
                    ++result.scored;
                }
            }
            cursor = stop;
        }
        result.rmse = result.scored == 0 ? 0.0 : std::sqrt(squared / result.scored);
        result.milliseconds = elapsedMs(start);
        result.status = "ok";
        return result;
    }
    for (std::size_t cursor = common; cursor < finish; cursor += patch) {
        const auto length = cursor - begin;
        timestral::ForwardInput input;
        input.context = timestral::Tensor({rows, length});
        input.observed.assign(rows * length, 0);
        input.groupIds.assign(rows, 0);
        input.roles.assign(rows, 0);
        input.horizonPatches = 1;
        input.hasFuture = true;
        input.future = timestral::Tensor({rows, patch});
        input.futureObserved.assign(rows * patch, 0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t t = 0; t < length; ++t) {
                input.context.values[row * length + t] = feedback.values[row * width + t];
                input.observed[row * length + t] = masks[row * width + t];
            }
            if (ends[row] >= cursor + patch) {
                input.roles[row] = 2;
                for (std::size_t t = 0; t < patch; ++t) {
                    input.future.values[row * patch + t] = feedback.values[row * width + length + t];
                    input.futureObserved[row * patch + t] = masks[row * width + length + t];
                }
            }
        }
        timestral::Tensor prediction;
        if (frozen) {
            // 后端只读pool接口，旧KV不会因aux的角色或未来值变化而重算
            prediction = model->predictFrozen(
                input,
                pool,
                table,
                static_cast<double>(begin / patch)
            );
        } else {
            prediction = model->predictBlock(input);
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t t = 0; t < patch; ++t) {
                const auto index = row * width + length + t;
                const bool known = masks[index] != 0;
                if (!known) {
                    const auto value = prediction.values[(row * patch + t) * knots.size() + median];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("nonfinite streaming prediction");
                    }
                    feedback.values[index] = value;
                    masks[index] = 1;
                }
                if (row == 0) {
                    const auto step = cursor - common + t;
                    result.values[step] = feedback.values[index];
                    const auto base = t * knots.size();
                    result.p10[step] = known ? feedback.values[index] : prediction.values[base + lower];
                    result.p90[step] = known ? feedback.values[index] : prediction.values[base + upper];
                    if (!known) {
                        const auto time = static_cast<double>(cursor + t) * config.interval;
                        const double error = result.values[step] - config.truth(0, time);
                        squared += error * error;
                        ++result.scored;
                    }
                }
            }
        }
    }
    result.rmse = result.scored == 0 ? 0.0 : std::sqrt(squared / result.scored);
    result.milliseconds = elapsedMs(start);
    result.status = "ok";
    return result;
}

void StreamingDemo::update() {
    const auto common = *std::min_element(ends.begin(), ends.end());
    if (common > 0) {
        appendCommon(common);
    } else {
        newPages = 0;
        droppedPages = 0;
        hits = table.size();
        appendMilliseconds = 0.0;
    }
    nativeResult = predict(false, common);
    fastResult = predict(true, common);
    fastResult.milliseconds += appendMilliseconds;
    const auto keep = common - std::min(common, config.window);
    while (dataBegin < keep) {
        for (auto& channel : data) {
            channel.pop_front();
        }
        ++dataBegin;
    }
}

const std::vector<std::size_t>& StreamingDemo::progress() const {
    return ends;
}

const StreamingPrediction& StreamingDemo::native() const {
    return nativeResult;
}

const StreamingPrediction& StreamingDemo::accelerated() const {
    return fastResult;
}

std::size_t StreamingDemo::retainedPages() const {
    return table.size();
}

std::size_t StreamingDemo::appendedPages() const {
    return newPages;
}

std::size_t StreamingDemo::evictedPages() const {
    return droppedPages;
}

std::size_t StreamingDemo::reusedPages() const {
    return hits;
}

std::uint64_t StreamingDemo::cacheDigest() const {
    timestral::FnvHash hash;
    for (const auto index : table) {
        const auto& page = pool.page(index);
        for (const auto& values : page.keys) {
            hash.addValues(values);
        }
        for (const auto& values : page.values) {
            hash.addValues(values);
        }
    }
    return hash.value();
}

std::vector<std::uint64_t> StreamingDemo::pageDigests() const {
    std::vector<std::uint64_t> result;
    for (const auto index : table) {
        timestral::FnvHash hash;
        const auto& page = pool.page(index);
        for (const auto& values : page.keys) {
            hash.addValues(values);
        }
        for (const auto& values : page.values) {
            hash.addValues(values);
        }
        result.push_back(hash.value());
    }
    return result;
}

void StreamingDemo::setView(bool bands, bool color, std::size_t width, std::size_t height) {
    showBands = bands;
    config.color = color;
    config.width = std::clamp(width, std::size_t{16}, std::size_t{240});
    config.height = std::clamp(height, std::size_t{5}, std::size_t{60});
}

void StreamingDemo::render(std::ostream& output) const {
    if (config.color) {
        output << "\x1b[2J\x1b[H";
    }
    const auto common = *std::min_element(ends.begin(), ends.end());
    const auto longest = *std::max_element(ends.begin(), ends.end());
    const auto begin = common - std::min(common, config.window);
    const auto finish = common + config.horizon;
    const auto span = finish - begin;
    const auto slotAt = [&](std::size_t column) {
        return begin + column * span / config.width;
    };
    output << "\nwindow=" << config.window << " horizon(aux+predict)=" << config.horizon
        << " common=" << common << " aux=" << longest - common
        << " predict=" << config.horizon - (longest - common) << '\n';
    output << "ends:";
    for (std::size_t channel = 0; channel < ends.size(); ++channel) {
        output << ' ' << channel + 1 << '=' << ends[channel];
    }
    output << '\n';
    for (std::size_t channel = 0; channel < ends.size(); ++channel) {
        output << channel + 1 << " |";
        for (std::size_t col = 0; col < config.width; ++col) {
            const auto slot = slotAt(col);
            output << (slot < ends[channel] ? (std::isfinite(observation(channel, slot)) ? '#' : '.') : '=');
        }
        output << " end=" << ends[channel] << '\n';
    }
    output << "- |";
    for (std::size_t col = 0; col < config.width; ++col) {
        const auto slot = slotAt(col);
        if (config.color) {
            output << (slot < common ? "\x1b[32m" : (slot < longest ? "\x1b[33m" : "\x1b[31m"));
        }
        output << (slot < common ? 'C' : (slot < longest ? 'A' : 'P'));
    }
    if (config.color) {
        output << "\x1b[0m";
    }
    output << "\n  | C=cached A=aux P=predict  #=observed .=missing ==not-arrived\n";
    const auto missing = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> curves(7, std::vector<double>(config.width, missing));
    double low = std::numeric_limits<double>::infinity();
    double high = -low;
    for (std::size_t col = 0; col < config.width; ++col) {
        const auto slot = slotAt(col);
        curves[0][col] = config.truth(0, static_cast<double>(slot) * config.interval);
        for (std::size_t c = 1; c < 3; ++c) {
            const auto& prediction = c == 1 ? nativeResult : fastResult;
            curves[c][col] = slot < common ? observation(0, slot)
                : (prediction.values.empty() ? missing : prediction.values[slot - common]);
            if (showBands && slot >= common && !prediction.p10.empty()) {
                curves[c * 2 + 1][col] = prediction.p10[slot - common];
                curves[c * 2 + 2][col] = prediction.p90[slot - common];
            }
        }
        for (const auto& curve : curves) {
            if (std::isfinite(curve[col])) {
                low = std::min(low, curve[col]);
                high = std::max(high, curve[col]);
            }
        }
    }
    const auto range = std::max(high - low, 1e-6);
    const auto rowAt = [&](double value) {
        return static_cast<std::size_t>(std::clamp(std::llround(
            (high - value) / range * static_cast<double>(config.height - 1)
        ), 0LL, static_cast<long long>(config.height - 1)));
    };
    std::vector<std::string> canvas(config.height, std::string(config.width, ' '));
    // cached区间画带噪输入，N与K在此完全相同，R不重合只是噪声与无噪声真值的差别
    for (std::size_t col = 0; col < config.width; ++col) {
        const auto slot = slotAt(col);
        if (slot < common) {
            const auto value = observation(0, slot);
            if (std::isfinite(value)) {
                canvas[rowAt(value)][col] = 'o';
            }
        }
    }
    if (showBands) {
        for (std::size_t c = 1; c < 3; ++c) {
            for (std::size_t col = 0; col < config.width; ++col) {
                const auto lower = curves[c * 2 + 1][col];
                const auto upper = curves[c * 2 + 2][col];
                if (!std::isfinite(lower) || !std::isfinite(upper)) {
                    continue;
                }
                const auto first = std::min(rowAt(lower), rowAt(upper));
                const auto last = std::max(rowAt(lower), rowAt(upper));
                for (auto row = first; row <= last; ++row) {
                    auto& cell = canvas[row][col];
                    cell = cell == ' ' ? (c == 1 ? '.' : ':') : '+';
                }
                canvas[first][col] = c == 1 ? 'n' : 'k';
                canvas[last][col] = c == 1 ? 'n' : 'k';
            }
        }
    }
    const std::string symbols = "RNK";
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t col = 0; col < config.width; ++col) {
            const auto slot = slotAt(col);
            if (c > 0 && slot < common) {
                continue;
            }
            if (std::isfinite(curves[c][col])) {
                auto& cell = canvas[rowAt(curves[c][col])][col];
                cell = cell == 'R' || cell == 'N' || cell == 'K' || cell == '*' || cell == 'o'
                    ? '*' : symbols[c];
            }
        }
    }
    for (const auto& row : canvas) {
        output << "  |";
        for (const char symbol : row) {
            if (config.color) {
                if (symbol == 'N') {
                    output << "\x1b[1;36m";
                } else if (symbol == 'K') {
                    output << "\x1b[1;35m";
                } else if (symbol == 'n' || symbol == '.') {
                    output << "\x1b[2;36m";
                } else if (symbol == 'k' || symbol == ':') {
                    output << "\x1b[2;35m";
                } else if (symbol == '*') {
                    output << "\x1b[1;33m";
                } else if (symbol == 'o') {
                    output << "\x1b[2;32m";
                } else if (symbol == '+') {
                    output << "\x1b[2;34m";
                } else {
                    output << "\x1b[0m";
                }
            }
            output << symbol;
        }
        if (config.color) {
            output << "\x1b[0m";
        }
        output << '\n';
    }
    output << std::fixed << std::setprecision(5)
        << "R=truth N=native p50 K=FIFO p50 *=overlap o=cached noisy input\n"
        << (showBands ? "n/.=native p10-p90 k/:=FIFO p10-p90 +=band overlap\n"
            : "p10-p90 bands hidden (press b to show)\n")
        << "cached: o is the shared noisy input of N/K; o vs R is observation noise (no prediction is cached)\n"
        << "N: " << nativeResult.status << " rmse="
        << (nativeResult.scored == 0 ? "n/a" : std::to_string(nativeResult.rmse))
        << " scored=" << nativeResult.scored << " ms=" << nativeResult.milliseconds << '\n'
        << "K: " << fastResult.status << " rmse="
        << (fastResult.scored == 0 ? "n/a" : std::to_string(fastResult.rmse))
        << " scored=" << fastResult.scored << " ms(includes append)=" << fastResult.milliseconds << '\n'
        << "pages retained=" << table.size() << " reused=" << hits << " appended=" << newPages
        << " fifo_evicted=" << droppedPages << " (C is causal-only; approximation in aux/after eviction)\n"
        << "axis=[" << begin << ',' << finish << ") low=" << low << " high=" << high << '\n';
    const auto printPercentiles = [&](const std::string& label, const StreamingPrediction& prediction) {
        if (!prediction.p10.empty()) {
            output << label << " p10/p50/p90 @first=" << prediction.p10.front()
                << '/' << prediction.values.front() << '/' << prediction.p90.front() << '\n';
        }
    };
    printPercentiles("N", nativeResult);
    printPercentiles("K", fastResult);
}
