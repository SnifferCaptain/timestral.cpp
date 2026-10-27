/***************
 * @file signal.cpp
 * @brief 噪声加多正弦信号上的离线预测验证：真实权重对照无噪声真值
 ***************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "timestral/timestral.hpp"

const double twoPi = 2.0 * std::acos(-1.0);

// 参数边界，context上限取模型contextLength，horizon上限按演示显示宽度固定
constexpr float minF0 = 1.0f / 2048.0f;
constexpr float maxF0 = 1.0f / 64.0f;
constexpr std::size_t minHarmonics = 1;
constexpr std::size_t maxHarmonics = 6;
constexpr float noiseStep = 0.025f;
constexpr std::size_t noiseMaxSteps = 20;
constexpr std::size_t minContext = 64;
constexpr std::size_t minHorizon = 32;
constexpr std::size_t maxHorizon = 512;
constexpr std::size_t displayContext = 192;
constexpr std::size_t plotWidth = 100;
constexpr std::size_t plotHeight = 16;
constexpr double sampleInterval = 0.01; // 秒，演示采样间隔

static const char* const legend = ". 上下文  o 预测中位数  : 80%区间  + 真值(无噪声)";

// @brief 一次演示的输入参数，长度字段在clampParams中按模型与演示上限钳制
struct SignalParams {
    float f0 = 1.0f / 128.0f;
    std::size_t harmonics = 3;
    float noise = 0.1f;
    std::size_t contextLength = 256;
    std::size_t horizonSteps = 128;
    std::uint64_t seed = 42;
};

// @brief 指标汇总，corr与ampRatio的分母为0时保持NaN
struct Metrics {
    double mae = 0.0;
    double rmse = 0.0;
    double corr = std::numeric_limits<double>::quiet_NaN();
    double ampRatio = std::numeric_limits<double>::quiet_NaN();
    double cov80 = 0.0;
    double cov90 = 0.0;
    double forecastMs = 0.0;
};

// 带噪上下文观测，以及未来horizonSteps步的确定性真值（无噪声）
struct GeneratedSignal {
    std::vector<float> context;
    std::vector<double> truth;
};

// @brief 按seed生成带噪上下文与无噪声真值，相位与Box-Muller噪声由同一个mt19937_64依次抽取
// 信号y(t)=Σ(1/k)sin(2πf0kt+φ_k)，seed变化时整个信号重新实现
// @return 前contextLength点带噪观测与后horizonSteps点无噪声真值
static GeneratedSignal generateSignal(const SignalParams& params) {
    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<double> phaseDist(0.0, twoPi);
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);
    std::vector<double> phase(params.harmonics + 1, 0.0);
    for (std::size_t k = 1; k <= params.harmonics; ++k) {
        phase[k] = phaseDist(rng);
    }
    auto yAt = [&](double t) {
        double y = 0.0;
        for (std::size_t k = 1; k <= params.harmonics; ++k) {
            y += std::sin(twoPi * params.f0 * static_cast<double>(k) * t + phase[k])
                / static_cast<double>(k);
        }
        return y;
    };
    GeneratedSignal signal;
    signal.context.resize(params.contextLength);
    signal.truth.resize(params.horizonSteps);
    double spare = 0.0;
    bool haveSpare = false;
    for (std::size_t t = 0; t < params.contextLength; ++t) {
        double z;
        if (haveSpare) {
            z = spare;
            haveSpare = false;
        } else {
            double u1 = uniformDist(rng);
            if (u1 <= 0.0) {
                u1 = std::numeric_limits<double>::denorm_min();
            }
            const double u2 = uniformDist(rng);
            const double radius = std::sqrt(-2.0 * std::log(u1));
            z = radius * std::cos(twoPi * u2);
            spare = radius * std::sin(twoPi * u2);
            haveSpare = true;
        }
        signal.context[t] = static_cast<float>(yAt(static_cast<double>(t)) + params.noise * z);
    }
    for (std::size_t step = 0; step < params.horizonSteps; ++step) {
        signal.truth[step] = yAt(static_cast<double>(params.contextLength + step));
    }
    return signal;
}

// @brief 将命令行参数钳制到演示允许范围
// @param contextLimit 模型支持的最长上下文，优先于演示默认上限
// @return 钳制后的参数副本，不修改入参
static SignalParams clampParams(const SignalParams& requested, std::size_t contextLimit) {
    SignalParams clamped = requested;
    clamped.f0 = std::min(std::max(requested.f0, minF0), maxF0);
    clamped.harmonics = std::min(
        std::max(requested.harmonics, minHarmonics),
        maxHarmonics
    );
    const int noiseSteps = static_cast<int>(
        std::lround(static_cast<double>(requested.noise) / noiseStep)
    );
    clamped.noise = static_cast<float>(
        std::min(std::max(noiseSteps, 0), static_cast<int>(noiseMaxSteps))) * noiseStep;
    clamped.contextLength = std::min(std::max(requested.contextLength, minContext), contextLimit);
    clamped.horizonSteps = std::min(std::max(requested.horizonSteps, minHorizon), maxHorizon);
    return clamped;
}

// @brief 指标全部用FP64累加，分母为0时corr/ampRatio记NaN
// @param truth 与预测逐步对齐的无噪声真值
// @param forecastMs 本次forecast耗时，原样写入指标
// @return 误差、相关、幅值比与区间覆盖率
static Metrics computeMetrics(
    const timestral::Timestral::Result& forecast,
    const std::vector<double>& truth,
    double forecastMs
) {
    const double stepCount = static_cast<double>(truth.size());
    double sumAbsolute = 0.0;
    double sumSquared = 0.0;
    double sumForecast = 0.0;
    double sumTruth = 0.0;
    double sumForecastSquared = 0.0;
    double sumTruthSquared = 0.0;
    double sumProduct = 0.0;
    for (std::size_t t = 0; t < truth.size(); ++t) {
        const double median = forecast.median(t);
        const double truthValue = truth[t];
        const double difference = median - truthValue;
        sumAbsolute += std::abs(difference);
        sumSquared += difference * difference;
        sumForecast += median;
        sumTruth += truthValue;
        sumForecastSquared += median * median;
        sumTruthSquared += truthValue * truthValue;
        sumProduct += median * truthValue;
    }
    Metrics metrics;
    metrics.mae = sumAbsolute / stepCount;
    metrics.rmse = std::sqrt(sumSquared / stepCount);
    const double meanForecast = sumForecast / stepCount;
    const double meanTruth = sumTruth / stepCount;
    const double varianceForecast = sumForecastSquared / stepCount - meanForecast * meanForecast;
    const double varianceTruth = sumTruthSquared / stepCount - meanTruth * meanTruth;
    const double covariance = sumProduct / stepCount - meanForecast * meanTruth;
    if (varianceForecast > 0.0 && varianceTruth > 0.0) {
        metrics.corr = covariance / std::sqrt(varianceForecast * varianceTruth);
    }
    if (varianceTruth > 0.0) {
        const double amplitude = varianceForecast > 0.0 ? varianceForecast : 0.0;
        metrics.ampRatio = std::sqrt(amplitude) / std::sqrt(varianceTruth);
    }
    auto coverage = [&](double level) {
        double insideCount = 0.0;
        for (std::size_t t = 0; t < truth.size(); ++t) {
            if (truth[t] >= forecast.lower(t, level) && truth[t] <= forecast.upper(t, level)) {
                insideCount += 1.0;
            }
        }
        return insideCount / stepCount;
    };
    metrics.cov80 = coverage(0.8);
    metrics.cov90 = coverage(0.9);
    metrics.forecastMs = forecastMs;
    return metrics;
}

// @brief 按相关、幅值与RMSE给出跟随结论，无噪声时改用绝对误差阈值
static std::string verdict(const Metrics& metrics, float noise) {
    const bool rmseOk = noise > 0.0f ? metrics.rmse <= 1.5 * noise : metrics.rmse <= 1e-3;
    if (metrics.corr >= 0.9 && metrics.ampRatio >= 0.7 && metrics.ampRatio <= 1.3 && rmseOk) {
        if (noise > 0.0f) {
            return "跟随信号良好：中位数延续周期结构，误差接近噪声水平";
        }
        return "跟随信号良好：中位数延续周期结构";
    }
    if (metrics.corr < 0.7 || metrics.ampRatio < 0.5) {
        return "未有效跟随：中位数未延续周期结构或幅值明显衰减";
    }
    return "部分跟随：周期结构部分延续，见指标";
}

// @brief 定点格式化，非有限值统一输出nan或-nan，不交给printf
static std::string formatFixed(double value, int decimals) {
    if (!std::isfinite(value)) {
        return value < 0.0 ? "-nan" : "nan";
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return buffer;
}

// @brief 纯文本绘图规格：横轴为按采样间隔计的时间索引，纵轴为读数
// 上下文、预测中位数、区间上下界与真值长度各自独立，truth为空表示不绘制真值
struct PlotSpec {
    std::size_t width = plotWidth;
    std::size_t height = plotHeight;
    double xMin = 0.0;
    double xMax = 0.0;
    float yMin = 0.0f;
    float yMax = 0.0f;
    std::vector<float> context;
    std::vector<float> median;
    std::vector<float> bandLo;
    std::vector<float> bandHi;
    std::vector<float> truth;
};

// @brief Y轴刻度统一三位小数，非有限值输出nan或-nan
static std::string formatTick(float value) {
    if (!std::isfinite(value)) {
        return value < 0.0f ? "-nan" : "nan";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3f", value);
    return std::string(buffer);
}

// @brief 符号优先级，同格绘制时高优先级覆盖低优先级
static int symbolPriority(char symbol) {
    switch (symbol) {
        case '+': return 4;
        case 'o': return 3;
        case '.': return 2;
        case ':': return 1;
        default: return 0;
    }
}

// @brief 返回纯文本绘图区：左侧Y轴刻度（yMax/中值/yMin，3位小数）、绘图区与底部X轴刻度
// 列=时间，行=值，上高下低，context为'.'，区间为':'，median为'o'，truth为'+'，
// 同格优先级truth>median>context>band，非有限值跳过。输出固定width列
static std::string plot(const PlotSpec& spec) {
    const std::size_t width = spec.width;
    const std::size_t height = spec.height;
    if (width == 0 || height == 0) {
        return std::string();
    }

    const std::string topLabel = formatTick(spec.yMax);
    const std::string midLabel = formatTick(0.5f * (spec.yMax + spec.yMin));
    const std::string bottomLabel = formatTick(spec.yMin);
    const std::size_t gutter = std::max(
        std::size_t(3),
        std::max(topLabel.size(), std::max(midLabel.size(), bottomLabel.size()))
    );

    const double span = spec.xMax - spec.xMin;
    const double columnSpan = width > 1 ? static_cast<double>(width - 1) : 1.0;
    const std::size_t contextCount = spec.context.size();
    const std::size_t horizonCount = spec.median.size();

    auto columnForTime = [&](double time) -> std::size_t {
        double column = span > 0.0 ? (time - spec.xMin) * columnSpan / span : 0.0;
        column = std::round(column);
        if (column < 0.0) {
            column = 0.0;
        }
        if (column > static_cast<double>(width - 1)) {
            column = static_cast<double>(width - 1);
        }
        return static_cast<std::size_t>(column);
    };
    auto rowForValue = [&](float value) -> long {
        if (!std::isfinite(value)) {
            return -1;
        }
        if (spec.yMax <= spec.yMin) {
            return static_cast<long>(height / 2);
        }
        const double ratio =
            (static_cast<double>(spec.yMax) - value) / (static_cast<double>(spec.yMax) - spec.yMin);
        long row = static_cast<long>(std::llround(ratio * static_cast<double>(height - 1)));
        if (row < 0) {
            row = 0;
        }
        if (row > static_cast<long>(height - 1)) {
            row = static_cast<long>(height - 1);
        }
        return row;
    };

    std::vector<char> grid(width * height, ' ');
    auto place = [&](std::size_t column, long row, char symbol) {
        if (row < 0) {
            return;
        }
        const std::size_t cellIndex = static_cast<std::size_t>(row) * width + column;
        if (symbolPriority(symbol) > symbolPriority(grid[cellIndex])) {
            grid[cellIndex] = symbol;
        }
    };

    // 区间按列时间取最近预测步的bandLo..bandHi，填充区间内的格
    if (horizonCount > 0
        && spec.bandLo.size() == horizonCount
        && spec.bandHi.size() == horizonCount) {
        const double forecastStart = spec.xMax - static_cast<double>(horizonCount);
        for (std::size_t column = 0; column < width; ++column) {
            const double time = span > 0.0
                ? spec.xMin + span * static_cast<double>(column) / columnSpan
                : spec.xMin;
            if (time < forecastStart || time > spec.xMax) {
                continue;
            }
            long step = static_cast<long>(std::llround(time - forecastStart));
            if (step < 0) {
                step = 0;
            }
            if (step > static_cast<long>(horizonCount - 1)) {
                step = static_cast<long>(horizonCount - 1);
            }
            const long top = rowForValue(spec.bandHi[step]);
            const long bottom = rowForValue(spec.bandLo[step]);
            if (top < 0 || bottom < 0) {
                continue;
            }
            for (long row = std::min(top, bottom); row <= std::max(top, bottom); ++row) {
                place(column, row, ':');
            }
        }
    }
    for (std::size_t i = 0; i < contextCount; ++i) {
        place(
            columnForTime(spec.xMin + static_cast<double>(i)),
            rowForValue(spec.context[i]),
            '.'
        );
    }
    for (std::size_t j = 0; j < horizonCount; ++j) {
        place(
            columnForTime(spec.xMax - static_cast<double>(horizonCount) + static_cast<double>(j)),
            rowForValue(spec.median[j]),
            'o'
        );
    }
    if (spec.truth.size() == horizonCount) {
        for (std::size_t j = 0; j < horizonCount; ++j) {
            place(
                columnForTime(spec.xMax - static_cast<double>(horizonCount) + static_cast<double>(j)),
                rowForValue(spec.truth[j]),
                '+'
            );
        }
    }

    std::string frame;
    frame.reserve((gutter + width + 1) * (height + 1));
    const std::size_t midRow = height / 2;
    for (std::size_t rowIndex = 0; rowIndex < height; ++rowIndex) {
        std::string label;
        if (rowIndex == 0) {
            label = topLabel;
        } else if (rowIndex == midRow) {
            label = midLabel;
        } else if (rowIndex + 1 == height) {
            label = bottomLabel;
        }
        frame.append(gutter - label.size(), ' ');
        frame += label;
        frame.append(grid.data() + rowIndex * width, width);
        frame += '\n';
    }

    // X轴刻度：xMin、1/4、1/2、3/4、xMax处的整数时间索引，以刻度列居中放置，
    // 自左向右不重叠，末端空间不足时左移整体，保证完整可见
    std::string axis(gutter + width, ' ');
    const std::size_t end = gutter + width;
    std::size_t nextFree = gutter;
    for (int k = 0; k < 5; ++k) {
        const double time = spec.xMin + span * static_cast<double>(k) / 4.0;
        const std::string text = std::to_string(static_cast<long long>(std::llround(time)));
        if (text.size() > end) {
            continue;
        }
        const std::size_t column = gutter + columnForTime(time);
        const std::size_t half = (text.size() - 1) / 2;
        std::size_t start = column >= half ? column - half : 0;
        if (start < nextFree) {
            start = nextFree;
        }
        if (start + text.size() > end) {
            start = end - text.size();
        }
        if (start < nextFree) {
            continue;
        }
        axis.replace(start, text.size(), text);
        nextFree = start + text.size() + 1;
    }
    frame += axis;
    frame += '\n';
    return frame;
}

// @brief 由显示数据确定y范围并加5%留白，全为单一值时上下各扩1
// @param yMin 输出下界，需由调用方持有
// @param yMax 输出上界，需由调用方持有
static void computeYRange(
    const std::vector<float>& context,
    const std::vector<float>& median,
    const std::vector<float>& bandLo,
    const std::vector<float>& bandHi,
    const std::vector<float>& truth,
    float& yMin,
    float& yMax
) {
    double minimum = 0.0;
    double maximum = 0.0;
    bool hasValue = false;
    auto addValue = [&](double value) {
        if (!std::isfinite(value)) {
            return;
        }
        if (!hasValue) {
            minimum = value;
            maximum = value;
            hasValue = true;
        } else {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    };
    for (float value : context) {
        addValue(value);
    }
    for (float value : median) {
        addValue(value);
    }
    for (float value : bandLo) {
        addValue(value);
    }
    for (float value : bandHi) {
        addValue(value);
    }
    for (float value : truth) {
        addValue(value);
    }
    if (!hasValue) {
        yMin = -1.0f;
        yMax = 1.0f;
        return;
    }
    if (maximum - minimum < 1e-6) {
        minimum -= 1.0;
        maximum += 1.0;
    } else {
        const double padding = 0.05 * (maximum - minimum);
        minimum -= padding;
        maximum += padding;
    }
    yMin = static_cast<float>(minimum);
    yMax = static_cast<float>(maximum);
}

// @brief 参数行含patch分辨率说明：周期低于2个patch的高频分量会混叠
static std::string paramsLine(const SignalParams& params, std::size_t patchSize) {
    return "f0=" + formatFixed(params.f0, 4)
        + " harmonics=" + std::to_string(params.harmonics)
        + " noise=" + formatFixed(params.noise, 3)
        + " context=" + std::to_string(params.contextLength)
        + " horizon=" + std::to_string(params.horizonSteps)
        + " seed=" + std::to_string(params.seed)
        + " | patch_size=" + std::to_string(patchSize)
        + "，时间分辨率以patch计，周期<2个patch的高频分量会混叠，基频周期建议至少4个patch";
}

// @brief 汇总指标为一行，供脚本按key=value读取
static std::string metricsLine(const Metrics& metrics) {
    return "MAE=" + formatFixed(metrics.mae, 3)
        + " RMSE=" + formatFixed(metrics.rmse, 3)
        + " corr=" + formatFixed(metrics.corr, 3)
        + " ampRatio=" + formatFixed(metrics.ampRatio, 3)
        + " cov80=" + formatFixed(metrics.cov80, 3)
        + " cov90=" + formatFixed(metrics.cov90, 3)
        + " forecast_ms=" + formatFixed(metrics.forecastMs, 1);
}

// @brief 生成带噪上下文与无噪声真值，调用一次forecast，打印参数、指标、判定与曲线
static int runDemo(const std::filesystem::path& modelDirectory, const SignalParams& requested) {
    const auto loadStart = std::chrono::steady_clock::now();
    timestral::Timestral model(modelDirectory);
    const auto loadFinish = std::chrono::steady_clock::now();
    const double loadMs = std::chrono::duration<double, std::milli>(loadFinish - loadStart).count();

    const std::size_t contextLimit = std::max(model.contextLength(), minContext);
    const SignalParams params = clampParams(requested, contextLimit);
    const GeneratedSignal signal = generateSignal(params);

    std::vector<timestral::Sample> history(signal.context.size());
    for (std::size_t t = 0; t < signal.context.size(); ++t) {
        history[t] = {static_cast<double>(t) * sampleInterval, signal.context[t]};
    }
    const double horizonSeconds = static_cast<double>(params.horizonSteps) * sampleInterval;
    const auto forecastStart = std::chrono::steady_clock::now();
    timestral::Timestral::Request request;
    request.variables = {history};
    request.horizonSeconds = horizonSeconds;
    const timestral::Timestral::Result forecast = model.predict(std::move(request));
    const auto forecastFinish = std::chrono::steady_clock::now();
    const double forecastMs =
        std::chrono::duration<double, std::milli>(forecastFinish - forecastStart).count();

    const Metrics metrics = computeMetrics(forecast, signal.truth, forecastMs);

    const std::size_t windowCount = std::min(params.contextLength, displayContext);
    const std::size_t windowStart = params.contextLength - windowCount;
    std::vector<float> window(windowCount);
    for (std::size_t i = 0; i < windowCount; ++i) {
        window[i] = static_cast<float>(history[windowStart + i].value);
    }
    std::vector<float> median(forecast.steps());
    std::vector<float> bandLo(forecast.steps());
    std::vector<float> bandHi(forecast.steps());
    for (std::size_t t = 0; t < forecast.steps(); ++t) {
        median[t] = static_cast<float>(forecast.median(t));
        bandLo[t] = static_cast<float>(forecast.lower(t, 0.8));
        bandHi[t] = static_cast<float>(forecast.upper(t, 0.8));
    }
    std::vector<float> truth(signal.truth.size());
    for (std::size_t t = 0; t < signal.truth.size(); ++t) {
        truth[t] = static_cast<float>(signal.truth[t]);
    }

    PlotSpec spec;
    spec.xMin = static_cast<double>(windowStart);
    spec.xMax = static_cast<double>(params.contextLength + params.horizonSteps);
    computeYRange(window, median, bandLo, bandHi, truth, spec.yMin, spec.yMax);
    const std::size_t gutter = std::max(
        std::size_t(3),
        std::max(
            formatTick(spec.yMax).size(),
            std::max(formatTick(0.5f * (spec.yMax + spec.yMin)).size(), formatTick(spec.yMin).size())
        )
    );
    spec.width = plotWidth > gutter + 1 ? plotWidth - gutter : 1;
    spec.context = std::move(window);
    spec.median = std::move(median);
    spec.bandLo = std::move(bandLo);
    spec.bandHi = std::move(bandHi);
    spec.truth = std::move(truth);

    std::cout << "model=" << modelDirectory.string()
        << " load_ms=" << formatFixed(loadMs, 1) << '\n';
    std::cout << paramsLine(params, model.patchSize()) << '\n';
    std::cout << metricsLine(metrics) << '\n';
    std::cout << "判定: " << verdict(metrics, params.noise) << '\n';
    std::cout << plot(spec);
    std::cout << legend << '\n';
    std::cout.flush();
    return 0;
}

// @brief 命令行解析结果，参数尚未按模型与演示上限钳制
struct Args {
    std::filesystem::path modelDirectory;
    SignalParams params;
};

// @brief 只接受十进制数字，拒绝符号、空白和溢出
static std::uint64_t parseUint64(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("invalid unsigned integer " + text);
    }
    try {
        return std::stoull(text);
    } catch (const std::exception&) {
        throw std::invalid_argument("integer out of range " + text);
    }
}

// @brief 在size_t范围内解析无符号参数
static std::size_t parseSize(const std::string& text) {
    const std::uint64_t value = parseUint64(text);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("integer out of range " + text);
    }
    return static_cast<std::size_t>(value);
}

// @brief 解析有限浮点数，末尾残留字符按非法参数拒绝
static float parseFloat(const std::string& text) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) {
            throw std::invalid_argument("invalid number " + text);
        }
        return static_cast<float>(value);
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid number " + text);
    }
}

// @brief 参数解析，缺失或非法参数抛invalid_argument，由main打印usage并退出
static Args parseArgs(int argc, char** argv) {
    Args args;
    if (argc < 2) {
        throw std::invalid_argument("missing MODEL_DIRECTORY");
    }
    args.modelDirectory = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        auto nextValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            ++i;
            return argv[i];
        };
        if (option == "--seed") {
            args.params.seed = parseUint64(nextValue("--seed"));
        } else if (option == "--context") {
            args.params.contextLength = parseSize(nextValue("--context"));
        } else if (option == "--horizon") {
            args.params.horizonSteps = parseSize(nextValue("--horizon"));
        } else if (option == "--f0") {
            args.params.f0 = parseFloat(nextValue("--f0"));
        } else if (option == "--harmonics") {
            args.params.harmonics = parseSize(nextValue("--harmonics"));
        } else if (option == "--noise") {
            args.params.noise = parseFloat(nextValue("--noise"));
        } else {
            throw std::invalid_argument("unknown option " + option);
        }
    }
    return args;
}

// @brief 解析参数并运行一次离线验证，参数错误返回2，运行错误返回1
int main(int argc, char** argv) {
    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::cerr << "usage: timestralSignal MODEL_DIRECTORY [--seed N] [--context N]"
            " [--horizon N] [--f0 F] [--harmonics K] [--noise S]\n";
        return 2;
    }
    try {
        return runDemo(args.modelDirectory, args.params);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
