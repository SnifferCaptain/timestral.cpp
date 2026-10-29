/***************
 * @file model.cpp
 * @brief ytensor可选后端单块前向、分页增量前向与因果归一化，语义与reference/model.cpp逐段对应，
 * 矩阵乘法与线性投影使用ytensor
 * 移植自Mistral AI Timestral layers.py/model.py/masks.py/scaler.py，
 * Apache-2.0，固定基线a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
 ***************/

#include "timestral/ytensor/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

// 该单头在模板解析阶段产生的C4702不在MSVC /external:W0的抑制范围内，只在此处最小屏蔽，
// 本仓库其它编译单元仍按/W4检查
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4702)
// MSVC的/arch:AVX2已启用FMA指令但不定义__FMA__，且不支持YBLAS微内核使用的
// GNU __attribute__语法（仅作内联提示，无语义影响），在此补齐让ytensor选中AVX2路径，包含后移除
#if defined(__AVX2__) && !defined(__FMA__)
#define __FMA__
#define TIMESTRAL_MSVC_FMA_SHIM
#endif
#define __attribute__(x)
#endif

#include "ytensor.hpp"

#if defined(_MSC_VER)
#pragma warning(pop)
#undef __attribute__
#ifdef TIMESTRAL_MSVC_FMA_SHIM
#undef __FMA__
#undef TIMESTRAL_MSVC_FMA_SHIM
#endif
#endif

#include "timestral/reference/kernels.hpp"
#include "timestral/utils/patch.hpp"

// 矩阵乘法统一使用ytensor的strided matmul：非拥有strided视图包装本后端FP32存储，
// 地址合同与yt::blas::sgemm的元素地址p*rsb+j*csb一致，步幅映射逐项为：
// linear的B=W^T取base=W.data、rsb=1、csb=inDim，A=input行主序、C=(row,outDim)紧凑，
// 注意力头内行距time轴为modelDim、variate轴为slots*modelDim，QK^T的B=kHead^T取(1,行距)，
// (softmax)V的B=vHead取(行距,1)，C沿用头内行距写回attended

namespace timestral::ytensor {

// 默认跟随ytensor自身的defaultMatmulBackend，本仓库不定义YT_USE_YBLAS、YT_USE_EIGEN
// 等特性宏，后端由ytensor按编译环境选择；MSVC下经/arch:AVX2与上方兼容补齐选中YBLAS，
// 未启用AVX2的环境仍回退Eigen strided或Naive
constexpr yt::info::MatmulBackend matmulBackend = yt::info::defaultMatmulBackend;

// sdpa后端按编译期选择：AVX2+FMA已启用且YBLAS在列时走FA2融合内核（掩码与softmax在核内完成），
// 无AVX2的环境回退MATH分解路径（掩码matmul加softmax加matmul），数值语义一致
#if defined(__AVX2__) && defined(__FMA__) && YT_USE_YBLAS
static constexpr yt::function::sdpaBackend sdpaBackend = yt::function::sdpaBackend::FLASH_AVX2;
#else
static constexpr yt::function::sdpaBackend sdpaBackend = yt::function::sdpaBackend::MATH;
#endif

// YBLAS矩阵乘与FA2核共用进程级线程数（yt::blas::g_num_threads），
// 语义与Eigen后端一致：5M的小GEMM在逻辑核全开时会被OpenMP调度开销拖慢，
// 0取保守自动值（上限4），1强制串行，N为精确线程数；未链接OpenMP时并行区不生效，保持串行
void setThreads(std::size_t threads) {
    if (threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("ytensor thread count out of range");
    }
    constexpr int defaultThreadCap = 4;
    yt::blas::set_num_threads(
        threads != 0 ? static_cast<int>(threads)
                     : std::min(defaultThreadCap, yt::blas::default_num_threads())
    );
}

static std::vector<bool> attentionAxes(const std::string& kind) {
    if (kind == "time") {
        return {true};
    }
    if (kind == "variate") {
        return {false};
    }
    if (kind == "time_variate") {
        return {true, false};
    }
    if (kind == "variate_time") {
        return {false, true};
    }
    throw std::invalid_argument("unknown block_schedule kind: " + kind);
}

// reference/model.cpp的文件内实现不可见，按reference实现复制
static void addInPlace(Tensor& target, const Tensor& value) {
    if (target.shape != value.shape) {
        throw std::invalid_argument("residual: shape mismatch");
    }
    for (std::size_t i = 0; i < target.size(); ++i) {
        target.values[i] += value.values[i];
    }
}

static void validateRows(
    const Tensor& input,
    const std::vector<std::uint8_t>& observed,
    std::size_t patchSize
) {
    input.validate();
    if (input.shape.size() != 2 || input.shape[0] == 0 || input.shape[1] == 0
        || input.shape[1] % patchSize != 0 || observed.size() != input.size()) {
        throw std::invalid_argument("model: expected nonempty (R,T), patch-aligned T and matching observation mask");
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (observed[i] > 1 || (observed[i] != 0 && !std::isfinite(input.values[i]))) {
            throw std::invalid_argument("model: invalid observation mask or nonfinite observed value");
        }
    }
}

// 非拥有strided视图：包装外部FP32存储供ytensor strided matmul读写，offset以元素计，
// deleter为空操作，存储生命周期由调用方张量负责
static yt::YTensorBase stridedView(
    std::vector<float>& data,
    std::size_t offset,
    const std::vector<int>& shape,
    const std::vector<int>& stride
) {
    yt::YTensorBase view;
    yt::YMemory memory(
        reinterpret_cast<char*>(data.data() + offset),
        data.size() * sizeof(float),
        "cpu",
        [](char*) {}
    );
    yt::strided::BaseViewAccess::setView(view, shape, stride, 0, memory, sizeof(float), "float32");
    return view;
}

// 只读版本：ytensor的内存句柄只接受char*，这里只构造被matmul读取的视图
static yt::YTensorBase stridedView(
    const std::vector<float>& data,
    std::size_t offset,
    const std::vector<int>& shape,
    const std::vector<int>& stride
) {
    return stridedView(const_cast<std::vector<float>&>(data), offset, shape, stride);
}

Model::Model(ModelConfig loadedConfig, utils::Safetensors weights)
    : modelConfig(std::move(loadedConfig)),
      weights(std::move(weights)) {
    modelConfig.validate();
    const auto& config = modelConfig;
    const std::size_t modelDim = config.modelDim;
    std::set<std::string> expected;
    const auto require = [&](const std::string& name, const std::vector<std::size_t>& shape) {
        expected.insert(name);
        const auto& weight = this->weights.at(name);
        if (weight.shape != shape || weight.values.size() != elementCount(shape)) {
            throw std::invalid_argument("weight shape mismatch: " + name);
        }
        for (float value : weight.values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("nonfinite weight: " + name);
            }
        }
    };
    const auto projection = [&](const std::string& name, std::size_t in, std::size_t out, bool bias) {
        require(name + ".weight", {out, in});
        if (bias) {
            require(name + ".bias", {out});
        }
    };
    const auto block = [&](const std::string& name, std::size_t in, std::size_t hidden, std::size_t out, bool bias) {
        projection(name + ".hidden_layer", in, hidden, bias);
        projection(name + ".output_layer", hidden, out, bias);
        projection(name + ".residual_layer", in, out, bias);
    };
    block(
        "representation.patch_proj",
        elementCount({2, config.patchSize}),
        elementCount({4, modelDim}),
        modelDim,
        true
    );
    require("representation.forecast_query", {modelDim});
    block(
        "head.projection",
        modelDim,
        elementCount({4, modelDim}),
        elementCount({config.patchSize, config.quantileKnots.size()}),
        false
    );
    require("transformer.out_norm.scale", {modelDim});
    for (std::size_t layer = 0; layer < config.blockSchedule.size(); ++layer) {
        const std::string prefix = "transformer.layers." + std::to_string(layer);
        const auto axes = attentionAxes(config.blockSchedule[layer]);
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            const std::string index = std::to_string(axis);
            require(prefix + ".attn_norms." + index + ".scale", {modelDim});
            const std::string attn = prefix + ".attns." + index;
            projection(attn + ".wQKV", modelDim, elementCount({3, modelDim}), config.attnBias);
            projection(attn + ".wO", modelDim, modelDim, config.attnBias);
            if ((axes[axis] ? config.timeAttnGate : config.variateAttnGate) == "sdpa_sigmoid") {
                projection(attn + "._gate", modelDim, modelDim, config.attnBias);
            }
        }
        require(prefix + ".mlp_norm.scale", {modelDim});
        projection(prefix + ".mlp.w12", modelDim, elementCount({2, config.mlpHiddenDim}), config.mlpBias);
        projection(prefix + ".mlp.w3", config.mlpHiddenDim, modelDim, config.mlpBias);
    }
    for (const auto& entry : this->weights.tensors()) {
        if (expected.count(entry.first) == 0) {
            throw std::invalid_argument("unexpected weight: " + entry.first);
        }
    }
}

const ModelConfig& Model::config() const {
    return modelConfig;
}

std::size_t Model::threadCount() const {
    return static_cast<std::size_t>(std::max(1, yt::blas::get_num_threads()));
}

// 权重布局(out,in)，输出每行与reference::linear相同：先累加点积再加偏置
Tensor Model::project(const Tensor& input, const std::string& name, bool bias) const {
    const auto& weight = weights.at(name + ".weight");
    input.validate();
    if (input.shape.empty() || weight.shape.size() != 2
        || input.shape.back() != weight.shape[1]
        || weight.values.size() != elementCount(weight.shape)) {
        throw std::invalid_argument("ytensor linear: incompatible input or weight shape");
    }
    const std::size_t inDim = weight.shape[1];
    const std::size_t outDim = weight.shape[0];
    if (inDim == 0) {
        throw std::invalid_argument("ytensor linear: invalid input width or bias");
    }
    if (bias) {
        const auto& biasWeight = weights.at(name + ".bias");
        if (biasWeight.shape != std::vector<std::size_t>{outDim}
            || biasWeight.values.size() != outDim) {
            throw std::invalid_argument("ytensor linear: invalid input width or bias");
        }
    }
    const std::size_t row = input.size() / inDim;
    auto shape = input.shape;
    shape.back() = outDim;
    // 当前YBLAS微核为6行，统一整行块以避免GEMV及尾块改变K分块累加顺序
    constexpr std::size_t rowBlock = 6;
    if (row > static_cast<std::size_t>(std::numeric_limits<int>::max()) - rowBlock + 1
        || inDim > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || outDim > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("ytensor linear: dimensions exceed the backend integer range");
    }
    const std::size_t paddedRows = (row + rowBlock - 1) / rowBlock * rowBlock;
    Tensor padded;
    if (row != paddedRows) {
        padded = Tensor({paddedRows, inDim});
        std::copy(input.values.begin(), input.values.end(), padded.values.begin());
    }
    Tensor output(shape);
    if (row != 0 && outDim != 0) {
        const int rowInt = static_cast<int>(paddedRows);
        const int inDimInt = static_cast<int>(inDim);
        const int outDimInt = static_cast<int>(outDim);
        const auto& values = padded.values.empty() ? input.values : padded.values;
        const auto left = stridedView(values, 0, {rowInt, inDimInt}, {inDimInt, 1});
        // W布局(out,in)：B=W^T按行步幅1、列步幅inDim表达，无需显式转置
        const auto right = stridedView(weight.values, 0, {inDimInt, outDimInt}, {1, inDimInt});
        const auto result = yt::strided::matmul(left, right, matmulBackend);
        std::copy(result.data<float>(), result.data<float>() + row * outDim, output.values.begin());
        if (bias) {
            const auto& biasWeight = weights.at(name + ".bias");
            for (std::size_t i = 0; i < row; ++i) {
                for (std::size_t j = 0; j < outDim; ++j) {
                    output.values[i * outDim + j] += biasWeight.values[j];
                }
            }
        }
    }
    return output;
}

Tensor Model::residual(const Tensor& input, const std::string& name, bool bias) const {
    Tensor hidden = project(input, name + ".hidden_layer", bias);
    for (float& value : hidden.values) {
        value = reference::silu(value);
    }
    Tensor result = project(hidden, name + ".output_layer", bias);
    addInPlace(result, project(input, name + ".residual_layer", bias));
    return result;
}

ScaleResult Model::scale(const Tensor& context, const std::vector<std::uint8_t>& observed) const {
    const auto& config = modelConfig;
    validateRows(context, observed, config.patchSize);
    ScaleResult result{Tensor(context.shape), Tensor(context.shape), Tensor(context.shape)};
    const std::size_t time = context.shape[1];
    for (std::size_t row = 0; row < context.shape[0]; ++row) {
        double count = 0.0;
        double sum = 0.0;
        double mean = 0.0;
        double m2 = 0.0;
        for (std::size_t end = config.patchSize; end <= time; end += config.patchSize) {
            for (std::size_t t = end - config.patchSize; t < end; ++t) {
                const std::size_t index = row * time + t;
                if (observed[index] != 0) {
                    const double value = context.values[index];
                    const double previous = mean;
                    count += 1.0;
                    sum += value;
                    mean = sum / count;
                    m2 += (value - previous) * (value - mean);
                }
            }
            const float loc = static_cast<float>(mean);
            const float scale = static_cast<float>(std::max(
                std::sqrt(m2 / std::max(count - 1.0, 1.0)),
                static_cast<double>(config.minimumScale)
            ));
            for (std::size_t t = end - config.patchSize; t < end; ++t) {
                const std::size_t index = row * time + t;
                result.loc.values[index] = loc;
                result.scale.values[index] = scale;
                float value = observed[index] != 0
                    ? (context.values[index] - loc) / std::max(scale, std::numeric_limits<float>::epsilon()) : 0.0f;
                if (config.useAsinh) {
                    value = std::asinh(value);
                }
                result.scaled.values[index] = value;
            }
        }
    }
    return result;
}

Tensor Model::forward(const ForwardInput& input) const {
    ModelTrace scratch;
    return forwardQuantiles(input, false, false, scratch);
}

Tensor Model::forward(const ForwardInput& input, ModelTrace& trace) const {
    return forwardQuantiles(input, false, true, trace);
}

Tensor Model::predictBlock(const ForwardInput& input) const {
    ModelTrace scratch;
    return forwardQuantiles(input, true, false, scratch);
}

Tensor Model::headOutput(
    const Tensor& hidden,
    std::size_t patchCount,
    const std::vector<float>& loc,
    const std::vector<float>& scale,
    bool denormalize
) const {
    const auto& config = modelConfig;
    const std::size_t rowCount = hidden.shape[0];
    const std::size_t horizon = elementCount({patchCount, config.patchSize});
    const std::size_t quantiles = config.quantileKnots.size();
    Tensor raw = residual(hidden, "head.projection", false);
    raw.shape = {rowCount, horizon, quantiles};
    Tensor output = reference::monotoneQuantiles(raw, config.quantileKnots, config.quantileWidthFactor);
    if (!denormalize) {
        return output;
    }
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t t = 0; t < horizon; ++t) {
            const std::size_t base = (row * horizon + t) * quantiles;
            float minimum = std::numeric_limits<float>::infinity();
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < quantiles; ++j) {
                float value = output.values[base + j];
                if (config.useAsinh) {
                    value = std::sinh(value);
                }
                value = value * scale[row] + loc[row];
                output.values[base + j] = value;
                if (std::isfinite(value)) {
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
            }
            for (std::size_t j = 0; j < quantiles; ++j) {
                float& value = output.values[base + j];
                if (std::isinf(value)) {
                    value = value > 0.0f ? maximum : minimum;
                    if (std::isinf(value)) {
                        value = 0.0f;
                    }
                }
            }
        }
    }
    return output;
}

PageGeometry Model::pageGeometry(std::size_t rowCount) const {
    PageGeometry geometry;
    geometry.rowCount = rowCount;
    geometry.headCount = modelConfig.headCount;
    geometry.headDim = modelConfig.headDim();
    geometry.timeLayers = timeLayerCount(modelConfig);
    return geometry;
}

std::uint64_t Model::pageFingerprint() const {
    return modelFingerprint(modelConfig);
}

bool Model::supportsIncremental() const {
    return true;
}

void Model::appendPages(
    const CacheAppendInput& input,
    PagePool& pool,
    std::vector<std::size_t>& table
) const {
    const auto& config = modelConfig;
    if (!std::isfinite(input.positionOffset) || input.positionOffset < 0.0) {
        throw std::invalid_argument("invalid rolling position offset");
    }
    validateRows(input.context, input.observed, config.patchSize);
    const std::size_t rowCount = input.context.shape[0];
    const std::size_t time = input.context.shape[1];
    const std::size_t patchCount = time / config.patchSize;
    if (input.groupIds.size() != rowCount) {
        throw std::invalid_argument("model: appendPages requires one group id per row");
    }
    if (table.size() > patchCount) {
        throw std::invalid_argument("model: page table exceeds context patches");
    }
    verifyPoolGeometry(pool, config, rowCount);
    std::map<std::size_t, std::size_t> first = firstObservedPatches(
        input.observed,
        input.groupIds,
        rowCount,
        time,
        config.patchSize
    );
    std::vector<std::uint8_t> keyValid;
    buildKeyValid(input.observed, rowCount, time, config.patchSize, config.attendMissingPatches, keyValid);
    if (input.positionOffset > 0.0) {
        for (auto& entry : first) {
            entry.second = 0;
        }
        for (std::size_t row = 0; row < rowCount; ++row) {
            if (config.attendMissingPatches && !table.empty()
                && pool.page(table.front()).scaleStates[row].count > 0.0) {
                std::fill_n(keyValid.begin() + row * patchCount, patchCount, std::uint8_t{1});
            }
        }
    }
    verifyTableGrid(pool, table, rowCount, keyValid, input.groupIds, first, input.coordinates, input.positionOffset);
    std::vector<PageScaleState> running(rowCount, PageScaleState{});
    if (!table.empty()) {
        running = pool.page(table.back()).scaleStates;
    }
    const std::vector<int> targetRoles(rowCount, 0);
    const std::size_t modelDim = config.modelDim;
    ModelTrace scratch;
    std::vector<float> scaled;
    std::vector<float> loc;
    std::vector<float> scaleValues;
    const std::size_t pendingCount = patchCount - table.size();
    // 先行按父键链生成各待补页的页键，用peek判定命中，定位尾部连续未命中段
    std::vector<std::uint64_t> patchParentKeys(pendingCount);
    std::vector<std::uint64_t> patchKeys(pendingCount);
    std::vector<std::uint64_t> patchFingerprints(pendingCount);
    std::vector<std::size_t> patchFound(pendingCount);
    std::vector<std::vector<std::uint8_t>> patchMaskBits(pendingCount);
    std::vector<std::vector<double>> patchPositions(pendingCount);
    std::size_t runStart = 0;
    {
        std::uint64_t parentKey = table.empty() ? 0 : pool.page(table.back()).key;
        for (std::size_t i = 0; i < pendingCount; ++i) {
            const std::size_t patch = table.size() + i;
            patchMaskBits[i].assign(rowCount, std::uint8_t{0});
            for (std::size_t row = 0; row < rowCount; ++row) {
                patchMaskBits[i][row] = keyValid[row * patchCount + patch];
            }
            buildPositions(rowCount, 1, patch, patchCount, false, input.groupIds, first, patchPositions[i], input.coordinates);
            for (double& position : patchPositions[i]) {
                position += input.positionOffset;
            }
            pageKeys(
                input.context,
                input.observed,
                rowCount,
                time,
                patch,
                config.patchSize,
                patchMaskBits[i],
                patchPositions[i],
                parentKey,
                pool.fingerprint(),
                patchKeys[i],
                patchFingerprints[i]
            );
            patchParentKeys[i] = parentKey;
            patchFound[i] = pool.peek(patchKeys[i], parentKey, patchFingerprints[i]);
            if (patchFound[i] != pool.capacity()) {
                runStart = i + 1;
            }
            parentKey = patchKeys[i];
        }
    }
    for (std::size_t i = 0; i < runStart; ++i) {
        const std::size_t patch = table.size();
        std::vector<PageScaleState> states;
        normalizePatches(
            input.context,
            input.observed,
            running,
            patch,
            patch + 1,
            config,
            scaled,
            loc,
            scaleValues,
            states
        );
        const std::vector<PageScaleState> pageStates(
            states.begin(),
            states.begin() + static_cast<std::ptrdiff_t>(rowCount)
        );
        const std::size_t found = pool.find(patchKeys[i], patchParentKeys[i], patchFingerprints[i]);
        if (found != pool.capacity()) {
            const std::vector<PageScaleState>& stored = pool.page(found).scaleStates;
            for (std::size_t row = 0; row < rowCount; ++row) {
                if (stored[row].count != pageStates[row].count || stored[row].sum != pageStates[row].sum
                    || stored[row].m2 != pageStates[row].m2) {
                    throw std::runtime_error("model: cached page normalization state does not match prefix");
                }
            }
            table.push_back(found);
            running = stored;
            continue;
        }
        const std::size_t index = pool.acquire();
        if (index == pool.capacity()) {
            throw std::runtime_error("model: page pool is full; capacity must cover the cached prefix");
        }
        CachePage& entry = pool.page(index);
        entry.parentKey = patchParentKeys[i];
        entry.key = patchKeys[i];
        entry.inputFingerprint = patchFingerprints[i];
        entry.maskBits = patchMaskBits[i];
        entry.positions = patchPositions[i];
        entry.scaleStates = pageStates;
        std::vector<float> patchValues;
        assemblePatchInput(scaled, input.observed, rowCount, time, patch, 1, config.patchSize, patchValues);
        Tensor patchInput({rowCount, 1, elementCount({2, config.patchSize})}, std::move(patchValues));
        Tensor encoded = residual(patchInput, "representation.patch_proj", true);
        Tensor x({rowCount, 1, modelDim});
        x.values = encoded.values;
        const std::vector<std::size_t> newPages(1, index);
        runLayers(
            x,
            table.size(),
            patchCount,
            patchCount,
            patchPositions[i],
            keyValid,
            false,
            targetRoles,
            input.groupIds,
            pool,
            table,
            newPages,
            false,
            false,
            scratch
        );
        table.push_back(index);
        running = pageStates;
    }
    if (runStart < pendingCount) {
        // 尾部连续未命中段一次性归一化、投影与写页，等价于整段prefill
        const std::size_t runBegin = table.size();
        const std::size_t runLength = pendingCount - runStart;
        std::vector<PageScaleState> states;
        normalizePatches(
            input.context,
            input.observed,
            running,
            runBegin,
            patchCount,
            config,
            scaled,
            loc,
            scaleValues,
            states
        );
        std::vector<double> runPositions;
        buildPositions(rowCount, runLength, runBegin, patchCount, false, input.groupIds, first, runPositions, input.coordinates);
        for (double& position : runPositions) {
            position += input.positionOffset;
        }
        std::vector<std::size_t> newPages(runLength);
        for (std::size_t j = 0; j < runLength; ++j) {
            const std::size_t i = runStart + j;
            const std::size_t index = pool.acquire();
            if (index == pool.capacity()) {
                throw std::runtime_error("model: page pool is full; capacity must cover the cached prefix");
            }
            newPages[j] = index;
            CachePage& entry = pool.page(index);
            entry.parentKey = patchParentKeys[i];
            entry.key = patchKeys[i];
            entry.inputFingerprint = patchFingerprints[i];
            entry.maskBits = patchMaskBits[i];
            entry.positions = patchPositions[i];
            std::vector<PageScaleState> pageStates(rowCount);
            for (std::size_t row = 0; row < rowCount; ++row) {
                pageStates[row] = states[row * runLength + j];
            }
            entry.scaleStates = pageStates;
        }
        std::vector<float> patchValues;
        assemblePatchInput(scaled, input.observed, rowCount, time, runBegin, runLength, config.patchSize, patchValues);
        Tensor patchInput({rowCount, runLength, elementCount({2, config.patchSize})}, std::move(patchValues));
        Tensor encoded = residual(patchInput, "representation.patch_proj", true);
        Tensor x({rowCount, runLength, modelDim});
        x.values = encoded.values;
        runLayers(
            x,
            runBegin,
            patchCount,
            patchCount,
            runPositions,
            keyValid,
            false,
            targetRoles,
            input.groupIds,
            pool,
            table,
            newPages,
            false,
            false,
            scratch
        );
        for (std::size_t j = 0; j < runLength; ++j) {
            table.push_back(newPages[j]);
        }
        for (std::size_t row = 0; row < rowCount; ++row) {
            running[row] = states[row * runLength + runLength - 1];
        }
    }

}

Tensor Model::decodePages(
    const CacheAppendInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table
) const {
    return decodePagesImpl(input, pool, table, false);
}

Tensor Model::decodePagesShared(
    const CacheAppendInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table
) const {
    return decodePagesImpl(input, pool, table, true);
}

Tensor Model::decodePagesImpl(
    const CacheAppendInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    bool shared
) const {
    const auto& config = modelConfig;
    if (!std::isfinite(input.positionOffset) || input.positionOffset < 0.0) {
        throw std::invalid_argument("invalid rolling position offset");
    }
    validateRows(input.context, input.observed, config.patchSize);
    const std::size_t rowCount = input.context.shape[0];
    const std::size_t time = input.context.shape[1];
    const std::size_t patchCount = time / config.patchSize;
    const std::size_t patches = input.horizonPatches;
    if (input.groupIds.size() != rowCount) {
        throw std::invalid_argument("model: decodePages requires one group id per row");
    }
    if (patches == 0) {
        throw std::invalid_argument("model: decodePages requires a positive horizon");
    }
    if (table.size() > patchCount) {
        throw std::invalid_argument("model: page table exceeds context patches");
    }
    if (shared && input.positionOffset != 0.0) {
        throw std::invalid_argument("rolling offsets cannot use shared request pages");
    }
    if (shared && table.empty()) {
        throw std::invalid_argument("model: decodePagesShared requires a nonempty page table");
    }
    // 共享页保存一组通道，批内请求按相同通道顺序读取
    verifyPoolGeometry(pool, config, shared ? pool.geometry().rowCount : rowCount);
    std::map<std::size_t, std::size_t> first = firstObservedPatches(
        input.observed,
        input.groupIds,
        rowCount,
        time,
        config.patchSize
    );
    std::vector<std::uint8_t> keyValid;
    buildKeyValid(input.observed, rowCount, time, config.patchSize, config.attendMissingPatches, keyValid);
    if (shared) {
        verifySharedTableGrid(pool, table, rowCount, keyValid, input.groupIds, first, input.coordinates);
        // 页内K/V只由观测到的前缀值与掩码决定，未观测点不参与归一化
        const std::size_t prefixLength = table.size() * config.patchSize;
        for (std::size_t row = pool.geometry().rowCount; row < rowCount; ++row) {
            const std::size_t sourceRow = row % pool.geometry().rowCount;
            for (std::size_t t = 0; t < prefixLength; ++t) {
                const std::size_t index = row * time + t;
                if (input.observed[index] != input.observed[sourceRow * time + t]) {
                    throw std::invalid_argument("model: shared prefix rows must be identical");
                }
                if (input.observed[index] != 0
                    && input.context.values[index] != input.context.values[sourceRow * time + t]) {
                    throw std::invalid_argument("model: shared prefix rows must be identical");
                }
            }
        }
    } else {
        verifyTableGrid(pool, table, rowCount, keyValid, input.groupIds, first, input.coordinates, input.positionOffset);
    }
    const std::size_t base = table.size();
    std::vector<PageScaleState> running(rowCount, PageScaleState{});
    if (base > 0) {
        if (shared) {
            // 每条请求从对应通道的累计量继续归一化
            for (std::size_t row = 0; row < rowCount; ++row) {
                running[row] = pool.page(table.back()).scaleStates[row % pool.geometry().rowCount];
            }
        } else {
            running = pool.page(table.back()).scaleStates;
        }
    }
    const std::size_t freshPatches = patchCount - base;
    std::vector<float> scaled;
    std::vector<float> loc;
    std::vector<float> scaleValues;
    std::vector<PageScaleState> states;
    if (freshPatches > 0) {
        normalizePatches(
            input.context,
            input.observed,
            running,
            base,
            patchCount,
            config,
            scaled,
            loc,
            scaleValues,
            states
        );
    }
    const std::size_t modelDim = config.modelDim;
    const std::size_t count = freshPatches + patches;
    const auto& query = weights.at("representation.forecast_query").values;
    Tensor x({rowCount, count, modelDim});
    if (freshPatches > 0) {
        std::vector<float> patchValues;
        assemblePatchInput(scaled, input.observed, rowCount, time, base, freshPatches, config.patchSize, patchValues);
        Tensor patchInput({rowCount, freshPatches, elementCount({2, config.patchSize})}, std::move(patchValues));
        Tensor encoded = residual(patchInput, "representation.patch_proj", true);
        for (std::size_t row = 0; row < rowCount; ++row) {
            for (std::size_t patch = 0; patch < freshPatches; ++patch) {
                for (std::size_t j = 0; j < modelDim; ++j) {
                    x.values[(row * count + patch) * modelDim + j] = encoded.values[(row * freshPatches + patch) * modelDim + j];
                }
            }
        }
    }
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t patch = 0; patch < patches; ++patch) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                x.values[(row * count + freshPatches + patch) * modelDim + j] = query[j];
            }
        }
    }
    std::vector<double> positions;
    buildPositions(rowCount, count, base, patchCount, false, input.groupIds, first, positions, input.coordinates);
    for (double& position : positions) {
        position += input.positionOffset;
    }
    const std::vector<std::size_t> newPages(count, pool.capacity());
    const std::vector<int> targetRoles(rowCount, 0);
    ModelTrace scratch;
    runLayers(
        x,
        base,
        patchCount,
        patchCount,
        positions,
        keyValid,
        false,
        targetRoles,
        input.groupIds,
        pool,
        table,
        newPages,
        shared,
        false,
        scratch
    );
    Tensor decoder({rowCount, patches, modelDim});
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t patch = 0; patch < patches; ++patch) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                decoder.values[(row * patches + patch) * modelDim + j] = x.values[(row * count + freshPatches + patch) * modelDim + j];
            }
        }
    }
    std::vector<float> frameLoc(rowCount, 0.0f);
    std::vector<float> frameScale(rowCount, config.minimumScale);
    for (std::size_t row = 0; row < rowCount; ++row) {
        const PageScaleState& state = freshPatches > 0
            ? states[row * freshPatches + (freshPatches - 1)]
            : pool.page(table.back()).scaleStates[shared ? row % pool.geometry().rowCount : row];
        const double mean = state.count > 0.0 ? state.sum / state.count : 0.0;
        frameLoc[row] = static_cast<float>(mean);
        frameScale[row] = static_cast<float>(std::max(
            std::sqrt(state.m2 / std::max(state.count - 1.0, 1.0)),
            static_cast<double>(config.minimumScale)
        ));
    }
    return headOutput(decoder, patches, frameLoc, frameScale, true);
}

Tensor Model::predictFrozen(
    const ForwardInput& input,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    double positionOffset
) const {
    ModelTrace trace;
    return forwardQuantiles(input, true, false, trace, pool, table, positionOffset);
}

Tensor Model::predictBlock(const ForwardInput& input, ModelTrace& trace) const {
    return forwardQuantiles(input, true, true, trace);
}

Tensor Model::forwardQuantiles(
    const ForwardInput& input,
    bool denormalize,
    bool recordTrace,
    ModelTrace& trace
) const {
    PagePool pool;
    const std::vector<std::size_t> table;
    return forwardQuantiles(input, denormalize, recordTrace, trace, pool, table, 0.0);
}

Tensor Model::forwardQuantiles(
    const ForwardInput& input,
    bool denormalize,
    bool recordTrace,
    ModelTrace& trace,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    double positionOffset
) const {
    const auto& config = modelConfig;
    const Tensor& context = input.context;
    const std::vector<std::uint8_t>& observed = input.observed;
    const std::vector<std::size_t>& groupIds = input.groupIds;
    const std::size_t horizonPatches = input.horizonPatches;
    const std::vector<int>& roles = input.roles;
    const std::vector<std::uint8_t>& indicators = input.indicators;
    validateRows(context, observed, config.patchSize);
    const std::size_t rowCount = context.shape[0];
    const std::size_t time = context.shape[1];
    const std::size_t contextPatches = time / config.patchSize;
    const std::size_t horizon = elementCount({horizonPatches, config.patchSize});
    if (horizonPatches == 0 || groupIds.size() != rowCount || (!roles.empty() && roles.size() != rowCount)
        || (!indicators.empty() && indicators.size() != rowCount)) {
        throw std::invalid_argument("model: invalid horizon, groupIds, roles or indicators");
    }
    bool covariates = false;
    for (int role : roles) {
        if (role < 0 || role > 2) {
            throw std::invalid_argument("model: role must be Target=0, PastCov=1 or KnownCov=2");
        }
        covariates = covariates || role != 0;
    }
    covariates = covariates && config.covariateEnabled;
    if (covariates && horizonPatches > config.blockSize) {
        throw std::invalid_argument("model: covariate horizon exceeds blockSize");
    }
    if (input.hasFuture) {
        validateRows(input.future, input.futureObserved, config.patchSize);
        if (input.future.shape != std::vector<std::size_t>{rowCount, horizon}) {
            throw std::invalid_argument("model: future must have shape (R,horizonPatches*P)");
        }
    } else if (!input.futureObserved.empty()) {
        throw std::invalid_argument("model: futureObserved requires future");
    }
    if (contextPatches > std::numeric_limits<std::size_t>::max() - horizonPatches) {
        throw std::overflow_error("model: patch count overflow");
    }
    const std::size_t encodedCount = covariates ? contextPatches + horizonPatches : contextPatches;
    const std::size_t decodedCount = covariates ? encodedCount : horizonPatches;
    if (encodedCount > std::numeric_limits<std::size_t>::max() - decodedCount) {
        throw std::overflow_error("model: token count overflow");
    }
    const std::size_t slots = encodedCount + decodedCount;
    if (slots > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("model: positions exceed int64 range");
    }
    const std::size_t width = elementCount({encodedCount, config.patchSize});
    Tensor inputs({rowCount, width});
    std::vector<std::uint8_t> fullObserved(inputs.size(), 0);
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t t = 0; t < time; ++t) {
            inputs.values[row * width + t] = observed[row * time + t] != 0 ? context.values[row * time + t] : 0.0f;
            fullObserved[row * width + t] = observed[row * time + t];
        }
        if (covariates && input.hasFuture) {
            for (std::size_t t = 0; t < horizon; ++t) {
                const std::size_t index = row * horizon + t;
                inputs.values[row * width + time + t] = input.futureObserved[index] != 0
                    ? input.future.values[index] : 0.0f;
                fullObserved[row * width + time + t] = input.futureObserved[index];
            }
        }
    }
    ScaleResult normalization = scale(inputs, fullObserved);
    // known-future行按完整窗口总体方差归一化，二值行固定(0,1)，不做asinh
    if (covariates) {
        for (std::size_t row = 0; row < rowCount; ++row) {
            if (roles[row] != 2) {
                continue;
            }
            float count = 0.0f;
            float sum = 0.0f;
            bool binary = true;
            for (std::size_t t = 0; t < width; ++t) {
                const std::size_t index = row * width + t;
                if (fullObserved[index] != 0) {
                    const float value = inputs.values[index];
                    sum += value;
                    count += 1.0f;
                    binary = binary && (value == 0.0f || value == 1.0f);
                }
            }
            binary = binary && count > 0.0f;
            float loc = sum / std::max(count, 1.0f);
            float variance = 0.0f;
            for (std::size_t t = 0; t < width; ++t) {
                const std::size_t index = row * width + t;
                if (fullObserved[index] != 0) {
                    const float difference = inputs.values[index] - loc;
                    variance += difference * difference;
                }
            }
            float scale = std::max(std::sqrt(variance / std::max(count, 1.0f)), config.minimumScale);
            if (binary) {
                loc = 0.0f;
                scale = 1.0f;
            }
            for (std::size_t t = 0; t < width; ++t) {
                const std::size_t index = row * width + t;
                normalization.loc.values[index] = loc;
                normalization.scale.values[index] = scale;
                float value = fullObserved[index] != 0
                    ? (inputs.values[index] - loc) / std::max(scale, std::numeric_limits<float>::epsilon()) : 0.0f;
                if (config.useAsinh && !binary) {
                    value = std::asinh(value);
                }
                normalization.scaled.values[index] = value;
            }
        }
    }
    const std::size_t modelDim = config.modelDim;
    const std::size_t base = table.size();
    if (base > contextPatches || !std::isfinite(positionOffset) || positionOffset < 0.0) {
        throw std::invalid_argument("invalid frozen prefix or position offset");
    }
    if (base > 0) {
        verifyPoolGeometry(pool, config, rowCount);
    }
    const std::size_t activeSlots = slots - base;
    Tensor patchInput({rowCount, encodedCount - base, elementCount({2, config.patchSize})});
    std::vector<std::uint8_t> keyValid(elementCount({rowCount, encodedCount}), 0);
    std::map<std::size_t, std::size_t> firstObserved;
    for (std::size_t row = 0; row < rowCount; ++row) {
        auto first = firstObserved.emplace(groupIds[row], encodedCount).first;
        bool started = false;
        for (std::size_t patch = 0; patch < encodedCount; ++patch) {
            bool any = false;
            for (std::size_t t = 0; t < config.patchSize; ++t) {
                const std::size_t index = row * width + patch * config.patchSize + t;
                any = any || fullObserved[index] != 0;
            }
            if (patch >= base) {
                const std::size_t local = patch - base;
                for (std::size_t t = 0; t < config.patchSize; ++t) {
                    const std::size_t index = row * width + patch * config.patchSize + t;
                    const std::size_t out = (row * (encodedCount - base) + local) * 2 * config.patchSize + t;
                    patchInput.values[out] = normalization.scaled.values[index];
                    patchInput.values[out + config.patchSize] = fullObserved[index] != 0 ? 0.0f : 1.0f;
                }
            }
            if (any) {
                first->second = std::min(first->second, patch);
            }
            started = started || any;
            keyValid[row * encodedCount + patch] = any || (config.attendMissingPatches && started);
        }
    }
    if (positionOffset > 0.0) {
        for (auto& entry : firstObserved) {
            entry.second = 0;
        }
        for (std::size_t slot = 0; slot < base; ++slot) {
            for (std::size_t row = 0; row < rowCount; ++row) {
                keyValid[row * encodedCount + slot] = pool.page(table[slot]).maskBits[row];
            }
        }
    }
    Tensor encoded = residual(patchInput, "representation.patch_proj", true);
    Tensor x({rowCount, activeSlots, modelDim});
    std::vector<double> positions;
    const auto& query = weights.at("representation.forecast_query").values;
    for (std::size_t row = 0; row < rowCount; ++row) {
        if (firstObserved.at(groupIds[row]) == encodedCount) {
            firstObserved[groupIds[row]] = 0;
        }
        for (std::size_t slot = base; slot < slots; ++slot) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                x.values[(row * activeSlots + slot - base) * modelDim + j] = slot < encodedCount
                    ? encoded.values[(row * (encodedCount - base) + slot - base) * modelDim + j] : query[j];
            }
        }
    }
    buildPositions(rowCount, activeSlots, base, encodedCount, covariates, groupIds, firstObserved, positions, input.coordinates);
    for (double& position : positions) {
        position += positionOffset;
    }
    if (recordTrace) {
        trace.embedding = x;
        trace.layers.clear();
        trace.normalized = Tensor();
    }
    if (base > 0) {
        verifyTableGrid(
            pool,
            table,
            rowCount,
            keyValid,
            groupIds,
            firstObserved,
            input.coordinates,
            positionOffset
        );
    }
    const std::vector<std::size_t> newPages(activeSlots, pool.capacity());
    const std::vector<int> targetRoles(rowCount, 0);
    runLayers(
        x,
        base,
        encodedCount,
        contextPatches,
        positions,
        keyValid,
        covariates,
        roles.empty() ? targetRoles : roles,
        groupIds,
        pool,
        table,
        newPages,
        false,
        recordTrace,
        trace
    );
    Tensor decoder({rowCount, horizonPatches, modelDim});
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t patch = 0; patch < horizonPatches; ++patch) {
            for (std::size_t j = 0; j < modelDim; ++j) {
                const std::size_t slot = encodedCount + (covariates ? contextPatches : 0) + patch;
                decoder.values[(row * horizonPatches + patch) * modelDim + j] = x.values[(row * activeSlots + slot - base) * modelDim + j];
            }
        }
    }
    std::vector<float> loc(rowCount, 0.0f);
    std::vector<float> scaleValues(rowCount, 1.0f);
    for (std::size_t row = 0; row < rowCount; ++row) {
        loc[row] = normalization.loc.values[row * width + time - 1];
        scaleValues[row] = normalization.scale.values[row * width + time - 1];
    }
    return headOutput(decoder, horizonPatches, loc, scaleValues, denormalize);
}

void Model::runLayers(
    Tensor& x,
    std::size_t base,
    std::size_t encoded,
    std::size_t contextPatches,
    const std::vector<double>& positions,
    const std::vector<std::uint8_t>& keyValid,
    bool covariates,
    const std::vector<int>& roles,
    const std::vector<std::size_t>& groupIds,
    PagePool& pool,
    const std::vector<std::size_t>& table,
    const std::vector<std::size_t>& newPages,
    bool sharedPages,
    bool recordTrace,
    ModelTrace& trace
) const {
    const auto& config = modelConfig;
    const std::size_t rowCount = x.shape[0];
    const std::size_t count = x.shape[1];
    const std::size_t slots = base + count;
    const std::size_t modelDim = config.modelDim;
    if (table.size() != base || newPages.size() != count
        || positions.size() != rowCount * count || keyValid.size() != rowCount * encoded
        || roles.size() != rowCount || groupIds.size() != rowCount) {
        throw std::invalid_argument("model: inconsistent incremental layer input");
    }
    // time轴K/V按(rowCount,slots,modelDim)全局布局：缓存页填[0,base)，本批写入[base,slots)

    // 布尔掩码以谓词计算，避免存储完整S*S矩阵，对角始终开放
    const auto timeAllowed = [&](std::size_t row, std::size_t q, std::size_t k) {
        if (q < encoded) {
            return k < encoded && (q == k || (keyValid[row * encoded + k] != 0
                && ((covariates && roles[row] == 2) || k <= q)));
        }
        if (k < encoded) {
            return (!covariates || q - encoded >= contextPatches) && k < contextPatches
                && keyValid[row * encoded + k] != 0;
        }
        const std::size_t qDec = q - encoded;
        const std::size_t kDec = k - encoded;
        return q == k || (kDec <= qDec && (!covariates || (qDec >= contextPatches && kDec >= contextPatches)));
    };
    const auto variateAllowed = [&](std::size_t slot, std::size_t q, std::size_t k) {
        if (q == k) {
            return true;
        }
        if (groupIds[q] != groupIds[k] || (slot < encoded && keyValid[k * encoded + slot] == 0)) {
            return false;
        }
        return !covariates || !((roles[q] != 0 && roles[k] == 0) || (roles[q] == 2 && roles[k] != 2));
    };
    std::size_t timeLayer = 0;
    for (std::size_t layer = 0; layer < config.blockSchedule.size(); ++layer) {
        const std::string prefix = "transformer.layers." + std::to_string(layer);
        const auto axes = attentionAxes(config.blockSchedule[layer]);
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            const bool isTime = axes[axis];
            const std::string attn = prefix + ".attns." + std::to_string(axis);
            Tensor norm = reference::rmsNorm(
                x,
                weights.at(prefix + ".attn_norms." + std::to_string(axis) + ".scale"),
                config.normEps
            );
            Tensor fused = project(norm, attn + ".wQKV", config.attnBias);
            Tensor q({rowCount, count, config.headCount, config.headDim()});
            Tensor k(q.shape);
            Tensor v(q.shape);
            // 发布权重按(qkv,headDim,headCount)排列，attention输出按(headCount,headDim)展平
            for (std::size_t token = 0; token < rowCount * count; ++token) {
                for (std::size_t head = 0; head < config.headCount; ++head) {
                    for (std::size_t j = 0; j < config.headDim(); ++j) {
                        const std::size_t source = token * 3 * modelDim + j * config.headCount + head;
                        const std::size_t dest = token * modelDim + head * config.headDim() + j;
                        q.values[dest] = fused.values[source];
                        k.values[dest] = fused.values[source + modelDim];
                        v.values[dest] = fused.values[source + 2 * modelDim];
                    }
                }
            }
            Tensor attended({rowCount, count, modelDim});
            const std::size_t headDim = config.headDim();
            if (isTime) {
                reference::rotary(q, positions, config.ropeDim(), config.ropeTheta);
                reference::rotary(k, positions, config.ropeDim(), config.ropeTheta);
                const std::size_t prefixRows = sharedPages ? pool.geometry().rowCount : rowCount;
                std::vector<float> prefixKeys(elementCount({prefixRows, base, modelDim}));
                std::vector<float> prefixValues(prefixKeys.size());
                for (std::size_t slot = 0; slot < base; ++slot) {
                    const auto& page = pool.page(table[slot]);
                    for (std::size_t row = 0; row < prefixRows; ++row) {
                        std::copy_n(page.keys[timeLayer].begin() + row * modelDim, modelDim,
                            prefixKeys.begin() + (row * base + slot) * modelDim);
                        std::copy_n(page.values[timeLayer].begin() + row * modelDim, modelDim,
                            prefixValues.begin() + (row * base + slot) * modelDim);
                    }
                }
                for (std::size_t row = 0; row < rowCount; ++row) {
                    for (std::size_t local = 0; local < count; ++local) {
                        if (!sharedPages && base + local < encoded && newPages[local] != pool.capacity()) {
                            auto& page = pool.page(newPages[local]);
                            std::copy_n(k.values.begin() + (row * count + local) * modelDim, modelDim,
                                page.keys[timeLayer].begin() + row * modelDim);
                            std::copy_n(v.values.begin() + (row * count + local) * modelDim, modelDim,
                                page.values[timeLayer].begin() + row * modelDim);
                        }
                    }
                    AttentionMask mask;
                    mask.allowed = [&](std::size_t query, std::size_t key) {
                        return timeAllowed(row, base + query, key);
                    };
                    for (std::size_t head = 0; head < config.headCount; ++head) {
                        const std::size_t offset = row * count * modelDim + head * headDim;
                        const std::size_t prefixOffset = (row % prefixRows) * base * modelDim + head * headDim;
                        const AttentionView view{
                            q.values, k.values, v.values, attended.values,
                            count, slots, headDim, offset, offset, offset, offset,
                            modelDim, modelDim, modelDim, modelDim,
                            prefixKeys, prefixValues, base, prefixOffset, modelDim
                        };
                        reference::attention(view, mask);
                    }
                }
                ++timeLayer;
            } else {
                for (const auto& range : timestral::attentionGroups(groupIds)) {
                    for (std::size_t local = 0; local < count; ++local) {
                        AttentionMask mask;
                        mask.allowed = [&](std::size_t query, std::size_t key) {
                            return variateAllowed(base + local, range.first + query, range.first + key);
                        };
                        for (std::size_t head = 0; head < config.headCount; ++head) {
                            const std::size_t offset = (range.first * count + local) * modelDim + head * headDim;
                            const std::size_t stride = count * modelDim;
                            const AttentionView view{
                                q.values, k.values, v.values, attended.values,
                                range.second, range.second, headDim, offset, offset, offset, offset,
                                stride, stride, stride, stride
                            };
                            reference::attention(view, mask);
                        }
                    }
                }
            }
            if ((isTime ? config.timeAttnGate : config.variateAttnGate) == "sdpa_sigmoid") {
                Tensor gate = project(norm, attn + "._gate", config.attnBias);
                for (std::size_t i = 0; i < attended.size(); ++i) {
                    attended.values[i] *= 1.0f / (1.0f + std::exp(-gate.values[i]));
                }
            }
            addInPlace(x, project(attended, attn + ".wO", config.attnBias));
        }
        Tensor norm = reference::rmsNorm(x, weights.at(prefix + ".mlp_norm.scale"), config.normEps);
        Tensor hidden = reference::swiGLU(project(norm, prefix + ".mlp.w12", config.mlpBias));
        addInPlace(x, project(hidden, prefix + ".mlp.w3", config.mlpBias));
        if (recordTrace) {
            trace.layers.push_back(x);
        }
    }
    x = reference::rmsNorm(x, weights.at("transformer.out_norm.scale"), config.normEps);
    if (recordTrace) {
        trace.normalized = x;
    }
}

std::unique_ptr<timestral::Model> makeModel(ModelConfig config, utils::Safetensors weights) {
    return std::make_unique<Model>(std::move(config), std::move(weights));
}

} // namespace timestral::ytensor
