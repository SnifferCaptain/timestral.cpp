/***************
 * @file model.hpp
 * @brief ytensor可选后端单块前向、分页增量前向与因果归一化。
 * 语义与reference/model.cpp逐段对应，矩阵乘法与线性投影使用ytensor。
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "timestral/utils/cache.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/model.hpp"
#include "timestral/utils/safetensors.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral::ytensor {

using ScaleResult = timestral::ScaleResult;
using ModelTrace = timestral::ModelTrace;

class Model final : public timestral::Model {
public:
    // @brief 拥有配置与权重并校验权重契约。
    // @param config 已解析的模型配置。
    // @param weights 已载入的权重，所有权转移给模型。
    // @throw std::invalid_argument 权重名称、形状或有限值不满足契约时抛出。
    Model(ModelConfig config, utils::Safetensors weights);

    // @brief 模型配置。
    // @return 构造时校验过的配置。
    const ModelConfig& config() const override;

    // @brief 后端实际使用的计算线程数。
    // @return YBLAS/FA2核的进程级线程数，未设置时返回1。
    std::size_t threadCount() const override;

    // @brief 对上下文做因果归一化。
    // @param context 形状(rowCount,T)，T为正patch倍数。
    // @param observed 与context同形，非0表示观测。
    // @return 归一化值、每点loc与scale，三者均为(rowCount,T)。
    // @throw std::invalid_argument 形状或掩码非法时抛出。
    ScaleResult scale(const Tensor& context, const std::vector<std::uint8_t>& observed) const override;

    // @brief 运行一次前向，返回归一化分位数。
    // @param input 前向输入，context为(rowCount,T)。
    // @return 形状(rowCount,horizonPatches*P,Q)，值为归一化空间。
    // @note 不修改输入，允许并发只读调用。
    Tensor forward(const ForwardInput& input) const override;

    // @brief 运行一次前向并记录中间张量。
    // @param input 前向输入。
    // @param trace 输出参数，写入embedding、各层输出与归一化快照。
    // @return 与无trace重载相同。
    Tensor forward(const ForwardInput& input, ModelTrace& trace) const override;

    // @brief 预测一块并反归一化最后上下文patch帧。
    // @param input 前向输入。
    // @return 形状(rowCount,horizonPatches*P,Q)，为原始量纲。
    // @note 与上游predict_block一致：处理sinh溢出后的无穷值。
    Tensor predictBlock(const ForwardInput& input) const override;

    // @brief 预测一块并记录中间张量。
    // @param input 前向输入。
    // @param trace 输出参数，写入embedding、各层输出与归一化快照。
    // @return 与无trace重载相同，为原始量纲。
    Tensor predictBlock(const ForwardInput& input, ModelTrace& trace) const override;

    // @brief 按行数给出页池几何。
    // @param rowCount 一次预测的轨迹数。
    // @return 每页K/V的行数、头数与time轴子层数。
    PageGeometry pageGeometry(std::size_t rowCount) const override;
    // @brief 模型配置的页键指纹。
    // @return 影响time轴K/V的配置字段哈希。
    std::uint64_t pageFingerprint() const override;

    // @brief 是否支持分页增量接口。
    // @return 本后端恒为true。
    bool supportsIncremental() const override;

    // @brief 把尚未入页的整patch编码进页池。
    // @param input 前缀输入，网格左对齐。
    // @param pool 页池，几何必须与模型一致。
    // @param table 页表，按patch顺序追加页号。
    // @throw std::invalid_argument 页池几何或页表网格不一致时抛出。
    // @throw std::runtime_error 页池容量不足或缓存页状态与前缀不符时抛出。
    void appendPages(
        const CacheAppendInput& input,
        PagePool& pool,
        std::vector<std::size_t>& table
    ) const override;

    // @brief 用前缀页与未入页的新patch解码horizonPatches个patch。
    // @param input 前缀输入；horizonPatches为要解码的patch数。
    // @param pool 页池。
    // @param table 已入页的页号。
    // @return 反归一化的(rowCount,horizonPatches*P,Q)。
    // @throw std::invalid_argument 页池几何或页表网格不一致时抛出。
    Tensor decodePages(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table
    ) const override;

    // @brief 用整组页表解码多条共享前缀的请求。
    // @param input context为(B*V,T)，对应通道前缀相同，groupIds隔离请求。
    // @param pool 以pageGeometry(V)建立、页宽为一组通道的页池。
    // @param table 共享前缀页号，按patch顺序。
    // @return 反归一化的(rowCount,horizonPatches*P,Q)。
    // @note 第r行读取页内r%V通道的K/V，共享路径不写页。
    // @throw std::invalid_argument 页池几何、组布局、对应通道网格或前缀不一致时抛出。
    Tensor decodePagesShared(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table
    ) const override;

    // @brief 冻结table中的前缀KV后重算条件尾部，契约见Model::predictFrozen。
    // @param input 前向输入。
    // @param pool 页池。
    // @param table 冻结前缀页号，按patch顺序。
    // @param positionOffset 滑动窗口绝对位置偏移。
    // @return 反归一化的(rowCount,horizonPatches*P,Q)。
    Tensor predictFrozen(
        const ForwardInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        double positionOffset
    ) const override;

private:
    ModelConfig modelConfig;
    utils::Safetensors weights;

    // @brief 按名称取权重做线性投影。
    // @param input 形状(...,in)的输入。
    // @param name 权重名前缀，读取name.weight与可选name.bias。
    // @param bias 为true时加偏置。
    // @return 形状(...,out)的输出。
    Tensor project(const Tensor& input, const std::string& name, bool bias) const;

    // @brief 带残差连接的投影块。
    // @param input 形状(...,in)的输入。
    // @param name 权重名前缀。
    // @param bias 为true时三层投影都加偏置。
    // @return 形状(...,out)的输出。
    Tensor residual(const Tensor& input, const std::string& name, bool bias) const;

    // @brief 把解码槽隐藏状态映射到输出分位数。
    // @param hidden 形状(rowCount,patchCount,d)的隐藏状态。
    // @param patchCount 解码patch数。
    // @param loc 每行的归一化均值。
    // @param scale 每行的归一化尺度。
    // @param denormalize 为true时反归一化并处理sinh溢出。
    // @return 形状(rowCount,patchCount*P,Q)的分位数。
    Tensor headOutput(
        const Tensor& hidden,
        std::size_t patchCount,
        const std::vector<float>& loc,
        const std::vector<float>& scale,
        bool denormalize
    ) const;

    // @brief 执行前向并输出分位数。
    // @param input 前向输入。
    // @param denormalize 为true时按最后上下文patch帧反归一化。
    // @param recordTrace 为true时写入trace。
    // @param trace 输出参数，recordTrace为false时不写入。
    // @return 形状(rowCount,horizonPatches*P,Q)的分位数。
    Tensor forwardQuantiles(
        const ForwardInput& input,
        bool denormalize,
        bool recordTrace,
        ModelTrace& trace
    ) const;

    // @brief 执行前向并输出分位数，前缀K/V按页表读取。
    // @param input 前向输入。
    // @param denormalize 为true时按最后上下文patch帧反归一化。
    // @param recordTrace 为true时写入trace。
    // @param trace 输出参数，recordTrace为false时不写入。
    // @param pool 页池，空页表时不读页。
    // @param table 冻结前缀页号，按patch顺序。
    // @param positionOffset 滑动窗口绝对位置偏移。
    // @return 形状(rowCount,horizonPatches*P,Q)的分位数。
    Tensor forwardQuantiles(
        const ForwardInput& input,
        bool denormalize,
        bool recordTrace,
        ModelTrace& trace,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        double positionOffset
    ) const;

    // @brief 逐层执行time/variate注意力、门控与MLP。
    // @param x 输入输出隐藏状态，形状(rowCount,count,d)。
    // @param base 本批首个槽的全局下标，[0,base)的time轴K/V取自页表。
    // @param encoded 上下文编码槽数。
    // @param contextPatches 上下文patch数。
    // @param positions 每行每槽的rotary位置。
    // @param keyValid 每个上下文patch的有效位。
    // @param covariates 是否带协变量行。
    // @param roles 每行的角色，Target=0、PastCov=1、KnownCov=2。
    // @param groupIds 每行一个组号。
    // @param pool 页池。
    // @param table 已入页的页号。
    // @param newPages 本批写入的页号，等于pool.capacity()时不落页。
    // @param sharedPages 为true时页内K/V对所有行取第0行，且不写页。
    // @param recordTrace 为true时按层追加trace。
    // @param trace 输出参数，recordTrace为false时不写入。
    void runLayers(
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
    ) const;

    // @brief decodePages与decodePagesShared的共用主体。
    // @param input 前缀输入。
    // @param pool 页池，shared为true时按单行几何解释。
    // @param table 前缀页表。
    // @param shared 为true时所有行读取页的第0行K/V，且不写入页。
    // @return 反归一化的(rowCount,horizonPatches*P,Q)。
    Tensor decodePagesImpl(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        bool shared
    ) const;
};

// @brief 创建ytensor后端Model。
// @param config 模型配置。
// @param weights 已载入的权重，所有权转移。
// @return 多态模型句柄。
// @throw std::invalid_argument 权重契约不满足时抛出。
std::unique_ptr<timestral::Model> makeModel(ModelConfig config, utils::Safetensors weights);

// @brief 设置YBLAS矩阵乘与FA2内核的进程级线程数。
// @param threads 0取保守自动值（最多4），1强制串行，其余为精确线程数。
// @throw std::invalid_argument threads超出int范围时抛出。
void setThreads(std::size_t threads);

} // namespace timestral::ytensor
