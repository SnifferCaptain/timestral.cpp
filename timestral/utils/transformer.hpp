/***************
 * @file transformer.hpp
 * @brief 各数值后端共用的Timestral模型编排与分页执行
 * 移植自Mistral AI Timestral，Apache-2.0，基线a534bdbd
 ***************/

#pragma once

#include <functional>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "timestral/utils/attention.hpp"
#include "timestral/utils/cache.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/model.hpp"
#include "timestral/utils/safetensors.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

// 内部数值阶段仅由调用链显式传递，不根据M或共享可变状态判断
enum class ProjectionPhase {
    Float32,
    Decode
};

class Transformer : public Model {
public:
    // @brief 拥有配置与权重并校验权重契约
    // @param loadedConfig 已解析的模型配置
    // @param weights 已载入的权重，所有权转移给模型
    // @throw std::invalid_argument 权重名称、形状或有限值不满足契约时抛出
    Transformer(ModelConfig loadedConfig, utils::Safetensors weights);

    // @brief 设置time轴附加可见性谓词，须在模型共享给其它线程前调用
    // @param mask 参数为从上下文网格起点计的patch下标，horizon接续计数，与原生掩码取与，空函数表示不附加
    // @note 每个token对自身恒可见，谓词须可重入
    void setTimeMask(std::function<bool(std::size_t, std::size_t)> mask);

    // @brief 模型配置
    // @return 构造时校验过的配置
    const ModelConfig& config() const override;

    // @brief 对上下文做因果归一化
    // @param context 形状(rowCount,T)，T为正patch倍数
    // @param observed 与context同形，非0表示观测
    // @return 归一化值、每点loc与scale，三者均为(rowCount,T)
    // @throw std::invalid_argument 形状或掩码非法时抛出
    ScaleResult scale(const Tensor& context, const std::vector<std::uint8_t>& observed) const override;

    // @brief 运行一次前向，返回归一化分位数
    // @param input 前向输入，context为(rowCount,T)
    // @return 形状(rowCount,horizonPatches*P,Q)，值为归一化空间
    // @note 不修改输入，允许并发只读调用
    Tensor forward(const ForwardInput& input) const override;

    // @brief 运行一次前向并记录中间张量
    // @param input 前向输入
    // @param trace 输出参数，写入embedding、各层输出与归一化快照
    // @return 与无trace重载相同
    Tensor forward(const ForwardInput& input, ModelTrace& trace) const override;

    // @brief 预测一块并反归一化最后上下文patch帧
    // @param input 前向输入
    // @return 形状(rowCount,horizonPatches*P,Q)，为原始量纲
    // @note 与上游predict_block一致：处理sinh溢出后的无穷值
    Tensor predictBlock(const ForwardInput& input) const override;

    // @brief 预测一块并记录中间张量
    // @param input 前向输入
    // @param trace 输出参数，写入embedding、各层输出与归一化快照
    // @return 与无trace重载相同，为原始量纲
    Tensor predictBlock(const ForwardInput& input, ModelTrace& trace) const override;

    // @brief 按行数给出页池几何
    // @param rowCount 一次预测的轨迹数
    // @return 每页K/V的行数、头数与time轴子层数
    PageGeometry pageGeometry(std::size_t rowCount) const override;

    // @brief 模型配置的页键指纹
    // @return 影响time轴K/V的配置字段哈希
    std::uint64_t pageFingerprint() const override;

    // @brief 是否支持分页增量接口
    // @return 本后端恒为true
    bool supportsIncremental() const override;

    // @brief 把尚未入页的整patch编码进页池
    // @param input 前缀输入，网格左对齐
    // @param pool 页池，几何必须与模型一致
    // @param table 页表，按patch顺序追加页号
    // @note 尾部连续未命中段按整段prefill一次编码落页，段中孤立未命中逐patch编码
    // @throw std::invalid_argument 页池几何或页表网格不一致时抛出
    // @throw std::runtime_error 页池容量不足或缓存页状态与前缀不符时抛出
    void appendPages(
        const CacheAppendInput& input,
        PagePool& pool,
        std::vector<std::size_t>& table
    ) const override;

    // @brief 用前缀页与未入页的新patch解码horizonPatches个patch
    // @param input 前缀输入，horizonPatches为要解码的patch数
    // @param pool 页池
    // @param table 已入页的页号
    // @return 反归一化的(rowCount,horizonPatches*P,Q)
    Tensor decodePages(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table
    ) const override;

    // @brief 用整组通道页解码共享前缀的请求
    // @param input context为(B*V,T)，各请求的对应通道前缀必须相同
    // @param pool 以pageGeometry(V)建立的整组通道页池
    // @param table 共享前缀页号，按patch顺序
    // @return 反归一化的(rowCount,horizonPatches*P,Q)
    // @throw std::invalid_argument 页池几何、请求分组、网格或对应通道前缀不同时抛出
    Tensor decodePagesShared(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table
    ) const override;

    // 冻结table中的前缀KV后重算条件尾部，契约见Model::predictFrozen
    Tensor predictFrozen(
        const ForwardInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        double positionOffset
    ) const override;

    // 实验：attention sink页表追加，契约见Model::appendPagesSink
    void appendPagesSink(
        const CacheAppendInput& input,
        PagePool& pool,
        std::vector<std::size_t>& table
    ) const override;

    // 实验：冻结含sink首位页的页表后重算条件尾部，契约见Model::predictFrozenSink
    Tensor predictFrozenSink(
        const ForwardInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        double positionOffset
    ) const override;

protected:
    const ModelConfig modelConfig;
    const utils::Safetensors weights;
    std::function<bool(std::size_t, std::size_t)> timeMask;  // 附加time轴可见性，空为原生掩码

    // 输入最后一维为in，权重布局为(out,in)，输出最后一维为out
    virtual Tensor project(const Tensor& input, const std::string& name, bool bias) const = 0;

    // 默认仍为FP32，只有全部历史已入页的显式decode调用允许后端选择低精度权重
    virtual Tensor projectPhase(
        const Tensor& input,
        const std::string& name,
        bool bias,
        ProjectionPhase phase
    ) const;

    virtual Tensor normalize(const Tensor& input, const utils::Weight& scale, float epsilon) const = 0;

    // 默认保持先RMS再投影的FP32中间舍入，后端可在A打包时融合
    virtual Tensor projectNormalized(
        const Tensor& input,
        const utils::Weight& scale,
        float epsilon,
        const std::string& name,
        bool bias,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const;

    // QKV与gate读取同一个axis输入的RMS，不跨越注意力残差更新
    virtual void projectAttention(
        const Tensor& input,
        const utils::Weight& scale,
        float epsilon,
        const std::string& attn,
        bool bias,
        bool gated,
        Tensor& fused,
        Tensor& gate,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const;

    // 纯prefill最后time轴只要求K/V，默认保留完整QKV投影但不计算gate
    virtual void prepareCacheKv(
        const Tensor& input,
        const utils::Weight& scale,
        float epsilon,
        const std::string& attn,
        bool bias,
        Tensor& k,
        Tensor& v,
        const std::vector<double>& positions
    ) const;

    // 把发布的(qkv,headDim,headCount)布局转换到(headCount,headDim)
    // 仅time轴按positions对Q/K应用部分维RoPE，V不旋转
    // fused来自同一后端的projectAttention，行宽至少3*modelDim，前3*modelDim列为QKV
    virtual void prepareQkv(
        const Tensor& fused,
        Tensor& q,
        Tensor& k,
        Tensor& v,
        const std::vector<double>& positions,
        bool isTime
    ) const = 0;

    virtual void attention(const AttentionView& view, const AttentionMask& mask) const = 0;

    // 独立输出头或通道组的遍历，默认串行，后端可按互不重叠的输出并行
    virtual void attentionWork(std::size_t tasks, const std::function<void(std::size_t)>& function) const;

    // 原位attended *= sigmoid(gate)，二者同形
    virtual void applyGate(Tensor& attended, const Tensor& gate) const;

    // 返回前半*SiLU(后半)，输入最后一维为两倍隐藏宽度
    virtual Tensor activate(const Tensor& fused) const;

    // @brief 带残差连接的投影块
    // @param input 形状(...,in)的输入
    // @param name 权重名前缀
    // @param bias 为true时三层投影都加偏置
    // @param phase 显式数值阶段，默认保持三次投影与CPU SiLU的顺序
    // @return 形状(...,out)的输出
    virtual Tensor residual(
        const Tensor& input,
        const std::string& name,
        bool bias,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const;

    // x += project(input)，默认先投影再逐元素相加，后端可把残差作为GEMM的beta=1初值
    virtual void projectResidual(
        Tensor& x,
        const Tensor& input,
        const std::string& name,
        bool bias,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const;

    // 仅纯页编码允许在最后time轴K/V落页后截断，默认保留完整reference执行
    virtual bool optimizedPrefill() const;

    // 允许后端通过分段视图读取整组共享前缀，默认保留reference连续KV缓冲
    virtual bool broadcastPrefix() const;

private:
    // @brief decodePages与decodePagesShared的共用主体
    // @param input 前缀输入
    // @param pool 页池，shared为true时按整组通道几何解释
    // @param table 前缀页表
    // @param shared 为true时第r行读取页内r%V通道，且不写入页
    // @return 反归一化的(rowCount,horizonPatches*P,Q)
    Tensor decodePagesImpl(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        bool shared
    ) const;

    // @brief appendPages与appendPagesSink的共用主体
    // @param sinkPrefix 为true时页表首位允许位置不对齐网格的sink页
    void appendPagesImpl(
        const CacheAppendInput& input,
        PagePool& pool,
        std::vector<std::size_t>& table,
        bool sinkPrefix
    ) const;

    Tensor forwardQuantiles(
        const ForwardInput& input,
        bool denormalize,
        bool recordTrace,
        ModelTrace& trace
    ) const;

    // @brief 执行前向并输出分位数
    // @param input 前向输入
    // @param denormalize 为true时按最后上下文patch帧反归一化
    // @param recordTrace 为true时写入trace
    // @param trace 输出参数，recordTrace为false时不写入
    // @param sinkPrefix 为true时页表首位允许位置不对齐网格的sink页
    // @return 形状(rowCount,horizonPatches*P,Q)的分位数
    Tensor forwardQuantiles(
        const ForwardInput& input,
        bool denormalize,
        bool recordTrace,
        ModelTrace& trace,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        double positionOffset,
        bool sinkPrefix
    ) const;

protected:
    // @brief 逐层执行time/variate注意力、门控与MLP
    // @param x 输入输出隐藏状态，形状(rowCount,count,d)
    // @param base 本批首个槽的全局下标，[0,base)的time轴K/V取自页表
    // @param encoded 上下文编码槽数
    // @param contextPatches 上下文patch数
    // @param positions 每行每槽的rotary位置
    // @param keyValid 每个上下文patch的有效位
    // @param covariates 是否带协变量行
    // @param roles 每行的角色，Target=0、PastCov=1、KnownCov=2
    // @param groupIds 每行一个组号
    // @param pool 页池
    // @param table 已入页的页号
    // @param newPages 本批写入的页号，等于pool.capacity()时不落页
    // @param sharedPages 为true时第r行读取页内r%V通道，且不写页
    // @param prefill 仅appendPages传true，允许后端在最后time轴K/V落页后截断
    // @param recordTrace 为true时按层追加trace
    // @param trace 输出参数，recordTrace为false时不写入
    virtual void runLayers(
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
        bool prefill,
        bool recordTrace,
        ModelTrace& trace,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const;

private:
    // @brief 把解码槽隐藏状态映射到输出分位数
    // @param hidden 形状(rowCount,patchCount,d)的隐藏状态
    // @param patchCount 解码patch数
    // @param loc 每行的归一化均值
    // @param scale 每行的归一化尺度
    // @param denormalize 为true时反归一化并处理sinh溢出
    // @return 形状(rowCount,patchCount*P,Q)的分位数
    Tensor headOutput(
        const Tensor& hidden,
        std::size_t patchCount,
        const std::vector<float>& loc,
        const std::vector<float>& scale,
        bool denormalize,
        ProjectionPhase phase = ProjectionPhase::Float32
    ) const;
};

} // namespace timestral
