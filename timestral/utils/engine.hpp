/***************
 * @file engine.hpp
 * @brief 内部张量级推理引擎：分块自回归、分页增量与协变量拼装
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "timestral/timestral.hpp"
#include "timestral/utils/cache.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/model.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

// 张量级预测。输入(B,V,T)或(1,T)，NaN表示缺失，输出(B,V,H,Q)行主序
// 内部类型：只供入口实现与张量级对照测试使用，不进入公开头文件
class Engine {
public:
    // @brief 持有模型并解析载入期配置
    // @param model 已加载的后端模型，不得为空
    // @param settings 上下文、解码块与位置编码配置
    // @throw std::invalid_argument 模型为空或patchBatch为0时抛出
    Engine(std::shared_ptr<const Model> model, const Timestral::Settings& settings);

    // @brief 模型配置
    // @return 后端模型配置
    const ModelConfig& config() const;

    // @brief 生效的上下文上限
    // @return 载入时解析的maxContext，0已替换为模型contextLength
    std::size_t maxContext() const;

    // @brief 无缓存的完整预测
    // @param input 形状(B,V,T)或(1,T)，NaN表示缺失
    // @param predictionLength 预测步数，必须为正
    // @param request 后处理开关与采样间隔
    // @param pastCovariates 形状(B,C,T)的past行，零元素表示不提供
    // @param knownCovariates 形状(B,C,T+predictionLength)的已知未来行，零元素表示不提供
    // @param times 连续模式的(B,T)时间戳，离散模式可空
    // @param interval 预测采样间隔，0尝试使用request.sampleInterval
    // @return 形状(B,V,predictionLength,Q)的分位数
    // @note 协变量只作为模型行输入，公开入口不再产生past行，条件补全只使用known行
    // @throw std::invalid_argument 维度、协变量或分位点不合法时抛出
    Tensor predict(
        const Tensor& input,
        std::size_t predictionLength,
        const Timestral::Request& request = {},
        const Tensor& pastCovariates = {},
        const Tensor& knownCovariates = {},
        const std::vector<double>& times = {},
        double interval = 0.0
    ) const;

    // @brief 分页增量预测
    // @param context 形状(B,V,T)，网格左对齐
    // @param predictionLength 预测步数
    // @param request 后处理开关与采样间隔，只支持目标行
    // @param pool 页池，几何必须与模型一致
    // @param table 整patch前缀页表，按patch顺序
    // @param timing 输出，累加本次decode耗时，prefill由调用方统计
    // @param times 连续模式的(B,T)时间戳，离散模式可空
    // @param interval 预测采样间隔，0尝试使用request.sampleInterval
    // @return 形状(B,V,predictionLength,Q)的分位数
    // @throw std::runtime_error 后端不支持分页增量时抛出
    Tensor predictStreamed(
        const Tensor& context,
        std::size_t predictionLength,
        const Timestral::Request& request,
        PagePool& pool,
        std::vector<std::size_t>& table,
        TimingStats& timing,
        const std::vector<double>& times = {},
        double interval = 0.0
    ) const;

    // @brief 分页增量预测，镜像pass读取取负前缀的页表
    // @param mirrorTable 按取负前缀编码的整patch页表，空表时镜像pass全量重算，其余参数同上
    // @note 只在request.symmetricAveraging为true时读取mirrorTable，截断时同样由引擎释放
    Tensor predictStreamed(
        const Tensor& context,
        std::size_t predictionLength,
        const Timestral::Request& request,
        PagePool& pool,
        std::vector<std::size_t>& table,
        std::vector<std::size_t>& mirrorTable,
        TimingStats& timing,
        const std::vector<double>& times = {},
        double interval = 0.0
    ) const;

    // deprecated：跨未来块共享冻结历史并独立平移RoPE，跳过块间反馈
    // 仅保留历史实验与回归，不等于正常32patch单块前向，正式调用使用块AR
    // 历史按patch对齐，预测长度须为整帧，历史加预测不超过maxContext，前缀页只读
    [[deprecated("Use block AR with Settings::patchBatch; cross-frame RoPE Parallel is historical only")]]
    Tensor predictParallel(
        const Tensor& context,
        std::size_t predictionLength,
        const Timestral::Request& request,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        TimingStats& timing
    ) const;

    // @brief 是否支持分页增量接口
    // @return 后端支持时返回true
    bool supportsIncremental() const;

    // @brief 生效的块AR每轮patch数
    // @return 构造时钳制到blockSize后的patchBatch
    std::size_t patchBatch() const;

    // @brief 是否使用实验性连续patch位置
    bool continuousRope() const;

    // @brief 参考采样间隔，0表示采用请求间隔
    double ropeReferenceInterval() const;

    // @brief 按行数给出页池几何
    // @param rowCount 一次预测的轨迹数
    // @return 页池几何
    PageGeometry pageGeometry(std::size_t rowCount) const;

    // @brief 配置指纹
    // @return 后端模型的页键指纹
    std::uint64_t pageFingerprint() const;

    // @brief 把整patch前缀编码进页池
    // @param input 前缀输入
    // @param pool 页池
    // @param table 页表，按patch顺序追加页号
    void appendPages(
        const CacheAppendInput& input,
        PagePool& pool,
        std::vector<std::size_t>& table
    ) const;

    // @brief 用前缀页解码horizonPatches个patch
    // @param input 前缀输入
    // @param pool 页池
    // @param table 已入页的页号
    // @return 反归一化的分位数
    Tensor decodePages(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table
    ) const;

    // @brief 用整组页表解码共享前缀的多条请求
    // @param context 形状(B,T)或(B,V,T)，组间对应通道前缀相同，T对齐patch
    // @param predictionLength 预测步数，必须为正
    // @param request 后处理开关与采样间隔，只支持目标行
    // @param pool 以pageGeometry(V)建立的整组页池，二维输入按V=1解释
    // @param table 共享前缀页号，按patch顺序，非空
    // @param timing 输出，累加本次decode耗时
    // @param times 连续模式的(B,T)时间戳，离散模式可空
    // @param interval 预测采样间隔，0尝试使用request.sampleInterval
    // @return 形状(B,V,predictionLength,Q)的分位数
    // @note 页表引用仍归调用方所有，本方法不增删引用，上下文被maxContext截断时改走全量重算
    // @throw std::runtime_error 后端不支持分页增量时抛出
    // @throw std::invalid_argument 形状、协变量、页表或页池几何不合法时抛出
    Tensor decodeShared(
        const Tensor& context,
        std::size_t predictionLength,
        const Timestral::Request& request,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        TimingStats& timing,
        const std::vector<double>& times = {},
        double interval = 0.0
    ) const;

    // @brief 共享解码同时复用取负前缀，镜像页的引用也归调用方所有
    // @param mirrorTable 取负后对应同一共享前缀的页表，空表时镜像pass全量重算
    Tensor decodeShared(
        const Tensor& context,
        std::size_t predictionLength,
        const Timestral::Request& request,
        PagePool& pool,
        const std::vector<std::size_t>& table,
        const std::vector<std::size_t>& mirrorTable,
        TimingStats& timing,
        const std::vector<double>& times = {},
        double interval = 0.0
    ) const;

private:
    // @brief 全量与增量共用的分块自回归流程
    // @param input 形状(B,V,T)或(1,T)
    // @param predictionLength 预测步数
    // @param request 后处理开关与采样间隔
    // @param pastCovariates 形状(B,C,T)的past行，零元素表示不提供
    // @param knownCovariates 形状(B,C,T+predictionLength)的已知未来行，零元素表示不提供
    // @param pool 页池，全量时为空池
    // @param table 页表，全量时为空表
    // @param mirrorTable 镜像pass的页表，空表时镜像pass全量重算
    // @param timing 输出，累加耗时
    // @param incremental 为true时按页解码，否则整块前向
    // @param sharedPages 为true时页按单行几何解释，table的所有权仍归调用方
    // @param times 连续模式的(B,T)时间戳，离散模式可空
    // @param interval 预测采样间隔，0尝试使用request.sampleInterval
    // @return 分位数结果
    Tensor computePrediction(
        const Tensor& input,
        std::size_t predictionLength,
        const Timestral::Request& request,
        const Tensor& pastCovariates,
        const Tensor& knownCovariates,
        PagePool& pool,
        std::vector<std::size_t>& table,
        std::vector<std::size_t>& mirrorTable,
        TimingStats& timing,
        bool incremental,
        bool sharedPages,
        const std::vector<double>& times,
        double interval
    ) const;

    std::shared_ptr<const Model> model;
    std::size_t contextLimit = 0; // 构造时0已替换为模型contextLength
    std::size_t patchBlock = 16; // 构造时已钳制到blockSize
    bool continuousTime = false;
    double referenceInterval = 0.0;
};

// @brief 读取本地模型目录并构造引擎
// @param directory 含config.json与model.safetensors的目录
// @param settings 载入期参数
// @return 引擎句柄
// @throw std::runtime_error 配置文件或权重缺失、非法时抛出
std::shared_ptr<const Engine> loadEngine(
    const std::filesystem::path& directory,
    const Timestral::Settings& settings
);

// @brief 使用默认Settings加载引擎，不在声明处构造嵌套配置类型
std::shared_ptr<const Engine> loadEngine(const std::filesystem::path& directory);

} // namespace timestral
