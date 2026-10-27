/***************
 * @file model.hpp
 * @brief 各计算后端共用的Model接口及ForwardInput、CacheAppendInput、ScaleResult结构
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "timestral/utils/cache.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/tensor.hpp"

namespace timestral {

// 三个张量均为(R,T)，loc/scale在每个patch内重复
struct ScaleResult {
    Tensor scaled;
    Tensor loc;
    Tensor scale;
};

// 逐层快照，只由带ModelTrace的重载填充
struct ModelTrace {
    Tensor embedding;
    std::vector<Tensor> layers;
    Tensor normalized;
};

// 一次前向的完整输入。context为(R,T)，T必须为正patch倍数，observed与context同形，
// roles为空等价于全Target(0)，PastCov=1，KnownCov=2。hasFuture为true时future必须为
// (R,horizonPatches*P)且futureObserved同时提供，indicators为可选R项提示，
// 二值判断仍按上游检查实际观测值
struct ForwardInput {
    Tensor context;
    std::vector<std::uint8_t> observed;
    std::vector<std::size_t> groupIds;
    std::size_t horizonPatches = 0;
    std::vector<int> roles;
    Tensor future;
    bool hasFuture = false;
    std::vector<std::uint8_t> futureObserved;
    std::vector<std::uint8_t> indicators;
    std::vector<double> coordinates; // (R,contextPatches+horizonPatches)，空则使用离散位置
};

// 分页增量接口的输入：context为(R,T)，T为正patch倍数，网格从左对齐（首个样本落在
// slot 0），observed与context同形，groupIds每行一个组号。horizonPatches为
// decodePages要解码的patch数，对appendPages无意义
struct CacheAppendInput {
    Tensor context;
    std::vector<std::uint8_t> observed;
    std::vector<std::size_t> groupIds;
    std::size_t horizonPatches = 0;
    std::vector<double> coordinates; // 每行的实数patch坐标，空则使用离散位置
    double positionOffset = 0.0; // 实验滑窗保留绝对RoPE位置，不重算旧KV
};

// 各后端实现同一前向语义，构造时严格校验权重契约，方法允许并发只读调用
class Model {
public:
    virtual ~Model() = default;

    // @brief 模型配置
    // @return 构造时校验过的配置，生命周期与模型一致
    virtual const ModelConfig& config() const = 0;

    // @brief 对上下文做因果归一化
    // @param context 形状(R,T)，T为正patch倍数
    // @param observed 与context同形，非0表示观测
    // @return 归一化值、每点loc与scale，三者均为(R,T)
    // @throw std::invalid_argument 形状、patch对齐或掩码非法时抛出
    virtual ScaleResult scale(const Tensor& context, const std::vector<std::uint8_t>& observed) const = 0;

    // @brief 运行一次前向，返回归一化分位数
    // @param input 前向输入，context为(R,T)
    // @return 形状(R,horizonPatches*P,Q)，值为归一化空间
    // @note 不修改输入，允许并发只读调用
    virtual Tensor forward(const ForwardInput& input) const = 0;

    // @brief 运行一次前向并记录中间张量
    // @param input 前向输入
    // @param trace 输出参数，写入embedding、各层输出与归一化快照
    // @return 与无trace重载相同，为归一化分位数
    virtual Tensor forward(const ForwardInput& input, ModelTrace& trace) const = 0;

    // @brief 预测一块并反归一化最后上下文patch帧
    // @param input 前向输入
    // @return 形状(R,horizonPatches*P,Q)，为原始量纲
    // @note 与上游predict_block一致：处理sinh溢出后的无穷值
    virtual Tensor predictBlock(const ForwardInput& input) const = 0;

    // @brief 预测一块并记录中间张量
    // @param input 前向输入
    // @param trace 输出参数，写入embedding、各层输出与归一化快照
    // @return 与无trace重载相同，为原始量纲
    virtual Tensor predictBlock(const ForwardInput& input, ModelTrace& trace) const = 0;

    // @brief 按行数给出页池几何
    // @param rowCount 一次预测的轨迹数
    // @return 每页K/V的行数、头数、头维与time轴子层数
    // @note 页池构造时使用，运行期几何不再变化
    virtual PageGeometry pageGeometry(std::size_t rowCount) const = 0;

    // @brief 模型配置的页键指纹
    // @return 影响time轴K/V的配置字段哈希，配置变化后旧页不再命中
    virtual std::uint64_t pageFingerprint() const = 0;

    // @brief 是否支持分页增量接口
    // @return 支持时返回true，否则调用方回退到forward/predictBlock全量重算
    virtual bool supportsIncremental() const = 0;

    // @brief 后端实际使用的计算线程数
    // @return 单线程后端返回1，Eigen与ytensor返回当前OpenMP线程设置
    virtual std::size_t threadCount() const {
        return 1;
    }

    // 冻结因果前缀KV，按原生掩码重算条件尾部，角色及统计差异属于实验近似
    // table中的通道顺序须与input完全一致，返回原始量纲分位数，不改写旧页
    virtual Tensor predictFrozen(
        const ForwardInput&,
        PagePool&,
        const std::vector<std::size_t>&,
        double
    ) const {
        throw std::runtime_error("backend does not support frozen-prefix experiments");
    }

    // @brief 把context中尚未入页的整patch编码进页池
    // @param input 前缀输入，context为(R,T)，网格左对齐
    // @param pool 页池，几何必须与模型一致
    // @param table 页表，按patch顺序追加页号
    // @note 只支持无协变量的目标行，table中的页必须与本次context的网格一致
    // @note 尾部连续未命中段按整段prefill一次编码落页，段中孤立未命中逐patch编码
    // @throw std::invalid_argument 页池几何或页表网格不一致时抛出
    virtual void appendPages(
        const CacheAppendInput& input,
        PagePool& pool,
        std::vector<std::size_t>& table
    ) const = 0;

    // @brief 用前缀页与未入页的新patch解码horizonPatches个patch
    // @param input 前缀输入，horizonPatches为要解码的patch数
    // @param pool 页池，table中的前缀页参与time轴注意力
    // @param table 已入页的页号，按patch顺序
    // @return 反归一化的(R,horizonPatches*P,Q)，与predictBlock同语义
    // @throw std::invalid_argument 页池几何或页表网格不一致时抛出
    virtual Tensor decodePages(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table
    ) const = 0;

    // @brief 用整组页表解码多条共享前缀的请求
    // @param input context为(B*V,T)，对应通道前缀相同，groupIds隔离请求
    // @param pool 以pageGeometry(V)建立、页宽为一组通道的页池
    // @param table 共享前缀页号，按patch顺序
    // @return 反归一化的(R,horizonPatches*P,Q)，与predictBlock同语义
    // @note 第r行读取页内r%V通道的K/V，共享路径不写页
    // @throw std::invalid_argument 页池几何、组布局、对应通道网格或前缀不一致时抛出
    virtual Tensor decodePagesShared(
        const CacheAppendInput& input,
        PagePool& pool,
        const std::vector<std::size_t>& table
    ) const = 0;
};

} // namespace timestral
