/***************
 * @file cache.hpp
 * @brief 分页前缀缓存：页池、页键、命中与计时统计
 ***************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "timestral/utils/config.hpp"

namespace timestral {

// 因果归一化在一个patch末尾的累计量：均值=sum/count，样本方差=m2/(count-1)
// 页保存该状态，使后续patch不必重算整条前缀的Welford统计
struct PageScaleState {
    double count = 0.0;
    double sum = 0.0;
    double m2 = 0.0;
};

// 页池几何：一页只保存time轴注意力子层的K/V，每子层布局为(rowCount,headCount,headDim)
// timeLayers按blockSchedule中time轴子层的出现顺序编号
struct PageGeometry {
    std::size_t rowCount = 0;
    std::size_t headCount = 0;
    std::size_t headDim = 0;
    std::size_t timeLayers = 0;
};

// 一页 = 一个patch在全部time轴注意力子层的K/V，以及每行的掩码位、rotary位置
// 与归一化累计量。写满即只读，引用计数为0时可被LRU回收，可被多条流共享
struct CachePage {
    std::vector<std::vector<float>> keys; // 每子层rowCount*headCount*headDim，行主序
    std::vector<std::vector<float>> values;
    std::vector<std::uint8_t> maskBits; // 每行该patch能否被后续槽关注
    std::vector<double> positions; // 每行rotary位置
    std::vector<PageScaleState> scaleStates; // 每行该patch末尾的归一化累计量
    std::uint64_t parentKey = 0;
    std::uint64_t key = 0;
    std::uint64_t inputFingerprint = 0;
    std::size_t references = 0;
    std::uint64_t lastUse = 0;
    bool occupied = false;
};

// 一次预测的耗时拆分，单位秒。prefill为页编码，decode为按页解码，
// total为整次调用（含后处理），因此total不小于prefill与decode之和
struct TimingStats {
    double prefillSeconds = 0.0;
    double decodeSeconds = 0.0;
    double totalSeconds = 0.0;
};

// FNV-1a 64位增量哈希，按字节区间和整数累积，用于页键与输入指纹
class FnvHash {
public:
    // @brief 以给定offset basis初始化
    // @param seed 起始状态，默认FNV-1a 64位offset basis
    explicit FnvHash(std::uint64_t seed = 0xcbf29ce484222325ULL);

    // @brief 累积一段字节
    // @param bytes 待哈希的字节序列
    void addBytes(const std::vector<std::uint8_t>& bytes);

    // @brief 按位模式累积浮点值
    // @param values 待哈希的数值，按位模式比较，不比较浮点值
    // @note 按IEEE 754位模式累积，NaN载荷与有符号零都会影响哈希
    void addValues(const std::vector<float>& values);

    // @brief 累积字符串的每个字节
    // @param text 待哈希文本
    void addString(const std::string& text);

    // @brief 按小端字节序累积无符号整数
    // @param value 待哈希整数
    void addUnsigned(std::uint64_t value);

    // @brief 按补码位模式累积有符号整数
    // @param value 待哈希整数
    void addSigned(std::int64_t value);

    // @brief 按FP64位模式累积实数位置
    // @param position 无量纲patch坐标
    void addPosition(double position);

    // @brief 当前哈希值
    // @return 累积后的64位状态
    std::uint64_t value() const;

private:
    std::uint64_t state = 0xcbf29ce484222325ULL;
};

// @brief 计算影响time轴K/V的配置指纹
// @param config 模型配置
// @return 页键指纹，只覆盖决定K/V与归一化的字段，分位数后处理参数不参与页复用
std::uint64_t modelFingerprint(const ModelConfig& config);

// 固定容量页池：capacity是页数上限，页的K/V缓冲在首次占用时分配并在回收后复用
// 查找为按页键的线性扫描（容量按页数计，规模很小，不引入额外索引）
class PagePool {
public:
    PagePool() = default;

    // @brief 按几何与容量建立页池，K/V缓冲按需分配
    // @param geometry 每页的行数、头数、头维与time轴子层数
    // @param fingerprint 配置指纹，页键校验时使用
    // @param capacity 页数上限
    PagePool(PageGeometry geometry, std::uint64_t fingerprint, std::size_t capacity);

    // @brief 页数上限
    // @return 构造时给定的capacity
    std::size_t capacity() const;

    // @brief 已占用页数
    // @return 当前被占用的页数
    std::size_t size() const;

    // @brief 可回收页数
    // @return 已占用且引用计数为0的页数
    std::size_t cachedPages() const;

    // @brief 未占用页数
    // @return 当前可以直接写入的页数
    std::size_t freePages() const;

    // @brief 可回收页数
    // @return 已占用且引用计数为0的页数
    std::size_t reclaimablePages() const;

    // @brief 最久未用的可回收页
    // @return 页号，没有可回收页时返回capacity()
    // @note 淘汰顺序由调用方决定，PagePool只提供LRU候选与页状态
    std::size_t leastRecentlyUsed() const;

    // @brief 页键命中累计次数
    // @return 命中次数，用于区分复用页与新算页
    std::size_t hitCount() const;

    // @brief 配置指纹
    // @return 构造页池时传入的指纹
    std::uint64_t fingerprint() const;

    // @brief 页池几何
    // @return 每页的行数、头数与time轴子层数
    const PageGeometry& geometry() const;

    // @brief 取一页用于写入
    // @return 页号，优先未占用页，否则回收引用计数为0的最久未用页。取到后引用计数为1
    // @note 没有可用页时返回capacity()，调用方必须按容量不足处理
    std::size_t acquire();

    // @brief 归还一次引用
    // @param index 页号，越界或页未占用时忽略
    // @note 引用计数归零后页保留内容，仍可按页键命中
    void release(std::size_t index);

    // @brief 调度器选中淘汰页后释放该页
    // @param index 页号
    // @note 只接受已占用且引用计数为0的页，越界或有引用时忽略
    void discard(std::size_t index);

    // @brief 按页键查询且不改变命中计数与引用计数
    // @param key 页键
    // @param parentKey 父页键，用于排除键冲突
    // @param inputFingerprint 输入指纹，用于排除键冲突
    // @return 页号，未命中时返回capacity()
    std::size_t peek(
        std::uint64_t key,
        std::uint64_t parentKey,
        std::uint64_t inputFingerprint
    ) const;

    // @brief 按页键查找可复用页
    // @param key 页键
    // @param parentKey 父页键，用于排除键冲突
    // @param inputFingerprint 输入指纹，用于排除键冲突
    // @return 页号，未命中时返回capacity()。命中即增加一次引用
    std::size_t find(std::uint64_t key, std::uint64_t parentKey, std::uint64_t inputFingerprint);

    // @brief 按页号取可写页
    // @param index 页号，须小于capacity()
    // @return 页引用，生命周期与页池一致
    // @throw std::out_of_range 页号越界时抛出
    CachePage& page(std::size_t index);

    // @brief 按页号取只读页
    // @param index 页号，须小于capacity()
    // @return 页的常量引用
    // @throw std::out_of_range 页号越界时抛出
    const CachePage& page(std::size_t index) const;

private:
    // @brief 标记为占用并清空元数据
    // @param index 页号
    // @note 首次使用时先分配K/V，成功后才登记占用状态
    void occupy(std::size_t index);

    PageGeometry shape;
    std::uint64_t configFingerprint = 0;
    std::vector<CachePage> pages;
    std::size_t occupied = 0;
    std::size_t hits = 0;
    std::uint64_t clock = 0;
};

} // namespace timestral
