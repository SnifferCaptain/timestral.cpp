/***************
 * @file safetensors.hpp
 * @brief 本地safetensors的F32权重读取及边界校验。
 ***************/

#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace timestral::utils {

// 单个张量：形状为各维长度，空表示标量。
struct Weight {
    std::vector<std::size_t> shape;
    std::vector<float> values; // 连续行主序，自有存储
};

// 只读的本地safetensors权重集合，张量名保留文件中的原名。
class Safetensors {
public:
    // @brief 读取文件中的全部F32张量。
    // @param path safetensors文件路径。
    // @return 权重集合。
    // @note 仅支持IEEE 754 F32；头部上限100MB，数据区必须连续且无覆盖；元数据仅允许字符串对象。
    // @throw std::runtime_error 文件缺失、头部非法、偏移越界或dtype不支持时抛出，消息含路径与张量名。
    static Safetensors load(const std::filesystem::path& path);

    // @brief 全部张量。
    // @return 名称到权重的映射，按名称字典序。
    const std::map<std::string, Weight>& tensors() const;

    // @brief 按名称取张量。
    // @param name 文件中的张量原名。
    // @return 对应权重，生命周期与本对象一致。
    // @throw std::runtime_error 名称不存在时抛出。
    const Weight& at(const std::string& name) const;

private:
    // 数据区区间：名称、字节偏移与元素数，只在load的解析阶段使用。
    struct TensorRange;
    std::map<std::string, Weight> weights;
};

} // namespace timestral::utils
