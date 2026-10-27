/***************
 * @file config.hpp
 * @brief 上游模型配置读取及推理约束校验。
 ***************/

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace timestral {

namespace utils {
class Json;
}

// 上游config.json的字段；名称与默认值保持上游拼写。
struct ModelConfig {
    std::size_t patchSize = 0; // 每个patch的采样点数
    std::size_t modelDim = 0; // 隐藏维d_model
    std::size_t headCount = 0; // 注意力头数
    std::size_t mlpHiddenDim = 0; // MLP中间维
    std::size_t blockSize = 0; // 单次前向最多处理的patch数
    std::size_t contextLength = 0; // 训练上下文长度，按步计
    std::vector<std::string> blockSchedule; // 每层注意力轴顺序
    std::vector<float> quantileKnots; // 分位点，升序且含0.5
    float ropeTheta = 10000.0f; // RoPE基频
    float ropePartialFactor = 0.5f; // 参与旋转的头维比例
    float normEps = 1e-6f; // RMSNorm的epsilon
    float minimumScale = 1e-6f; // 因果归一化尺度下界
    float quantileWidthFactor = 2.0f; // 分位数头宽度系数
    bool attnBias = false; // 注意力投影是否带偏置
    bool mlpBias = false; // MLP投影是否带偏置
    bool useAsinh = true; // 归一化后是否取asinh
    bool stabilizeWithGlobal = false; // 上游全局稳定选项，本实现要求false
    bool covariateEnabled = true; // checkpoint是否支持协变量
    bool splitContextHorizonAttn = true; // 上下文与horizon的time轴注意力是否分开
    bool attendMissingPatches = true; // 缺失patch能否被后续槽关注
    std::string timeAttnGate = "sdpa_sigmoid"; // time轴门控类型：none或sdpa_sigmoid
    std::string variateAttnGate = "sdpa_sigmoid"; // variate轴门控类型：none或sdpa_sigmoid

    // @brief 读取并校验config.json。
    // @param path 配置文件路径。
    // @return 校验后的配置。
    // @note 只接受固定上游字段，未知字段与缺失必填项均视为错误。
    // @throw std::runtime_error 文件不可读、字段缺失或取值非法时抛出。
    static ModelConfig load(std::filesystem::path path);

    // @brief 从已解析的JSON构造配置。
    // @param json 配置对象。
    // @return 校验后的配置。
    // @throw std::invalid_argument 字段未知、缺失或类型不符时抛出。
    static ModelConfig fromJson(const utils::Json& json);

    // @brief 校验维度、RoPE、分位数及上游不支持的全局稳定选项。
    // @throw std::invalid_argument 任一约束不满足时抛出。
    void validate() const;

    // @brief 单头维度。
    // @return modelDim / headCount。
    // @throw std::invalid_argument 维度为零或不能整除时抛出。
    std::size_t headDim() const;

    // @brief 参与RoPE的维度。
    // @return headDim与ropePartialFactor乘积向下取整，偶数且非零。
    // @throw std::invalid_argument 比例或结果非法时抛出。
    std::size_t ropeDim() const;

    // @brief 中位数在分位点中的下标。
    // @return quantileKnots中0.5的位置。
    // @throw std::invalid_argument 分位点不含0.5时抛出。
    std::size_t medianIndex() const;

    // @brief 是否支持known-future协变量。
    // @return 协变量启用且某层time轴注意力能看到更晚variate轴时返回true。
    bool supportsKnownCovariates() const;
};

} // namespace timestral
