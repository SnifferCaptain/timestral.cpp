/***************
 * @file json.hpp
 * @brief 严格JSON解析及无精度损失的整数读取。
 ***************/

#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace timestral::utils {

// 严格JSON值；数字保留原串，整数转换不经浮点。
class Json {
public:
    using Object = std::map<std::string, Json>;
    using Array = std::vector<Json>;

    Json() = default;

    // @brief 解析一段JSON文本。
    // @param text 完整文本，必须恰好包含一个值。
    // @return 解析后的值。
    // @note 容器最多嵌套256层；语法错误包含字节位置。
    // @throw std::runtime_error 语法非法、重复键、控制字符或编码错误时抛出。
    static Json parse(const std::string& text);

    // @brief 读取并解析本地JSON文件。
    // @param path 文件路径。
    // @return 解析后的值。
    // @throw std::runtime_error 文件不可读或内容非法时抛出，消息另含文件路径。
    static Json load(const std::filesystem::path& path);

    // @brief 判断是否为对象。
    // @return 是对象时返回true。
    bool isObject() const;

    // @brief 判断是否为数组。
    // @return 是数组时返回true。
    bool isArray() const;

    // @brief 判断是否为字符串。
    // @return 是字符串时返回true。
    bool isString() const;

    // @brief 判断是否为数字。
    // @return 是数字时返回true。
    bool isNumber() const;

    // @brief 判断是否为布尔值。
    // @return 是布尔值时返回true。
    bool isBool() const;

    // @brief 判断是否为null。
    // @return 是null时返回true。
    bool isNull() const;

    // @brief 取对象内容。
    // @return 当前值持有的对象引用。
    // @throw std::runtime_error 当前值不是对象时抛出。
    const Object& asObject() const;

    // @brief 取数组内容。
    // @return 当前值持有的数组引用。
    // @throw std::runtime_error 当前值不是数组时抛出。
    const Array& asArray() const;

    // @brief 取字符串内容。
    // @return 当前值持有的字符串引用。
    // @throw std::runtime_error 当前值不是字符串时抛出。
    const std::string& asString() const;

    // @brief 取布尔值。
    // @return 当前值持有的布尔量。
    // @throw std::runtime_error 当前值不是布尔值时抛出。
    bool asBool() const;

    // @brief 把数字转成double。
    // @return 数值。
    // @note 拒绝超出有限double范围的数值。
    // @throw std::runtime_error 当前值不是数字或数值溢出时抛出。
    double asDouble() const;

    // @brief 把数字转成无符号整数。
    // @return 整数数值。
    // @note 仅接受非负十进制整数原串，不接受负号、小数或指数，并检查size_t溢出。
    // @throw std::runtime_error 原串不是非负整数或超过size_t时抛出。
    std::size_t asSize() const;

    // @brief 按键取对象成员。
    // @param key 对象键。
    // @return 对应值，生命周期与当前Json一致。
    // @throw std::runtime_error 当前值不是对象或键不存在时抛出。
    const Json& at(const std::string& key) const;

private:
    // 数字保留原串，asDouble与asSize按需转换。
    struct Number {
        std::string text;
    };

    std::variant<std::nullptr_t, bool, std::string, Number, Object, Array> value = nullptr;

    friend class JsonParser;
};

} // namespace timestral::utils
