/***************
 * @file json.cpp
 * @brief 严格JSON解析器：语法、字符串转义与数字原串。
 ***************/

#include "timestral/utils/json.hpp"

#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "timestral/utils/path.hpp"

namespace timestral::utils {

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text(text) {
    }

    Json parse() {
        Json result = parseValue(0);
        skipSpace();
        if (position != text.size()) {
            fail("trailing data");
        }
        return result;
    }

private:
    const std::string& text;
    std::size_t position = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error("JSON byte " + std::to_string(position) + ": " + message);
    }

    void skipSpace() {
        while (position < text.size()) {
            const char ch = text[position];
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
                break;
            }
            ++position;
        }
    }

    bool consume(char ch) {
        if (position < text.size() && text[position] == ch) {
            ++position;
            return true;
        }
        return false;
    }

    void expect(char ch) {
        if (!consume(ch)) {
            fail(std::string("expected '") + ch + "'");
        }
    }

    static bool isDigit(char ch) {
        return ch >= '0' && ch <= '9';
    }

    unsigned int hexUnit() {
        unsigned int unit = 0;
        for (int i = 0; i < 4; ++i) {
            if (position == text.size()) {
                fail("truncated Unicode escape");
            }
            const char ch = text[position++];
            unsigned int digit = 0;
            if (ch >= '0' && ch <= '9') {
                digit = static_cast<unsigned int>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                digit = static_cast<unsigned int>(ch - 'a' + 10);
            } else if (ch >= 'A' && ch <= 'F') {
                digit = static_cast<unsigned int>(ch - 'A' + 10);
            } else {
                fail("invalid Unicode escape");
            }
            unit = unit * 16 + digit;
        }
        return unit;
    }

    static void appendCodepoint(std::string& result, unsigned int codepoint) {
        if (codepoint <= 0x7f) {
            result.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            result.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            result.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            result.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    // 只接受最短编码，拒绝过长编码、代理区码点与截断序列，避免同一字符有多种字节表示。
    void appendUtf8(std::string& result) {
        const std::size_t start = position;
        const unsigned char lead = static_cast<unsigned char>(text[position++]);
        unsigned int codepoint = 0;
        unsigned int minimum = 0;
        int remaining = 0;
        if (lead >= 0xc2 && lead <= 0xdf) {
            codepoint = lead & 0x1f;
            minimum = 0x80;
            remaining = 1;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            codepoint = lead & 0x0f;
            minimum = 0x800;
            remaining = 2;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            codepoint = lead & 0x07;
            minimum = 0x10000;
            remaining = 3;
        } else {
            fail("invalid UTF-8 leading byte");
        }
        for (int i = 0; i < remaining; ++i) {
            if (position == text.size()) {
                fail("truncated UTF-8 sequence");
            }
            const unsigned char ch = static_cast<unsigned char>(text[position++]);
            if ((ch & 0xc0) != 0x80) {
                fail("invalid UTF-8 continuation");
            }
            codepoint = (codepoint << 6) | (ch & 0x3f);
        }
        if (codepoint < minimum || codepoint > 0x10ffff
            || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            fail("invalid UTF-8 codepoint");
        }
        result.append(text, start, position - start);
    }

    std::string parseString() {
        expect('"');
        std::string result;
        while (position < text.size()) {
            const unsigned char ch = static_cast<unsigned char>(text[position]);
            if (ch == '"') {
                ++position;
                return result;
            }
            if (ch < 0x20) {
                fail("unescaped control character");
            }
            if (ch >= 0x80) {
                appendUtf8(result);
                continue;
            }
            ++position;
            if (ch != '\\') {
                result.push_back(static_cast<char>(ch));
                continue;
            }
            if (position == text.size()) {
                fail("truncated escape");
            }
            const char escape = text[position++];
            switch (escape) {
                case '"':
                    result.push_back('"');
                    break;
                case '\\':
                    result.push_back('\\');
                    break;
                case '/':
                    result.push_back('/');
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                case 'u': {
                    unsigned int codepoint = hexUnit();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        expect('\\');
                        expect('u');
                        const unsigned int low = hexUnit();
                        if (low < 0xdc00 || low > 0xdfff) {
                            fail("invalid low surrogate");
                        }
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        fail("unpaired low surrogate");
                    }
                    appendCodepoint(result, codepoint);
                    break;
                }
                default:
                    fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    Json parseNumber() {
        const std::size_t start = position;
        consume('-');
        if (!consume('0')) {
            if (position == text.size() || text[position] < '1' || text[position] > '9') {
                fail("expected number digit");
            }
            while (position < text.size() && isDigit(text[position])) {
                ++position;
            }
        }
        if (consume('.')) {
            const std::size_t digits = position;
            while (position < text.size() && isDigit(text[position])) {
                ++position;
            }
            if (digits == position) {
                fail("missing fraction digits");
            }
        }
        if (consume('e') || consume('E')) {
            if (!consume('+')) {
                consume('-');
            }
            const std::size_t digits = position;
            while (position < text.size() && isDigit(text[position])) {
                ++position;
            }
            if (digits == position) {
                fail("missing exponent digits");
            }
        }
        Json result;
        result.value = Json::Number{text.substr(start, position - start)};
        return result;
    }

    Json parseValue(std::size_t depth) {
        skipSpace();
        if (position == text.size()) {
            fail("expected value");
        }
        Json result;
        const char ch = text[position];
        if (ch == '{' || ch == '[') {
            if (depth >= 256) {
                fail("container nesting exceeds 256");
            }
            ++position;
            skipSpace();
            if (ch == '{') {
                Json::Object object;
                if (!consume('}')) {
                    do {
                        skipSpace();
                        const std::string key = parseString();
                        skipSpace();
                        expect(':');
                        Json member = parseValue(depth + 1);
                        if (!object.emplace(key, std::move(member)).second) {
                            fail("duplicate object key: " + key);
                        }
                        skipSpace();
                    } while (consume(','));
                    expect('}');
                }
                result.value = std::move(object);
            } else {
                Json::Array array;
                if (!consume(']')) {
                    do {
                        array.push_back(parseValue(depth + 1));
                        skipSpace();
                    } while (consume(','));
                    expect(']');
                }
                result.value = std::move(array);
            }
        } else if (ch == '"') {
            result.value = parseString();
        } else if (ch == '-' || isDigit(ch)) {
            return parseNumber();
        } else {
            const std::string literal = ch == 't' ? "true" : ch == 'f' ? "false" : "null";
            if (text.compare(position, literal.size(), literal) != 0) {
                fail("invalid value");
            }
            position += literal.size();
            if (ch != 'n') {
                result.value = ch == 't';
            }
        }
        return result;
    }
};

Json Json::parse(const std::string& text) {
    return JsonParser(text).parse();
}

Json Json::load(const std::filesystem::path& path) {
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open file");
        }
        const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (input.bad()) {
            throw std::runtime_error("file read failed");
        }
        return parse(text);
    } catch (const std::exception& error) {
        throw std::runtime_error("JSON file " + pathText(path) + ": " + error.what());
    }
}

bool Json::isObject() const {
    return std::holds_alternative<Object>(value);
}

bool Json::isArray() const {
    return std::holds_alternative<Array>(value);
}

bool Json::isString() const {
    return std::holds_alternative<std::string>(value);
}

bool Json::isNumber() const {
    return std::holds_alternative<Number>(value);
}

bool Json::isBool() const {
    return std::holds_alternative<bool>(value);
}

bool Json::isNull() const {
    return std::holds_alternative<std::nullptr_t>(value);
}

const Json::Object& Json::asObject() const {
    if (!isObject()) {
        throw std::runtime_error("JSON: expected object");
    }
    return std::get<Object>(value);
}

const Json::Array& Json::asArray() const {
    if (!isArray()) {
        throw std::runtime_error("JSON: expected array");
    }
    return std::get<Array>(value);
}

const std::string& Json::asString() const {
    if (!isString()) {
        throw std::runtime_error("JSON: expected string");
    }
    return std::get<std::string>(value);
}

bool Json::asBool() const {
    if (!isBool()) {
        throw std::runtime_error("JSON: expected boolean");
    }
    return std::get<bool>(value);
}

double Json::asDouble() const {
    if (!isNumber()) {
        throw std::runtime_error("JSON: expected number for double conversion");
    }
    const std::string& text = std::get<Number>(value).text;
    std::istringstream input(text);
    input.imbue(std::locale::classic());
    double result = 0.0;
    input >> result;
    if (input.fail() || !input.eof() || !std::isfinite(result)) {
        throw std::runtime_error("JSON number outside double range: " + text);
    }
    return result;
}

std::size_t Json::asSize() const {
    if (!isNumber()) {
        throw std::runtime_error("JSON: expected number for size conversion");
    }
    const std::string& text = std::get<Number>(value).text;
    std::size_t result = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            throw std::runtime_error("JSON: expected nonnegative integer, got " + text);
        }
        const std::size_t digit = static_cast<std::size_t>(ch - '0');
        if (result > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            throw std::runtime_error("JSON integer exceeds size_t: " + text);
        }
        result = result * 10 + digit;
    }
    return result;
}

const Json& Json::at(const std::string& key) const {
    if (!isObject()) {
        throw std::runtime_error("JSON key " + key + ": expected object");
    }
    const auto found = asObject().find(key);
    if (found == asObject().end()) {
        throw std::runtime_error("JSON: missing key " + key);
    }
    return found->second;
}

} // namespace timestral::utils
