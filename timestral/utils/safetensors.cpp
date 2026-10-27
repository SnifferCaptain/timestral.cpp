/***************
 * @file safetensors.cpp
 * @brief safetensors头部解析、数据区校验与小端F32解码。
 ***************/

#include "timestral/utils/safetensors.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "timestral/utils/json.hpp"
#include "timestral/utils/path.hpp"

namespace timestral::utils {

// 解析期数据区区间；只在Safetensors::load内使用。
struct Safetensors::TensorRange {
    std::string name;
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t count = 0;
};

// 读满bytes个字节写入destination起始处；EOF或短读抛错。
// 元素按字节读入，由调用方决定如何解释，不依赖主机字节序。
template <typename Container>
static void readBytes(std::ifstream& input, Container& destination, std::size_t bytes) {
    char* cursor = reinterpret_cast<char*>(destination.data());
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
    while (bytes > 0) {
        const std::size_t chunk = std::min(bytes, maximum);
        input.read(cursor, static_cast<std::streamsize>(chunk));
        if (!input || input.gcount() != static_cast<std::streamsize>(chunk)) {
            throw std::runtime_error("truncated file or read failure");
        }
        cursor += chunk;
        bytes -= chunk;
    }
}

template <typename Container>
static void readExact(std::ifstream& input, Container& destination) {
    readBytes(input, destination, destination.size() * sizeof(destination[0]));
}

static std::uint64_t decodeLength(const std::array<unsigned char, 8>& bytes) {
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        result |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return result;
}

Safetensors Safetensors::load(const std::filesystem::path& path) {
    static_assert(
        sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
        "safetensors F32 requires IEEE 754 binary32"
    );
    try {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            throw std::runtime_error("cannot open file");
        }
        const std::streamoff fileLength = input.tellg();
        if (fileLength < 8) {
            throw std::runtime_error("truncated 8-byte length prefix");
        }
        if (static_cast<std::uintmax_t>(fileLength) > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("file size exceeds size_t");
        }
        const std::size_t fileSize = static_cast<std::size_t>(fileLength);
        input.seekg(0);
        std::array<unsigned char, 8> prefix{};
        readExact(input, prefix);
        const std::uint64_t headerLength = decodeLength(prefix);
        if (headerLength == 0 || headerLength > 100000000 || headerLength > fileSize - 8) {
            throw std::runtime_error("invalid or truncated header length " + std::to_string(headerLength));
        }
        std::string header(static_cast<std::size_t>(headerLength), '\0');
        readExact(input, header);
        if (header.front() != '{') {
            throw std::runtime_error("header must begin with '{'");
        }
        const Json root = Json::parse(header);
        const std::size_t dataSize = fileSize - 8 - header.size();
        Safetensors result;
        std::vector<TensorRange> ranges;
        for (const auto& entry : root.asObject()) {
            const std::string& name = entry.first;
            try {
                if (name == "__metadata__") {
                    for (const auto& metadata : entry.second.asObject()) {
                        if (!metadata.second.isString()) {
                            throw std::runtime_error("metadata key " + metadata.first + " must be a string");
                        }
                    }
                    continue;
                }
                const auto& descriptor = entry.second.asObject();
                if (descriptor.size() != 3) {
                    throw std::runtime_error("expected exactly dtype, shape and data_offsets fields");
                }
                const std::string& dtype = entry.second.at("dtype").asString();
                if (dtype != "F32") {
                    throw std::runtime_error("unsupported dtype " + dtype + "; only F32 is supported");
                }
                Weight weight;
                bool empty = false;
                for (const Json& dimension : entry.second.at("shape").asArray()) {
                    const std::size_t size = dimension.asSize();
                    weight.shape.push_back(size);
                    empty = empty || size == 0;
                }
                std::size_t count = empty ? 0 : 1;
                if (!empty) {
                    for (std::size_t dimension : weight.shape) {
                        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
                            throw std::runtime_error("shape element count overflow");
                        }
                        count *= dimension;
                    }
                }
                if (count > std::numeric_limits<std::size_t>::max() / 4) {
                    throw std::runtime_error("F32 byte count overflow");
                }
                const auto& offsets = entry.second.at("data_offsets").asArray();
                if (offsets.size() != 2) {
                    throw std::runtime_error("data_offsets must have two integers");
                }
                const std::size_t begin = offsets[0].asSize();
                const std::size_t end = offsets[1].asSize();
                if (begin > end || end > dataSize) {
                    throw std::runtime_error("data_offsets out of data section bounds");
                }
                if (end - begin != count * 4) {
                    throw std::runtime_error("shape byte count does not match data_offsets");
                }
                ranges.push_back(TensorRange{name, begin, end, count});
                result.weights.emplace(name, std::move(weight));
            } catch (const std::exception& error) {
                throw std::runtime_error("tensor " + name + ": " + error.what());
            }
        }
        std::sort(ranges.begin(), ranges.end(), [](const TensorRange& left, const TensorRange& right) {
            if (left.begin != right.begin) {
                return left.begin < right.begin;
            }
            return left.end < right.end;
        });
        std::size_t cursor = 0;
        for (const TensorRange& range : ranges) {
            if (range.begin != cursor) {
                throw std::runtime_error("tensor " + range.name + ": overlapping or noncontiguous data_offsets");
            }
            cursor = range.end;
        }
        if (cursor != dataSize) {
            throw std::runtime_error("data section has unclaimed trailing bytes");
        }

        // 按数据偏移顺序读取，并逐值显式解码小端，不依赖主机字节序或对齐。
        for (const TensorRange& range : ranges) {
            try {
                auto& values = result.weights.at(range.name).values;
                values.resize(range.count);
                std::array<unsigned char, 65536> buffer{};
                std::size_t index = 0;
                while (index < values.size()) {
                    const std::size_t count = std::min(values.size() - index, buffer.size() / 4);
                    readBytes(input, buffer, count * 4);
                    for (std::size_t i = 0; i < count; ++i) {
                        const std::size_t offset = i * 4;
                        const std::uint32_t bits = static_cast<std::uint32_t>(buffer[offset])
                            | (static_cast<std::uint32_t>(buffer[offset + 1]) << 8)
                            | (static_cast<std::uint32_t>(buffer[offset + 2]) << 16)
                            | (static_cast<std::uint32_t>(buffer[offset + 3]) << 24);
                        std::memcpy(&values[index + i], &bits, sizeof(bits));
                    }
                    index += count;
                }
            } catch (const std::exception& error) {
                throw std::runtime_error("tensor " + range.name + ": " + error.what());
            }
        }
        if (input.peek() != std::char_traits<char>::eof() || input.bad()) {
            throw std::runtime_error("file changed during read or trailing data");
        }
        return result;
    } catch (const std::exception& error) {
        throw std::runtime_error("safetensors file " + pathText(path) + ": " + error.what());
    }
}

const std::map<std::string, Weight>& Safetensors::tensors() const {
    return weights;
}

const Weight& Safetensors::at(const std::string& name) const {
    const auto found = weights.find(name);
    if (found == weights.end()) {
        throw std::runtime_error("safetensors: missing tensor " + name);
    }
    return found->second;
}

} // namespace timestral::utils
