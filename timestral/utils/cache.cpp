/***************
 * @file cache.cpp
 * @brief 分页前缀缓存：FNV哈希、页池查找与LRU回收
 ***************/

#include "timestral/utils/cache.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include "timestral/utils/tensor.hpp"

namespace timestral {

// FNV-1a 64位参数（offset basis与prime）
static constexpr std::uint64_t fnvOffsetBasis = 0xcbf29ce484222325ULL;
static constexpr std::uint64_t fnvPrime = 0x100000001b3ULL;

FnvHash::FnvHash(std::uint64_t seed)
    : state(seed) {
}

void FnvHash::addBytes(const std::vector<std::uint8_t>& bytes) {
    for (std::uint8_t byte : bytes) {
        state ^= static_cast<std::uint64_t>(byte);
        state *= fnvPrime;
    }
}

void FnvHash::addValues(const std::vector<float>& values) {
    std::uint8_t raw[sizeof(float)] = {0, 0, 0, 0};
    for (float value : values) {
        std::memcpy(raw, &value, sizeof(float));
        for (std::uint8_t byte : raw) {
            state ^= static_cast<std::uint64_t>(byte);
            state *= fnvPrime;
        }
    }
}

void FnvHash::addString(const std::string& text) {
    for (char character : text) {
        state ^= static_cast<std::uint64_t>(static_cast<std::uint8_t>(character));
        state *= fnvPrime;
    }
}

void FnvHash::addUnsigned(std::uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) {
        state ^= (value >> (8 * i)) & 0xffULL;
        state *= fnvPrime;
    }
}

void FnvHash::addSigned(std::int64_t value) {
    addUnsigned(static_cast<std::uint64_t>(value));
}

void FnvHash::addPosition(double position) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(position), "FP64 coordinates require 64-bit double");
    std::memcpy(&bits, &position, sizeof(bits));
    addUnsigned(bits);
}

std::uint64_t FnvHash::value() const {
    return state;
}

// 只覆盖决定time轴K/V与归一化的字段，分位数后处理参数不参与页复用
std::uint64_t modelFingerprint(const ModelConfig& config) {
    FnvHash hash(fnvOffsetBasis);
    hash.addUnsigned(config.patchSize);
    hash.addUnsigned(config.modelDim);
    hash.addUnsigned(config.headCount);
    hash.addUnsigned(config.mlpHiddenDim);
    hash.addUnsigned(config.headDim());
    hash.addUnsigned(config.ropeDim());
    hash.addValues({config.ropeTheta});
    hash.addValues({config.normEps});
    hash.addValues({config.minimumScale});
    hash.addUnsigned(config.useAsinh ? 1 : 0);
    hash.addUnsigned(config.attnBias ? 1 : 0);
    hash.addUnsigned(config.mlpBias ? 1 : 0);
    hash.addUnsigned(config.attendMissingPatches ? 1 : 0);
    hash.addUnsigned(config.covariateEnabled ? 1 : 0);
    hash.addString(config.timeAttnGate);
    hash.addString(config.variateAttnGate);
    for (const std::string& kind : config.blockSchedule) {
        hash.addString(kind);
    }
    return hash.value();
}

PagePool::PagePool(PageGeometry geometry, std::uint64_t fingerprint, std::size_t capacity)
    : shape(geometry),
      configFingerprint(fingerprint),
      pages(capacity) {
    elementCount({shape.rowCount, shape.headCount, shape.headDim});
    elementCount({shape.rowCount, shape.headCount, shape.headDim, shape.timeLayers});
    // K/V缓冲在页首次占用时分配，容量只是上限，未用到的页不占内存
    for (CachePage& entry : pages) {
        entry.maskBits.assign(shape.rowCount, 0);
        entry.positions.assign(shape.rowCount, 0);
        entry.scaleStates.assign(shape.rowCount, PageScaleState{});
    }
}

std::size_t PagePool::capacity() const {
    return pages.size();
}

std::size_t PagePool::size() const {
    return occupied;
}

std::size_t PagePool::cachedPages() const {
    std::size_t count = 0;
    for (const CachePage& entry : pages) {
        count += entry.occupied && entry.references == 0 ? 1 : 0;
    }
    return count;
}

std::size_t PagePool::freePages() const {
    std::size_t count = 0;
    for (const CachePage& entry : pages) {
        count += entry.occupied ? 0 : 1;
    }
    return count;
}

std::size_t PagePool::reclaimablePages() const {
    return cachedPages();
}

std::size_t PagePool::leastRecentlyUsed() const {
    std::size_t victim = pages.size();
    for (std::size_t i = 0; i < pages.size(); ++i) {
        if (!pages[i].occupied || pages[i].references != 0) {
            continue;
        }
        if (victim == pages.size() || pages[i].lastUse < pages[victim].lastUse) {
            victim = i;
        }
    }
    return victim;
}

std::size_t PagePool::hitCount() const {
    return hits;
}

std::uint64_t PagePool::fingerprint() const {
    return configFingerprint;
}

const PageGeometry& PagePool::geometry() const {
    return shape;
}

std::size_t PagePool::acquire() {
    for (std::size_t i = 0; i < pages.size(); ++i) {
        if (!pages[i].occupied) {
            occupy(i);
            ++occupied;
            return i;
        }
    }
    const std::size_t victim = leastRecentlyUsed();
    if (victim == pages.size()) {
        return pages.size();
    }
    occupy(victim);
    return victim;
}

void PagePool::release(std::size_t index) {
    if (index >= pages.size()) {
        return;
    }
    CachePage& entry = pages[index];
    if (!entry.occupied || entry.references == 0) {
        return;
    }
    --entry.references;
    if (entry.references == 0) {
        // 刚归还的页仍是最新使用过的，LRU应最后回收它
        entry.lastUse = ++clock;
    }
}

void PagePool::discard(std::size_t index) {
    if (index >= pages.size()) {
        return;
    }
    CachePage& entry = pages[index];
    if (!entry.occupied || entry.references != 0) {
        return;
    }
    entry.occupied = false;
    entry.parentKey = 0;
    entry.key = 0;
    entry.inputFingerprint = 0;
    --occupied;
}

std::size_t PagePool::peek(
    std::uint64_t key,
    std::uint64_t parentKey,
    std::uint64_t inputFingerprint
) const {
    for (std::size_t i = 0; i < pages.size(); ++i) {
        const CachePage& entry = pages[i];
        if (!entry.occupied || entry.key != key) {
            continue;
        }
        if (entry.parentKey != parentKey || entry.inputFingerprint != inputFingerprint) {
            return pages.size();
        }
        return i;
    }
    return pages.size();
}

std::size_t PagePool::find(std::uint64_t key, std::uint64_t parentKey, std::uint64_t inputFingerprint) {
    for (std::size_t i = 0; i < pages.size(); ++i) {
        CachePage& entry = pages[i];
        if (!entry.occupied || entry.key != key) {
            continue;
        }
        // 64位键可能冲突：父键与输入指纹不一致时按未命中处理，绝不复用别的页
        if (entry.parentKey != parentKey || entry.inputFingerprint != inputFingerprint) {
            return pages.size();
        }
        ++hits;
        ++entry.references;
        entry.lastUse = ++clock;
        return i;
    }
    return pages.size();
}

CachePage& PagePool::page(std::size_t index) {
    return pages.at(index);
}

const CachePage& PagePool::page(std::size_t index) const {
    return pages.at(index);
}

void PagePool::occupy(std::size_t index) {
    CachePage& entry = pages[index];
    if (entry.keys.size() != shape.timeLayers) {
        const std::size_t slotSize = elementCount({shape.rowCount, shape.headCount, shape.headDim});
        std::vector<std::vector<float>> keys(shape.timeLayers, std::vector<float>(slotSize, 0.0f));
        std::vector<std::vector<float>> values(shape.timeLayers, std::vector<float>(slotSize, 0.0f));
        entry.keys = std::move(keys);
        entry.values = std::move(values);
    } else {
        for (std::size_t layer = 0; layer < entry.keys.size(); ++layer) {
            std::fill(entry.keys[layer].begin(), entry.keys[layer].end(), 0.0f);
            std::fill(entry.values[layer].begin(), entry.values[layer].end(), 0.0f);
        }
    }
    entry.maskBits.assign(shape.rowCount, 0);
    entry.positions.assign(shape.rowCount, 0);
    entry.scaleStates.assign(shape.rowCount, PageScaleState{});
    entry.occupied = true;
    entry.references = 1;
    entry.lastUse = ++clock;
    entry.parentKey = 0;
    entry.key = 0;
    entry.inputFingerprint = 0;
}

} // namespace timestral
