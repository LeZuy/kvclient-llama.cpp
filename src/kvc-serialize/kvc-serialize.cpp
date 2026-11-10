#include "kvc-serialize.h"

#include "kvc-chunk.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

namespace kvclient::detail {
namespace {

constexpr uint8_t magic[] = { 'K', 'V', 'C', '1' };

void append_u32(std::vector<uint8_t> & output, uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t> & output, uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<uint8_t>(value >> shift));
    }
}

bool read_u32(const std::vector<uint8_t> & input, size_t & cursor, uint32_t & value) {
    if (input.size() - cursor < sizeof(uint32_t)) {
        return false;
    }

    value = 0;
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        value |= uint32_t(input[cursor++]) << shift;
    }
    return true;
}

bool read_u64(const std::vector<uint8_t> & input, size_t & cursor, uint64_t & value) {
    if (input.size() - cursor < sizeof(uint64_t)) {
        return false;
    }

    value = 0;
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        value |= uint64_t(input[cursor++]) << shift;
    }
    return true;
}

} // namespace

bool serialize(const kvc_chunk & chunk, std::vector<uint8_t> & data) {
    constexpr size_t header_size = sizeof(magic) + 5 * sizeof(uint32_t) + sizeof(uint64_t);
    if (!kvc_chunk_is_valid(chunk)
            || chunk.model_profile.size() > std::numeric_limits<uint32_t>::max()
            || chunk.content_hash.size() > std::numeric_limits<uint32_t>::max()
            || chunk.model_profile.size() > std::numeric_limits<size_t>::max() - header_size
            || chunk.content_hash.size() > std::numeric_limits<size_t>::max() - header_size - chunk.model_profile.size()
            || chunk.state_data.size() > std::numeric_limits<size_t>::max() - header_size
                    - chunk.model_profile.size() - chunk.content_hash.size()) {
        return false;
    }

    std::vector<uint8_t> result;
    result.reserve(header_size + chunk.model_profile.size()
            + chunk.content_hash.size() + chunk.state_data.size());
    result.insert(result.end(), std::begin(magic), std::end(magic));
    append_u32(result, chunk.format_version);
    append_u32(result, chunk.offset);
    append_u32(result, chunk.n_tokens);
    append_u32(result, static_cast<uint32_t>(chunk.model_profile.size()));
    append_u32(result, static_cast<uint32_t>(chunk.content_hash.size()));
    append_u64(result, chunk.state_data.size());
    result.insert(result.end(), chunk.model_profile.begin(), chunk.model_profile.end());
    result.insert(result.end(), chunk.content_hash.begin(), chunk.content_hash.end());
    result.insert(result.end(), chunk.state_data.begin(), chunk.state_data.end());

    data = std::move(result);
    return true;
}

bool deserialize(const std::vector<uint8_t> & data, kvc_chunk & chunk) {
    constexpr size_t header_size = sizeof(magic) + 5 * sizeof(uint32_t) + sizeof(uint64_t);
    if (data.size() < header_size
            || !std::equal(std::begin(magic), std::end(magic), data.begin())) {
        return false;
    }

    size_t cursor = sizeof(magic);
    kvc_chunk result;
    uint32_t profile_size = 0;
    uint32_t hash_size = 0;
    uint64_t state_size = 0;
    if (!read_u32(data, cursor, result.format_version)
            || !read_u32(data, cursor, result.offset)
            || !read_u32(data, cursor, result.n_tokens)
            || !read_u32(data, cursor, profile_size)
            || !read_u32(data, cursor, hash_size)
            || !read_u64(data, cursor, state_size)) {
        return false;
    }

    const uint64_t remaining = data.size() - cursor;
    if (profile_size > remaining || hash_size > remaining - profile_size
            || state_size != remaining - profile_size - hash_size
            || state_size > std::numeric_limits<size_t>::max()) {
        return false;
    }

    result.model_profile.assign(
            reinterpret_cast<const char *>(data.data() + cursor), profile_size);
    cursor += profile_size;
    result.content_hash.assign(
            reinterpret_cast<const char *>(data.data() + cursor), hash_size);
    cursor += hash_size;
    result.state_data.assign(data.begin() + cursor, data.end());

    if (!kvc_chunk_is_valid(result)) {
        return false;
    }

    chunk = std::move(result);
    return true;
}

} // namespace kvclient::detail
