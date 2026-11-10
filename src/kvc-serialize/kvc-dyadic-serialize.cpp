#include "kvc-dyadic-serialize.h"

#include "kvc-dyadic-internal.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

namespace kvclient::detail {
namespace {

constexpr uint8_t magic[] = { 'K', 'V', 'D', '1' };

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
    if (cursor > input.size() || input.size() - cursor < sizeof(uint32_t)) {
        return false;
    }
    value = 0;
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        value |= uint32_t(input[cursor++]) << shift;
    }
    return true;
}

bool read_u64(const std::vector<uint8_t> & input, size_t & cursor, uint64_t & value) {
    if (cursor > input.size() || input.size() - cursor < sizeof(uint64_t)) {
        return false;
    }
    value = 0;
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        value |= uint64_t(input[cursor++]) << shift;
    }
    return true;
}

bool append_string(std::vector<uint8_t> & output, const std::string & value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    append_u32(output, static_cast<uint32_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
    return true;
}

void append_bytes(std::vector<uint8_t> & output, const std::vector<uint8_t> & value) {
    append_u64(output, value.size());
    output.insert(output.end(), value.begin(), value.end());
}

bool read_string(const std::vector<uint8_t> & input, size_t & cursor, std::string & value) {
    uint32_t size = 0;
    if (!read_u32(input, cursor, size) || cursor > input.size() || size > input.size() - cursor) {
        return false;
    }
    value.assign(reinterpret_cast<const char *>(input.data() + cursor), size);
    cursor += size;
    return true;
}

bool read_bytes(const std::vector<uint8_t> & input, size_t & cursor, std::vector<uint8_t> & value) {
    uint64_t size = 0;
    if (!read_u64(input, cursor, size) || cursor > input.size()
            || size > input.size() - cursor || size > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value.assign(input.begin() + cursor, input.begin() + cursor + static_cast<size_t>(size));
    cursor += static_cast<size_t>(size);
    return true;
}

} // namespace

bool serialize_dyadic(const kvc_dyadic_chunk & chunk, std::vector<uint8_t> & data) {
    if (!kvc_dyadic_chunk_is_valid(chunk)
            || chunk.layers.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    std::vector<uint8_t> result;
    result.insert(result.end(), std::begin(magic), std::end(magic));
    append_u32(result, chunk.format_version);
    append_u32(result, chunk.offset);
    append_u32(result, chunk.n_tokens);
    append_u32(result, chunk.level);
    append_u64(result, chunk.index);
    if (!append_string(result, chunk.model_profile)
            || !append_string(result, chunk.prefix_hash)
            || !append_string(result, chunk.content_hash)
            || !append_string(result, chunk.payload_hash)) {
        return false;
    }
    append_u32(result, static_cast<uint32_t>(chunk.token_ids.size()));
    for (const int32_t token : chunk.token_ids) {
        append_u32(result, static_cast<uint32_t>(token));
    }
    append_u32(result, static_cast<uint32_t>(chunk.layers.size()));
    for (const auto & layer : chunk.layers) {
        append_u32(result, layer.layer_idx);
        if (!append_string(result, layer.type_k) || !append_string(result, layer.type_v)) {
            return false;
        }
        append_bytes(result, layer.k_data);
        append_bytes(result, layer.v_data);
    }

    data = std::move(result);
    return true;
}

bool deserialize_dyadic(const std::vector<uint8_t> & data, kvc_dyadic_chunk & chunk) {
    if (data.size() < sizeof(magic)
            || !std::equal(std::begin(magic), std::end(magic), data.begin())) {
        return false;
    }

    size_t cursor = sizeof(magic);
    kvc_dyadic_chunk result;
    uint32_t token_count = 0;
    if (!read_u32(data, cursor, result.format_version)
            || !read_u32(data, cursor, result.offset)
            || !read_u32(data, cursor, result.n_tokens)
            || !read_u32(data, cursor, result.level)
            || !read_u64(data, cursor, result.index)
            || !read_string(data, cursor, result.model_profile)
            || !read_string(data, cursor, result.prefix_hash)
            || !read_string(data, cursor, result.content_hash)
            || !read_string(data, cursor, result.payload_hash)
            || !read_u32(data, cursor, token_count)
            || token_count > (data.size() - cursor) / sizeof(uint32_t)) {
        return false;
    }

    result.token_ids.reserve(token_count);
    for (uint32_t i = 0; i < token_count; ++i) {
        uint32_t token = 0;
        if (!read_u32(data, cursor, token)) {
            return false;
        }
        result.token_ids.push_back(static_cast<int32_t>(token));
    }
    uint32_t n_layers = 0;
    if (!read_u32(data, cursor, n_layers) || n_layers > (data.size() - cursor) / 28) {
        return false;
    }
    result.layers.reserve(n_layers);
    for (uint32_t i = 0; i < n_layers; ++i) {
        kvc_dyadic_layer layer;
        if (!read_u32(data, cursor, layer.layer_idx)
                || !read_string(data, cursor, layer.type_k)
                || !read_string(data, cursor, layer.type_v)
                || !read_bytes(data, cursor, layer.k_data)
                || !read_bytes(data, cursor, layer.v_data)) {
            return false;
        }
        result.layers.push_back(std::move(layer));
    }

    if (cursor != data.size() || !kvc_dyadic_chunk_is_valid(result)) {
        return false;
    }
    chunk = std::move(result);
    return true;
}

} // namespace kvclient::detail
