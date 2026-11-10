#include "kvclient/kvc-dyadic.h"

#include "kvc-chunk.h"
#include "kvc-dyadic-internal.h"

#include <limits>
#include <set>

namespace {

void append_u32(std::vector<uint8_t> & data, uint32_t value) {
    for (unsigned int i = 0; i < 4; ++i) {
        data.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
}

void append_u64(std::vector<uint8_t> & data, uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) {
        data.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
}

void append_string(std::vector<uint8_t> & data, const std::string & value) {
    append_u64(data, value.size());
    data.insert(data.end(), value.begin(), value.end());
}

void append_bytes(std::vector<uint8_t> & data, const std::vector<uint8_t> & value) {
    append_u64(data, value.size());
    data.insert(data.end(), value.begin(), value.end());
}

} // namespace

namespace kvclient {

std::vector<kvc_dyadic_interval> kvc_dyadic_partition(uint32_t offset, uint32_t n_tokens) {
    std::vector<kvc_dyadic_interval> result;
    if (n_tokens == 0 || uint64_t(offset) + n_tokens > uint64_t(std::numeric_limits<int32_t>::max()) + 1) {
        return result;
    }

    uint32_t consumed = 0;
    while (consumed < n_tokens) {
        const uint32_t remaining = n_tokens - consumed;
        uint32_t size = 1;
        while (size <= remaining / 2) {
            size *= 2;
        }

        uint32_t level = 0;
        for (uint32_t value = size; value > 1; value >>= 1) {
            ++level;
        }

        result.push_back({
            offset + consumed,
            size,
            level,
            consumed / size,
        });
        consumed += size;
    }

    return result;
}

std::vector<kvc_dyadic_cache_key> kvc_dyadic_cache_keys(
        const std::string & model_profile,
        const std::vector<int32_t> & token_ids) {
    std::vector<kvc_dyadic_cache_key> result;
    if (model_profile.empty() || token_ids.empty()
            || token_ids.size() > std::numeric_limits<uint32_t>::max()) {
        return result;
    }

    const auto intervals = kvc_dyadic_partition(0, static_cast<uint32_t>(token_ids.size()));
    result.reserve(intervals.size());
    std::string previous_hash = KVC_DYADIC_HASH_SEED;
    for (const auto & interval : intervals) {
        kvc_dyadic_chunk chunk;
        chunk.offset = interval.offset;
        chunk.n_tokens = interval.n_tokens;
        chunk.level = interval.level;
        chunk.index = interval.index;
        chunk.model_profile = model_profile;
        chunk.prefix_hash = previous_hash;
        chunk.token_ids.assign(token_ids.begin() + interval.offset,
                token_ids.begin() + interval.offset + interval.n_tokens);
        chunk.content_hash = detail::kvc_dyadic_content_hash(chunk);
        result.push_back({ interval, previous_hash, chunk.content_hash });
        previous_hash = chunk.content_hash;
    }
    return result;
}

} // namespace kvclient

namespace kvclient::detail {

std::string kvc_dyadic_content_hash(const kvc_dyadic_chunk & chunk) {
    std::vector<uint8_t> data;
    append_string(data, chunk.prefix_hash);
    append_u32(data, chunk.level);
    append_u64(data, chunk.index);
    append_string(data, chunk.model_profile);
    append_u64(data, chunk.token_ids.size());
    for (const int32_t token : chunk.token_ids) {
        append_u32(data, static_cast<uint32_t>(token));
    }
    return kvc_hash_state(data);
}

std::string kvc_dyadic_payload_hash(const kvc_dyadic_chunk & chunk) {
    std::vector<uint8_t> data;
    append_u64(data, chunk.layers.size());
    for (const auto & layer : chunk.layers) {
        append_u32(data, layer.layer_idx);
        append_string(data, layer.type_k);
        append_string(data, layer.type_v);
        append_bytes(data, layer.k_data);
        append_bytes(data, layer.v_data);
    }
    return kvc_hash_state(data);
}

bool kvc_dyadic_chunk_is_valid(const kvc_dyadic_chunk & chunk) {
    if (chunk.level >= 32 || (uint64_t(1) << chunk.level) != chunk.n_tokens
            || uint64_t(chunk.offset) + chunk.n_tokens > uint64_t(std::numeric_limits<int32_t>::max()) + 1) {
        return false;
    }
    if (chunk.format_version != KVC_DYADIC_FORMAT_VERSION || chunk.n_tokens == 0
            || chunk.token_ids.size() != chunk.n_tokens || chunk.model_profile.empty()
            || chunk.prefix_hash.empty() || chunk.content_hash.empty() || chunk.payload_hash.empty()
            || chunk.layers.empty()) {
        return false;
    }
    std::set<uint32_t> layer_ids;
    for (const auto & layer : chunk.layers) {
        if (layer.type_k.empty() || layer.type_v.empty() || layer.k_data.empty()
                || layer.v_data.empty() || !layer_ids.insert(layer.layer_idx).second) {
            return false;
        }
    }
    return chunk.content_hash == kvc_dyadic_content_hash(chunk)
            && chunk.payload_hash == kvc_dyadic_payload_hash(chunk);
}

} // namespace kvclient::detail
