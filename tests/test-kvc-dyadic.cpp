#include "kvclient/kvc-dyadic.h"

#include "kvc-dyadic-internal.h"
#include "kvc-dyadic-serialize.h"

#include <iostream>

namespace {

bool expect(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

} // namespace

int main() {
    using kvclient::kvc_dyadic_partition;

    const auto intervals = kvc_dyadic_partition(0, 524);
    if (!expect(intervals.size() == 3, "524 tokens should produce three dyadic intervals")
            || !expect(intervals[0].offset == 0 && intervals[0].n_tokens == 512
                    && intervals[0].level == 9 && intervals[0].index == 0,
                    "first 524-token interval should cover [1, 512]")
            || !expect(intervals[1].offset == 512 && intervals[1].n_tokens == 8
                    && intervals[1].level == 3 && intervals[1].index == 64,
                    "second 524-token interval should cover [513, 520]")
            || !expect(intervals[2].offset == 520 && intervals[2].n_tokens == 4
                    && intervals[2].level == 2 && intervals[2].index == 130,
                    "third 524-token interval should cover [521, 524]")) {
        return 1;
    }

    if (!expect(kvc_dyadic_partition(0, 0).empty(), "zero-length ranges must be rejected")
            || !expect(kvc_dyadic_partition(0, 1).size() == 1, "one token should be one level-zero interval")
            || !expect(kvc_dyadic_partition(0, 512).size() == 1, "a power-of-two prefix should be one interval")
            || !expect(kvc_dyadic_partition(0, 513).size() == 2, "one token past a power of two should split")
            || !expect(kvc_dyadic_partition(2147483647u, 2).empty(), "positions beyond llama_pos must be rejected")) {
        return 1;
    }

    const std::vector<int32_t> cache_tokens = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    const auto cache_keys = kvclient::kvc_dyadic_cache_keys("stable-model-profile", cache_tokens);
    const auto repeated_keys = kvclient::kvc_dyadic_cache_keys("stable-model-profile", cache_tokens);
    const auto changed_keys = kvclient::kvc_dyadic_cache_keys(
            "stable-model-profile", std::vector<int32_t>{ 11, 12, 13, 14, 15, 16, 17, 18, 20 });
    if (!expect(cache_keys.size() == 2 && repeated_keys.size() == cache_keys.size(),
                    "cache key helper should follow the canonical partition")
            || !expect(cache_keys[0].content_hash == repeated_keys[0].content_hash
                    && cache_keys[1].content_hash == repeated_keys[1].content_hash,
                    "same model profile and prompt should produce stable cache keys")
            || !expect(cache_keys[0].content_hash == changed_keys[0].content_hash
                    && cache_keys[1].content_hash != changed_keys[1].content_hash,
                    "a changed suffix should retain earlier prefix hits and invalidate later chunks")) {
        return 1;
    }

    kvclient::kvc_dyadic_chunk source;
    source.offset = 0;
    source.n_tokens = 1;
    source.level = 0;
    source.index = 0;
    source.model_profile = "test-model";
    source.prefix_hash = kvclient::KVC_DYADIC_HASH_SEED;
    source.token_ids = { 42 };
    kvclient::kvc_dyadic_layer layer;
    layer.layer_idx = 3;
    layer.type_k = "f16";
    layer.type_v = "f16";
    layer.k_data = { 1, 2, 3, 4 };
    layer.v_data = { 5, 6, 7, 8 };
    source.layers.push_back(layer);
    source.content_hash = kvclient::detail::kvc_dyadic_content_hash(source);
    source.payload_hash = kvclient::detail::kvc_dyadic_payload_hash(source);

    std::vector<uint8_t> encoded;
    kvclient::kvc_dyadic_chunk decoded;
    if (!expect(kvclient::detail::serialize_dyadic(source, encoded), "valid dyadic chunk should serialize")
            || !expect(kvclient::detail::deserialize_dyadic(encoded, decoded),
                    "serialized dyadic chunk should deserialize")
            || !expect(decoded.content_hash == source.content_hash
                    && decoded.payload_hash == source.payload_hash
                    && decoded.layers[0].k_data == source.layers[0].k_data
                    && decoded.layers[0].v_data == source.layers[0].v_data,
                    "dyadic codec should preserve interval hashes and K/V payloads")) {
        return 1;
    }

    encoded.back() ^= 1;
    decoded.model_profile = "unchanged-on-failure";
    if (!expect(!kvclient::detail::deserialize_dyadic(encoded, decoded),
                    "dyadic codec should reject payload corruption")
            || !expect(decoded.model_profile == "unchanged-on-failure",
                    "failed decoding must preserve the output chunk")) {
        return 1;
    }
    return 0;
}
