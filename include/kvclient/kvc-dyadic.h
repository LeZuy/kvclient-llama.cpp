#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kvclient {

inline constexpr uint32_t KVC_DYADIC_FORMAT_VERSION = 1;
inline constexpr const char * KVC_DYADIC_HASH_SEED = "kvclient-dyadic-v1";

struct kvc_dyadic_interval {
    uint32_t offset = 0;
    uint32_t n_tokens = 0;
    uint32_t level = 0;
    uint64_t index = 0;
};

struct kvc_dyadic_cache_key {
    kvc_dyadic_interval interval;
    std::string prefix_hash;
    std::string content_hash;
};

struct kvc_dyadic_layer {
    uint32_t layer_idx = 0;
    std::string type_k;
    std::string type_v;
    std::vector<uint8_t> k_data;
    std::vector<uint8_t> v_data;
};

struct kvc_dyadic_chunk {
    uint32_t format_version = KVC_DYADIC_FORMAT_VERSION;
    uint32_t offset = 0;
    uint32_t n_tokens = 0;
    uint32_t level = 0;
    uint64_t index = 0;

    std::string model_profile;
    std::string prefix_hash;
    std::string content_hash;
    std::string payload_hash;

    std::vector<int32_t> token_ids;
    std::vector<kvc_dyadic_layer> layers;
};

std::vector<kvc_dyadic_interval> kvc_dyadic_partition(uint32_t offset, uint32_t n_tokens);
std::vector<kvc_dyadic_cache_key> kvc_dyadic_cache_keys(
        const std::string & model_profile,
        const std::vector<int32_t> & token_ids);

} // namespace kvclient
