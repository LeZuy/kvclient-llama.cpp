#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kvclient {

inline constexpr uint32_t KVC_CHUNK_FORMAT_VERSION = 1;

struct kvc_chunk {
    uint32_t format_version = KVC_CHUNK_FORMAT_VERSION;
    uint32_t offset = 0;
    uint32_t n_tokens = 0;

    // Process-local model-instance identity and cache-layout compatibility.
    std::string model_profile;
    std::string content_hash;

    // Native llama.cpp sequence state; restore with the same model instance and build.
    std::vector<uint8_t> state_data;
};

} // namespace kvclient
