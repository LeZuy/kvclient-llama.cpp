#pragma once

#include "kvc-chunk.h"

#include <cstdint>
#include <vector>

struct llama_context;

namespace kvclient {

enum class kvc_status {
    OK,
    INVALID_ARGUMENT,
    MISSING_POSITIONS,
    INCOMPATIBLE,
    UNSUPPORTED,
    BACKEND_ERROR,
};

struct kvc_metadata {
    uint32_t seq_id;
    uint32_t offset;
    uint32_t n_tokens;
};

// Export/import the entire contiguous sequence range described by metadata.
// Import requires an empty destination sequence and the same loaded model instance.
kvc_status kvc_export_chunk(llama_context * ctx, const kvc_metadata & metadata, kvc_chunk & chunk);
kvc_status kvc_import_chunk(llama_context * ctx, uint32_t seq_id, const kvc_chunk & chunk);

// Encode a chunk for local storage; the native state is not a network interchange format.
bool kvc_chunk_serialize(const kvc_chunk & chunk, std::vector<uint8_t> & data);
bool kvc_chunk_deserialize(const std::vector<uint8_t> & data, kvc_chunk & chunk);

} // namespace kvclient
