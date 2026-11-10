#include "llama-wrapper.h"

#include "kvc-chunk.h"
#include "kvc-context.h"

#include "llama.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace kvclient::detail {

kvc_status kv_chunk_from_llama(llama_context * ctx, const kvc_metadata * metadata, kvc_chunk & chunk) {
    if (!ctx || !metadata || metadata->n_tokens == 0
            || metadata->seq_id >= llama_n_ctx_seq(ctx)) {
        return kvc_status::INVALID_ARGUMENT;
    }

    if (!kvc_sequence_matches_range(ctx, *metadata)) {
        return kvc_status::MISSING_POSITIONS;
    }

    std::string profile;
    if (!kvc_get_model_profile(ctx, profile)) {
        return kvc_status::BACKEND_ERROR;
    }

    const llama_seq_id seq_id = static_cast<llama_seq_id>(metadata->seq_id);
    llama_synchronize(ctx);
    const size_t state_size = llama_state_seq_get_size(ctx, seq_id);
    if (state_size == 0) {
        return kvc_status::MISSING_POSITIONS;
    }

    kvc_chunk result;
    result.offset = metadata->offset;
    result.n_tokens = metadata->n_tokens;
    result.model_profile = std::move(profile);
    result.state_data.resize(state_size);

    const size_t bytes_written = llama_state_seq_get_data(
            ctx, result.state_data.data(), result.state_data.size(), seq_id);
    if (bytes_written != state_size) {
        return kvc_status::BACKEND_ERROR;
    }

    result.content_hash = kvc_hash_state(result.state_data);
    chunk = std::move(result);
    return kvc_status::OK;
}

kvc_status kv_chunk_to_llama(llama_context * ctx, uint32_t seq_id, const kvc_chunk & chunk) {
    if (!ctx || seq_id >= llama_n_ctx_seq(ctx)) {
        return kvc_status::INVALID_ARGUMENT;
    }
    if (chunk.format_version != KVC_CHUNK_FORMAT_VERSION) {
        return kvc_status::INCOMPATIBLE;
    }
    if (!kvc_chunk_is_valid(chunk)) {
        return kvc_status::INVALID_ARGUMENT;
    }

    std::string profile;
    if (!kvc_get_model_profile(ctx, profile)) {
        return kvc_status::BACKEND_ERROR;
    }
    if (profile != chunk.model_profile) {
        return kvc_status::INCOMPATIBLE;
    }

    const uint64_t end = uint64_t(chunk.offset) + chunk.n_tokens;
    if (end > static_cast<uint64_t>(llama_n_ctx(ctx))) {
        return kvc_status::INCOMPATIBLE;
    }

    const llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        return kvc_status::BACKEND_ERROR;
    }

    const llama_seq_id destination = static_cast<llama_seq_id>(seq_id);
    if (llama_memory_seq_pos_min(memory, destination) >= 0
            || llama_memory_seq_pos_max(memory, destination) >= 0) {
        return kvc_status::INVALID_ARGUMENT;
    }

    const size_t bytes_read = llama_state_seq_set_data(
            ctx, chunk.state_data.data(), chunk.state_data.size(), destination);
    if (bytes_read != chunk.state_data.size()) {
        llama_memory_seq_rm(memory, destination, 0, -1);
        return kvc_status::INCOMPATIBLE;
    }

    const llama_pos expected_min = static_cast<llama_pos>(chunk.offset);
    const llama_pos expected_max = static_cast<llama_pos>(end - 1);
    if (llama_memory_seq_pos_min(memory, destination) != expected_min
            || llama_memory_seq_pos_max(memory, destination) != expected_max) {
        llama_memory_seq_rm(memory, destination, 0, -1);
        return kvc_status::INCOMPATIBLE;
    }

    return kvc_status::OK;
}

} // namespace kvclient::detail
