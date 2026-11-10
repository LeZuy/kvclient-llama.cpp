#include "kvclient/kvclient.h"

#include "kvc-chunk.h"
#include "wrapper/llama-wrapper.h"
#include "core/kvc-serialize.h"

#include <utility>

namespace kvclient {

kvc_status kvc_export_chunk(llama_context * ctx, const kvc_metadata & metadata, kvc_chunk & chunk) {
    if (!ctx || metadata.n_tokens == 0) {
        return kvc_status::INVALID_ARGUMENT;
    }

    kvc_chunk result;
    const kvc_status status = detail::kv_chunk_from_llama(ctx, &metadata, result);
    if (status != kvc_status::OK) {
        return status;
    }

    chunk = std::move(result);
    return kvc_status::OK;
}

kvc_status kvc_import_chunk(llama_context * ctx, uint32_t seq_id, const kvc_chunk & chunk) {
    if (!ctx) {
        return kvc_status::INVALID_ARGUMENT;
    }
    return detail::kv_chunk_to_llama(ctx, seq_id, chunk);
}

bool kvc_chunk_serialize(const kvc_chunk & chunk, std::vector<uint8_t> & data) {
    return detail::serialize(chunk, data);
}

bool kvc_chunk_deserialize(const std::vector<uint8_t> & data, kvc_chunk & chunk) {
    return detail::deserialize(data, chunk);
}

} // namespace kvclient
