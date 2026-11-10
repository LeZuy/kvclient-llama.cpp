#include "kvclient/kvclient.h"

#include "kvc-chunk.h"
#include "kvc-dyadic-internal.h"
#include "llama-wrapper/llama-wrapper.h"
#include "core/kvc-serialize.h"
#include "core/kvc-dyadic-serialize.h"

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

kvc_status kvc_export_dyadic_chunks(
        llama_context * ctx,
        const kvc_metadata & metadata,
        const std::vector<int32_t> & token_ids,
        std::vector<kvc_dyadic_chunk> & chunks) {
    if (!ctx || metadata.n_tokens == 0 || token_ids.size() != metadata.n_tokens) {
        return kvc_status::INVALID_ARGUMENT;
    }

    std::vector<kvc_dyadic_chunk> result;
    const kvc_status status = detail::kv_dyadic_chunks_from_llama(ctx, metadata, token_ids, result);
    if (status != kvc_status::OK) {
        return status;
    }

    chunks = std::move(result);
    return kvc_status::OK;
}

kvc_status kvc_export_dyadic_chunks(
        llama_context * ctx,
        const kvc_metadata & metadata,
        const std::vector<int32_t> & token_ids,
        const std::string & model_identity,
        std::vector<kvc_dyadic_chunk> & chunks) {
    if (!ctx || metadata.n_tokens == 0 || token_ids.size() != metadata.n_tokens || model_identity.empty()) {
        return kvc_status::INVALID_ARGUMENT;
    }

    std::vector<kvc_dyadic_chunk> result;
    const kvc_status status = detail::kv_dyadic_chunks_from_llama(
            ctx, metadata, token_ids, model_identity, result);
    if (status != kvc_status::OK) {
        return status;
    }
    chunks = std::move(result);
    return kvc_status::OK;
}

kvc_status kvc_import_dyadic_chunks(
        llama_context * ctx,
        uint32_t seq_id,
        const std::vector<kvc_dyadic_chunk> & chunks) {
    if (!ctx) {
        return kvc_status::INVALID_ARGUMENT;
    }
    return detail::kv_dyadic_chunks_to_llama(ctx, seq_id, chunks);
}

kvc_status kvc_import_dyadic_chunks(
        llama_context * ctx,
        uint32_t seq_id,
        const std::vector<kvc_dyadic_chunk> & chunks,
        const std::string & model_identity) {
    if (!ctx || model_identity.empty()) {
        return kvc_status::INVALID_ARGUMENT;
    }
    return detail::kv_dyadic_chunks_to_llama(ctx, seq_id, chunks, model_identity);
}

kvc_status kvc_dyadic_get_cache_keys(
        llama_context * ctx,
        const std::string & model_identity,
        const std::vector<int32_t> & token_ids,
        std::vector<kvc_dyadic_cache_key> & keys) {
    if (!ctx || model_identity.empty() || token_ids.empty()) {
        return kvc_status::INVALID_ARGUMENT;
    }

    std::string profile;
    if (!detail::kvc_get_dyadic_model_profile(ctx, model_identity, profile)) {
        return kvc_status::UNSUPPORTED;
    }

    std::vector<kvc_dyadic_cache_key> result = kvc_dyadic_cache_keys(profile, token_ids);
    if (result.empty()) {
        return kvc_status::INVALID_ARGUMENT;
    }
    keys = std::move(result);
    return kvc_status::OK;
}

bool kvc_chunk_serialize(const kvc_chunk & chunk, std::vector<uint8_t> & data) {
    return detail::serialize(chunk, data);
}

bool kvc_chunk_deserialize(const std::vector<uint8_t> & data, kvc_chunk & chunk) {
    return detail::deserialize(data, chunk);
}

bool kvc_dyadic_chunk_serialize(const kvc_dyadic_chunk & chunk, std::vector<uint8_t> & data) {
    return detail::serialize_dyadic(chunk, data);
}

bool kvc_dyadic_chunk_deserialize(const std::vector<uint8_t> & data, kvc_dyadic_chunk & chunk) {
    return detail::deserialize_dyadic(data, chunk);
}

} // namespace kvclient
