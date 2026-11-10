#pragma once

#include "kvclient/kvclient.h"

namespace kvclient::detail {

kvc_status kv_chunk_from_llama(llama_context * ctx, const kvc_metadata * metadata, kvc_chunk & chunk);
kvc_status kv_chunk_to_llama(llama_context * ctx, uint32_t seq_id, const kvc_chunk & chunk);

kvc_status kv_dyadic_chunks_from_llama(
        llama_context * ctx,
        const kvc_metadata & metadata,
        const std::vector<int32_t> & token_ids,
        std::vector<kvc_dyadic_chunk> & chunks);
kvc_status kv_dyadic_chunks_from_llama(
        llama_context * ctx,
        const kvc_metadata & metadata,
        const std::vector<int32_t> & token_ids,
        const std::string & model_identity,
        std::vector<kvc_dyadic_chunk> & chunks);
kvc_status kv_dyadic_chunks_to_llama(
        llama_context * ctx,
        uint32_t seq_id,
        const std::vector<kvc_dyadic_chunk> & chunks);
kvc_status kv_dyadic_chunks_to_llama(
        llama_context * ctx,
        uint32_t seq_id,
        const std::vector<kvc_dyadic_chunk> & chunks,
        const std::string & model_identity);

} // namespace kvclient::detail