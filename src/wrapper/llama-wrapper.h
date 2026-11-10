#pragma once

#include "kvclient/kvclient.h"

namespace kvclient::detail{
    kvc_status kv_chunk_from_llama(llama_context * ctx, const kvc_metadata * kvmtd, kvc_chunk & chunk);
    kvc_status kv_chunk_to_llama(llama_context * ctx, uint32_t seq_id, const kvc_chunk & chunk);
}