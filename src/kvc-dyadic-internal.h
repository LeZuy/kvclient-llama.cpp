#pragma once

#include "kvclient/kvc-dyadic.h"

struct llama_context;

namespace kvclient::detail {

bool kvc_get_dyadic_model_profile(
        const llama_context * ctx,
        const std::string & model_identity,
        std::string & profile);
std::string kvc_dyadic_content_hash(const kvc_dyadic_chunk & chunk);
std::string kvc_dyadic_payload_hash(const kvc_dyadic_chunk & chunk);
bool kvc_dyadic_chunk_is_valid(const kvc_dyadic_chunk & chunk);

} // namespace kvclient::detail
