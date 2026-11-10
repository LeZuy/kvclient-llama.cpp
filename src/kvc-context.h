#pragma once

#include "kvclient/kvclient.h"

#include <string>

namespace kvclient::detail {

bool kvc_get_model_profile(const llama_context * ctx, std::string & profile);
bool kvc_sequence_matches_range(const llama_context * ctx, const kvc_metadata & metadata);

} // namespace kvclient::detail
