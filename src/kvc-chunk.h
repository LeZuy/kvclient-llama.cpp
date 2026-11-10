#pragma once

#include "kvclient/kvc-chunk.h"

#include <string>
#include <vector>

namespace kvclient::detail {

bool kvc_chunk_is_valid(const kvc_chunk & chunk);
std::string kvc_hash_state(const std::vector<uint8_t> & state_data);

} // namespace kvclient::detail
