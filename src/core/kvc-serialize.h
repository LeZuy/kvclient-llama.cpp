#pragma once

#include "kvclient/kvc-chunk.h"

#include <cstdint>
#include <vector>

namespace kvclient::detail {

bool serialize(const kvc_chunk & chunk, std::vector<uint8_t> & data);
bool deserialize(const std::vector<uint8_t> & data, kvc_chunk & chunk);

} // namespace kvclient::detail
