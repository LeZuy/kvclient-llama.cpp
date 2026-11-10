#pragma once

#include "kvclient/kvc-dyadic.h"

#include <vector>

namespace kvclient::detail {

bool serialize_dyadic(const kvc_dyadic_chunk & chunk, std::vector<uint8_t> & data);
bool deserialize_dyadic(const std::vector<uint8_t> & data, kvc_dyadic_chunk & chunk);

} // namespace kvclient::detail
