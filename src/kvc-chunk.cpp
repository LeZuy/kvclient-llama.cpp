#include "kvc-chunk.h"

#include <iomanip>
#include <limits>
#include <sstream>

namespace kvclient::detail {

std::string kvc_hash_state(const std::vector<uint8_t> & state_data) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const uint8_t byte : state_data) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }

    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << hash;
    return result.str();
}

bool kvc_chunk_is_valid(const kvc_chunk & chunk) {
    const uint64_t end = uint64_t(chunk.offset) + chunk.n_tokens;
    return chunk.format_version == KVC_CHUNK_FORMAT_VERSION
            && chunk.n_tokens > 0
            && end <= uint64_t(std::numeric_limits<int32_t>::max()) + 1
            && !chunk.model_profile.empty()
            && !chunk.state_data.empty()
            && chunk.content_hash == kvc_hash_state(chunk.state_data);
}

} // namespace kvclient::detail
