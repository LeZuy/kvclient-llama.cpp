#include "kvc-storage.h"

#include "kvc-chunk.h"

#include <memory>
#include <utility>

namespace kvclient {

size_t kvc_storage::size() const {
    return chunks_.size();
}

bool kvc_storage::exists(const std::string & content_hash) const {
    return chunks_.find(content_hash) != chunks_.end();
}

bool kvc_storage::put_chunk(kvc_chunk chunk) {
    if (!detail::kvc_chunk_is_valid(chunk)) {
        return false;
    }

    const std::string key = chunk.content_hash;
    chunks_[key] = std::make_shared<const kvc_chunk>(std::move(chunk));
    return true;
}

std::shared_ptr<const kvc_chunk> kvc_storage::get_chunk(const std::string & content_hash) const {
    const auto it = chunks_.find(content_hash);
    return it == chunks_.end() ? nullptr : it->second;
}

void kvc_storage::remove(const std::string & content_hash) {
    chunks_.erase(content_hash);
}

} // namespace kvclient
