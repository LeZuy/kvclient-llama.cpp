#pragma once

#include "kvc-chunk.h"
#include "kvc-dyadic.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace kvclient {

class kvc_storage {
public:
    size_t size() const;
    bool exists(const std::string & content_hash) const;
    bool put_chunk(kvc_chunk chunk);
    std::shared_ptr<const kvc_chunk> get_chunk(const std::string & content_hash) const;
    void remove(const std::string & content_hash);

private:
    std::unordered_map<std::string, std::shared_ptr<const kvc_chunk>> chunks_;
};

class kvc_dyadic_storage {
public:
    size_t size() const;
    bool exists(const std::string & content_hash) const;
    bool put_chunk(kvc_dyadic_chunk chunk);
    std::shared_ptr<const kvc_dyadic_chunk> get_chunk(const std::string & content_hash) const;
    void remove(const std::string & content_hash);

private:
    std::unordered_map<std::string, std::shared_ptr<const kvc_dyadic_chunk>> chunks_;
};

} // namespace kvclient
