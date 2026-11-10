#pragma once

#include "kvc-dyadic.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace kvclient {

enum class kvc_file_storage_status {
    OK,
    INVALID_ARGUMENT,
    NOT_FOUND,
    CORRUPT,
    CONFLICT,
    IO_ERROR,
};

class kvc_dyadic_file_storage {
public:
    explicit kvc_dyadic_file_storage(std::filesystem::path directory);

    const std::filesystem::path & directory() const;

    kvc_file_storage_status put(
            const kvc_dyadic_chunk & chunk,
            std::string * error = nullptr) const;
    kvc_file_storage_status get(
            const std::string & content_hash,
            kvc_dyadic_chunk & chunk,
            std::string * error = nullptr) const;
    kvc_file_storage_status remove(
            const std::string & content_hash,
            std::string * error = nullptr) const;
    kvc_file_storage_status list(
            std::vector<std::string> & content_hashes,
            std::string * error = nullptr) const;
    kvc_file_storage_status size(
            size_t & count,
            std::string * error = nullptr) const;

private:
    std::filesystem::path directory_;
};

} // namespace kvclient
