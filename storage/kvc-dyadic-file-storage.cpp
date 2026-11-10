#include "kvclient/kvc-file-storage.h"

#include "core/kvc-dyadic-serialize.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <limits>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kvclient {
namespace {

std::atomic<uint64_t> temp_sequence{ 0 };

void set_error(std::string * error, std::string message) {
    if (error) {
        *error = std::move(message);
    }
}

bool valid_content_hash(const std::string & hash) {
    if (hash.size() != 16) {
        return false;
    }
    return std::all_of(hash.begin(), hash.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::filesystem::path chunk_path(const std::filesystem::path & directory, const std::string & hash) {
    return directory / (hash + ".kvd");
}

bool read_file(const std::filesystem::path & path, std::vector<uint8_t> & data, std::string * error) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        set_error(error, "Unable to open chunk file for reading: " + path.string());
        return false;
    }

    const std::streampos end = input.tellg();
    if (end < 0 || static_cast<uintmax_t>(end) > std::numeric_limits<size_t>::max()
            || static_cast<uintmax_t>(end) > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        set_error(error, "Chunk file size is invalid or too large: " + path.string());
        return false;
    }

    std::vector<uint8_t> result(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!result.empty()) {
        input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(result.size()));
    }
    if (!input || input.peek() != std::char_traits<char>::eof()) {
        set_error(error, "Unable to read complete chunk file: " + path.string());
        return false;
    }
    data = std::move(result);
    return true;
}

bool write_file_exclusive(
        const std::filesystem::path & path,
        const std::vector<uint8_t> & data,
        std::string * error) {
#ifdef _WIN32
    const int fd = _wopen(path.c_str(), _O_BINARY | _O_WRONLY | _O_CREAT | _O_EXCL, _S_IREAD | _S_IWRITE);
#else
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
#endif
    if (fd < 0) {
        set_error(error, "Unable to create temporary chunk file: " + path.string());
        return false;
    }

    size_t written = 0;
    bool ok = true;
    while (written < data.size()) {
        const size_t remaining = data.size() - written;
#ifdef _WIN32
        const unsigned int count = static_cast<unsigned int>(
                std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<int>::max())));
        const int result = _write(fd, data.data() + written, count);
#else
        const size_t count = std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t result = write(fd, data.data() + written, count);
#endif
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            ok = false;
            break;
        }
        written += static_cast<size_t>(result);
    }

#ifdef _WIN32
    if (ok && _commit(fd) != 0) {
        ok = false;
    }
    const int close_result = _close(fd);
#else
    if (ok && fsync(fd) != 0) {
        ok = false;
    }
    const int close_result = close(fd);
#endif
    if (close_result != 0) {
        ok = false;
    }
    if (!ok) {
        set_error(error, "Unable to write and flush temporary chunk file: " + path.string());
    }
    return ok;
}

bool ensure_directory(const std::filesystem::path & path, std::string * error) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        set_error(error, "Unable to create storage directory: " + ec.message());
        return false;
    }
    const auto status = std::filesystem::status(path, ec);
    if (ec || !std::filesystem::is_directory(status)) {
        set_error(error, "Storage path is not a directory: " + path.string());
        return false;
    }
    return true;
}

kvc_file_storage_status read_and_validate(
        const std::filesystem::path & path,
        const std::string & expected_hash,
        std::vector<uint8_t> & bytes,
        kvc_dyadic_chunk & chunk,
        std::string * error) {
    if (!read_file(path, bytes, error)) {
        return kvc_file_storage_status::IO_ERROR;
    }
    if (!detail::deserialize_dyadic(bytes, chunk)) {
        set_error(error, "Stored chunk has invalid KVD1 data or checksum: " + path.string());
        return kvc_file_storage_status::CORRUPT;
    }
    if (chunk.content_hash != expected_hash) {
        set_error(error, "Stored chunk content hash does not match its filename: " + path.string());
        return kvc_file_storage_status::CORRUPT;
    }
    return kvc_file_storage_status::OK;
}

} // namespace

kvc_dyadic_file_storage::kvc_dyadic_file_storage(std::filesystem::path directory)
    : directory_(std::move(directory)) {
}

const std::filesystem::path & kvc_dyadic_file_storage::directory() const {
    return directory_;
}

kvc_file_storage_status kvc_dyadic_file_storage::put(
        const kvc_dyadic_chunk & chunk,
        std::string * error) const {
    set_error(error, "");
    if (!valid_content_hash(chunk.content_hash)) {
        set_error(error, "Chunk content hash must be 16 lowercase hexadecimal characters");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }

    std::vector<uint8_t> bytes;
    if (!detail::serialize_dyadic(chunk, bytes)) {
        set_error(error, "Chunk failed KVD1 validation or serialization");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }
    if (directory_.empty()) {
        set_error(error, "Storage directory must not be empty");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }
    if (!ensure_directory(directory_, error)) {
        return kvc_file_storage_status::IO_ERROR;
    }

    const std::filesystem::path destination = chunk_path(directory_, chunk.content_hash);
    std::error_code ec;
    const bool destination_exists = std::filesystem::exists(destination, ec);
    if (ec) {
        set_error(error, "Unable to inspect existing chunk: " + ec.message());
        return kvc_file_storage_status::IO_ERROR;
    }
    if (destination_exists) {
        std::vector<uint8_t> existing_bytes;
        kvc_dyadic_chunk existing_chunk;
        const auto status = read_and_validate(
                destination, chunk.content_hash, existing_bytes, existing_chunk, error);
        if (status != kvc_file_storage_status::OK) {
            return status;
        }
        if (existing_bytes == bytes) {
            return kvc_file_storage_status::OK;
        }
        set_error(error, "A different serialized chunk already uses this content hash");
        return kvc_file_storage_status::CONFLICT;
    }

    std::filesystem::path temp_directory;
    bool temp_directory_created = false;
    for (unsigned int attempt = 0; attempt < 32; ++attempt) {
        const uint64_t sequence = temp_sequence.fetch_add(1, std::memory_order_relaxed);
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        temp_directory = directory_ / ("." + chunk.content_hash + ".tmp."
                + std::to_string(ticks) + "." + std::to_string(sequence));
        ec.clear();
        temp_directory_created = std::filesystem::create_directory(temp_directory, ec);
        if (temp_directory_created) {
            break;
        }
        if (ec && ec != std::errc::file_exists) {
            set_error(error, "Unable to create temporary storage directory: " + ec.message());
            return kvc_file_storage_status::IO_ERROR;
        }
    }
    if (!temp_directory_created) {
        set_error(error, "Unable to allocate a unique temporary storage directory");
        return kvc_file_storage_status::IO_ERROR;
    }

    const std::filesystem::path temp_file = temp_directory / "chunk.kvd";
    if (!write_file_exclusive(temp_file, bytes, error)) {
        std::filesystem::remove(temp_file, ec);
        std::filesystem::remove(temp_directory, ec);
        return kvc_file_storage_status::IO_ERROR;
    }

    ec.clear();
    std::filesystem::create_hard_link(temp_file, destination, ec);
    if (ec) {
        std::error_code exists_error;
        const auto destination_status = std::filesystem::symlink_status(destination, exists_error);
        const bool destination_exists = !exists_error && std::filesystem::exists(destination_status);
        std::vector<uint8_t> existing_bytes;
        kvc_dyadic_chunk existing_chunk;
        if (!exists_error && destination_exists
                && read_and_validate(destination, chunk.content_hash, existing_bytes, existing_chunk, nullptr)
                        == kvc_file_storage_status::OK
                && existing_bytes == bytes) {
            std::filesystem::remove(temp_file, exists_error);
            std::filesystem::remove(temp_directory, exists_error);
            return kvc_file_storage_status::OK;
        }
        std::filesystem::remove(temp_file, exists_error);
        std::filesystem::remove(temp_directory, exists_error);
        if (destination_exists) {
            set_error(error, "A different serialized chunk already uses this content hash");
            return kvc_file_storage_status::CONFLICT;
        }
        set_error(error, "Unable to atomically publish chunk file: " + ec.message());
        return kvc_file_storage_status::IO_ERROR;
    }

    std::filesystem::remove(temp_file, ec);
    std::filesystem::remove(temp_directory, ec);
    return kvc_file_storage_status::OK;
}

kvc_file_storage_status kvc_dyadic_file_storage::get(
        const std::string & content_hash,
        kvc_dyadic_chunk & chunk,
        std::string * error) const {
    set_error(error, "");
    if (directory_.empty()) {
        set_error(error, "Storage directory must not be empty");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }
    if (!valid_content_hash(content_hash)) {
        set_error(error, "Content hash must be 16 lowercase hexadecimal characters");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }

    const auto path = chunk_path(directory_, content_hash);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        if (ec) {
            set_error(error, "Unable to inspect chunk path: " + ec.message());
            return kvc_file_storage_status::IO_ERROR;
        }
        set_error(error, "Chunk not found: " + content_hash);
        return kvc_file_storage_status::NOT_FOUND;
    }

    std::vector<uint8_t> bytes;
    kvc_dyadic_chunk result;
    const auto status = read_and_validate(path, content_hash, bytes, result, error);
    if (status != kvc_file_storage_status::OK) {
        return status;
    }
    chunk = std::move(result);
    return kvc_file_storage_status::OK;
}

kvc_file_storage_status kvc_dyadic_file_storage::remove(
        const std::string & content_hash,
        std::string * error) const {
    set_error(error, "");
    if (directory_.empty()) {
        set_error(error, "Storage directory must not be empty");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }
    if (!valid_content_hash(content_hash)) {
        set_error(error, "Content hash must be 16 lowercase hexadecimal characters");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }

    std::error_code ec;
    const bool removed = std::filesystem::remove(chunk_path(directory_, content_hash), ec);
    if (ec) {
        set_error(error, "Unable to remove chunk: " + ec.message());
        return kvc_file_storage_status::IO_ERROR;
    }
    if (!removed) {
        set_error(error, "Chunk not found: " + content_hash);
        return kvc_file_storage_status::NOT_FOUND;
    }
    return kvc_file_storage_status::OK;
}

kvc_file_storage_status kvc_dyadic_file_storage::list(
        std::vector<std::string> & content_hashes,
        std::string * error) const {
    set_error(error, "");
    if (directory_.empty()) {
        set_error(error, "Storage directory must not be empty");
        return kvc_file_storage_status::INVALID_ARGUMENT;
    }
    std::error_code ec;
    if (!std::filesystem::exists(directory_, ec)) {
        if (ec) {
            set_error(error, "Unable to inspect storage directory: " + ec.message());
            return kvc_file_storage_status::IO_ERROR;
        }
        content_hashes.clear();
        return kvc_file_storage_status::OK;
    }

    std::vector<std::string> result;
    std::filesystem::directory_iterator it(directory_, ec);
    const std::filesystem::directory_iterator end;
    if (ec) {
        set_error(error, "Unable to enumerate storage directory: " + ec.message());
        return kvc_file_storage_status::IO_ERROR;
    }
    for (; it != end; it.increment(ec)) {
        if (ec) {
            set_error(error, "Unable to enumerate storage directory: " + ec.message());
            return kvc_file_storage_status::IO_ERROR;
        }
        const auto status = it->symlink_status(ec);
        if (ec) {
            set_error(error, "Unable to inspect storage entry: " + ec.message());
            return kvc_file_storage_status::IO_ERROR;
        }
        if (!std::filesystem::is_regular_file(status) || it->path().extension() != ".kvd") {
            continue;
        }
        const std::string hash = it->path().stem().string();
        if (valid_content_hash(hash)) {
            result.push_back(hash);
        }
    }
    std::sort(result.begin(), result.end());
    content_hashes = std::move(result);
    return kvc_file_storage_status::OK;
}

kvc_file_storage_status kvc_dyadic_file_storage::size(size_t & count, std::string * error) const {
    set_error(error, "");
    std::vector<std::string> hashes;
    const auto status = list(hashes, error);
    if (status == kvc_file_storage_status::OK) {
        count = hashes.size();
    }
    return status;
}

} // namespace kvclient
