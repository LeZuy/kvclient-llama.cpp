#include "kvclient/kvc-file-storage.h"

#include "kvc-dyadic-internal.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>

namespace {

bool expect(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

struct temporary_directory {
    explicit temporary_directory(std::filesystem::path value)
        : path(std::move(value)) {
    }

    ~temporary_directory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    std::filesystem::path path;
};

kvclient::kvc_dyadic_chunk make_chunk() {
    kvclient::kvc_dyadic_chunk chunk;
    chunk.offset = 0;
    chunk.n_tokens = 1;
    chunk.level = 0;
    chunk.index = 0;
    chunk.model_profile = "storage-test-model";
    chunk.prefix_hash = kvclient::KVC_DYADIC_HASH_SEED;
    chunk.token_ids = { 17 };

    kvclient::kvc_dyadic_layer layer;
    layer.layer_idx = 2;
    layer.type_k = "f16";
    layer.type_v = "f16";
    layer.k_data = { 1, 2, 3, 4 };
    layer.v_data = { 5, 6, 7, 8 };
    chunk.layers.push_back(std::move(layer));

    chunk.content_hash = kvclient::detail::kvc_dyadic_content_hash(chunk);
    chunk.payload_hash = kvclient::detail::kvc_dyadic_payload_hash(chunk);
    return chunk;
}

} // namespace

int main() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto test_path = std::filesystem::temp_directory_path()
            / ("kvclient-file-storage-test-" + std::to_string(suffix));
    std::error_code ec;
    if (!expect(std::filesystem::create_directory(test_path, ec) && !ec,
                    "should create an isolated storage test directory")) {
        return 1;
    }
    const temporary_directory cleanup(test_path);

    const kvclient::kvc_dyadic_chunk source = make_chunk();
    const kvclient::kvc_dyadic_file_storage storage(test_path);
    std::string error;
    if (!expect(storage.put(source, &error) == kvclient::kvc_file_storage_status::OK,
                    "should persist a valid chunk")
            || !expect(std::filesystem::exists(test_path / (source.content_hash + ".kvd")),
                    "persisted chunk should have a content-addressed file")
            || !expect(storage.put(source, &error) == kvclient::kvc_file_storage_status::OK,
                    "putting the identical chunk should be idempotent")) {
        std::cerr << error << '\n';
        return 1;
    }

    kvclient::kvc_dyadic_chunk loaded;
    const kvclient::kvc_dyadic_file_storage reopened(test_path);
    if (!expect(reopened.get(source.content_hash, loaded, &error) == kvclient::kvc_file_storage_status::OK,
                    "a new store instance should load the persisted chunk")
            || !expect(loaded.layers.size() == 1
                    && loaded.layers[0].k_data == source.layers[0].k_data
                    && loaded.layers[0].v_data == source.layers[0].v_data,
                    "storage should preserve and extract both K and V tensors")) {
        std::cerr << error << '\n';
        return 1;
    }

    std::vector<std::string> hashes;
    size_t count = 0;
    if (!expect(reopened.list(hashes, &error) == kvclient::kvc_file_storage_status::OK
                    && hashes.size() == 1 && hashes[0] == source.content_hash,
                    "list should return the persisted content hash")
            || !expect(reopened.size(count, &error) == kvclient::kvc_file_storage_status::OK && count == 1,
                    "size should count persisted chunks")) {
        std::cerr << error << '\n';
        return 1;
    }

    kvclient::kvc_dyadic_chunk conflict = source;
    conflict.layers[0].k_data[0] ^= 0xff;
    conflict.payload_hash = kvclient::detail::kvc_dyadic_payload_hash(conflict);
    if (!expect(reopened.put(conflict, &error) == kvclient::kvc_file_storage_status::CONFLICT,
                    "different chunk data must not overwrite an existing content key")) {
        std::cerr << error << '\n';
        return 1;
    }

    const auto file_path = test_path / (source.content_hash + ".kvd");
    {
        std::fstream file(file_path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(-1, std::ios::end);
        char byte = 0;
        file.write(&byte, 1);
        file.flush();
        if (!expect(static_cast<bool>(file), "should corrupt the stored payload for validation test")) {
            return 1;
        }
    }
    loaded.model_profile = "unchanged-on-failure";
    if (!expect(reopened.get(source.content_hash, loaded, &error) == kvclient::kvc_file_storage_status::CORRUPT,
                    "read should reject a corrupted chunk")
            || !expect(loaded.model_profile == "unchanged-on-failure",
                    "failed read should not replace its output chunk")) {
        std::cerr << error << '\n';
        return 1;
    }

    if (!expect(reopened.remove(source.content_hash, &error) == kvclient::kvc_file_storage_status::OK,
                    "remove should delete the stored chunk")
            || !expect(reopened.get(source.content_hash, loaded, &error)
                    == kvclient::kvc_file_storage_status::NOT_FOUND,
                    "deleted chunks should not be readable")
            || !expect(reopened.remove(source.content_hash, &error)
                    == kvclient::kvc_file_storage_status::NOT_FOUND,
                    "removing an absent chunk should report not found")) {
        std::cerr << error << '\n';
        return 1;
    }

    return 0;
}
