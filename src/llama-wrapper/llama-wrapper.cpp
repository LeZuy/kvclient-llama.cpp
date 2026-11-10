#include "llama-wrapper.h"

#include "kvc-chunk.h"
#include "kvc-context.h"
#include "kvc-dyadic-internal.h"

#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace kvclient::detail {

namespace {

llama_kv_cache * get_kv_cache(llama_context * ctx) {
    return ctx ? dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx)) : nullptr;
}

bool get_contiguous_slots(
        const llama_kv_cache & cache,
        llama_seq_id seq_id,
        uint32_t offset,
        uint32_t n_tokens,
        std::vector<uint32_t> & slots) {
    const auto & positions = cache.get_cells(seq_id).seq_pos_get(seq_id);
    if (positions.size() != n_tokens) {
        return false;
    }

    slots.clear();
    slots.reserve(n_tokens);
    auto it = positions.begin();
    for (uint32_t i = 0; i < n_tokens; ++i, ++it) {
        if (it->first != static_cast<llama_pos>(uint64_t(offset) + i)) {
            return false;
        }
        slots.push_back(it->second);
    }
    return true;
}

bool get_row_bytes(const ggml_tensor * tensor, size_t & row_bytes) {
    if (!tensor || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0) {
        return false;
    }

    row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    return row_bytes != 0 && tensor->nb[1] == row_bytes;
}

bool tensor_range_fits(const ggml_tensor * tensor, uint32_t stream, uint32_t slot, size_t size) {
    if (!tensor || stream >= static_cast<uint32_t>(tensor->ne[2]) || slot >= static_cast<uint32_t>(tensor->ne[1])) {
        return false;
    }
    const uint64_t offset = uint64_t(stream) * tensor->nb[2] + uint64_t(slot) * tensor->nb[1];
    return offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset;
}

std::vector<std::pair<size_t, size_t>> contiguous_runs(const std::vector<uint32_t> & slots) {
    std::vector<std::pair<size_t, size_t>> runs;
    for (size_t first = 0; first < slots.size();) {
        size_t end = first + 1;
        while (end < slots.size() && slots[end - 1] != std::numeric_limits<uint32_t>::max()
                && slots[end] == slots[end - 1] + 1) {
            ++end;
        }
        runs.emplace_back(first, end);
        first = end;
    }
    return runs;
}

bool copy_row_range(
        ggml_tensor * tensor,
        uint32_t stream,
        const std::vector<uint32_t> & slots,
        size_t row_bytes,
        uint8_t * output,
        const uint8_t * input) {
    if (row_bytes == 0 || (output == nullptr) == (input == nullptr)) {
        return false;
    }

    for (const auto & run : contiguous_runs(slots)) {
        const size_t first = run.first;
        const size_t count = run.second - run.first;
        const size_t run_bytes = count * row_bytes;
        const uint32_t slot = slots[first];
        if (uint64_t(slot) + count > static_cast<uint64_t>(tensor->ne[1])
                || !tensor_range_fits(tensor, stream, slot, run_bytes)) {
            return false;
        }
        const size_t offset = size_t(stream) * tensor->nb[2] + size_t(slot) * tensor->nb[1];
        if (input) {
            ggml_backend_tensor_set(tensor, input + first * row_bytes, offset, run_bytes);
        } else {
            ggml_backend_tensor_get(tensor, output + first * row_bytes, offset, run_bytes);
        }
    }
    return true;
}

bool copy_transposed_v_range(
        ggml_tensor * tensor,
        uint32_t stream,
        uint32_t head_dim,
        const std::vector<uint32_t> & slots,
        uint8_t * output,
        const uint8_t * input) {
    if (!tensor || ggml_blck_size(tensor->type) != 1 || head_dim == 0
            || tensor->ne[0] % head_dim != 0 || (output == nullptr) == (input == nullptr)) {
        return false;
    }

    const size_t element_bytes = ggml_type_size(tensor->type);
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    if (element_bytes == 0) {
        return false;
    }

    const uint32_t n_heads = static_cast<uint32_t>(tensor->ne[0] / head_dim);
    const size_t feature_stride = ggml_row_size(tensor->type, tensor->ne[1]);
    const size_t head_stride = ggml_row_size(tensor->type, uint64_t(tensor->ne[1]) * head_dim);

    for (const auto & run : contiguous_runs(slots)) {
        const size_t first = run.first;
        const size_t count = run.second - run.first;
        const size_t run_bytes = count * element_bytes;
        const uint32_t slot = slots[first];
        if (stream >= static_cast<uint32_t>(tensor->ne[2]) || slot >= static_cast<uint32_t>(tensor->ne[1])
                || uint64_t(slot) + count > static_cast<uint64_t>(tensor->ne[1])) {
            return false;
        }

        std::vector<uint8_t> values(run_bytes);
        for (uint32_t head = 0; head < n_heads; ++head) {
            for (uint32_t feature = 0; feature < head_dim; ++feature) {
                const size_t offset = size_t(stream) * tensor->nb[2]
                        + size_t(head) * head_stride
                        + size_t(feature) * feature_stride
                        + size_t(slot) * element_bytes;
                if (offset > ggml_nbytes(tensor) || run_bytes > ggml_nbytes(tensor) - offset) {
                    return false;
                }

                if (input) {
                    for (size_t i = 0; i < count; ++i) {
                        std::memcpy(values.data() + i * element_bytes,
                                input + (first + i) * row_bytes
                                        + (size_t(head) * head_dim + feature) * element_bytes,
                                element_bytes);
                    }
                    ggml_backend_tensor_set(tensor, values.data(), offset, run_bytes);
                } else {
                    ggml_backend_tensor_get(tensor, values.data(), offset, run_bytes);
                    for (size_t i = 0; i < count; ++i) {
                        std::memcpy(output + (first + i) * row_bytes
                                        + (size_t(head) * head_dim + feature) * element_bytes,
                                values.data() + i * element_bytes,
                                element_bytes);
                    }
                }
            }
        }
    }
    return true;
}

bool cache_supports_dyadic(const llama_kv_cache * cache, const std::vector<uint32_t> & layer_ids) {
    if (!cache || layer_ids.empty() || cache->get_has_shift() || cache->has_cell_ext()) {
        return false;
    }
    for (const uint32_t layer_idx : layer_ids) {
        ggml_tensor * k = cache->get_k_storage(static_cast<int32_t>(layer_idx));
        ggml_tensor * v = cache->get_v_storage(static_cast<int32_t>(layer_idx));
        if (!k || !v || (cache->get_v_trans() && ggml_blck_size(v->type) != 1)) {
            return false;
        }
    }
    return true;
}

} // namespace

bool kvc_get_dyadic_model_profile(
        const llama_context * ctx,
        const std::string & model_identity,
        std::string & profile) {
    if (!ctx || model_identity.empty()) {
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    const auto * cache = dynamic_cast<const llama_kv_cache *>(llama_get_memory(ctx));
    const llama_vocab * vocab = model ? llama_model_get_vocab(model) : nullptr;
    if (!model || !cache || !vocab) {
        return false;
    }

    std::array<char, 256> description{};
    if (llama_model_desc(model, description.data(), description.size()) < 0) {
        return false;
    }
    description.back() = '\0';

    std::string result = std::string(llama_version()) + "|" + description.data()
            + "|identity=" + model_identity
            + "|layers=" + std::to_string(llama_model_n_layer(model))
            + "|embd=" + std::to_string(llama_model_n_embd(model))
            + "|vocab=" + std::to_string(llama_vocab_n_tokens(vocab))
            + "|type_k=" + ggml_type_name(cache->type_k())
            + "|type_v=" + ggml_type_name(cache->type_v())
            + "|v_trans=" + (cache->get_v_trans() ? "1" : "0");
    for (const uint32_t layer_idx : cache->get_layer_ids()) {
        ggml_tensor * k = cache->get_k_storage(static_cast<int32_t>(layer_idx));
        ggml_tensor * v = cache->get_v_storage(static_cast<int32_t>(layer_idx));
        if (!k || !v) {
            return false;
        }
        result += "|layer=" + std::to_string(layer_idx)
                + ":" + ggml_type_name(k->type)
                + ":" + ggml_type_name(v->type)
                + ":" + std::to_string(k->ne[0])
                + ":" + std::to_string(v->ne[0]);
    }
    profile = std::move(result);
    return true;
}

kvc_status kv_chunk_from_llama(llama_context * ctx, const kvc_metadata * metadata, kvc_chunk & chunk) {
    if (!ctx || !metadata || metadata->n_tokens == 0
            || metadata->seq_id >= llama_n_ctx_seq(ctx)) {
        return kvc_status::INVALID_ARGUMENT;
    }

    if (!kvc_sequence_matches_range(ctx, *metadata)) {
        return kvc_status::MISSING_POSITIONS;
    }

    std::string profile;
    if (!kvc_get_model_profile(ctx, profile)) {
        return kvc_status::BACKEND_ERROR;
    }

    const llama_seq_id seq_id = static_cast<llama_seq_id>(metadata->seq_id);
    llama_synchronize(ctx);
    const size_t state_size = llama_state_seq_get_size(ctx, seq_id);
    if (state_size == 0) {
        return kvc_status::MISSING_POSITIONS;
    }

    kvc_chunk result;
    result.offset = metadata->offset;
    result.n_tokens = metadata->n_tokens;
    result.model_profile = std::move(profile);
    result.state_data.resize(state_size);

    const size_t bytes_written = llama_state_seq_get_data(
            ctx, result.state_data.data(), result.state_data.size(), seq_id);
    if (bytes_written != state_size) {
        return kvc_status::BACKEND_ERROR;
    }

    result.content_hash = kvc_hash_state(result.state_data);
    chunk = std::move(result);
    return kvc_status::OK;
}

kvc_status kv_chunk_to_llama(llama_context * ctx, uint32_t seq_id, const kvc_chunk & chunk) {
    if (!ctx || seq_id >= llama_n_ctx_seq(ctx)) {
        return kvc_status::INVALID_ARGUMENT;
    }
    if (chunk.format_version != KVC_CHUNK_FORMAT_VERSION) {
        return kvc_status::INCOMPATIBLE;
    }
    if (!kvc_chunk_is_valid(chunk)) {
        return kvc_status::INVALID_ARGUMENT;
    }

    std::string profile;
    if (!kvc_get_model_profile(ctx, profile)) {
        return kvc_status::BACKEND_ERROR;
    }
    if (profile != chunk.model_profile) {
        return kvc_status::INCOMPATIBLE;
    }

    const uint64_t end = uint64_t(chunk.offset) + chunk.n_tokens;
    if (end > static_cast<uint64_t>(llama_n_ctx(ctx))) {
        return kvc_status::INCOMPATIBLE;
    }

    const llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        return kvc_status::BACKEND_ERROR;
    }

    const llama_seq_id destination = static_cast<llama_seq_id>(seq_id);
    if (llama_memory_seq_pos_min(memory, destination) >= 0
            || llama_memory_seq_pos_max(memory, destination) >= 0) {
        return kvc_status::INVALID_ARGUMENT;
    }

    const size_t bytes_read = llama_state_seq_set_data(
            ctx, chunk.state_data.data(), chunk.state_data.size(), destination);
    if (bytes_read != chunk.state_data.size()) {
        llama_memory_seq_rm(memory, destination, 0, -1);
        return kvc_status::INCOMPATIBLE;
    }

    const llama_pos expected_min = static_cast<llama_pos>(chunk.offset);
    const llama_pos expected_max = static_cast<llama_pos>(end - 1);
    if (llama_memory_seq_pos_min(memory, destination) != expected_min
            || llama_memory_seq_pos_max(memory, destination) != expected_max) {
        llama_memory_seq_rm(memory, destination, 0, -1);
        return kvc_status::INCOMPATIBLE;
    }

    return kvc_status::OK;
}

kvc_status kv_dyadic_chunks_from_llama(
        llama_context * ctx,
        const kvc_metadata & metadata,
        const std::vector<int32_t> & token_ids,
        std::vector<kvc_dyadic_chunk> & chunks) {
    return kv_dyadic_chunks_from_llama(ctx, metadata, token_ids, std::string(), chunks);
}

kvc_status kv_dyadic_chunks_from_llama(
        llama_context * ctx,
        const kvc_metadata & metadata,
        const std::vector<int32_t> & token_ids,
        const std::string & model_identity,
        std::vector<kvc_dyadic_chunk> & chunks) {
    if (!ctx || metadata.offset != 0 || metadata.n_tokens == 0 || metadata.seq_id >= llama_n_ctx_seq(ctx)
            || token_ids.size() != metadata.n_tokens) {
        return kvc_status::INVALID_ARGUMENT;
    }

    const auto intervals = kvc_dyadic_partition(metadata.offset, metadata.n_tokens);
    if (intervals.empty()) {
        return kvc_status::INVALID_ARGUMENT;
    }

    llama_kv_cache * cache = get_kv_cache(ctx);
    if (!cache) {
        return kvc_status::UNSUPPORTED;
    }
    const std::vector<uint32_t> layer_ids = cache->get_layer_ids();
    if (!cache_supports_dyadic(cache, layer_ids)) {
        return kvc_status::UNSUPPORTED;
    }

    std::vector<uint32_t> source_slots;
    const llama_seq_id seq_id = static_cast<llama_seq_id>(metadata.seq_id);
    if (!get_contiguous_slots(*cache, seq_id, metadata.offset, metadata.n_tokens, source_slots)) {
        return kvc_status::MISSING_POSITIONS;
    }

    std::string profile;
    const bool profile_ok = model_identity.empty()
            ? kvc_get_model_profile(ctx, profile)
            : kvc_get_dyadic_model_profile(ctx, model_identity, profile);
    if (!profile_ok) {
        return kvc_status::BACKEND_ERROR;
    }

    const uint32_t stream = cache->get_stream(seq_id);
    llama_synchronize(ctx);
    std::vector<kvc_dyadic_chunk> result;
    result.reserve(intervals.size());
    std::string previous_hash = KVC_DYADIC_HASH_SEED;

    for (const auto & interval : intervals) {
        kvc_dyadic_chunk chunk;
        chunk.offset = interval.offset;
        chunk.n_tokens = interval.n_tokens;
        chunk.level = interval.level;
        chunk.index = interval.index;
        chunk.model_profile = profile;
        chunk.prefix_hash = previous_hash;
        chunk.token_ids.assign(token_ids.begin() + (interval.offset - metadata.offset),
                token_ids.begin() + (interval.offset - metadata.offset + interval.n_tokens));

        const auto first = source_slots.begin() + (interval.offset - metadata.offset);
        const std::vector<uint32_t> chunk_slots(first, first + interval.n_tokens);
        chunk.layers.reserve(layer_ids.size());
        for (const uint32_t layer_idx : layer_ids) {
            ggml_tensor * k = cache->get_k_storage(static_cast<int32_t>(layer_idx));
            ggml_tensor * v = cache->get_v_storage(static_cast<int32_t>(layer_idx));
            size_t k_row_bytes = 0;
            size_t v_row_bytes = 0;
            if (!get_row_bytes(k, k_row_bytes) || !get_row_bytes(v, v_row_bytes)
                    || interval.n_tokens > std::numeric_limits<size_t>::max() / k_row_bytes
                    || interval.n_tokens > std::numeric_limits<size_t>::max() / v_row_bytes) {
                return kvc_status::UNSUPPORTED;
            }

            kvc_dyadic_layer layer;
            layer.layer_idx = layer_idx;
            layer.type_k = ggml_type_name(k->type);
            layer.type_v = ggml_type_name(v->type);
            layer.k_data.resize(size_t(interval.n_tokens) * k_row_bytes);
            layer.v_data.resize(size_t(interval.n_tokens) * v_row_bytes);
            if (!copy_row_range(k, stream, chunk_slots, k_row_bytes, layer.k_data.data(), nullptr)) {
                return kvc_status::BACKEND_ERROR;
            }
            if (cache->get_v_trans()) {
                if (ggml_blck_size(v->type) != 1
                        || cache->get_n_embd_head_v(layer_idx) == 0
                        || v->ne[0] % cache->get_n_embd_head_v(layer_idx) != 0) {
                    return kvc_status::UNSUPPORTED;
                }
                if (!copy_transposed_v_range(v, stream, cache->get_n_embd_head_v(layer_idx),
                        chunk_slots, layer.v_data.data(), nullptr)) {
                    return kvc_status::BACKEND_ERROR;
                }
            } else if (!copy_row_range(v, stream, chunk_slots, v_row_bytes, layer.v_data.data(), nullptr)) {
                return kvc_status::BACKEND_ERROR;
            }
            chunk.layers.push_back(std::move(layer));
        }

        chunk.content_hash = kvc_dyadic_content_hash(chunk);
        chunk.payload_hash = kvc_dyadic_payload_hash(chunk);
        previous_hash = chunk.content_hash;
        result.push_back(std::move(chunk));
    }

    chunks = std::move(result);
    return kvc_status::OK;
}

kvc_status kv_dyadic_chunks_to_llama(
        llama_context * ctx,
        uint32_t seq_id,
        const std::vector<kvc_dyadic_chunk> & chunks) {
    return kv_dyadic_chunks_to_llama(ctx, seq_id, chunks, std::string());
}

kvc_status kv_dyadic_chunks_to_llama(
        llama_context * ctx,
        uint32_t seq_id,
        const std::vector<kvc_dyadic_chunk> & chunks,
        const std::string & model_identity) {
    if (!ctx || seq_id >= llama_n_ctx_seq(ctx) || chunks.empty()) {
        return kvc_status::INVALID_ARGUMENT;
    }
    if (chunks.front().offset != 0) {
        return kvc_status::INCOMPATIBLE;
    }

    llama_kv_cache * cache = get_kv_cache(ctx);
    if (!cache) {
        return kvc_status::UNSUPPORTED;
    }
    const std::vector<uint32_t> layer_ids = cache->get_layer_ids();
    if (!cache_supports_dyadic(cache, layer_ids)) {
        return kvc_status::UNSUPPORTED;
    }

    std::string profile;
    const bool profile_ok = model_identity.empty()
            ? kvc_get_model_profile(ctx, profile)
            : kvc_get_dyadic_model_profile(ctx, model_identity, profile);
    if (!profile_ok) {
        return kvc_status::BACKEND_ERROR;
    }

    uint64_t total_tokens = 0;
    for (const auto & chunk : chunks) {
        if (!kvc_dyadic_chunk_is_valid(chunk) || chunk.model_profile != profile) {
            return kvc_status::INCOMPATIBLE;
        }
        if (chunk.offset != chunks.front().offset + total_tokens) {
            return kvc_status::INCOMPATIBLE;
        }
        total_tokens += chunk.n_tokens;
        if (total_tokens > std::numeric_limits<uint32_t>::max()) {
            return kvc_status::INCOMPATIBLE;
        }
    }

    const auto expected_intervals = kvc_dyadic_partition(
            chunks.front().offset, static_cast<uint32_t>(total_tokens));
    if (expected_intervals.size() != chunks.size()) {
        return kvc_status::INCOMPATIBLE;
    }
    std::string expected_prefix = KVC_DYADIC_HASH_SEED;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto & chunk = chunks[i];
        const auto & interval = expected_intervals[i];
        if (chunk.offset != interval.offset || chunk.n_tokens != interval.n_tokens
                || chunk.level != interval.level || chunk.index != interval.index
                || chunk.prefix_hash != expected_prefix
                || chunk.layers.size() != layer_ids.size()) {
            return kvc_status::INCOMPATIBLE;
        }
        for (size_t layer = 0; layer < layer_ids.size(); ++layer) {
            const auto & payload = chunk.layers[layer];
            const uint32_t layer_idx = layer_ids[layer];
            ggml_tensor * k = cache->get_k_storage(static_cast<int32_t>(layer_idx));
            ggml_tensor * v = cache->get_v_storage(static_cast<int32_t>(layer_idx));
            size_t k_row_bytes = 0;
            size_t v_row_bytes = 0;
            if (payload.layer_idx != layer_idx || payload.type_k != ggml_type_name(k->type)
                    || payload.type_v != ggml_type_name(v->type)
                    || !get_row_bytes(k, k_row_bytes) || !get_row_bytes(v, v_row_bytes)
                    || chunk.n_tokens > std::numeric_limits<size_t>::max() / k_row_bytes
                    || chunk.n_tokens > std::numeric_limits<size_t>::max() / v_row_bytes
                    || payload.k_data.size() != size_t(chunk.n_tokens) * k_row_bytes
                    || payload.v_data.size() != size_t(chunk.n_tokens) * v_row_bytes
                    || (cache->get_v_trans()
                            && ggml_blck_size(v->type) != 1)) {
                return kvc_status::INCOMPATIBLE;
            }
        }
        expected_prefix = chunk.content_hash;
    }

    const uint64_t end = uint64_t(chunks.front().offset) + total_tokens;
    if (end > static_cast<uint64_t>(llama_n_ctx(ctx))
            || end > cache->get_size()) {
        return kvc_status::INCOMPATIBLE;
    }

    const llama_seq_id destination = static_cast<llama_seq_id>(seq_id);
    auto & cells = cache->get_cells_mutable(destination);
    if (!cells.seq_pos_get(destination).empty()) {
        return kvc_status::INVALID_ARGUMENT;
    }

    std::vector<uint32_t> destination_slots;
    destination_slots.reserve(static_cast<size_t>(total_tokens));
    for (uint32_t i = 0; i < cells.size() && destination_slots.size() < total_tokens; ++i) {
        if (cells.is_empty(i)) {
            destination_slots.push_back(i);
        }
    }
    if (destination_slots.size() != total_tokens) {
        return kvc_status::BACKEND_ERROR;
    }

    llama_synchronize(ctx);
    const uint32_t stream = cache->get_stream(destination);
    for (const auto & chunk : chunks) {
        const size_t chunk_index = static_cast<size_t>(chunk.offset - chunks.front().offset);
        const std::vector<uint32_t> chunk_slots(
                destination_slots.begin() + chunk_index,
                destination_slots.begin() + chunk_index + chunk.n_tokens);

        for (size_t layer = 0; layer < layer_ids.size(); ++layer) {
            ggml_tensor * k = cache->get_k_storage(static_cast<int32_t>(layer_ids[layer]));
            ggml_tensor * v = cache->get_v_storage(static_cast<int32_t>(layer_ids[layer]));
            size_t k_row_bytes = 0;
            size_t v_row_bytes = 0;
            if (!get_row_bytes(k, k_row_bytes) || !get_row_bytes(v, v_row_bytes)
                    || !copy_row_range(k, stream, chunk_slots, k_row_bytes,
                            nullptr, chunk.layers[layer].k_data.data())) {
                return kvc_status::BACKEND_ERROR;
            }

            const bool v_ok = cache->get_v_trans()
                    ? copy_transposed_v_range(v, stream, cache->get_n_embd_head_v(layer_ids[layer]),
                            chunk_slots, nullptr, chunk.layers[layer].v_data.data())
                    : copy_row_range(v, stream, chunk_slots, v_row_bytes, nullptr,
                            chunk.layers[layer].v_data.data());
            if (!v_ok) {
                return kvc_status::BACKEND_ERROR;
            }
        }
    }

    llama_synchronize(ctx);
    size_t token_index = 0;
    for (const auto & chunk : chunks) {
        for (uint32_t i = 0; i < chunk.n_tokens; ++i, ++token_index) {
            const uint32_t slot = destination_slots[token_index];
            cells.pos_set(slot, static_cast<llama_pos>(uint64_t(chunk.offset) + i));
            cells.seq_add(slot, destination);
        }
    }
    return kvc_status::OK;
}

} // namespace kvclient::detail
