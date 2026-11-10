#include "kvc-context.h"

#include "llama.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace kvclient::detail {

bool kvc_get_model_profile(const llama_context * ctx, std::string & profile) {
    if (!ctx) {
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    if (!model) {
        return false;
    }

    std::array<char, 256> description{};
    const int32_t description_size = llama_model_desc(model, description.data(), description.size());
    if (description_size < 0) {
        return false;
    }
    description.back() = '\0';

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        return false;
    }

    profile = std::string(llama_version()) + "|" + description.data()
            + "|layers=" + std::to_string(llama_model_n_layer(model))
            + "|embd=" + std::to_string(llama_model_n_embd(model))
            + "|vocab=" + std::to_string(llama_vocab_n_tokens(vocab))
            + "|model-instance="
            + std::to_string(reinterpret_cast<std::uintptr_t>(model));
    return true;
}

bool kvc_sequence_matches_range(const llama_context * ctx, const kvc_metadata & metadata) {
    if (!ctx || metadata.seq_id >= llama_n_ctx_seq(ctx) || metadata.n_tokens == 0) {
        return false;
    }

    const uint64_t end = uint64_t(metadata.offset) + metadata.n_tokens;
    if (end == 0 || end - 1 > static_cast<uint64_t>(std::numeric_limits<llama_pos>::max())) {
        return false;
    }

    const llama_memory_t memory = llama_get_memory(ctx);
    if (!memory) {
        return false;
    }

    const llama_pos min_pos = llama_memory_seq_pos_min(memory, static_cast<llama_seq_id>(metadata.seq_id));
    const llama_pos max_pos = llama_memory_seq_pos_max(memory, static_cast<llama_seq_id>(metadata.seq_id));
    return min_pos == static_cast<llama_pos>(metadata.offset)
            && max_pos == static_cast<llama_pos>(end - 1);
}

} // namespace kvclient::detail
