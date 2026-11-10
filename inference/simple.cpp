#include "llama.h"
#include "kvclient/kvclient.h"
#include "kvc-inspect.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
};

const char * status_name(kvclient::kvc_status status) {
    switch (status) {
        case kvclient::kvc_status::OK: return "OK";
        case kvclient::kvc_status::INVALID_ARGUMENT: return "invalid argument";
        case kvclient::kvc_status::MISSING_POSITIONS: return "sequence positions are incomplete";
        case kvclient::kvc_status::INCOMPATIBLE: return "incompatible model or chunk";
        case kvclient::kvc_status::UNSUPPORTED: return "unsupported cache format";
        case kvclient::kvc_status::BACKEND_ERROR: return "llama backend error";
    }
    return "unknown error";
}

bool token_piece(const llama_vocab * vocab, llama_token token, std::string & piece) {
    std::vector<char> buffer(128);
    int32_t n = llama_token_to_piece(
            vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    if (n < 0) {
        buffer.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(
                vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    }
    if (n < 0) {
        return false;
    }
    piece.assign(buffer.data(), static_cast<size_t>(n));
    return true;
}

bool make_model_identity(const std::string & model_path, std::string & identity) {
    std::error_code ec;
    const std::filesystem::path canonical_path = std::filesystem::canonical(model_path, ec);
    if (ec) {
        std::fprintf(stderr, "error: unable to canonicalize model path: %s\n", ec.message().c_str());
        return false;
    }

    const uintmax_t file_size = std::filesystem::file_size(canonical_path, ec);
    if (ec) {
        std::fprintf(stderr, "error: unable to read model file size: %s\n", ec.message().c_str());
        return false;
    }
    const auto modified = std::filesystem::last_write_time(canonical_path, ec);
    if (ec) {
        std::fprintf(stderr, "error: unable to read model modification time: %s\n", ec.message().c_str());
        return false;
    }

    identity = canonical_path.generic_string()
            + "|bytes=" + std::to_string(file_size)
            + "|mtime=" + std::to_string(modified.time_since_epoch().count());
    return true;
}

bool decode_tokens_at_positions(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        size_t first_token) {
    if (!ctx || first_token >= tokens.size()
            || tokens.size() - first_token > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    const int32_t n_tokens = static_cast<int32_t>(tokens.size() - first_token);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return false;
    }

    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const size_t token_index = first_token + static_cast<size_t>(i);
        batch.token[i] = tokens[token_index];
        batch.pos[i] = static_cast<llama_pos>(token_index);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == n_tokens - 1 ? 1 : 0;
    }

    const int32_t result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return result == 0;
}

bool parse_arguments(
        int argc,
        char ** argv,
        std::string & model_path,
        std::string & prompt,
        std::string & storage_directory,
        int & n_predict,
        bool & use_dyadic,
        bool & show_help) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::printf("Usage: inference [--model PATH] [--prompt TEXT] [--predict TOKENS] [--dyadic] [--storage DIR]\n");
            show_help = true;
            return true;
        }
        if (arg == "--dyadic") {
            use_dyadic = true;
            continue;
        }
        if (arg != "--model" && arg != "--prompt" && arg != "--predict" && arg != "--storage") {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
        if (++i >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
            return false;
        }

        if (arg == "--model") {
            model_path = argv[i];
        } else if (arg == "--prompt") {
            prompt = argv[i];
        } else if (arg == "--storage") {
            storage_directory = argv[i];
        } else {
            char * end = nullptr;
            const long value = std::strtol(argv[i], &end, 10);
            if (!end || *end != '\0' || value < 1 || value > std::numeric_limits<int>::max()) {
                std::fprintf(stderr, "--predict must be a positive integer\n");
                return false;
            }
            n_predict = static_cast<int>(value);
        }
    }
    if (!storage_directory.empty() && !use_dyadic) {
        std::fprintf(stderr, "--storage requires --dyadic\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path = "./weights/Qwen/Qwen3-8b.gguf";
    std::string prompt = "What is Blockchain? Tell about application";
    std::string storage_directory;
    int n_predict = 3;
    bool use_dyadic = false;
    bool show_help = false;
    if (!parse_arguments(argc, argv, model_path, prompt, storage_directory,
                n_predict, use_dyadic, show_help)) {
        return 1;
    }
    if (show_help) {
        return 0;
    }

    backend_guard backend;
    llama_model_params model_params = llama_model_default_params();

    std::fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
    if (!model) {
        std::fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    if (llama_model_has_encoder(model.get())) {
        std::fprintf(stderr, "error: this local KV round-trip demo supports decoder-only models\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    if (!vocab || prompt.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        std::fprintf(stderr, "error: invalid vocabulary or prompt size\n");
        return 1;
    }

    const int32_t token_count = llama_tokenize(
            vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (token_count >= 0) {
        std::fprintf(stderr, "error: failed to determine prompt token count\n");
        return 1;
    }

    if (token_count == std::numeric_limits<int32_t>::min()) {
        std::fprintf(stderr, "error: prompt token count overflow\n");
        return 1;
    }
    const int32_t n_prompt = -token_count;
    if (n_prompt <= 0 || int64_t(n_prompt) + n_predict > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "error: prompt or generation length is out of range\n");
        return 1;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
    const int32_t tokenized = llama_tokenize(
            vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(), n_prompt, true, true);
    if (tokenized < 0 || tokenized != n_prompt) {
        std::fprintf(stderr, "error: failed to tokenize the prompt\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_prompt + n_predict);
    ctx_params.n_batch = static_cast<uint32_t>(n_prompt);
    ctx_params.no_perf = false;

    const kvclient::kvc_metadata metadata { 0, 0, static_cast<uint32_t>(n_prompt) };
    kvclient::kvc_chunk chunk;
    std::vector<kvclient::kvc_dyadic_chunk> dyadic_chunks;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(nullptr, llama_free);
    if (use_dyadic && !storage_directory.empty()) {
        std::string model_identity;
        if (!make_model_identity(model_path, model_identity)) {
            return 1;
        }

        ctx.reset(llama_init_from_model(model.get(), ctx_params));
        if (!ctx) {
            std::fprintf(stderr, "error: failed to create the inference context\n");
            return 1;
        }

        const std::vector<int32_t> dyadic_tokens(prompt_tokens.begin(), prompt_tokens.end());
        std::vector<kvclient::kvc_dyadic_cache_key> cache_keys;
        const auto key_status = kvclient::kvc_dyadic_get_cache_keys(
                ctx.get(), model_identity, dyadic_tokens, cache_keys);
        if (key_status != kvclient::kvc_status::OK) {
            std::fprintf(stderr, "error: unable to derive KV cache keys: %s\n", status_name(key_status));
            return 1;
        }

        kvclient::kvc_dyadic_file_storage disk_storage(storage_directory);
        size_t cached_tokens = 0;
        std::vector<kvclient::kvc_dyadic_chunk> cached_chunks;
        cached_chunks.reserve(cache_keys.size());
        for (const auto & key : cache_keys) {
            kvclient::kvc_dyadic_chunk cached_chunk;
            std::string storage_error;
            const auto storage_status = disk_storage.get(key.content_hash, cached_chunk, &storage_error);
            if (storage_status == kvclient::kvc_file_storage_status::NOT_FOUND) {
                break;
            }
            if (storage_status != kvclient::kvc_file_storage_status::OK) {
                std::fprintf(stderr, "error: unable to read cached KV chunk: %s\n", storage_error.c_str());
                return 1;
            }

            const size_t token_offset = key.interval.offset;
            if (cached_chunk.offset != key.interval.offset
                    || cached_chunk.n_tokens != key.interval.n_tokens
                    || cached_chunk.level != key.interval.level
                    || cached_chunk.index != key.interval.index
                    || cached_chunk.content_hash != key.content_hash
                    || cached_chunk.token_ids.size() != key.interval.n_tokens
                    || !std::equal(cached_chunk.token_ids.begin(), cached_chunk.token_ids.end(),
                            dyadic_tokens.begin() + token_offset)) {
                std::fprintf(stderr, "error: cached KV chunk metadata does not match the requested prompt\n");
                return 1;
            }
            cached_tokens += cached_chunk.n_tokens;
            cached_chunks.push_back(std::move(cached_chunk));
        }

        if (!cached_chunks.empty()) {
            const auto import_status = kvclient::kvc_import_dyadic_chunks(
                    ctx.get(), 0, cached_chunks, model_identity);
            if (import_status != kvclient::kvc_status::OK) {
                std::fprintf(stderr, "error: cached KV import failed: %s\n", status_name(import_status));
                return 1;
            }
        }
        std::fprintf(stderr, "KV cache hit: %zu/%zu chunks, reused %zu/%d prompt tokens\n",
                cached_chunks.size(), cache_keys.size(), cached_tokens, n_prompt);

        if (cached_tokens == static_cast<size_t>(n_prompt)) {
            const llama_pos last_prompt_pos = n_prompt - 1;
            if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, last_prompt_pos, -1)) {
                std::fprintf(stderr, "error: failed to prepare cached KV for final-token evaluation\n");
                return 1;
            }
            if (!decode_tokens_at_positions(ctx.get(), prompt_tokens, static_cast<size_t>(n_prompt - 1))) {
                std::fprintf(stderr, "error: failed to evaluate the final prompt token from cache\n");
                return 1;
            }
        } else if (!decode_tokens_at_positions(ctx.get(), prompt_tokens, cached_tokens)) {
            std::fprintf(stderr, "error: prompt prefill/decode failed after cached prefix\n");
            return 1;
        }

        if (cached_tokens < static_cast<size_t>(n_prompt)) {
            const auto export_status = kvclient::kvc_export_dyadic_chunks(
                    ctx.get(), metadata, dyadic_tokens, model_identity, dyadic_chunks);
            if (export_status != kvclient::kvc_status::OK) {
                std::fprintf(stderr, "error: dyadic KV export failed: %s\n", status_name(export_status));
                return 1;
            }

            for (const auto & dyadic_chunk : dyadic_chunks) {
                std::string storage_error;
                const auto storage_status = disk_storage.put(dyadic_chunk, &storage_error);
                if (storage_status != kvclient::kvc_file_storage_status::OK) {
                    std::fprintf(stderr, "error: unable to persist dyadic chunk: %s\n", storage_error.c_str());
                    return 1;
                }
                std::fprintf(stderr,
                        "Stored dyadic KV chunk offset=%u tokens=%u level=%u index=%llu key=%s\n",
                        dyadic_chunk.offset, dyadic_chunk.n_tokens, dyadic_chunk.level,
                        static_cast<unsigned long long>(dyadic_chunk.index),
                        dyadic_chunk.content_hash.c_str());
            }
            std::fprintf(stderr, "Persisted %zu dyadic chunks under %s\n",
                    dyadic_chunks.size(), storage_directory.c_str());
            std::vector<kvclient::kvc_dyadic_chunk>().swap(dyadic_chunks);
        }
        llama_perf_context_print(ctx.get());
    } else {
        std::unique_ptr<llama_context, decltype(&llama_free)> prefill_ctx(
                llama_init_from_model(model.get(), ctx_params), llama_free);
        if (!prefill_ctx) {
            std::fprintf(stderr, "error: failed to create the prefill context\n");
            return 1;
        }

        llama_batch prompt_batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
        if (llama_decode(prefill_ctx.get(), prompt_batch) != 0) {
            std::fprintf(stderr, "error: prompt prefill failed\n");
            return 1;
        }
        std::fprintf(stderr, "Prefilled %d prompt tokens\n", n_prompt);

        if (use_dyadic) {
            const std::vector<int32_t> dyadic_tokens(prompt_tokens.begin(), prompt_tokens.end());
            const auto export_status = kvclient::kvc_export_dyadic_chunks(
                    prefill_ctx.get(), metadata, dyadic_tokens, dyadic_chunks);
            if (export_status != kvclient::kvc_status::OK) {
                std::fprintf(stderr, "error: dyadic KV export failed: %s\n", status_name(export_status));
                return 1;
            }
            for (const auto & dyadic_chunk : dyadic_chunks) {
                std::fprintf(stderr,
                        "Exported dyadic KV chunk offset=%u tokens=%u level=%u index=%llu key=%s payload=%s\n",
                        dyadic_chunk.offset, dyadic_chunk.n_tokens, dyadic_chunk.level,
                        static_cast<unsigned long long>(dyadic_chunk.index),
                        dyadic_chunk.content_hash.c_str(), dyadic_chunk.payload_hash.c_str());
            }
            std::fprintf(stderr, "Exported %zu dyadic KV chunks\n", dyadic_chunks.size());
        } else {
            const auto export_status = kvclient::kvc_export_chunk(prefill_ctx.get(), metadata, chunk);
            if (export_status != kvclient::kvc_status::OK) {
                std::fprintf(stderr, "error: KV export failed: %s\n", status_name(export_status));
                return 1;
            }
            std::fprintf(stderr, "Exported KV state (%zu bytes, hash %s)\n",
                    chunk.state_data.size(), chunk.content_hash.c_str());
        }
        llama_perf_context_print(prefill_ctx.get());
        prefill_ctx.reset();

        ctx.reset(llama_init_from_model(model.get(), ctx_params));
        if (!ctx) {
            std::fprintf(stderr, "error: failed to create the inference context\n");
            return 1;
        }

        const auto import_status = use_dyadic
                ? kvclient::kvc_import_dyadic_chunks(ctx.get(), 0, dyadic_chunks)
                : kvclient::kvc_import_chunk(ctx.get(), 0, chunk);
        if (import_status != kvclient::kvc_status::OK) {
            std::fprintf(stderr, "error: KV import failed: %s\n", status_name(import_status));
            return 1;
        }
        std::fprintf(stderr, "Imported KV state into a fresh context\n");

        const llama_pos last_prompt_pos = n_prompt - 1;
        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, last_prompt_pos, -1)) {
            std::fprintf(stderr, "error: failed to prepare the imported context for decoding\n");
            return 1;
        }

        if (!decode_tokens_at_positions(ctx.get(), prompt_tokens, static_cast<size_t>(n_prompt - 1))) {
            std::fprintf(stderr, "error: failed to evaluate the final prompt token after import\n");
            return 1;
        }
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_chain_init(sampler_params), llama_sampler_free);
    if (!sampler) {
        std::fprintf(stderr, "error: failed to create the sampler\n");
        return 1;
    }
    llama_sampler * greedy = llama_sampler_init_greedy();
    if (!greedy) {
        std::fprintf(stderr, "error: failed to create the greedy sampler\n");
        return 1;
    }
    llama_sampler_chain_add(sampler.get(), greedy);

    std::printf("%s", prompt.c_str());
    for (int i = 0; i < n_predict; ++i) {
        llama_token next_token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, next_token)) {
            break;
        }

        std::string piece;
        if (!token_piece(vocab, next_token, piece)) {
            std::fprintf(stderr, "error: failed to convert generated token to text\n");
            return 1;
        }
        std::printf("%s", piece.c_str());
        std::fflush(stdout);

        if (i + 1 < n_predict) {
            llama_batch next_batch = llama_batch_get_one(&next_token, 1);
            if (llama_decode(ctx.get(), next_batch) != 0) {
                std::fprintf(stderr, "error: generation decode failed\n");
                return 1;
            }
        }
    }
    std::printf("\n");

    llama_debug_print_k_metadata(ctx.get(), 0);
    llama_perf_sampler_print(sampler.get());
    llama_perf_context_print(ctx.get());
    return 0;
}
