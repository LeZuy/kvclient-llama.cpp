#include "kvc-inspect.h"
#include "llama-kv-cache.h"

#include <cstdio>

extern "C" LLAMA_API void llama_debug_print_k_metadata(llama_context * ctx, llama_seq_id seq_id){
    llama_memory_t mem = llama_get_memory(ctx); 
    auto * kv = static_cast<llama_kv_cache *>(mem);
    const auto & cells = kv->get_cells(seq_id);

    std::printf("\nKV capacity: %u", kv->get_size());
    std::printf("\nNumber of stream: %u", kv->get_n_stream());
    std::printf("\nStream ID: %u", kv->get_stream(seq_id));
    std::printf("\nKey's type: %s", ggml_type_name(kv->type_k()));
    std::printf("\nValue's type: %s", ggml_type_name(kv->type_v()));

    for (const auto & [pos, slot] : cells.seq_pos_get(seq_id)){
        std::printf("\nPosition: %d -> slot %u", pos, slot);
    }

    std::printf("\nKey ggml tensor metadata:");
    for (uint32_t il : kv->get_layer_ids()){
        ggml_tensor * k = kv->get_k_storage(il);
        std::printf("\nLayer %u: (%lld, %lld, %lld), bytes=%zu", il, (long long)k->ne[0],(long long)k->ne[1], (long long)k->ne[2], ggml_nbytes(k));
    }
}
