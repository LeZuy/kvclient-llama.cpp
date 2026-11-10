#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C"{
#endif

LLAMA_API void llama_debug_print_k_metadata(llama_context * ctx, llama_seq_id seq_id);

#ifdef __cplusplus
}
#endif
