## kvclient-llama.cpp

#### Data structs:

- ```llama_context``` : manage resources required to perform inference (model, memmory, backend, ...)

- ```llama_kv_cache``` : manange key and value cache, extends ```llama_memmory_t```.

- ```kvc_context``` : contains model profile.

- ```kvc_chunk``` & ```kvc_dyadic_chunk``` : manage ```offset```, ```n_tokens``` and ``` state_data```  

#### API functions:

A basic inference pipeline is: decoding -> export kvc_chunk -> put() to local -> network peer exchaging -> get() from local -> import kvc_chunk -> decoding 

- ``` kvc_export_chunks() ``` : llama_context -> kvc_chunk

- ``` kvc_export_dyadic_chunks() ``` : llama_context -> kvc_dyadic_chunk

- ``` kvc_chunk_serialize() ``` :  kvc_chunk -> std::vector<uint8_t>

- ``` kvc_dyadic_chunk_serialize() ``` :  kvc_dyadic_chunk -> std::vector<uint8_t>

- ``` kvc_dyadic_file_storage::put/get/remove``` : read or write kv serialized tensor to local files.

- ``` kvc_chunk_deserialize() ``` : std::vector<uint8_t> ->  kvc_chunk

- ``` kvc_dyadic_chunk_deserialize() ``` : std::vector<uint8_t> -> kvc_dyadic_chunk

- ``` kvc_import_chunks() ``` :  kvc_chunk -> llama_context

- ``` kvc_import_dyadic_chunks() ``` :  kvc_dyadic_chunk -> llama_context

#### Examples

- Prompt: ```"What is Blockchain?"```

```
KV cache hit: 0/1 chunks, reused 0/4 prompt tokens
Stored dyadic KV chunk offset=0 tokens=4 level=2 index=0 key=fe40cbad28d20a17
...
eval time =   68970.57 ms /     4 tokens (17242.64 ms per token,     0.06 tokens per second)
```

- Prompt again: ```"What is Blockchain?"```

```
KV cache hit: 1/1 chunks, reused 4/4 prompt tokens
...
eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
```

- Prompt: ```"What is Blockchain? Tell about application"```
```
KV cache hit: 1/3 chunks, reused 4/7 prompt tokens
Stored dyadic KV chunk offset=0 tokens=4 level=2 index=0 key=fe40cbad28d20a17
Stored dyadic KV chunk offset=4 tokens=2 level=1 index=2 key=f44f87b307f5bb59
Stored dyadic KV chunk offset=6 tokens=1 level=0 index=6 key=02a131a8af2a5dda
...
eval time =   69911.46 ms /     3 tokens (23303.82 ms per token,     0.04 tokens per second)
```