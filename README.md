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

