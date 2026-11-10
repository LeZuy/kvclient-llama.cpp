# KV Client API

## Purpose and status

The KV client exposes local llama.cpp sequence-cache export and import operations, including a dyadic chunk path that reads and writes per-layer K/V rows. Dyadic chunks can be stored in memory or persisted as content-addressed `KVD1` files for later retrieval or transport. There is no network transport implementation yet.

The original `kvc_chunk` payload is llama.cpp's native serialized sequence state. The dyadic path uses explicit layer K/V rows and a stable model identity for local cross-run reuse, but its tensor layout remains llama.cpp/GGML-specific. Neither path is a portable network interchange format. The network functions in this document are a proposed transport contract; they must not be treated as implemented APIs.

## Public types

### `kvc_status`

| Value | Meaning |
|---|---|
| `OK` | The operation completed. |
| `INVALID_ARGUMENT` | A pointer, sequence ID, chunk, or destination state is invalid. |
| `MISSING_POSITIONS` | The requested positions are not present as one contiguous sequence range. |
| `INCOMPATIBLE` | The chunk version, model profile, context capacity, or restored position range does not match. |
| `UNSUPPORTED` | Reserved for cache formats or llama.cpp features that the implementation does not support. |
| `BACKEND_ERROR` | llama.cpp or its memory backend could not complete the operation. |

### `kvc_metadata`

Identifies a source sequence and the token-position interval to export:

- `seq_id`: llama.cpp sequence ID.
- `offset`: first token position.
- `n_tokens`: number of positions; must be nonzero.

### `kvc_chunk`

| Member | Meaning |
|---|---|
| `format_version` | Chunk schema version, currently `KVC_CHUNK_FORMAT_VERSION` (`1`). |
| `offset`, `n_tokens` | Position interval represented by the chunk. |
| `model_profile` | Process-local model/build compatibility profile. It currently includes model-instance identity and is not suitable for identifying a model on another process. |
| `content_hash` | 64-bit FNV-1a hash of `state_data`, represented as hexadecimal text. This detects accidental corruption only; it is not a cryptographic digest or authentication tag. |
| `state_data` | Opaque llama.cpp sequence-state bytes. Do not parse or modify this payload in application code. |

### `kvc_dyadic_interval`, `kvc_dyadic_layer`, and `kvc_dyadic_chunk`

`kvc_dyadic_interval` describes a dyadic block: `n_tokens` is `2^level`; `index` is the block index relative to the beginning of the prompt; and `offset` is its llama.cpp token position. `kvc_dyadic_layer` contains one layer's K and V bytes in token-row order plus the GGML type names. `kvc_dyadic_chunk` combines the interval, prompt token IDs, model profile, layer payloads, and hashes.

The prefix decomposition is canonical and uses largest powers of two first. For example, 524 prompt tokens become blocks of 512, 8, and 4 tokens (levels 9, 3, and 2). The key hash chains the previous prefix hash with the level, block index, model profile, and supplied token IDs. `payload_hash` independently covers layer IDs, GGML types, and K/V bytes. Both hashes currently use 64-bit FNV-1a for accidental-corruption detection only; neither is cryptographic.

## Local API

### `kvc_export_chunk`

```cpp
kvc_status kvc_export_chunk(
    llama_context * ctx,
    const kvc_metadata & metadata,
    kvc_chunk & chunk);
```

Exports the requested sequence state into `chunk`. The source sequence must contain exactly the contiguous position range `[offset, offset + n_tokens)`. The current implementation serializes the whole sequence state; it does not extract an arbitrary subrange of KV tensors. The output argument is only replaced on success.

### `kvc_import_chunk`

```cpp
kvc_status kvc_import_chunk(
    llama_context * ctx,
    uint32_t seq_id,
    const kvc_chunk & chunk);
```

Restores a chunk into `seq_id`. The destination sequence must be empty, and it must belong to the same loaded model instance used for export. The function validates the chunk hash, format version, model profile, context capacity, and restored position bounds. If restoration fails after modifying the sequence, it removes the partial sequence state.

### `kvc_chunk_serialize` and `kvc_chunk_deserialize`

```cpp
bool kvc_chunk_serialize(
    const kvc_chunk & chunk,
    std::vector<uint8_t> & data);

bool kvc_chunk_deserialize(
    const std::vector<uint8_t> & data,
    kvc_chunk & chunk);
```

Encode/decode the current `KVC1` local-storage format. Integer fields are written little-endian, followed by length-delimited profile, hash, and native state bytes. Decoding validates the magic, lengths, version, required fields, and content hash; the output chunk is only replaced when decoding succeeds.

This codec is useful for local persistence or same-process storage. It is **not a network interchange format**: in particular, its native llama.cpp payload and process-local model profile cannot be restored by an independent receiver.

### `kvc_dyadic_partition`

```cpp
std::vector<kvc_dyadic_interval> kvc_dyadic_partition(
    uint32_t offset,
    uint32_t n_tokens);
```

Returns the canonical prefix decomposition, or an empty vector for zero length or an out-of-range position. Chunk indexes are relative to the supplied range, while interval offsets retain the supplied llama.cpp position.

### `kvc_export_dyadic_chunks` and `kvc_import_dyadic_chunks`

```cpp
kvc_status kvc_export_dyadic_chunks(
    llama_context * ctx,
    const kvc_metadata & metadata,
    const std::vector<int32_t> & token_ids,
    std::vector<kvc_dyadic_chunk> & chunks);

kvc_status kvc_import_dyadic_chunks(
    llama_context * ctx,
    uint32_t seq_id,
    const std::vector<kvc_dyadic_chunk> & chunks);

kvc_status kvc_export_dyadic_chunks(
    llama_context * ctx,
    const kvc_metadata & metadata,
    const std::vector<int32_t> & token_ids,
    const std::string & model_identity,
    std::vector<kvc_dyadic_chunk> & chunks);

kvc_status kvc_import_dyadic_chunks(
    llama_context * ctx,
    uint32_t seq_id,
    const std::vector<kvc_dyadic_chunk> & chunks,
    const std::string & model_identity);

kvc_status kvc_dyadic_get_cache_keys(
    llama_context * ctx,
    const std::string & model_identity,
    const std::vector<int32_t> & token_ids,
    std::vector<kvc_dyadic_cache_key> & keys);
```

Dyadic export extracts the actual per-layer K/V cache for a prompt prefix starting at position zero and emits one independent chunk per canonical block. The source sequence must contain exactly the metadata range, with no additional positions. `token_ids` must be the exact tokens that were prefilled; llama.cpp does not retain ordinary prompt token IDs in its cache cells, so the API cannot independently prove this. The output vector is replaced only on success.

Import validates chunk versions, interval sequence, prefix chain, hashes, model profile, layer order, tensor types and row sizes before writing to an empty destination sequence. It installs each restored token position into the cache's cell metadata so normal llama.cpp decoding can continue. Like native import, it does not restore logits or sampler state; callers must reproduce any needed final-token evaluation.

The current implementation supports direct llama.cpp KV caches with no cache shifting or extended cell metadata. Transposed V storage is supported only when its GGML block size is one; other transposed V encodings, hybrid/recurrent memory wrappers, and missing K/V layers return `UNSUPPORTED`. K and non-transposed V rows retain their native GGML encoding. Import validates the model profile, compatible cache layout, and llama.cpp build.

The overloads taking `model_identity` use a deterministic profile derived from the caller-supplied identity, llama.cpp version, model description and dimensions, and active K/V cache layout. `kvc_dyadic_get_cache_keys` produces the expected canonical chunk keys before inference, without prefill, so a client can look up matching files first. The legacy overloads without `model_identity` retain a process-local model profile and are not suitable for persistent cache lookup.

### Inference cache reuse

Run the demo with `--dyadic --storage <directory>` to enable persistent cache lookup. Before prompt prefill it derives the dyadic keys from the tokenized prompt and stable model identity, then loads consecutive matching chunks from position zero. It imports the longest cached prefix, evaluates only the uncached prompt suffix, and stores the complete canonical chunk set when there was a miss. A full hit still re-evaluates the final prompt token to regenerate logits, which are not part of KV storage. Logs report the number of hit chunks and reused prompt tokens.

The demo's model identity consists of the canonical model path, file size, and last-write timestamp. This is a practical local invalidation key, not a cryptographic model checksum; replacing a model while preserving all three can incorrectly reuse stale KV. For production, provide a content-derived or trusted immutable model identifier. Cache keys also include the llama.cpp version and active K/V tensor layout. Cache storage from older versions with process-local keys naturally misses and is left untouched.

### `kvc_dyadic_chunk_serialize` and `kvc_dyadic_chunk_deserialize`

```cpp
bool kvc_dyadic_chunk_serialize(
    const kvc_dyadic_chunk & chunk,
    std::vector<uint8_t> & data);

bool kvc_dyadic_chunk_deserialize(
    const std::vector<uint8_t> & data,
    kvc_dyadic_chunk & chunk);
```

Encode/decode the `KVD1` local-storage format. Decoding validates lengths, version, required data, and both hashes; the output chunk changes only on success. This codec makes chunks addressable and storable locally, but does not make their model profile or cache rows portable across processes.

### `kvc_storage`

`kvc_storage` is an in-memory map keyed by `content_hash`:

- `put_chunk(chunk)` validates the chunk and stores an immutable shared copy; returns `false` for invalid chunks.
- `get_chunk(hash)` returns a shared immutable chunk, or `nullptr` if absent.
- `exists(hash)` and `size()` query the map.
- `remove(hash)` deletes an entry.

The store is not persistent, synchronized for concurrent access, or a network cache.

### `kvc_dyadic_storage`

The dyadic store is a separate in-memory map keyed by each chunk's `content_hash`. `put_chunk` validates both hashes and stores an immutable copy; `get_chunk`, `exists`, `remove`, and `size` provide lookup and management. It is not persistent, synchronized, or a network cache.

### `kvc_dyadic_file_storage`

```cpp
#include "kvclient/kvc-file-storage.h"

kvclient::kvc_dyadic_file_storage storage("D:/kvclient-cache");
std::string error;
const auto write_status = storage.put(chunk, &error);

kvclient::kvc_dyadic_chunk restored;
const auto read_status = storage.get(chunk.content_hash, restored, &error);
```

`kvc_dyadic_file_storage` writes one `<content_hash>.kvd` file per dyadic chunk under the configured directory. The file contains the self-describing `KVD1` encoding, including every layer's K and V tensor bytes and metadata. Writes go to a unique temporary directory on the same volume, flush the file to disk, then atomically publish it without replacing an existing file. Repeating an identical write is safe; a different serialized chunk under the same content hash returns `CONFLICT` instead of overwriting existing data.

`get` reads and validates the format and both hashes before replacing the output object. `list` returns sorted content hashes, `size` counts stored chunk files, and `remove` deletes a chunk. Methods return `kvc_file_storage_status` (`OK`, `INVALID_ARGUMENT`, `NOT_FOUND`, `CORRUPT`, `CONFLICT`, or `IO_ERROR`) and can provide a diagnostic through the optional error string. Storage files are independent of the process lifetime, but the currently exported model profile includes a process-local model address; a loaded file can be extracted after restart but cannot yet be imported into a different process/model instance. The K/V byte layout is also llama.cpp/GGML-specific, so this is not yet a portable network wire format.

Example local pipeline:

```powershell
.\build-ninja\bin\inference.exe --dyadic --storage D:\kvclient-cache
```

The demo writes each exported dyadic chunk to SSD, releases the in-memory exported chunks, reads the stored chunks back, then imports them into a fresh context. `storage/` contains the storage implementation; choose a separate data directory for cache files rather than writing generated K/V data into the source tree.

## Current constraints

1. Native-state export covers the complete sequence; it cannot represent an arbitrary KV subrange. The dyadic API is the path for per-range K/V chunks.
2. Both import APIs require an empty destination sequence. Callers must not use them as implicit replace operations.
3. Native `kvc_chunk` profile includes the address of the loaded model object. Dyadic profiles created through the `model_identity` overload are stable across runs when given the same identity and compatible model/cache build.
4. The FNV-1a hashes are not suitable for trust decisions, authentication, or protection against malicious modification. Use a cryptographic digest and authenticated transport before network deployment.
5. llama.cpp cache tensor layout is implementation-specific. A llama.cpp update may require a dyadic-format compatibility change and round-trip testing.
6. Dyadic export binds the chunks to caller-supplied token IDs; it does not introspect prompt tokens from the cache.
7. File storage is a local persistence layer only. Network exchange still needs a portable model fingerprint/cache-layout negotiation, authenticated transport, and explicit framing/size limits.

## Network transfer design (proposed, not implemented)

Do not send `kvc_chunk_serialize()` output over the network and then assume the remote process can import it. A useful network API needs a separately versioned, portable payload and explicit compatibility negotiation.

### Functions to add

The transport should stay independent of sockets and TLS libraries. The application supplies callbacks so it can use its chosen TCP, TLS, QUIC, or other channel:

```cpp
#include "kvclient/kvclient.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kvclient::transport {

struct kvc_peer_hello;
struct kvc_transfer_options;
struct kvc_transfer_limits;
struct kvc_wire_chunk;
struct kvc_frame;

struct kvc_transport_io {
    void * user_data;
    // Return the number of bytes transferred, 0 for EOF, or a negative value on error.
    std::ptrdiff_t (*read)(void * user_data, uint8_t * dst, size_t capacity);
    std::ptrdiff_t (*write)(void * user_data, const uint8_t * src, size_t size);
};

kvc_status kvc_export_wire_chunk(
    llama_context * ctx,
    const kvc_metadata & metadata,
    kvc_wire_chunk & chunk);

kvc_status kvc_import_wire_chunk(
    llama_context * ctx,
    uint32_t seq_id,
    const kvc_wire_chunk & chunk);

bool kvc_wire_encode(
    const kvc_wire_chunk & chunk,
    std::vector<uint8_t> & bytes);

bool kvc_wire_decode(
    const std::vector<uint8_t> & bytes,
    size_t max_bytes,
    kvc_wire_chunk & chunk);

kvc_status kvc_send_frame(
    const kvc_transport_io & io,
    uint32_t message_type,
    const uint8_t * payload,
    size_t payload_size,
    size_t max_frame_size);

kvc_status kvc_receive_frame(
    const kvc_transport_io & io,
    size_t max_frame_size,
    kvc_frame & frame);

kvc_status kvc_exchange_hello(
    const kvc_transport_io & io,
    const kvc_peer_hello & local,
    kvc_peer_hello & remote);

kvc_status kvc_send_wire_chunk(
    const kvc_transport_io & io,
    const kvc_wire_chunk & chunk,
    const kvc_transfer_options & options);

kvc_status kvc_receive_wire_chunk(
    const kvc_transport_io & io,
    const kvc_transfer_limits & limits,
    kvc_wire_chunk & chunk);

} // namespace kvclient::transport
```

These are illustrative declarations, not current headers or ABI. The definitions below describe their required responsibilities:

| Function | Responsibility |
|---|---|
| `kvc_exchange_hello` | Exchange protocol versions, portable cache-format versions, model fingerprints, cache parameters, maximum frame/chunk sizes, and supported encodings. Fail before transferring KV when the peers cannot interoperate. |
| `kvc_export_wire_chunk` | Export a **portable** KV representation from a context. This needs llama.cpp tensor/cache-layout support; it must not merely copy the current native `state_data` blob. |
| `kvc_import_wire_chunk` | Validate a portable chunk and install it into a compatible context and empty sequence. Roll back if any layer or tensor cannot be imported. |
| `kvc_wire_encode` | Convert portable chunk metadata and KV tensors into a canonical, versioned byte representation. Encode fields explicitly; never copy a C++ struct's raw memory. |
| `kvc_wire_decode` | Parse untrusted wire bytes with strict frame, field, and allocation limits. Return a chunk only after all structure and digest checks pass. |
| `kvc_send_frame` | Write one bounded frame containing its length/type header and payload. Loop on partial writes and report EOF/error/timeout distinctly. |
| `kvc_receive_frame` | Read a complete frame despite partial reads. Validate the declared size against configured limits before allocating its payload buffer. |
| `kvc_send_wire_chunk` | Compose wire encoding and framing: validate negotiated compatibility, split the encoded chunk into bounded frames, and send transfer metadata and integrity digest. |
| `kvc_receive_wire_chunk` | Receive and reassemble all frames for one transfer ID, reject missing/duplicate/out-of-order frames as specified by the protocol, verify the digest, and only then return the portable chunk. |

The read/write callbacks must support partial progress. The send/receive helpers must loop until the requested bytes are transferred, EOF occurs, or an I/O error is returned; a single socket read or write is not guaranteed to transfer a complete header or payload.

The future `kvc_wire_chunk` should be a separate type from the current `kvc_chunk`. It should carry a stable model fingerprint and portable per-layer K/V data with explicit tensor types, dimensions, and position metadata. The current native `kvc_chunk` can remain useful for same-process checkpoints without being represented as a network-ready object.

### Wire-format requirements

The portable network format should define, at minimum:

- A fixed magic value, protocol version, message type, and request/transfer ID.
- A declared header length and payload length in a documented byte order (network byte order is recommended).
- A maximum accepted payload/frame size and checked arithmetic before allocating memory.
- A stable model fingerprint (for example, a cryptographic digest of model files or a trusted model identity), plus all cache-layout parameters required to interpret the KV.
- A portable payload schema describing layer identity, K/V tensor types and dimensions, token-position range, and any layout or positional-encoding metadata necessary for correct reuse.
- A cryptographic digest over the canonical header fields and payload. If authenticity is required, use an authenticated channel or a message authentication/signature scheme; a checksum alone does not authenticate a peer.
- Explicit error/status messages, cancellation behavior, and cleanup rules for incomplete transfers.

Avoid serializing C++ structs by copying their memory representation: padding, alignment, endianness, and ABI vary across builds. Validate every length and enum value as untrusted input.

### Suggested send/receive flow

1. The sender exports a supported KV range with `kvc_export_wire_chunk`.
2. Both peers exchange capabilities and compare model and cache-format fingerprints.
3. The sender emits a bounded transfer header and one or more payload frames.
4. The receiver reads exact frame lengths, enforces limits, verifies integrity, and reassembles the chunk.
5. The receiver validates compatibility before importing with `kvc_import_wire_chunk` into an empty destination sequence.
6. The receiver acknowledges success only after import completes; otherwise it returns a specific failure and clears any partial destination state.

### Transport security and reliability

The transport layer should use authenticated TLS (or an equivalent authenticated encrypted channel) for confidentiality, peer identity, and in-transit integrity. The API should impose timeouts, frame/chunk size limits, and cancellation; it should not retry a partially applied import as if it were idempotent. Retries should use a transfer ID and only occur after the receiver has confirmed that the destination sequence is clean or the previous attempt was rolled back.

## Local inference round trip

The local example in `inference/inference.cpp` prefills a prompt, exports its sequence, creates a fresh context, imports the sequence, and evaluates the last prompt token to obtain logits for generation. KV state alone does not include the sampler history or guarantee that next-token logits are ready; callers must preserve any generation state they need and follow llama.cpp's decoding requirements.
