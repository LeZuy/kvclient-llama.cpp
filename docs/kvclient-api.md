# KV Client API

## Purpose and status

The KV client exposes local llama.cpp sequence-cache export and import operations. It also provides an in-memory chunk store and a local binary codec. There is no network transport implementation yet.

The current chunk payload is llama.cpp's native serialized sequence state, not a portable representation of individual K and V tensors. It can only be restored with the same loaded model instance and a compatible llama.cpp build. The network functions in this document are a proposed transport contract; they must not be treated as implemented APIs.

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

### `kvc_storage`

`kvc_storage` is an in-memory map keyed by `content_hash`:

- `put_chunk(chunk)` validates the chunk and stores an immutable shared copy; returns `false` for invalid chunks.
- `get_chunk(hash)` returns a shared immutable chunk, or `nullptr` if absent.
- `exists(hash)` and `size()` query the map.
- `remove(hash)` deletes an entry.

The store is not persistent, synchronized for concurrent access, or a network cache.

## Current constraints

1. Export currently requires the metadata interval to match the complete sequence range in the context. A request to export only a subrange must fail rather than claim that a whole-sequence state is a smaller chunk.
2. Import requires an empty destination sequence. Callers must not use it as an implicit replace operation.
3. `model_profile` includes the address of the loaded model object. It is useful only as a process-local guard; a different process will have a different address even when loading the same weights.
4. The FNV-1a `content_hash` is not suitable for trust decisions, authentication, or protection against malicious modification.
5. llama.cpp sequence state is an implementation-specific format. A llama.cpp update may require a cache-format version change and compatibility testing.

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
