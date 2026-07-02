# Maru CXL-Backed L1 Backend Design

This document describes the Maru L1 backend for LMCache multiprocess mode:
a CXL-backed L1 tier served by an external MaruServer process. It covers the
tier's placement in the L1 stack, the pass-through mode it activates in
`L1Manager`, the contracts it suspends, and its known limits.

## Goals

- Serve L1 KV cache from a CXL memory pool owned by an external MaruServer,
  instead of local pinned DRAM, Device-DAX, or a GDS slab file.
- Keep the maru runtime (`maru`, `maru_lmcache`) an optional dependency:
  nothing imports it unless the backend is selected, and no RPC is issued
  before the first KV-layout registration.
- Leave every other L1 tier and the full default controller stack untouched.

## Non-Goals

- L2 tiering under maru. The CXL pool *is* the capacity tier; MaruServer owns
  placement and eviction, so no L2 adapters are constructed. A maru L2
  adapter (for the standard controller flow) is a separate follow-up.
- P2P orchestration. P2P requires a single registerable local L1 buffer
  (`StorageManager.l1_memory_desc`), which does not exist for a CXL pool
  reached via per-page mmaps. The property raises `NotImplementedError` in
  maru mode.
- Multi-object-group models. The CXL pool is typed once from object group 0;
  `StorageManager.register_kv_layout` rejects `num_object_groups > 1`.

## Placement In The L1 Stack

The MP-mode L1 stack has three layers. Maru plugs into two of them:

```
L1Manager            <- state machine (object dict / TTLLock / eviction)
  L1ManagerProtocol  <- memory-manager tier (CPU | DevDax | GDS | Maru)
    Allocator        <- byte-level allocation
```

`MaruL1MemoryManager` (`memory_manager/maru_l1_memory_manager.py`) is a peer
of `L1MemoryManager` (CPU), `DevDaxL1MemoryManager`, and `GDSL1MemoryManager`,
selected by the same mutually-exclusive `if/elif` chain in
`L1Manager.__init__`. It owns `MaruMemoryAllocator`
(`maru_memory_allocator.py`), which wraps the maru handler and the
`CxlMemoryAdapter` pool.

Unlike the other tiers, selecting maru also flips `L1Manager` into
**pass-through mode**: a `MaruL1Dispatcher` (`maru_l1_dispatch.py`) is
constructed, and every state-machine entry point delegates to it.

## Why Pass-Through Instead Of A Plain Tier

The other tiers change *where bytes live*; ownership of the cache metadata
stays with LMCache, so the in-process state machine remains correct above
them. Maru changes *who owns the tier*:

- **Source of truth moves out of process.** MaruServer owns key existence,
  page lifecycle (`pin_kv` / `delete_kv`), and eviction, and may serve other
  clients. An in-process object dict and TTLLock cannot protect pages it does
  not own — the locks would be false guarantees, and a second eviction policy
  would fight the server's.
- **The operation shape differs.** `L1ManagerProtocol.allocate(layout, count)`
  hands out anonymous buffers that the caller binds to keys. Maru operations
  are key-addressed RPCs (`batch_store`, `batch_pin`, `batch_retrieve`,
  `get_by_location`, `create_store_handle`) — they do not fit the protocol
  without turning it into a lifecycle-aware key-value interface, which is
  exactly what `MaruL1Dispatcher` encapsulates instead.
- **The lock-transition flow differs.** The engine's maru read path stages
  `MemoryObj`s in `reserve_read` and drains them via
  `unsafe_read` / `finish_read`; the atomic write-to-read transition
  (`finish_write_and_reserve_read`) is never exercised.

## Pass-Through Contract

In maru mode the following `L1Manager` behaviors are suspended. Callers that
rely on them must treat maru as a distinct backend:

| Entry point | Default backends | Maru mode |
|---|---|---|
| `reserve_write` / `finish_write` | state machine + TTLLock | dispatcher RPC (`create_store_handle` → `batch_store`) |
| `reserve_read` / `unsafe_read` / `finish_read` | state machine + TTLLock | dispatcher RPC (`batch_pin` / `batch_retrieve` → side channel → `batch_unpin`) |
| `delete` / `clear` | frees local objects | dispatcher RPC; `clear` drops only local staging state |
| `register_listener` | listener invoked on events | silently dropped (no controllers run) |
| `touch_keys` | LRU bookkeeping | no-op (MaruServer owns eviction) |
| `is_key_evictable` | consults lock state | always `True` (never consulted on the hot path) |
| `get_object_state` | returns `L1ObjectState` | always `None` (no object dict) |
| `memcheck` | allocator bookkeeping check | always `True` (server owns consistency) |

`StorageManager` mirrors this at the next level up: in maru mode its
`__init__` returns early, so the `L1EvictionController`, `StoreController`,
`PrefetchController`, `L2EvictionController`, and all L2 adapters are never
constructed. The adapter registry stays initialized-but-empty, so the L2
lifecycle/query helpers (`report_status`, `reconfigurable_l2_backends`,
quota endpoints) degrade gracefully rather than crash.

## Lazy Pool Bring-Up

`MaruMemoryAllocator.__init__` performs no I/O. The pool is typed on the
first `register_kv_layout` call, which flows:

```
MPCacheEngine.register_kv_cache
  -> LMCacheDrivenTransferModule.register_kv_cache
  -> StorageManager.register_kv_layout   (rejects num_object_groups > 1)
  -> L1Manager.register_kv_layout        (maru tier only; others skip)
  -> MaruL1MemoryManager.register_kv_layout
  -> MaruMemoryAllocator.init_layout     (imports maru runtime, connects,
                                          types the CxlMemoryAdapter pool)
```

`init_layout` is idempotent for matching layouts and raises `ValueError` on a
layout mismatch (single-model constraint).

## Key Encoding

`object_key_to_string` serializes `ObjectKey` as
`model@kv_rank_hex@chunk_hash_hex[@salt]` — the same 3-segment format as the
(follow-up) maru L2 adapter, so both can address one MaruServer index.
Standard L2 adapters use a 4-segment format with an `object_group_id`
segment; maru omits it because the tier serves single-object-group models
only (enforced at registration).

## Configuration

- `--maru-server-url` selects the backend; `--maru-pool-size-gb` (required
  with it) sizes the pool request; `--maru-instance-id` is a stable client
  identity for ownership tracking and restart recovery.
- The DRAM L1 flags (`--l1-size-gb`, `--l1-use-lazy`, `--l1-init-size-gb`)
  are ignored; pass `--l1-size-gb 0`.
- Conflicting backend selections are rejected at config time with
  `ValueError`: maru + GDS (`gds_l1_config`) and maru + Device-DAX
  (`devdax_path`). Maru + P2P fails at startup with a descriptive
  `NotImplementedError` from `StorageManager.l1_memory_desc`.

## Failure Handling And Known Limits

- **Store-failure page leak.** If the engine's D2H store fails after
  `reserve_write`, `finish_write` is never submitted and the reserved CXL
  page (plus its `_pending_write_memobjs` entry) leaks: `free` is a no-op by
  design (MaruServer owns pages) and no write-TTL sweep exists yet.
  `report_status.pending_write_memobjs` makes the growth observable. The
  intended fix is a write-TTL sweep driven by the already-plumbed
  `_write_ttl_seconds`, or an explicit abort hook from the engine.
- **Observability.** `get_memory_usage` is best-effort: it forwards
  MaruHandler `get_stats` and degrades to `(0, 0)` before pool bring-up or on
  stats failure. Controller-based metrics (store/prefetch/eviction loops) do
  not exist in maru mode.
- **Restart semantics.** Pool contents are owned by MaruServer; LMCache holds
  no persistent index. Recovery behavior is governed by MaruServer and the
  `--maru-instance-id` identity, not by LMCache.
