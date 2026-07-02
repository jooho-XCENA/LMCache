# SPDX-License-Identifier: Apache-2.0
"""Maru (CXL-backed) L1 memory manager."""

# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.maru_memory_allocator import MaruMemoryAllocator
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)


class MaruL1MemoryManager(L1MemoryManager):
    """L1 memory manager for the Maru CXL-backed tier.

    A peer of
    :class:`~lmcache.v1.distributed.memory_manager.l1_memory_manager.L1MemoryManager`
    (CPU pinned-DRAM),
    :class:`~lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager.DevDaxL1MemoryManager`,
    and
    :class:`~lmcache.v1.distributed.memory_manager.gds_l1_memory_manager.GDSL1MemoryManager`.
    It owns the :class:`MaruMemoryAllocator` path: the L1 arena lives in CXL
    pages served by an external MaruServer rather than in local DRAM.

    The allocator starts lazily — no RPC is issued at construction time. The
    CXL pool is typed from the model's KV layout on the first
    :meth:`register_kv_layout` call (forwarded from
    ``MPCacheEngine.register_kv_cache``).

    Note that when this tier is active, ``L1Manager`` runs in pass-through
    mode and drives MaruServer directly via
    :class:`~lmcache.v1.distributed.maru_l1_dispatch.MaruL1Dispatcher`;
    MaruServer owns the page lifecycle and eviction.
    """

    def __init__(self, config: L1MemoryManagerConfig) -> None:
        """Create a Maru L1 memory manager.

        Args:
            config: L1 memory configuration with ``maru_config`` set. The
                DRAM sizing fields (``size_in_bytes`` / ``use_lazy`` /
                ``init_size_in_bytes``) are ignored — capacity is
                ``maru_config.pool_size_bytes``, owned by MaruServer.

        Raises:
            ValueError: If ``maru_config`` is not configured.
        """
        if config.maru_config is None:
            raise ValueError("MaruL1MemoryManager requires maru_config")

        logger.debug(
            "use maru memory allocator: server=%s pool_size=%d bytes",
            config.maru_config.server_url,
            config.maru_config.pool_size_bytes,
        )
        self._allocator = MaruMemoryAllocator(config.maru_config)
        self._size_in_bytes = config.maru_config.pool_size_bytes
        self._align_bytes = config.align_bytes

    @property
    def allocator(self) -> MaruMemoryAllocator:
        """The concrete maru allocator.

        Exposed for callers that need maru-specific extension methods not in
        :class:`~lmcache.v1.memory_management.MemoryAllocatorInterface` —
        ``L1Manager`` hands it to
        :class:`~lmcache.v1.distributed.maru_l1_dispatch.MaruL1Dispatcher`,
        which uses ``handler`` / ``get_by_location`` / ``create_store_handle``.

        Returns:
            The underlying :class:`MaruMemoryAllocator`.
        """
        return cast(MaruMemoryAllocator, self._allocator)

    def get_memory_usage(self) -> tuple[int, int]:
        """Return best-effort ``(used_bytes, total_bytes)`` from MaruHandler.

        Eviction is owned by MaruServer, so this is observability only;
        failures degrade to ``(0, 0)`` instead of crashing the caller.

        Returns:
            ``(used_bytes, total_bytes)`` as reported by the handler stats,
            or ``(0, 0)`` when the pool is not yet initialized or the stats
            query fails.
        """
        allocator = self.allocator
        # Lazy backend — the handler is not built until register_kv_layout.
        if not allocator.is_initialized:
            return 0, 0
        try:
            handler = allocator.handler
            stats = handler.get_stats() if hasattr(handler, "get_stats") else {}
            used = int(stats.get("used_bytes", 0))
            total = int(stats.get("pool_size_bytes", 0) or stats.get("pool_size", 0))
            return used, total
        except Exception:
            logger.exception("Failed to query Maru handler stats")
            return 0, 0

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Unsupported for the maru tier.

        Maru-backed L1 lives in CXL pages mmap'd via the handler; there is no
        single contiguous local buffer to describe, so RDMA-style
        registration of one base pointer does not apply.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "get_l1_memory_desc is not supported for the maru backend "
            "(L1 lives in CXL via mmap, not a single contiguous buffer)."
        )

    def register_kv_layout(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
        chunk_size_in_tokens: int,
    ) -> None:
        """Bind the KV layout to the maru allocator.

        Types the CXL pool via ``MaruMemoryAllocator.init_layout``.
        Idempotent for matching layouts; a layout mismatch on a subsequent
        call raises ``ValueError`` (maru single-model constraint).

        Args:
            shapes: KV chunk shapes (per-layer-group).
            dtypes: KV chunk dtypes aligned with ``shapes``.
            fmt: Memory format.
            chunk_size_in_tokens: LMCache chunk size in tokens.

        Raises:
            ValueError: If a different layout was already registered.
        """
        self.allocator.init_layout(shapes, dtypes, fmt, chunk_size_in_tokens)
