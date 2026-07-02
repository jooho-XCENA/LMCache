# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Maru L1 memory manager tier.

Coverage:

1. ``L1MemoryManagerConfig.maru_config`` — when set, the DRAM sizing
   fields are ignored and the init-size clamp is skipped.
2. ``MaruL1MemoryManager`` construction — requires ``maru_config``;
   the allocator is built lazily (no MaruServer RPC).
3. ``MaruL1MemoryManager.get_memory_usage()`` — best-effort forwarding
   to ``MaruHandler.get_stats``; short-circuits to ``(0, 0)`` before
   ``init_layout`` is called.
4. ``MaruL1MemoryManager.get_l1_memory_desc()`` — raises
   ``NotImplementedError`` (no contiguous local buffer).
5. ``MaruL1MemoryManager.register_kv_layout()`` — forwards to
   ``MaruMemoryAllocator.init_layout``.

The maru runtime (``maru``, ``maru_lmcache``) is NOT required: the
lazy ``MaruMemoryAllocator.__init__`` performs no RPC. Tests that
need an "initialized" allocator install ``MagicMock`` handler +
adapter directly on the instance.
"""

# Standard
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat

try:
    # First Party
    from lmcache.v1.distributed.maru_memory_allocator import (
        MaruL1Config,
        MaruMemoryAllocator,
    )
    from lmcache.v1.distributed.memory_manager import (
        MaruL1MemoryManager,
        create_memory_allocator,
    )
except ImportError:
    pytest.skip(
        "MaruL1MemoryManager / memory_manager could not be imported",
        allow_module_level=True,
    )


@pytest.fixture
def maru_cfg() -> MaruL1Config:
    """Plausible MaruL1Config — the lazy ``MaruMemoryAllocator.__init__``
    performs no MaruServer RPC, so these values are not exercised unless
    a test explicitly drives ``init_layout``.
    """
    return MaruL1Config(
        server_url="maru://localhost:5555",
        pool_size_bytes=60 * 1024**3,
        instance_id="test-mp",
    )


# Tiny allocations so the tests don't pin gigabytes of host memory
# and starve subsequent ``MixedMemoryAllocator`` tests in the same process.
# ``LazyMemoryAllocator.__init__`` eagerly calls ``torch.empty(final_size)``
# and ``cudaHostRegister`` on ``init_size`` — so we keep both ≤ 1MB.
_TINY_BYTES = 1 << 20  # 1MB


# =========================================================================
# (1) L1MemoryManagerConfig — maru_config field
# =========================================================================


class TestL1MemoryManagerConfigMaru:
    def test_default_has_no_maru_config(self):
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=False)
        assert cfg.maru_config is None

    def test_default_clamps_init_size(self):
        # init_size_in_bytes defaults to 20GB; size_in_bytes=1MB → clamp to 1MB.
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=False)
        assert cfg.init_size_in_bytes == _TINY_BYTES

    def test_maru_config_skips_clamp(self, maru_cfg):
        # size_in_bytes=0 is OK when maru_config is set (DRAM fields ignored).
        # The default init_size_in_bytes (20GB) should NOT be clamped to 0.
        cfg = L1MemoryManagerConfig(
            size_in_bytes=0, use_lazy=False, maru_config=maru_cfg
        )
        assert cfg.maru_config is maru_cfg
        assert cfg.init_size_in_bytes == 20 << 30


# =========================================================================
# (2) MaruL1MemoryManager construction
# =========================================================================


def _make_maru_manager(maru_cfg) -> MaruL1MemoryManager:
    """Build a ``MaruL1MemoryManager`` whose allocator is a freshly
    constructed (uninitialized) maru allocator. Tests that exercise
    handler stats need to install ``_handler`` and ``_cxl_adapter``
    mocks on the allocator.
    """
    cfg = L1MemoryManagerConfig(size_in_bytes=0, use_lazy=False, maru_config=maru_cfg)
    return MaruL1MemoryManager(cfg)


class TestMaruL1MemoryManagerConstruction:
    def test_requires_maru_config(self):
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=False)
        with pytest.raises(ValueError, match="maru_config"):
            MaruL1MemoryManager(cfg)

    def test_builds_lazy_maru_allocator(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        alloc = mgr.allocator
        assert isinstance(alloc, MaruMemoryAllocator)
        # Lazy: no MaruServer connection before init_layout.
        assert alloc.is_initialized is False

    def test_use_lazy_is_ignored_for_maru_tier(self, maru_cfg):
        # use_lazy sizes DRAM allocators only; the maru tier must not
        # allocate host memory regardless of the flag.
        cfg = L1MemoryManagerConfig(
            size_in_bytes=_TINY_BYTES, use_lazy=True, maru_config=maru_cfg
        )
        mgr = MaruL1MemoryManager(cfg)
        assert isinstance(mgr.allocator, MaruMemoryAllocator)

    def test_create_memory_allocator_stays_maru_free(self):
        # The generic factory serves the CPU tier only — maru routing
        # lives in the L1Manager tier selection, not here.
        cfg = L1MemoryManagerConfig(size_in_bytes=_TINY_BYTES, use_lazy=True)
        alloc = create_memory_allocator(cfg)
        try:
            assert isinstance(alloc, LazyMemoryAllocator)
        finally:
            alloc.close()


# =========================================================================
# (3) MaruL1MemoryManager.get_memory_usage()
# =========================================================================


def _fake_init_layout(allocator: MaruMemoryAllocator) -> None:
    """Install ``MagicMock`` handler + adapter so the allocator
    behaves as ``is_initialized`` without contacting MaruServer.
    """
    allocator._handler = mock.MagicMock()
    allocator._cxl_adapter = mock.MagicMock()


class TestGetMemoryUsageMaru:
    def test_returns_zero_before_init_layout(self, maru_cfg):
        # Allocator constructed but ``init_layout`` not yet called.
        mgr = _make_maru_manager(maru_cfg)
        assert mgr.get_memory_usage() == (0, 0)

    def test_returns_zero_when_handler_has_no_get_stats(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr.allocator)
        # spec=[] → mock has no attributes (no ``get_stats``)
        mgr.allocator._handler = mock.Mock(spec=[])
        assert mgr.get_memory_usage() == (0, 0)

    def test_forwards_used_and_pool_size_bytes(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr.allocator)
        mgr.allocator._handler.get_stats.return_value = {
            "used_bytes": 1234,
            "pool_size_bytes": 5678,
        }
        assert mgr.get_memory_usage() == (1234, 5678)

    def test_falls_back_to_pool_size_key(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr.allocator)
        mgr.allocator._handler.get_stats.return_value = {
            "used_bytes": 100,
            "pool_size": 999,
        }
        assert mgr.get_memory_usage() == (100, 999)

    def test_returns_zero_on_handler_exception(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        _fake_init_layout(mgr.allocator)
        mgr.allocator._handler.get_stats.side_effect = RuntimeError("boom")
        # Should swallow and return (0, 0) rather than crash.
        assert mgr.get_memory_usage() == (0, 0)


# =========================================================================
# (4) MaruL1MemoryManager.get_l1_memory_desc()
# =========================================================================


class TestGetL1MemoryDescMaru:
    def test_raises_not_implemented_for_maru(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        with pytest.raises(NotImplementedError, match="maru"):
            mgr.get_l1_memory_desc()


# =========================================================================
# (5) MaruL1MemoryManager.register_kv_layout()
# =========================================================================


class TestRegisterKvLayoutMaru:
    def test_forwards_to_allocator_init_layout(self, maru_cfg):
        mgr = _make_maru_manager(maru_cfg)
        shapes = [torch.Size([2, 32, 256, 128])]
        dtypes = [torch.float16]
        with mock.patch.object(MaruMemoryAllocator, "init_layout") as mock_init_layout:
            mgr.register_kv_layout(shapes, dtypes, MemoryFormat.KV_2LTD, 256)
        mock_init_layout.assert_called_once_with(
            shapes, dtypes, MemoryFormat.KV_2LTD, 256
        )
