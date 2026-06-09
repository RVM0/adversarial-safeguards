"""Tests for utility modules."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ml_stack


def test_get_device_returns_torch_device():
    import torch

    from advsafe.utils.device import get_device

    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_get_dtype_consistent_with_device():
    import torch

    from advsafe.utils.device import get_device, get_dtype

    dev = get_device()
    dt = get_dtype(dev)
    if dev.type == "cuda":
        assert dt in (torch.bfloat16, torch.float16)
    elif dev.type == "mps":
        assert dt == torch.float16
    else:
        assert dt == torch.float32


def test_describe_device_includes_required_fields():
    from advsafe.utils.device import describe_device

    info = describe_device()
    assert info.name in ("cuda", "mps", "cpu")
    assert info.backend_version


def test_set_global_seed_reproducible():
    import torch

    from advsafe.utils.seeds import set_global_seed

    set_global_seed(42)
    a = torch.randn(10)
    set_global_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_capture_rng_state_returns_snapshot():
    from advsafe.utils.seeds import capture_rng_state, set_global_seed

    set_global_seed(7)
    snap = capture_rng_state(7)
    assert snap.seed == 7
    assert snap.python_state_hash
    assert snap.numpy_state_hash
    assert snap.torch_state_hash
