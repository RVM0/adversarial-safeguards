"""Deterministic RNG seeding for reproducibility."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RngSnapshot:
    """Captured RNG state — written into every experiment manifest."""

    seed: int
    python_state_hash: str
    numpy_state_hash: str
    torch_state_hash: str
    cuda_state_hash: str | None


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that PyTorch's stack touches.

    Args:
        seed: Master seed; downstream RNGs are seeded from this.
        deterministic: If True, also set deterministic algorithms (slower but
            bit-reproducible). Set False during training (faster, still seeded).
    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS: no separate seeding API; torch.manual_seed covers it.

    if deterministic:
        # These knobs may slow things down but make CUDA bit-reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Newer PyTorch knob
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def capture_rng_state(seed: int) -> RngSnapshot:
    """Snapshot the current RNG state for inclusion in the manifest.

    Hashes the state rather than serializing it, to keep manifests small.
    """
    import hashlib

    import torch

    def _hash(state: object) -> str:
        if hasattr(state, "tobytes"):
            data = state.tobytes()  # type: ignore[union-attr]
        elif isinstance(state, (bytes, bytearray)):
            data = bytes(state)
        else:
            data = repr(state).encode()
        return hashlib.sha256(data).hexdigest()[:16]

    return RngSnapshot(
        seed=seed,
        python_state_hash=_hash(repr(random.getstate())),
        numpy_state_hash=_hash(repr(np.random.get_state())),
        torch_state_hash=_hash(torch.get_rng_state().numpy()),
        cuda_state_hash=(
            _hash(torch.cuda.get_rng_state().numpy()) if torch.cuda.is_available() else None
        ),
    )
