"""Device + dtype detection.

The framework is intentionally device-agnostic. Code paths that need device
specifics import these helpers rather than hard-coding `cuda` or `mps`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Captured device state, recorded into experiment manifests."""

    name: str  # "cuda", "mps", "cpu"
    index: int | None  # CUDA device index, if applicable
    backend_version: str
    available_memory_gb: float | None
    supports_bf16: bool
    supports_fp16: bool
    supports_quantization: bool  # bitsandbytes available


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch.device.

    Order of preference: explicit `prefer` arg > CUDA > MPS > CPU.

    Args:
        prefer: If provided, force a specific device name ("cuda", "mps", "cpu").
            Errors if not actually available.

    Returns:
        torch.device.
    """
    import torch

    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")
        if prefer == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unknown device: {prefer}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device | None = None) -> torch.dtype:
    """Pick an appropriate dtype for the given device.

    - CUDA with bf16 support: torch.bfloat16
    - CUDA fallback: torch.float16
    - MPS: torch.float16 (MPS bf16 is unreliable as of PyTorch 2.5)
    - CPU: torch.float32
    """
    import torch

    if device is None:
        device = get_device()

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def describe_device(device: torch.device | None = None) -> DeviceInfo:
    """Return structured info about the device for logging/manifests."""
    # MLX path: stays torch-free (the box may have no torch installed at all).
    if device is not None and getattr(device, "type", None) == "mlx":
        from advsafe.models.mlx_backend import describe_device_mlx

        return DeviceInfo(**describe_device_mlx())

    import torch

    if device is None:
        device = get_device()

    available_gb: float | None = None
    backend_version = torch.__version__
    supports_bf16 = False
    supports_fp16 = True
    supports_quant = False

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        available_gb = props.total_memory / (1024**3)
        backend_version = f"cuda {torch.version.cuda} / torch {torch.__version__}"
        supports_bf16 = torch.cuda.is_bf16_supported()
        try:
            import bitsandbytes  # noqa: F401

            supports_quant = True
        except ImportError:
            supports_quant = False
        return DeviceInfo(
            name="cuda",
            index=idx,
            backend_version=backend_version,
            available_memory_gb=available_gb,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            supports_quantization=supports_quant,
        )

    if device.type == "mps":
        # Apple Silicon doesn't expose a simple "total memory" via torch
        # We report unified memory as a heuristic via system call (best-effort).
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                available_gb = int(result.stdout.strip()) / (1024**3)
        except Exception:  # noqa: BLE001
            available_gb = None
        backend_version = f"mps / torch {torch.__version__}"
        return DeviceInfo(
            name="mps",
            index=None,
            backend_version=backend_version,
            available_memory_gb=available_gb,
            supports_bf16=False,
            supports_fp16=True,
            supports_quantization=False,
        )

    return DeviceInfo(
        name="cpu",
        index=None,
        backend_version=f"cpu / torch {torch.__version__}",
        available_memory_gb=None,
        supports_bf16=False,
        supports_fp16=False,
        supports_quantization=False,
    )
