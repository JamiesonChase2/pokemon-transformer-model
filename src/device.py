"""Device resolution helpers for training/inference scripts."""

from __future__ import annotations

import torch


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def resolve_device(requested: str | None) -> str:
    """Resolve a requested device string to an available torch device.

    Behavior:
    - ``auto`` prefers MPS, then CUDA, then CPU.
    - Explicit unavailable requests gracefully fall back to another available device.
    """

    req = (requested or "auto").strip().lower()

    if req in {"", "auto"}:
        if _mps_available():
            return "mps"
        if _cuda_available():
            return "cuda"
        return "cpu"

    if req == "mps" or req.startswith("mps:"):
        if _mps_available():
            return "mps"
        if _cuda_available():
            return "cuda"
        return "cpu"

    if req == "cuda" or req.startswith("cuda:"):
        if _cuda_available():
            return req
        if _mps_available():
            return "mps"
        return "cpu"

    if req == "cpu":
        return "cpu"

    return req
