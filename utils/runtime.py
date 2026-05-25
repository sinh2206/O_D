from __future__ import annotations

import os
from typing import Any, Tuple

import torch


def _cpu_limit_from_affinity() -> int:
    cpu_count = os.cpu_count() or 1
    try:
        affinity_count = len(os.sched_getaffinity(0))
        if affinity_count > 0:
            cpu_count = min(cpu_count, affinity_count)
    except (AttributeError, OSError):
        pass
    return max(1, int(cpu_count))


def resolve_num_workers(requested: int) -> Tuple[int, int]:
    max_safe = _cpu_limit_from_affinity()
    if "COLAB_GPU" in os.environ:
        max_safe = min(max_safe, 2)

    if requested < 0:
        if max_safe <= 2:
            return max_safe, max_safe
        return min(4, max_safe), max_safe

    resolved = max(0, min(int(requested), max_safe))
    return resolved, max_safe


def resolve_device(requested: str = "auto") -> torch.device:
    req = str(requested).strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            cuda_build = torch.version.cuda or "none"
            raise RuntimeError(
                "Requested CUDA device but torch.cuda.is_available() is False. "
                f"torch.version.cuda={cuda_build}. "
                "On Colab, enable GPU runtime (T4) in Runtime > Change runtime type."
            )
        return torch.device("cuda")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {requested}")


def should_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def create_grad_scaler(device: torch.device, enabled: bool) -> Any:
    amp_enabled = bool(enabled and device.type == "cuda")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=amp_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=amp_enabled)

    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def device_summary(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return str(device)
