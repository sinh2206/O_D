from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def resolve_device(preferred: str = "auto") -> torch.device:
    """Choose CPU or CUDA device from a simple user-facing preference string."""

    pref = str(preferred).strip().lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Check Colab runtime GPU setting.")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {preferred}")


def device_summary(device: torch.device) -> str:
    """Format a short human-readable description of the active device."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    mem_gb = props.total_memory / float(1024**3)
    return f"cuda:{idx} ({props.name}, {mem_gb:.1f} GB)"


def _cpu_limit_from_affinity() -> int:
    """Estimate a safe worker upper bound from OS CPU affinity settings."""

    cpu_count = os.cpu_count() or 1
    try:
        aff = len(os.sched_getaffinity(0))
        if aff > 0:
            cpu_count = min(cpu_count, aff)
    except (AttributeError, OSError):
        pass
    return max(1, int(cpu_count))


def resolve_num_workers(requested: int) -> Tuple[int, int]:
    """Resolve DataLoader workers while staying conservative on constrained hosts."""

    max_safe = _cpu_limit_from_affinity()
    if "COLAB_GPU" in os.environ:
        max_safe = min(max_safe, 2)

    if requested < 0:
        if max_safe <= 2:
            return max_safe, max_safe
        return min(4, max_safe), max_safe

    resolved = max(0, min(int(requested), max_safe))
    return resolved, max_safe


def should_pin_memory(device: torch.device) -> bool:
    """Return whether DataLoader pin_memory should be enabled."""

    return device.type == "cuda" and torch.cuda.is_available()


def create_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """Create an AMP GradScaler compatible with multiple PyTorch versions."""

    amp_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=amp_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    best_val_loss: float,
    classes: list[str],
    img_size: int,
) -> None:
    """Save model weights plus optional optimizer/scheduler resume state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "classes": list(classes),
        "img_size": int(img_size),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    map_location: Optional[torch.device] = None,
) -> Tuple[int, float]:
    """Load checkpoint contents into model and optional optimizer objects."""

    ckpt = torch.load(str(path), map_location=map_location)
    state_dict = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = int(ckpt.get("epoch", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return epoch, best_val_loss


def get_optimizer(
    model: torch.nn.Module,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    backbone_lr_factor: float = 0.1,
) -> torch.optim.Optimizer:
    """Build AdamW with separate learning rates for backbone and heads."""

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("backbone.", "backbone_fpn.")):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return AdamW(
        [
            {"params": backbone_params, "lr": float(lr) * float(backbone_lr_factor)},
            {"params": head_params, "lr": float(lr)},
        ],
        weight_decay=float(weight_decay),
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 3,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a warmup + cosine schedule for the full training duration."""

    total_epochs = max(1, int(epochs))
    warm = max(0, min(int(warmup_epochs), total_epochs - 1))

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warm),
        eta_min=float(min_lr_ratio) * optimizer.param_groups[0]["lr"],
    )
    if warm == 0:
        return cosine

    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warm)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warm])
