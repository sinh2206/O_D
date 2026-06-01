from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR


def resolve_device(preferred: str = "auto") -> torch.device:
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
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    mem_gb = props.total_memory / float(1024**3)
    return f"cuda:{idx} ({props.name}, {mem_gb:.1f} GB)"


def _cpu_limit_from_affinity() -> int:
    cpu_count = os.cpu_count() or 1
    try:
        aff = len(os.sched_getaffinity(0))
        if aff > 0:
            cpu_count = min(cpu_count, aff)
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


def configure_precision_runtime(enable_tf32: bool = True) -> None:
    tf32 = bool(enable_tf32)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high")


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False)
            except TypeError:
                pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "min"
    best: Optional[float] = None
    bad_epochs: int = 0

    def update(self, value: float) -> Tuple[bool, bool]:
        current = float(value)
        improved = False
        mode = str(self.mode).strip().lower()

        if self.best is None:
            improved = True
        elif mode == "min":
            improved = current < (float(self.best) - float(self.min_delta))
        elif mode == "max":
            improved = current > (float(self.best) + float(self.min_delta))
        else:
            raise ValueError(f"Unsupported EarlyStopping mode: {self.mode}")

        if improved:
            self.best = current
            self.bad_epochs = 0
            return True, False

        self.bad_epochs += 1
        should_stop = self.bad_epochs >= max(1, int(self.patience))
        return False, bool(should_stop)


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
    mode: str = "cosine",
    warmup_epochs: int = 3,
    min_lr_ratio: float = 0.05,
    plateau_factor: float = 0.5,
    plateau_patience: int = 2,
    plateau_min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    mode = str(mode).strip().lower()
    if mode == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(plateau_factor),
            patience=max(1, int(plateau_patience)),
            min_lr=float(plateau_min_lr),
        )

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
