from __future__ import annotations

import os
import inspect
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@dataclass(frozen=True)
class DistributedEnv:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int


def get_distributed_env() -> DistributedEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedEnv(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def init_distributed(
    env: DistributedEnv,
    backend: Optional[str] = None,
    timeout_minutes: float = 30.0,
) -> None:
    if not env.enabled:
        return
    if dist.is_initialized():
        return

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requested but CUDA is not available.")

    kwargs = {
        "backend": backend,
        "init_method": "env://",
        "rank": env.rank,
        "world_size": env.world_size,
        "timeout": timedelta(minutes=float(timeout_minutes)),
    }

    # Ensure each rank binds to the expected CUDA device before NCCL init.
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(int(env.local_rank))
        try:
            sig = inspect.signature(dist.init_process_group)
            has_device_id = "device_id" in sig.parameters
        except (TypeError, ValueError):
            has_device_id = False
        if has_device_id:
            kwargs["device_id"] = torch.device("cuda", int(env.local_rank))

    try:
        dist.init_process_group(**kwargs)
    except TypeError:
        # Compatibility fallback for torch versions without `device_id`.
        kwargs.pop("device_id", None)
        dist.init_process_group(**kwargs)


def cleanup_distributed() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()


def is_main_process(env: Optional[DistributedEnv] = None) -> bool:
    if env is not None:
        return int(env.rank) == 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_scalar(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


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


def cuda_inventory() -> list[str]:
    if not torch.cuda.is_available():
        return []
    out: list[str] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / float(1024**3)
        out.append(f"cuda:{idx} {props.name} ({mem_gb:.1f} GB)")
    return out


def compute_mode_label(
    device: torch.device,
    gpu_count: int,
    distributed: bool = False,
    world_size: int = 1,
    data_parallel: bool = False,
) -> str:
    if device.type != "cuda" or gpu_count <= 0:
        return "CPU"
    if distributed and world_size > 1:
        return f"GPUx{world_size} (DDP)"
    if data_parallel and gpu_count > 1:
        return f"GPUx{gpu_count} (DataParallel)"
    return "GPUx1"


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
    warmup_epochs: int = 3,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LRScheduler:
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
