from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import NUM_CLASSES, STRIDES
from .loss import DetectionLoss


def resolve_device(preferred: str = "auto") -> torch.device:
    preferred = str(preferred).lower()
    if preferred == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    return f"CUDA:{idx} {name} ({total_gb:.1f} GB)"


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    best_val_loss: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def get_optimizer(model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    backbone = list(getattr(model, "backbone_fpn", model).parameters()) if hasattr(model, "backbone_fpn") else []
    head_params: List[torch.nn.Parameter] = []
    for name in ("head_s16", "head_s32"):
        module = getattr(model, name, None)
        if module is not None:
            head_params.extend(list(module.parameters()))

    used = {id(p) for p in backbone + head_params}
    other_params = [p for p in model.parameters() if id(p) not in used]

    param_groups = []
    if backbone:
        param_groups.append({"params": backbone, "lr": lr * 0.5})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})

    return AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 0):
    del warmup_epochs
    return CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DetectionLoss,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp_enabled: bool = False,
) -> Dict[str, float]:
    model.train()
    stats = {"loss": 0.0, "loss_obj": 0.0, "loss_noobj": 0.0, "loss_reg": 0.0, "loss_cls": 0.0, "loss_ctr": 0.0}
    steps = 0

    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for images, targets in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device), "image_id": t.get("image_id", "")}
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        stats["loss"] += float(loss.detach().item())
        for key in ("loss_obj", "loss_noobj", "loss_reg", "loss_cls", "loss_ctr"):
            stats[key] += float(loss_dict.get(key, torch.tensor(0.0, device=device)).detach().item())
        steps += 1

    if steps == 0:
        return {k: float("inf") for k in stats}
    return {k: v / steps for k, v in stats.items()}


@torch.no_grad()
def validate_loss(
    model: torch.nn.Module,
    criterion: DetectionLoss,
    loader,
    device: torch.device,
    amp_enabled: bool = False,
) -> Dict[str, float]:
    model.eval()
    stats = {"loss": 0.0, "loss_obj": 0.0, "loss_noobj": 0.0, "loss_reg": 0.0, "loss_cls": 0.0, "loss_ctr": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = [
            {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device), "image_id": t.get("image_id", "")}
            for t in targets
        ]

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)

        stats["loss"] += float(loss_dict["loss"].detach().item())
        for key in ("loss_obj", "loss_noobj", "loss_reg", "loss_cls", "loss_ctr"):
            stats[key] += float(loss_dict.get(key, torch.tensor(0.0, device=device)).detach().item())
        steps += 1

    if steps == 0:
        return {k: float("inf") for k in stats}
    return {k: v / steps for k, v in stats.items()}


def train(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    device: torch.device,
    criterion: Optional[DetectionLoss] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    amp_enabled: bool = False,
    checkpoint_dir: Optional[Path] = None,
    classes: Optional[Sequence[str]] = None,
    img_size: int = 320,
) -> Dict[str, Any]:
    if criterion is None:
        criterion = DetectionLoss(num_classes=NUM_CLASSES, strides=STRIDES, img_size=img_size)
    if optimizer is None:
        optimizer = get_optimizer(model)
    if scheduler is None:
        scheduler = get_scheduler(optimizer, epochs=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best.pth"
        last_path = checkpoint_dir / "last.pth"
    else:
        best_path = last_path = None

    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device, scaler=scaler, amp_enabled=amp_enabled)
        val_metrics = validate_loss(model, criterion, val_loader, device, amp_enabled=amp_enabled)
        scheduler.step()
        history.append({"epoch": float(epoch), **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if last_path is not None:
            save_checkpoint(last_path, epoch=epoch, model=model, optimizer=optimizer, best_val_loss=best_val_loss, extra={"classes": list(classes or []), "img_size": int(img_size)})
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            if best_path is not None:
                save_checkpoint(best_path, epoch=epoch, model=model, optimizer=optimizer, best_val_loss=best_val_loss, extra={"classes": list(classes or []), "img_size": int(img_size)})

    return {"best_val_loss": best_val_loss, "history": history}
