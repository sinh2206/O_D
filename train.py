from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.config import (
    ANCHORS,
    ANCHOR_MASKS,
    CLASS_NAMES,
    CLASS_SAMPLER_WEIGHTS,
    DEFAULT_TRAIN_ANN,
    DEFAULT_VAL_ANN,
    IMG_SIZE,
    LABEL_SMOOTHING,
    STRIDES,
)
from utils.loss import DetectionLoss
from utils.model import YOLOv3
from utils.process import DetectionDataset, collate_fn
from utils.runtime import (
    device_summary,
    get_scheduler,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
    validate_loss,
)

CLASS_WEIGHT_MAP = {
    "person": 0.8,
    "car": 0.8,
    "dog": 1.0,
    "cat": 1.2,
    "chair": 1.5,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_classes_from_annotation(annotation_path: Path, fallback: Sequence[str] = CLASS_NAMES) -> List[str]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    classes = data.get("classes", [])
    if isinstance(classes, list) and classes:
        return [str(c) for c in classes]
    return [str(c) for c in fallback]


def compute_class_weights(class_names: Sequence[str]) -> torch.Tensor:
    weights = [float(CLASS_WEIGHT_MAP.get(str(name), 1.0)) for name in class_names]
    return torch.tensor(weights, dtype=torch.float32)


def build_sample_weights(records, class_weights: torch.Tensor, class_names: Sequence[str]) -> torch.Tensor:
    cw = class_weights.detach().cpu().numpy().astype(np.float64)
    sampler_weight_map = {
        str(name): float(CLASS_SAMPLER_WEIGHTS[i]) if i < len(CLASS_SAMPLER_WEIGHTS) else 1.0
        for i, name in enumerate(CLASS_NAMES)
    }

    weights = np.ones((len(records),), dtype=np.float64)
    for i, record in enumerate(records):
        labels = np.asarray(getattr(record, "labels", []), dtype=np.int64)
        if labels.size == 0:
            weights[i] = 0.9
            continue
        uniq = np.unique(labels)
        uniq = uniq[(uniq >= 0) & (uniq < cw.size)]
        if uniq.size == 0:
            weights[i] = 1.0
            continue
        combined: List[float] = []
        for idx in uniq.tolist():
            cls_name = str(class_names[int(idx)]) if 0 <= int(idx) < len(class_names) else str(idx)
            combined.append(float(cw[int(idx)] * sampler_weight_map.get(cls_name, 1.0)))
        weights[i] = float(np.max(np.asarray(combined, dtype=np.float64)))

    weights = weights / max(weights.mean(), 1e-12)
    weights = np.clip(weights, 0.25, 4.5)
    return torch.as_tensor(weights, dtype=torch.double)


def build_optimizer(model: YOLOv3, lr_backbone: float, lr_head: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone_fpn.parameters())
    head_params = list(model.head_s16.parameters()) + list(model.head_s32.parameters())
    used = {id(p) for p in backbone_params + head_params}
    other_params = [p for p in model.parameters() if id(p) not in used]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr_head})

    return AdamW(param_groups, weight_decay=weight_decay)


def resolve_num_workers(requested: int, device: torch.device) -> int:
    requested = max(0, int(requested))
    cpu_count = os.cpu_count() or 2
    soft_cap = max(1, cpu_count // 2)
    if device.type != "cuda":
        soft_cap = min(soft_cap, 2)
    return max(0, min(requested, soft_cap))


def make_dataloader(
    dataset: DetectionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: Optional[WeightedRandomSampler],
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv3 detector with balanced sampling and class weights.")
    parser.add_argument("--train_data", type=Path, default=DEFAULT_TRAIN_ANN)
    parser.add_argument("--val_data", type=Path, default=DEFAULT_VAL_ANN)
    parser.add_argument("--image_dir", type=Path, default=Path("public/train/images"))
    parser.add_argument("--val_image_dir", type=Path, default=Path("public/val/images"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models"))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--center_radius", type=float, default=1.5)
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    num_workers = resolve_num_workers(args.num_workers, device)

    if not args.train_data.exists():
        raise FileNotFoundError(f"Train annotation not found: {args.train_data}")
    if not args.val_data.exists():
        raise FileNotFoundError(f"Val annotation not found: {args.val_data}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Train image directory not found: {args.image_dir}")
    if not args.val_image_dir.exists():
        raise FileNotFoundError(f"Val image directory not found: {args.val_image_dir}")

    classes = load_classes_from_annotation(args.train_data)
    val_classes = load_classes_from_annotation(args.val_data, fallback=classes)
    if val_classes != classes:
        print(f"Warning: val classes differ from train classes. Using train class order: {classes}")

    train_ds = DetectionDataset(
        annotation_path=args.train_data,
        image_dir=args.image_dir,
        img_size=args.img_size,
        augment=True,
        class_names=classes,
    )
    val_ds = DetectionDataset(
        annotation_path=args.val_data,
        image_dir=args.val_image_dir,
        img_size=args.img_size,
        augment=False,
        class_names=classes,
    )

    num_classes = len(classes)
    class_weights = None if args.no_class_weights else compute_class_weights(classes)

    train_sampler = None
    if class_weights is not None and not args.no_balanced_sampling:
        sample_weights = build_sample_weights(train_ds.records, class_weights, classes)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        sampler=train_sampler,
        device=device,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        device=device,
    )

    model = YOLOv3(num_classes=num_classes, pretrained=False).to(device).to(memory_format=torch.channels_last)
    criterion = DetectionLoss(
        num_classes=num_classes,
        strides=STRIDES,
        class_weights=class_weights,
        label_smoothing=float(args.label_smoothing),
        center_radius=float(args.center_radius),
        use_scale_ranges=not args.no_scale_ranges,
        anchors=ANCHORS,
        anchor_masks=ANCHOR_MASKS,
        img_size=args.img_size,
    ).to(device)
    optimizer = build_optimizer(model, lr_backbone=args.lr_backbone, lr_head=args.lr_head, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, epochs=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        ckpt_classes = ckpt.get("classes")
        if ckpt_classes is not None and list(ckpt_classes) != list(classes):
            raise ValueError(f"Checkpoint classes {ckpt_classes} do not match dataset classes {classes}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"

    print(f"Device: {device_summary(device)}, AMP: {amp_enabled}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
    print(f"Balanced sampling: {not args.no_balanced_sampling}")
    print(f"Class weights enabled: {not args.no_class_weights}")
    if class_weights is not None:
        print(f"Class weights: {[round(float(x), 4) for x in class_weights.tolist()]}")
    print(f"Num workers: {num_workers}")

    no_improve_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_metrics = validate_loss(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(obj={train_metrics['loss_obj']:.4f}, noobj={train_metrics['loss_noobj']:.4f}, reg={train_metrics['loss_reg']:.4f}, cls={train_metrics['loss_cls']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        checkpoint_extra = {
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": classes,
            "img_size": int(args.img_size),
            "architecture": "yolov3",
            "strides": STRIDES,
            "anchors": ANCHORS,
            "anchor_masks": ANCHOR_MASKS,
        }
        save_checkpoint(
            path=last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            extra=checkpoint_extra,
        )

        if val_metrics["loss"] + float(args.early_stopping_min_delta) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            no_improve_epochs = 0
            save_checkpoint(
                path=best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                extra=checkpoint_extra,
            )
            print(f"Saved best checkpoint: {best_path} (val_loss={best_val_loss:.4f})")
        else:
            no_improve_epochs += 1
            if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} after {no_improve_epochs} epochs without improvement.")
                break

    if not best_path.exists():
        save_checkpoint(
            path=best_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            extra={
                "scheduler_state_dict": scheduler.state_dict(),
                "classes": classes,
                "img_size": int(args.img_size),
                "architecture": "yolov3",
                "strides": STRIDES,
                "anchors": ANCHORS,
                "anchor_masks": ANCHOR_MASKS,
            },
        )

    print(f"Training done. Best model: {best_path}")


if __name__ == "__main__":
    main()
