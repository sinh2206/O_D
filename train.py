from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.config import (
    CLASS_FREQ_PRIOR_TRAIN,
    CLASS_FREQ_PRIOR_VAL,
    CLASS_LOSS_WEIGHTS,
    CLASS_NAMES,
    CLASS_SAMPLER_WEIGHTS,
    IMG_SIZE,
    LABEL_SMOOTHING,
    MEAN,
    NUM_CLASSES,
    STD,
    STRIDES,
)
from utils.loss import DetectionLoss
from utils.model import AnchorFreeDetector
from utils.runtime import create_grad_scaler, device_summary, resolve_device, resolve_num_workers, should_pin_memory

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_id: str
    image_path: Path
    boxes: List[List[float]]
    labels: List[int]


def compute_class_weights(num_classes: int) -> torch.Tensor:
    if len(CLASS_FREQ_PRIOR_TRAIN) == num_classes and len(CLASS_FREQ_PRIOR_VAL) == num_classes:
        p = 0.5 * (np.asarray(CLASS_FREQ_PRIOR_TRAIN, dtype=np.float64) + np.asarray(CLASS_FREQ_PRIOR_VAL, dtype=np.float64))
        p = np.clip(p, 1e-6, None)
        p = p / p.sum()
        w = 1.0 / np.sqrt(p)
    else:
        w = np.ones((num_classes,), dtype=np.float64)

    if len(CLASS_LOSS_WEIGHTS) == num_classes:
        w = w * np.asarray(CLASS_LOSS_WEIGHTS, dtype=np.float64)

    w = w / max(w.mean(), 1e-12)
    w = np.clip(w, 0.55, 2.5)
    return torch.as_tensor(w, dtype=torch.float32)


def build_sample_weights(samples: List[Sample], class_weights: torch.Tensor) -> torch.Tensor:
    cw = class_weights.cpu().numpy().astype(np.float64)
    class_sampler_weights = np.asarray(CLASS_SAMPLER_WEIGHTS, dtype=np.float64)
    if class_sampler_weights.size != cw.size:
        class_sampler_weights = np.ones_like(cw)

    ws = np.ones((len(samples),), dtype=np.float64)
    for i, s in enumerate(samples):
        if len(s.labels) == 0:
            ws[i] = 0.9
            continue
        uniq = np.unique(np.asarray(s.labels, dtype=np.int64))
        uniq = uniq[(uniq >= 0) & (uniq < len(cw))]
        if uniq.size == 0:
            ws[i] = 1.0
            continue
        ws[i] = float(np.max(cw[uniq] * class_sampler_weights[uniq]))

    ws = ws / max(ws.mean(), 1e-12)
    ws = np.clip(ws, 0.25, 4.5)
    return torch.as_tensor(ws, dtype=torch.double)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def enhance_low_light_bgr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) >= 82.0:
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    lut = np.array([((i / 255.0) ** 0.82) * 255.0 for i in range(256)], dtype=np.float32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    out = cv2.LUT(out, lut)
    return out


def load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_samples(annotation_path: Path, image_dir: Path, class_names: Optional[List[str]] = None) -> Tuple[List[Sample], List[str]]:
    data = load_annotation(annotation_path)
    json_classes = data.get("classes", [])
    classes = class_names if class_names is not None else (json_classes if json_classes else CLASS_NAMES)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    ann_by_image: Dict[str, List[dict]] = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        ann_by_image.setdefault(image_id, []).append(ann)

    samples: List[Sample] = []
    for im in images:
        image_id = str(im.get("id"))
        file_name = Path(str(im.get("file_name", image_id))).name
        image_path = image_dir / file_name
        if not image_path.exists():
            fallback = image_dir / image_id
            if fallback.exists():
                image_path = fallback
            else:
                continue

        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in ann_by_image.get(image_id, []):
            cls_name = ann.get("class")
            if cls_name not in class_to_idx:
                continue
            x1, y1, x2, y2 = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(class_to_idx[cls_name])

        samples.append(Sample(image_id=image_id, image_path=image_path, boxes=boxes, labels=labels))

    if not samples:
        raise ValueError(f"No valid samples from {annotation_path} with image_dir={image_dir}")
    return samples, classes


def make_pad_if_needed(img_size: int):
    params = inspect.signature(A.PadIfNeeded.__init__).parameters
    kwargs = {
        "min_height": img_size,
        "min_width": img_size,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 1.0,
    }
    if "fill" in params:
        kwargs["fill"] = (114, 114, 114)
        if "fill_mask" in params:
            kwargs["fill_mask"] = 0
    elif "value" in params:
        kwargs["value"] = (114, 114, 114)
        if "mask_value" in params:
            kwargs["mask_value"] = 0
    return A.PadIfNeeded(**kwargs)


def get_train_transforms(img_size: int) -> A.Compose:
    affine_params = inspect.signature(A.Affine.__init__).parameters
    affine_kwargs = dict(
        scale=(0.92, 1.08),
        translate_percent=(-0.06, 0.06),
        rotate=(-5, 5),
        shear=(-1.5, 1.5),
        p=0.25,
    )
    if "border_mode" in affine_params:
        affine_kwargs["border_mode"] = cv2.BORDER_CONSTANT
    elif "mode" in affine_params:
        affine_kwargs["mode"] = cv2.BORDER_CONSTANT

    if "fill" in affine_params:
        affine_kwargs["fill"] = (114, 114, 114)
        if "fill_mask" in affine_params:
            affine_kwargs["fill_mask"] = 0
    elif "value" in affine_params:
        affine_kwargs["value"] = (114, 114, 114)
        if "mask_value" in affine_params:
            affine_kwargs["mask_value"] = 0
    elif "cval" in affine_params:
        affine_kwargs["cval"] = (114, 114, 114)
        if "cval_mask" in affine_params:
            affine_kwargs["cval_mask"] = 0

    affine = A.Affine(**affine_kwargs)

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=2.2, tile_grid_size=(8, 8), p=0.2),
            A.RandomGamma(gamma_limit=(88, 122), p=0.25),
            A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.1, hue=0.05, p=0.45),
            affine,
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=4.0,
            min_visibility=0.2,
            clip=True,
        ),
    )


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=1.0,
            min_visibility=0.0,
            clip=True,
        ),
    )


class DetectionDataset(Dataset):
    def __init__(self, samples: List[Sample], transforms: A.Compose):
        self.samples = samples
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = imread_unicode(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        image = enhance_low_light_bgr(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = [list(b) for b in sample.boxes]
        class_labels = list(sample.labels)

        transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        img_t = transformed["image"].float()
        out_boxes = transformed["bboxes"]
        out_labels = transformed["class_labels"]

        if len(out_boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.as_tensor(np.array(out_boxes, dtype=np.float32))
            labels_t = torch.as_tensor(np.array(out_labels, dtype=np.int64))

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": sample.image_id,
        }
        return img_t, target


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    targets = [x[1] for x in batch]
    return images, targets


def move_targets_to_device(targets: List[dict], device: torch.device) -> List[dict]:
    out = []
    for t in targets:
        out.append(
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "image_id": t["image_id"],
            }
        )
    return out


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

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

        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].item())
        running["loss_reg"] += float(loss_dict["loss_reg"].item())
        running["loss_ctr"] += float(loss_dict["loss_ctr"].item())
        steps += 1

    if steps == 0:
        return {k: float("inf") for k in running}
    return {k: v / steps for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].item())
        running["loss_reg"] += float(loss_dict["loss_reg"].item())
        running["loss_ctr"] += float(loss_dict["loss_ctr"].item())
        steps += 1

    if steps == 0:
        return {k: float("inf") for k in running}
    return {k: v / steps for k, v in running.items()}


def build_optimizer(model: AnchorFreeDetector, lr_backbone: float, lr_head: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone_fpn.parameters())
    head_params = list(model.head_s16.parameters()) + list(model.head_s32.parameters())

    return AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_loss: float,
    classes: List[str],
    img_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "classes": classes,
            "img_size": img_size,
            "strides": STRIDES,
        },
        str(path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor-free detector (ResNet18 + 2-level FPN).")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--val_image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models"))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr_backbone", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers. Use -1 for auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--center_radius", type=float, default=1.3)
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    pin_memory = should_pin_memory(device)
    num_workers, max_safe_workers = resolve_num_workers(args.num_workers)

    train_samples, classes = parse_samples(args.train_data, args.image_dir, class_names=None)
    val_samples, _ = parse_samples(args.val_data, args.val_image_dir, class_names=classes)
    num_classes = len(classes)

    train_ds = DetectionDataset(train_samples, transforms=get_train_transforms(args.img_size))
    val_ds = DetectionDataset(val_samples, transforms=get_val_transforms(args.img_size))

    class_weights = None if args.no_class_weights else compute_class_weights(num_classes=num_classes)
    train_sampler = None
    if not args.no_balanced_sampling and class_weights is not None:
        sample_weights = build_sample_weights(train_samples, class_weights)
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
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = AnchorFreeDetector(num_classes=num_classes, pretrained=True).to(device).to(memory_format=torch.channels_last)
    criterion = DetectionLoss(
        num_classes=num_classes,
        strides=STRIDES,
        class_weights=class_weights,
        label_smoothing=float(args.label_smoothing),
        center_radius=float(args.center_radius),
        use_scale_ranges=not args.no_scale_ranges,
    ).to(device)
    optimizer = build_optimizer(model, lr_backbone=args.lr_backbone, lr_head=args.lr_head, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = create_grad_scaler(device=device, enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"

    print(f"Device: {device_summary(device)}, AMP: {amp_enabled}")
    if args.num_workers >= 0 and args.num_workers != num_workers:
        print(f"Requested num_workers={args.num_workers} exceeds safe limit; using {num_workers}.")
    print(f"DataLoader workers: {num_workers} (max safe: {max_safe_workers}), pin_memory: {pin_memory}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
    print(f"Balanced sampling: {not args.no_balanced_sampling}")
    print(f"Class weights enabled: {not args.no_class_weights}")
    if class_weights is not None:
        print(f"Class weights: {[round(x, 4) for x in class_weights.tolist()]}")

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
        val_metrics = validate_one_epoch(
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
            f"(cls={train_metrics['loss_cls']:.4f}, reg={train_metrics['loss_reg']:.4f}, ctr={train_metrics['loss_ctr']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            classes=classes,
            img_size=args.img_size,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                classes=classes,
                img_size=args.img_size,
            )
            print(f"Saved best checkpoint: {best_path} (val_loss={best_val_loss:.4f})")

    if not best_path.exists():
        save_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            best_val_loss=best_val_loss,
            classes=classes,
            img_size=args.img_size,
        )

    print(f"Training done. Best model: {best_path}")


if __name__ == "__main__":
    main()
