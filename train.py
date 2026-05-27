from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.config import CLASS_NAMES, IMG_SIZE, MAX_OBJECTS_PER_IMAGE, MEAN, STD, STRIDE
from utils.loss import compute_loss
from utils.model import YOLOv2Detector
from utils.runtime import (
    create_grad_scaler,
    device_summary,
    get_optimizer,
    get_scheduler,
    load_checkpoint,
    resolve_device,
    resolve_num_workers,
    save_checkpoint,
    should_pin_memory,
)


@dataclass
class Sample:
    image_id: str
    image_path: Path
    boxes: List[List[float]]
    labels: List[int]


def _is_fully_inside(a: List[float], b: List[float]) -> bool:
    return bool(a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3])


def _apply_annotation_constraints(
    boxes: List[List[float]],
    labels: List[int],
    max_objects_per_image: int,
) -> Tuple[List[List[float]], List[int]]:
    if len(boxes) <= 1:
        return boxes[: max(0, int(max_objects_per_image))], labels[: max(0, int(max_objects_per_image))]

    max_keep = max(1, int(max_objects_per_image))
    order = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
        reverse=True,
    )

    kept_indices: List[int] = []
    for idx in order:
        candidate_box = boxes[idx]
        candidate_label = labels[idx]

        drop = False
        for kept_idx in kept_indices:
            if labels[kept_idx] != candidate_label:
                continue
            kept_box = boxes[kept_idx]
            if _is_fully_inside(candidate_box, kept_box) or _is_fully_inside(kept_box, candidate_box):
                drop = True
                break
        if drop:
            continue

        kept_indices.append(idx)
        if len(kept_indices) >= max_keep:
            break

    new_boxes = [boxes[i] for i in kept_indices]
    new_labels = [labels[i] for i in kept_indices]
    return new_boxes, new_labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(image_dir: Path, file_name: str, image_id: str) -> Optional[Path]:
    candidates = [
        image_dir / Path(file_name).name,
        image_dir / file_name,
        image_dir / image_id,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_samples(
    annotation_path: Path,
    image_dir: Path,
    class_names: Optional[List[str]] = None,
    max_objects_per_image: int = MAX_OBJECTS_PER_IMAGE,
) -> Tuple[List[Sample], List[str]]:
    from utils.process import imread_unicode

    data = load_annotation(annotation_path)
    classes = class_names if class_names is not None else list(data.get("classes", CLASS_NAMES))
    cls_to_idx = {name: i for i, name in enumerate(classes)}

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    ann_by_image: Dict[str, List[dict]] = {}
    for ann in annotations:
        image_id = str(ann.get("image_id"))
        ann_by_image.setdefault(image_id, []).append(ann)

    samples: List[Sample] = []
    for im in images:
        image_id = str(im.get("id"))
        file_name = str(im.get("file_name", image_id))
        image_path = resolve_image_path(image_dir, file_name=file_name, image_id=image_id)
        if image_path is None:
            continue
        if imread_unicode(image_path) is None:
            continue

        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in ann_by_image.get(image_id, []):
            cls_name = str(ann.get("class", ""))
            if cls_name not in cls_to_idx:
                continue
            box = ann.get("bbox", [0, 0, 0, 0])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_to_idx[cls_name])

        boxes, labels = _apply_annotation_constraints(
            boxes=boxes,
            labels=labels,
            max_objects_per_image=max_objects_per_image,
        )
        samples.append(Sample(image_id=image_id, image_path=image_path, boxes=boxes, labels=labels))

    if not samples:
        raise ValueError(f"No valid samples found in {annotation_path} for image_dir={image_dir}")
    return samples, classes


def make_pad_if_needed(img_size: int) -> A.BasicTransform:
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
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.1, hue=0.05, p=0.35),
            A.CLAHE(clip_limit=2.2, tile_grid_size=(8, 8), p=0.2),
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
    def __init__(self, samples: List[Sample], transforms: A.Compose) -> None:
        self.samples = samples
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from utils.process import enhance_low_light_bgr, imread_unicode

        sample = self.samples[idx]
        image = imread_unicode(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {sample.image_path}")

        image = enhance_low_light_bgr(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(
            image=image,
            bboxes=[list(b) for b in sample.boxes],
            class_labels=list(sample.labels),
        )

        img_t = transformed["image"].float()
        out_boxes = transformed["bboxes"]
        out_labels = transformed["class_labels"]

        if len(out_boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.as_tensor(np.asarray(out_boxes, dtype=np.float32))
            labels_t = torch.as_tensor(np.asarray(out_labels, dtype=np.int64))

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": sample.image_id}
        return img_t, target


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def move_targets_to_device(targets: List[dict], device: torch.device) -> List[dict]:
    moved: List[dict] = []
    for t in targets:
        moved.append(
            {
                "boxes": t["boxes"].to(device=device, dtype=torch.float32),
                "labels": t["labels"].to(device=device, dtype=torch.long),
                "image_id": t["image_id"],
            }
        )
    return moved


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )


def _empty_train_metrics() -> Dict[str, float]:
    return {
        "loss": float("inf"),
        "loss_obj": float("inf"),
        "loss_noobj": float("inf"),
        "loss_box": float("inf"),
        "loss_cls": float("inf"),
        "mean_iou_pos": 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    img_size: int,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_obj": 0.0, "loss_noobj": 0.0, "loss_box": 0.0, "loss_cls": 0.0, "mean_iou_pos": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss, loss_dict = compute_loss(
                predictions=outputs,
                targets_list=targets,
                device=device,
                stride=STRIDE,
                img_size=img_size,
            )

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for key in running:
            running[key] += float(loss_dict[key])
        steps += 1

    if steps == 0:
        return _empty_train_metrics()
    return {k: v / steps for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    img_size: int,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "loss_obj": 0.0, "loss_noobj": 0.0, "loss_box": 0.0, "loss_cls": 0.0, "mean_iou_pos": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss, loss_dict = compute_loss(
                predictions=outputs,
                targets_list=targets,
                device=device,
                stride=STRIDE,
                img_size=img_size,
            )

        if not torch.isfinite(loss):
            continue

        for key in running:
            running[key] += float(loss_dict[key])
        steps += 1

    if steps == 0:
        return _empty_train_metrics()
    return {k: v / steps for k, v in running.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv2-style detector (single stride-32 grid).")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--val_image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models"))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers. Use -1 for auto.")
    parser.add_argument("--max_objects_per_image", type=int, default=MAX_OBJECTS_PER_IMAGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.img_size % STRIDE != 0:
        raise ValueError(f"--img_size must be divisible by STRIDE={STRIDE}, got {args.img_size}")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    num_workers, max_safe_workers = resolve_num_workers(args.num_workers)
    pin_memory = should_pin_memory(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_samples, classes = parse_samples(
        args.train_data,
        args.image_dir,
        class_names=None,
        max_objects_per_image=max(1, int(args.max_objects_per_image)),
    )
    val_samples, _ = parse_samples(
        args.val_data,
        args.val_image_dir,
        class_names=classes,
        max_objects_per_image=max(1, int(args.max_objects_per_image)),
    )
    num_classes = len(classes)

    train_ds = DetectionDataset(train_samples, transforms=get_train_transforms(args.img_size))
    val_ds = DetectionDataset(val_samples, transforms=get_val_transforms(args.img_size))

    train_loader = make_dataloader(
        dataset=train_ds,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        dataset=val_ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = YOLOv2Detector(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device).to(memory_format=torch.channels_last)
    optimizer = get_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_factor=args.backbone_lr_factor,
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    scaler = create_grad_scaler(device=device, enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None and args.resume.exists():
        loaded_epoch, loaded_best = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_best

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"

    print(f"Device: {device_summary(device)}, AMP: {amp_enabled}")
    if args.num_workers >= 0 and args.num_workers != num_workers:
        print(f"Requested num_workers={args.num_workers} exceeds safe limit; using {num_workers}.")
    print(f"DataLoader workers: {num_workers} (max safe: {max_safe_workers}), pin_memory: {pin_memory}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            img_size=args.img_size,
        )
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            img_size=args.img_size,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(obj={train_metrics['loss_obj']:.4f}, noobj={train_metrics['loss_noobj']:.4f}, "
            f"box={train_metrics['loss_box']:.4f}, cls={train_metrics['loss_cls']:.4f}, iou={train_metrics['mean_iou_pos']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        save_checkpoint(
            path=last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            classes=classes,
            img_size=args.img_size,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                classes=classes,
                img_size=args.img_size,
            )
            print(f"Saved best checkpoint: {best_path} (val_loss={best_val_loss:.4f})")

    if not best_path.exists():
        save_checkpoint(
            path=best_path,
            epoch=args.epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            classes=classes,
            img_size=args.img_size,
        )

    print(f"Training done. Best model: {best_path}")


if __name__ == "__main__":
    main()
