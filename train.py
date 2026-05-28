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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.config import (
    CLASS_FREQ_PRIOR_TRAIN,
    CLASS_FREQ_PRIOR_VAL,
    CLASS_NAMES,
    IMG_SIZE,
    MEAN,
    NUM_CLASSES,
    STD,
    STRIDES,
)
from utils.image_ops import enhance_low_light_bgr
from utils.loss import DetectionLoss
from utils.model import AnchorFreeDetector

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_id: str
    image_path: Path
    boxes: List[List[float]]
    labels: List[int]


def compute_class_weights(samples: List[Sample], num_classes: int, use_dataset_prior: bool = True) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.float64)
    for s in samples:
        for y in s.labels:
            if 0 <= int(y) < num_classes:
                counts[int(y)] += 1.0

    if use_dataset_prior and len(CLASS_FREQ_PRIOR_TRAIN) == num_classes and len(CLASS_FREQ_PRIOR_VAL) == num_classes:
        prior_train = np.asarray(CLASS_FREQ_PRIOR_TRAIN, dtype=np.float64)
        prior_val = np.asarray(CLASS_FREQ_PRIOR_VAL, dtype=np.float64)
        prior = 0.5 * (prior_train + prior_val)
        prior = np.clip(prior, 1e-6, None)
        prior = prior / prior.sum()

        # Safer weighting: inverse sqrt frequency (less aggressive than 1/freq).
        weights = 1.0 / np.sqrt(prior)
        weights = weights / max(weights.mean(), 1e-12)
        weights = np.clip(weights, 0.5, 1.8)
    else:
        counts = np.maximum(counts, 1.0)
        beta = 0.999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / max(weights.mean(), 1e-12)
        weights = np.clip(weights, 0.7, 3.5)
    return torch.as_tensor(weights, dtype=torch.float32)


def build_sample_weights(samples: List[Sample], class_weights: torch.Tensor) -> torch.Tensor:
    cw = class_weights.cpu().numpy().astype(np.float64)
    ws = np.ones((len(samples),), dtype=np.float64)
    for i, s in enumerate(samples):
        if len(s.labels) == 0:
            ws[i] = 1.0
            continue
        uniq = np.unique(np.asarray(s.labels, dtype=np.int64))
        uniq = uniq[(uniq >= 0) & (uniq < len(cw))]
        if uniq.size == 0:
            ws[i] = 1.0
            continue
        # Emphasize images containing rare classes.
        ws[i] = float(np.max(cw[uniq]))
    ws = ws / max(ws.mean(), 1e-12)
    ws = np.clip(ws, 0.2, 6.0)
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


def load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_output_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Infer classification head output channels from a model state dict.

    We inspect the classification head weights because the checkpoint may come
    from either the legacy C-channel model or the new C+1 softmax-bg model.
    """
    for key in ("head_s16.cls_out.weight", "head_s32.cls_out.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            return int(tensor.shape[0])
    raise ValueError("Cannot infer output_num_classes from checkpoint state_dict.")


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
            A.RandomGamma(gamma_limit=(90, 120), p=0.2),
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
    def __init__(self, samples: List[Sample], transforms: A.Compose, use_low_light_enhance: bool = True):
        self.samples = samples
        self.transforms = transforms
        self.use_low_light_enhance = bool(use_low_light_enhance)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = imread_unicode(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        if self.use_low_light_enhance:
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
                "boxes": t["boxes"].to(device, non_blocking=True),
                "labels": t["labels"].to(device, non_blocking=True),
                "image_id": t["image_id"],
            }
        )
    return out


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: Optional[WeightedRandomSampler] = None,
    prefetch_factor: int = 4,
) -> DataLoader:
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(
        **loader_kwargs,
    )


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
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
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
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

    adamw_kwargs = {"weight_decay": weight_decay}
    adamw_sig = inspect.signature(AdamW.__init__).parameters
    if torch.cuda.is_available() and "fused" in adamw_sig:
        adamw_kwargs["fused"] = True
    elif "foreach" in adamw_sig:
        adamw_kwargs["foreach"] = True

    return AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        **adamw_kwargs,
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
    output_num_classes: int,
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
            "output_num_classes": int(output_num_classes),
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_backbone", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--center_radius", type=float, default=1.3)
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_prior_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--no_lowlight_enhance",
        action="store_true",
        help="Disable CPU low-light enhancement in the training dataloader for higher throughput.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    train_samples, classes = parse_samples(args.train_data, args.image_dir, class_names=None)
    val_samples, _ = parse_samples(args.val_data, args.val_image_dir, class_names=classes)
    num_classes = len(classes)

    train_ds = DetectionDataset(
        train_samples,
        transforms=get_train_transforms(args.img_size),
        use_low_light_enhance=not args.no_lowlight_enhance,
    )
    val_ds = DetectionDataset(
        val_samples,
        transforms=get_val_transforms(args.img_size),
        use_low_light_enhance=not args.no_lowlight_enhance,
    )

    class_weights = compute_class_weights(
        train_samples,
        num_classes=num_classes,
        use_dataset_prior=not args.no_prior_class_weights,
    )
    output_num_classes = num_classes + 1
    train_sampler = None
    if not args.no_balanced_sampling:
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
        num_workers=args.num_workers,
        sampler=train_sampler,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = AnchorFreeDetector(num_classes=output_num_classes, pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(str(args.resume), map_location=device)
        resume_output_num_classes = int(ckpt.get("output_num_classes", infer_output_num_classes(ckpt["model_state_dict"])))
        if resume_output_num_classes != output_num_classes:
            raise ValueError(
                f"Resume checkpoint uses output_num_classes={resume_output_num_classes}, "
                f"but current run expects {output_num_classes}. "
                f"Legacy checkpoints must be retrained for the new softmax+background format."
            )
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

    print(f"Device: {device}, AMP: {amp_enabled}, channels_last=True")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
    print(f"Model output classes (incl bg): {output_num_classes}")
    print(f"DataLoader workers: {args.num_workers}, prefetch_factor: {args.prefetch_factor}")
    print(f"Low-light enhance: {not args.no_lowlight_enhance}")
    print(f"Class weight mode: {'dataset_prior(train+val)' if not args.no_prior_class_weights else 'empirical_train'}")
    print(f"Class weights: {class_weights.tolist()}")

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
            output_num_classes=output_num_classes,
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
                output_num_classes=output_num_classes,
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
            output_num_classes=output_num_classes,
        )

    print(f"Training done. Best model: {best_path}")


if __name__ == "__main__":
    main()
