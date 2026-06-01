from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.distributed as dist
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from utils.config import (
    CENTER_RADIUS,
    CLASS_FREQ_PRIOR_TRAIN,
    CLASS_FREQ_PRIOR_VAL,
    CLASS_LOSS_WEIGHTS,
    CLASS_NAMES,
    CLASS_SAMPLER_WEIGHTS,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LOW_LIGHT_CLAHE_CLIP,
    LOW_LIGHT_GAMMA,
    LOW_LIGHT_MEAN_THRESH,
    MEAN,
    NUM_CLASSES,
    SMALL_OBJECT_AREA_RATIO,
    SMALL_OBJECT_BONUS,
    STD,
    STRIDES,
)
from utils.loss import DetectionLoss
from utils.model import AnchorFreeDetector
from utils.runtime import (
    cleanup_distributed,
    create_grad_scaler,
    cuda_inventory,
    compute_mode_label,
    get_distributed_env,
    init_distributed,
    is_main_process,
    reduce_scalar,
    resolve_num_workers,
    should_pin_memory,
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: List[List[float]]
    labels: List[int]


class DistributedWeightedSampler(Sampler[int]):
    """
    Weighted sampler compatible with DDP.

    Each rank samples from a common weighted index stream with independent
    striding, so global weighted distribution is preserved.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}]")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterable[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def compute_class_weights(num_classes: int) -> torch.Tensor:
    if num_classes <= 0:
        return torch.zeros((0,), dtype=torch.float32)

    freq_train = np.asarray(CLASS_FREQ_PRIOR_TRAIN, dtype=np.float64)
    freq_val = np.asarray(CLASS_FREQ_PRIOR_VAL, dtype=np.float64)
    if freq_train.size != num_classes or freq_val.size != num_classes:
        weights = np.ones((num_classes,), dtype=np.float64)
        return torch.as_tensor(weights, dtype=torch.float32)

    freq = 0.5 * (freq_train + freq_val)
    freq = np.clip(freq, 1e-6, None)

    # Base inverse-frequency weight, softened by sqrt to avoid exploding the
    # minority classes. A mild class-specific boost is applied afterwards.
    weights = np.power(freq.mean() / freq, 0.5)
    class_boost = np.asarray(CLASS_LOSS_WEIGHTS, dtype=np.float64)
    if class_boost.size != num_classes:
        class_boost = np.ones((num_classes,), dtype=np.float64)
    weights = weights * class_boost
    weights = weights / max(weights.mean(), 1e-12)
    # Keep dominant class penalized enough to reduce person-overprediction
    # while still allowing minority classes to be upweighted.
    weights = np.clip(weights, 0.45, 2.20)
    return torch.as_tensor(weights, dtype=torch.float32)


def build_sample_weights(samples: List[Sample], class_weights: torch.Tensor) -> torch.Tensor:
    cw = class_weights.cpu().numpy().astype(np.float64)
    class_sampler_weights = np.asarray(CLASS_SAMPLER_WEIGHTS, dtype=np.float64)
    if class_sampler_weights.size != cw.size:
        class_sampler_weights = np.ones_like(cw)

    ws = np.ones((len(samples),), dtype=np.float64)
    for i, s in enumerate(samples):
        if len(s.labels) == 0:
            # Do not downsample empty scenes; they are important for lowering
            # false positives on background-only images.
            ws[i] = 1.0
            continue
        labels = np.asarray(s.labels, dtype=np.int64)
        labels = labels[(labels >= 0) & (labels < len(cw))]
        if labels.size == 0:
            ws[i] = 1.0
            continue
        # Use all labels, not just unique classes, so crowded scenes with many
        # objects get sampled more often.
        base_weight = float(np.mean(cw[labels] * class_sampler_weights[labels]))
        crowd_bonus = 1.0 + 0.12 * float(min(labels.size, 10))
        small_bonus = 1.0
        if s.width > 0 and s.height > 0 and len(s.boxes) > 0:
            img_area = float(max(s.width * s.height, 1))
            box_areas = np.asarray(
                [
                    max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
                    for box in s.boxes
                ],
                dtype=np.float64,
            )
            if box_areas.size > 0:
                area_ratios = box_areas / img_area
                smallness = np.mean(
                    np.clip((float(SMALL_OBJECT_AREA_RATIO) - area_ratios) / float(SMALL_OBJECT_AREA_RATIO), 0.0, 1.0)
                )
                small_bonus = 1.0 + float(SMALL_OBJECT_BONUS) * float(smallness)

        ws[i] = max(1.0, base_weight * crowd_bonus * small_bonus)

    ws = ws / max(ws.mean(), 1e-12)
    ws = np.clip(ws, 0.25, 4.5)
    return torch.as_tensor(ws, dtype=torch.double)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataloader_worker_init_fn(worker_id: int) -> None:
    # Prevent OpenCV thread oversubscription/deadlocks in multi-worker loading.
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    base_seed = int(torch.initial_seed()) % (2**32)
    seed = base_seed + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def enhance_low_light_bgr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) >= float(LOW_LIGHT_MEAN_THRESH):
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(LOW_LIGHT_CLAHE_CLIP), tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    lut = np.array([((i / 255.0) ** float(LOW_LIGHT_GAMMA)) * 255.0 for i in range(256)], dtype=np.float32)
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

        width = int(im.get("width", 0) or 0)
        height = int(im.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            image = imread_unicode(image_path)
            if image is not None:
                height, width = image.shape[:2]

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

        samples.append(
            Sample(
                image_id=image_id,
                image_path=image_path,
                width=int(width),
                height=int(height),
                boxes=boxes,
                labels=labels,
            )
        )

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
        scale=(0.85, 1.25),
        translate_percent=(-0.04, 0.04),
        rotate=(-4, 4),
        shear=(-1.0, 1.0),
        p=0.20,
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

    downscale_params = inspect.signature(A.Downscale.__init__).parameters
    if "scale_range" in downscale_params:
        downscale_aug = A.Downscale(scale_range=(0.60, 0.85), p=1.0)
    else:
        downscale_aug = A.Downscale(scale_min=0.60, scale_max=0.85, p=1.0)

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.08),
            A.RandomRotate90(p=0.22),
            A.CLAHE(clip_limit=2.2, tile_grid_size=(8, 8), p=0.2),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    downscale_aug,
                ],
                p=0.30,
            ),
            A.RandomGamma(gamma_limit=(88, 122), p=0.25),
            A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.1, hue=0.05, p=0.45),
            affine,
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=0.0,
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
            min_area=0.0,
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
    sampler: Optional[Sampler[int]] = None,
    persistent_workers: bool = True,
) -> DataLoader:
    kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and persistent_workers),
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=dataloader_worker_init_fn,
        **kwargs,
    )


def sync_should_skip_step(local_skip: bool, device: torch.device) -> bool:
    """
    In DDP, every rank must follow the same backward/step path.
    If any rank has invalid loss, all ranks skip that iteration.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return bool(local_skip)
    flag = torch.tensor(1 if local_skip else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item() > 0)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
    accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0
    accum = max(1, int(accum_steps))

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        local_invalid = not bool(torch.isfinite(loss.detach()).item())
        if not local_invalid:
            for v in loss_dict.values():
                if torch.is_tensor(v) and not bool(torch.isfinite(v.detach()).all().item()):
                    local_invalid = True
                    break
        should_skip = sync_should_skip_step(local_invalid, device=device)
        if should_skip:
            optimizer.zero_grad(set_to_none=True)
            continue

        backward_loss = loss / float(accum)
        scaler.scale(backward_loss).backward()

        do_step = (batch_idx % accum == 0) or (batch_idx == len(loader))
        if do_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].item())
        running["loss_reg"] += float(loss_dict["loss_reg"].item())
        running["loss_ctr"] += float(loss_dict["loss_ctr"].item())
        steps += 1

    if steps == 0:
        return {**{k: 0.0 for k in running}, "steps": 0.0}
    return {**running, "steps": float(steps)}


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
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].item())
        running["loss_reg"] += float(loss_dict["loss_reg"].item())
        running["loss_ctr"] += float(loss_dict["loss_ctr"].item())
        steps += 1

    if steps == 0:
        return {**{k: 0.0 for k in running}, "steps": 0.0}
    return {**running, "steps": float(steps)}


def finalize_epoch_metrics(raw: Dict[str, float], device: torch.device, distributed: bool) -> Dict[str, float]:
    keys = ["loss", "loss_cls", "loss_reg", "loss_ctr", "steps"]
    reduced = {k: float(raw.get(k, 0.0)) for k in keys}

    if distributed:
        for k in keys:
            reduced[k] = reduce_scalar(reduced[k], device=device)

    steps = max(1.0, reduced["steps"])
    return {
        "loss": reduced["loss"] / steps,
        "loss_cls": reduced["loss_cls"] / steps,
        "loss_reg": reduced["loss_reg"] / steps,
        "loss_ctr": reduced["loss_ctr"] / steps,
    }


def build_optimizer(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> torch.optim.Optimizer:
    m = model.module if isinstance(model, DDP) else model
    backbone_params = list(m.backbone_fpn.parameters())
    head_params = (
        list(m.head_s8.parameters())
        + list(m.head_s16.parameters())
        + list(m.head_s32.parameters())
    )

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
    model_to_save = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
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
    parser = argparse.ArgumentParser(description="Train anchor-free detector (ResNet34 + 3-level FPN).")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--val_image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models"))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--ddp",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Use DistributedDataParallel. 'auto' enables when launched via torchrun.",
    )
    parser.add_argument(
        "--ddp_timeout_min",
        type=float,
        default=30.0,
        help="DDP init timeout in minutes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--center_radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def maybe_auto_relaunch_ddp(args: argparse.Namespace) -> None:
    """
    If user runs `python train.py ...` on a multi-GPU machine, relaunch
    automatically with torch distributed so all GPUs are used.
    """
    ddp_mode = str(args.ddp).lower()
    if ddp_mode not in {"auto", "on"}:
        return
    if os.environ.get("WORLD_SIZE"):
        return
    if os.environ.get("OD_AUTO_DDP_LAUNCHED") == "1":
        return
    if not torch.cuda.is_available():
        return

    gpu_count = int(torch.cuda.device_count())
    if gpu_count <= 1:
        return

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={gpu_count}",
        sys.argv[0],
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env["OD_AUTO_DDP_LAUNCHED"] = "1"

    cpu_count = max(1, int(os.cpu_count() or 1))
    omp_threads = max(1, cpu_count // max(gpu_count, 1))
    env.setdefault("OMP_NUM_THREADS", str(omp_threads))
    env.setdefault("MKL_NUM_THREADS", str(omp_threads))

    # Safer NCCL defaults for notebook/managed environments (Kaggle/Colab).
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
    env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    is_managed_notebook = ("KAGGLE_URL_BASE" in env) or ("COLAB_GPU" in env)
    if is_managed_notebook:
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")

    print(f"[Auto-DDP] Detected {gpu_count} GPUs. Relaunching with torch.distributed.run ...")
    print(
        "[Auto-DDP] NCCL env: "
        f"OMP_NUM_THREADS={env.get('OMP_NUM_THREADS')}, "
        f"NCCL_ASYNC_ERROR_HANDLING={env.get('NCCL_ASYNC_ERROR_HANDLING')}, "
        f"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC={env.get('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC')}, "
        f"NCCL_P2P_DISABLE={env.get('NCCL_P2P_DISABLE', '0')}, "
        f"NCCL_IB_DISABLE={env.get('NCCL_IB_DISABLE', '0')}"
    )
    exit_code = subprocess.call(cmd, env=env)
    raise SystemExit(exit_code)


def main() -> None:
    args = parse_args()
    maybe_auto_relaunch_ddp(args)
    dist_env = get_distributed_env()
    if args.ddp == "on" and not dist_env.enabled:
        raise RuntimeError(
            "DDP is forced but WORLD_SIZE is not set. Launch with torchrun, e.g. "
            "`torchrun --standalone --nproc_per_node=2 train.py ...`"
        )
    if args.ddp == "off":
        dist_env = dist_env.__class__(enabled=False, rank=0, local_rank=0, world_size=1)

    if dist_env.enabled and dist_env.world_size <= 1:
        raise RuntimeError(
            "DDP requested but WORLD_SIZE<=1. Launch with torchrun, e.g. "
            "`torchrun --standalone --nproc_per_node=2 train.py ...`"
        )

    if dist_env.enabled:
        init_distributed(dist_env, backend="nccl", timeout_minutes=float(args.ddp_timeout_min))

    try:
        local_seed = int(args.seed) + int(dist_env.rank)
        seed_everything(local_seed)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if dist_env.enabled:
            if not torch.cuda.is_available():
                raise RuntimeError("DDP requires CUDA GPUs on this script.")
            torch.cuda.set_device(dist_env.local_rank)
            device = torch.device(f"cuda:{dist_env.local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        amp_enabled = (device.type == "cuda") and (not args.no_amp)
        num_workers, max_workers = resolve_num_workers(int(args.num_workers))
        pin_memory = should_pin_memory(device)

        train_samples, classes = parse_samples(args.train_data, args.image_dir, class_names=None)
        val_samples, _ = parse_samples(args.val_data, args.val_image_dir, class_names=classes)
        num_classes = len(classes)

        train_ds = DetectionDataset(train_samples, transforms=get_train_transforms(args.img_size))
        val_ds = DetectionDataset(val_samples, transforms=get_val_transforms(args.img_size))

        global_batch_size = max(1, int(args.batch_size))
        local_batch_size = global_batch_size
        if dist_env.enabled:
            if global_batch_size < dist_env.world_size:
                raise ValueError(
                    f"--batch_size ({global_batch_size}) must be >= world_size ({dist_env.world_size}) in DDP mode."
                )
            local_batch_size = global_batch_size // dist_env.world_size
            if global_batch_size % dist_env.world_size != 0 and is_main_process(dist_env):
                effective = local_batch_size * dist_env.world_size
                print(
                    f"Warning: batch_size {global_batch_size} not divisible by world_size {dist_env.world_size}. "
                    f"Using effective global batch size {effective}."
                )

        class_weights = None if args.no_class_weights else compute_class_weights(num_classes=num_classes)

        train_sampler: Optional[Sampler[int]] = None
        val_sampler: Optional[Sampler[int]] = None

        if dist_env.enabled:
            if (not args.no_balanced_sampling) and class_weights is not None:
                sample_weights = build_sample_weights(train_samples, class_weights)
                per_rank_samples = int(math.ceil(len(sample_weights) / float(dist_env.world_size)))
                train_sampler = DistributedWeightedSampler(
                    weights=sample_weights,
                    num_samples=per_rank_samples,
                    num_replicas=dist_env.world_size,
                    rank=dist_env.rank,
                    replacement=True,
                    seed=int(args.seed),
                )
            else:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=dist_env.world_size,
                    rank=dist_env.rank,
                    shuffle=True,
                    seed=int(args.seed),
                    drop_last=False,
                )

            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=dist_env.world_size,
                rank=dist_env.rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            if not args.no_balanced_sampling and class_weights is not None:
                sample_weights = build_sample_weights(train_samples, class_weights)
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )

        train_loader = make_dataloader(
            train_ds,
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            persistent_workers=not dist_env.enabled,
        )
        val_loader = make_dataloader(
            val_ds,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=val_sampler,
            persistent_workers=not dist_env.enabled,
        )

        pretrained_backbone = (not dist_env.enabled) or (int(dist_env.rank) == 0)
        model = AnchorFreeDetector(num_classes=num_classes, pretrained=pretrained_backbone).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        criterion = DetectionLoss(
            num_classes=num_classes,
            strides=STRIDES,
            class_weights=class_weights,
            label_smoothing=float(args.label_smoothing),
            center_radius=float(args.center_radius),
            use_scale_ranges=not args.no_scale_ranges,
        ).to(device)

        if dist_env.enabled:
            model = DDP(
                model,
                device_ids=[dist_env.local_rank],
                output_device=dist_env.local_rank,
                find_unused_parameters=False,
            )

        optimizer = build_optimizer(model, lr_backbone=args.lr_backbone, lr_head=args.lr_head, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        scaler = create_grad_scaler(device=device, enabled=amp_enabled)

        start_epoch = 1
        best_val_loss = float("inf")
        if args.resume is not None and args.resume.exists():
            ckpt = torch.load(str(args.resume), map_location=device)
            model_state = ckpt.get("model_state_dict", ckpt)
            model_target = model.module if isinstance(model, DDP) else model
            try:
                model_target.load_state_dict(model_state, strict=True)
            except RuntimeError as exc:
                missing, unexpected = model_target.load_state_dict(model_state, strict=False)
                if is_main_process(dist_env):
                    print(f"Resume checkpoint partially loaded ({exc}).")
                    print(f"Missing keys: {missing}")
                    print(f"Unexpected keys: {unexpected}")
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = args.checkpoint_dir / "best.pth"
        last_path = args.checkpoint_dir / "last.pth"

        if is_main_process(dist_env):
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            mode = compute_mode_label(
                device=device,
                gpu_count=gpu_count,
                distributed=dist_env.enabled,
                world_size=dist_env.world_size,
                data_parallel=False,
            )
            print(f"Compute mode: {mode}")
            print(f"Device: {device}, AMP: {amp_enabled}")
            print(
                f"DDP: {dist_env.enabled}, world_size={dist_env.world_size}, "
                f"rank={dist_env.rank}, local_rank={dist_env.local_rank}"
            )
            if gpu_count > 0:
                print(f"CUDA inventory: {cuda_inventory()}")
            print(f"Num workers: {num_workers} (max safe: {max_workers}), pin_memory={pin_memory}")
            print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
            print(
                f"Batch size global/local: {global_batch_size}/{local_batch_size}, "
                f"grad_accum_steps={max(1, int(args.grad_accum_steps))}"
            )
            print(f"Balanced sampling: {not args.no_balanced_sampling}")
            print(f"Class weights enabled: {not args.no_class_weights}")
            if class_weights is not None:
                print(f"Class weights: {[round(x, 4) for x in class_weights.tolist()]}")

        for epoch in range(start_epoch, args.epochs + 1):
            if isinstance(train_sampler, (DistributedSampler, DistributedWeightedSampler)):
                train_sampler.set_epoch(epoch)

            train_raw = train_one_epoch(
                model=model,
                criterion=criterion,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                accum_steps=max(1, int(args.grad_accum_steps)),
            )
            val_raw = validate_one_epoch(
                model=model,
                criterion=criterion,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
            )

            train_metrics = finalize_epoch_metrics(train_raw, device=device, distributed=dist_env.enabled)
            val_metrics = finalize_epoch_metrics(val_raw, device=device, distributed=dist_env.enabled)
            scheduler.step()

            if is_main_process(dist_env):
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

            if dist_env.enabled:
                if device.type == "cuda":
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    dist.barrier()

        if is_main_process(dist_env):
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
    finally:
        if dist_env.enabled:
            cleanup_distributed()


if __name__ == "__main__":
    main()
