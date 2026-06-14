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
    CENTER_RADIUS,
    CLASS_FREQ_PRIOR_TRAIN,
    CLASS_FREQ_PRIOR_VAL,
    CLASS_LOSS_WEIGHTS,
    CLASS_NAMES,
    CLASS_SAMPLER_WEIGHTS,
    CONF_THRESH,
    DETAIL_FOCUS_AREA_RATIO,
    DETAIL_FOCUS_CONTEXT_RANGE,
    DETAIL_FOCUS_JITTER,
    DETAIL_FOCUS_MIN_VISIBLE,
    DETAIL_FOCUS_PROB,
    IMG_SIZE,
    INFER_CENTER_COMBINE,
    LABEL_SMOOTHING,
    LOW_LIGHT_CLAHE_CLIP,
    LOW_LIGHT_GAMMA,
    LOW_LIGHT_MEAN_THRESH,
    MEAN,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    PARTIAL_OCCLUSION_PROB,
    SMALL_OBJECT_AREA_RATIO,
    SMALL_OBJECT_BONUS,
    STD,
    STRIDES,
)
from utils.loss import DetectionLoss
from utils.model import AnchorFreeDetector
from utils.nms import postprocess_batch
from utils.runtime import create_grad_scaler, resolve_num_workers, should_pin_memory

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: List[List[float]]
    labels: List[int]


def compute_class_weights(
    num_classes: int,
    train_samples: Optional[List[Sample]] = None,
    val_samples: Optional[List[Sample]] = None,
) -> torch.Tensor:
    if num_classes <= 0:
        return torch.zeros((0,), dtype=torch.float32)

    def _freq_from_samples(samples: List[Sample]) -> np.ndarray:
        counts = np.zeros((num_classes,), dtype=np.float64)
        for s in samples:
            if not s.labels:
                continue
            labels = np.asarray(s.labels, dtype=np.int64)
            labels = labels[(labels >= 0) & (labels < num_classes)]
            if labels.size == 0:
                continue
            binc = np.bincount(labels, minlength=num_classes).astype(np.float64)
            counts += binc
        if counts.sum() <= 0:
            return np.zeros((num_classes,), dtype=np.float64)
        return counts / counts.sum()

    freq_train = _freq_from_samples(train_samples) if train_samples else np.asarray(CLASS_FREQ_PRIOR_TRAIN, dtype=np.float64)
    freq_val = _freq_from_samples(val_samples) if val_samples else np.asarray(CLASS_FREQ_PRIOR_VAL, dtype=np.float64)

    if freq_train.size != num_classes or freq_val.size != num_classes or freq_train.sum() <= 0 or freq_val.sum() <= 0:
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


def make_partial_occlusion(img_size: int) -> A.BasicTransform:
    prob = max(0.0, float(PARTIAL_OCCLUSION_PROB))
    if prob <= 0.0:
        return A.NoOp(p=1.0)

    params = inspect.signature(A.CoarseDropout.__init__).parameters
    min_size = max(6, int(round(img_size * 0.08)))
    max_size = max(min_size + 1, int(round(img_size * 0.22)))

    if "num_holes_range" in params:
        kwargs = {
            "num_holes_range": (1, 3),
            "hole_height_range": (0.08, 0.22),
            "hole_width_range": (0.08, 0.22),
            "p": prob,
        }
        if "fill" in params:
            kwargs["fill"] = (114, 114, 114)
        if "fill_mask" in params:
            kwargs["fill_mask"] = 0
        return A.CoarseDropout(**kwargs)

    kwargs = {
        "min_holes": 1,
        "max_holes": 3,
        "min_height": min_size,
        "max_height": max_size,
        "min_width": min_size,
        "max_width": max_size,
        "p": prob,
    }
    if "fill_value" in params:
        kwargs["fill_value"] = (114, 114, 114)
    elif "value" in params:
        kwargs["value"] = (114, 114, 114)
    if "mask_fill_value" in params:
        kwargs["mask_fill_value"] = 0
    elif "fill_mask" in params:
        kwargs["fill_mask"] = 0
    return A.CoarseDropout(**kwargs)


def maybe_focus_crop(
    image_rgb: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    focus_prob: float,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    if focus_prob <= 0.0 or random.random() >= float(focus_prob) or len(boxes) == 0 or len(labels) == 0:
        return image_rgb, boxes, labels

    height, width = image_rgb.shape[:2]
    if height <= 2 or width <= 2:
        return image_rgb, boxes, labels

    img_area = float(max(width * height, 1))
    sampler_boost = np.asarray(CLASS_SAMPLER_WEIGHTS, dtype=np.float64)
    focus_weights: List[float] = []

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area_ratio = (bw * bh) / img_area
        smallness = np.clip(
            (float(DETAIL_FOCUS_AREA_RATIO) - area_ratio) / max(float(DETAIL_FOCUS_AREA_RATIO), 1e-6),
            0.0,
            1.0,
        )
        class_bonus = float(sampler_boost[label]) if 0 <= int(label) < len(sampler_boost) else 1.0
        focus_weights.append(max(1e-3, class_bonus * (1.0 + 1.8 * float(smallness))))

    weights_np = np.asarray(focus_weights, dtype=np.float64)
    weights_np = weights_np / max(weights_np.sum(), 1e-12)
    focus_idx = int(np.random.choice(len(boxes), p=weights_np))
    fx1, fy1, fx2, fy2 = [float(v) for v in boxes[focus_idx]]
    bw = max(1.0, fx2 - fx1)
    bh = max(1.0, fy2 - fy1)
    cx = 0.5 * (fx1 + fx2)
    cy = 0.5 * (fy1 + fy2)

    context_min, context_max = DETAIL_FOCUS_CONTEXT_RANGE
    context_scale = random.uniform(float(context_min), float(context_max))
    crop_w = min(float(width), max(bw * context_scale, bw + 24.0))
    crop_h = min(float(height), max(bh * context_scale, bh + 24.0))
    cx += random.uniform(-float(DETAIL_FOCUS_JITTER), float(DETAIL_FOCUS_JITTER)) * bw
    cy += random.uniform(-float(DETAIL_FOCUS_JITTER), float(DETAIL_FOCUS_JITTER)) * bh

    crop_x1 = int(round(cx - crop_w * 0.5))
    crop_y1 = int(round(cy - crop_h * 0.5))
    crop_x1 = max(0, min(crop_x1, width - int(round(crop_w))))
    crop_y1 = max(0, min(crop_y1, height - int(round(crop_h))))
    crop_x2 = min(width, crop_x1 + max(2, int(round(crop_w))))
    crop_y2 = min(height, crop_y1 + max(2, int(round(crop_h))))

    if crop_x2 - crop_x1 < 2 or crop_y2 - crop_y1 < 2:
        return image_rgb, boxes, labels

    cropped = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    kept_boxes: List[List[float]] = []
    kept_labels: List[int] = []

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [float(v) for v in box]
        ix1 = max(x1, float(crop_x1))
        iy1 = max(y1, float(crop_y1))
        ix2 = min(x2, float(crop_x2))
        iy2 = min(y2, float(crop_y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        visible = (ix2 - ix1) * (iy2 - iy1)
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        visible_ratio = visible / area
        if idx != focus_idx and visible_ratio < float(DETAIL_FOCUS_MIN_VISIBLE):
            continue

        kept_boxes.append([ix1 - crop_x1, iy1 - crop_y1, ix2 - crop_x1, iy2 - crop_y1])
        kept_labels.append(int(label))

    if len(kept_boxes) == 0:
        return image_rgb, boxes, labels
    return cropped, kept_boxes, kept_labels


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
        downscale_aug = A.Downscale(scale_range=(0.72, 0.90), p=1.0)
    else:
        downscale_aug = A.Downscale(scale_min=0.72, scale_max=0.90, p=1.0)

    try:
        crop_aug = A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, erosion_rate=0.0, p=0.16)
    except TypeError:
        crop_aug = A.NoOp(p=1.0)
    coarse_dropout = make_partial_occlusion(img_size)

    return A.Compose(
        [
            crop_aug,
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),
            A.RandomRotate90(p=0.12),
            coarse_dropout,
            A.CLAHE(clip_limit=2.2, tile_grid_size=(8, 8), p=0.2),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    downscale_aug,
                ],
                p=0.18,
            ),
            A.Sharpen(alpha=(0.15, 0.35), lightness=(0.85, 1.15), p=0.16),
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
    def __init__(self, samples: List[Sample], transforms: A.Compose, detail_focus_prob: float = 0.0):
        self.samples = samples
        self.transforms = transforms
        self.detail_focus_prob = float(detail_focus_prob)

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
        image, bboxes, class_labels = maybe_focus_crop(
            image_rgb=image,
            boxes=bboxes,
            labels=class_labels,
            focus_prob=self.detail_focus_prob,
        )

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
    sampler: Optional[WeightedRandomSampler] = None,
    pin_memory: bool = False,
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
    scaler: torch.cuda.amp.GradScaler,
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


def box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def evaluate_image_predictions(
    gt_boxes: List[dict],
    pred_boxes: List[dict],
    class_names: List[str],
    iou_thresh: float = 0.5,
    img_area: float = 1.0,
) -> Dict[str, object]:
    per_class = {
        cls_name: {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
        for cls_name in class_names
    }
    gt_small_flags: List[bool] = []
    key_classes = {"car", "dog", "cat"}

    for gt in gt_boxes:
        cls_name = str(gt.get("class", ""))
        if cls_name in per_class:
            per_class[cls_name]["gt"] += 1
        x1, y1, x2, y2 = [float(v) for v in gt.get("bbox", [0, 0, 0, 0])]
        area_ratio = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / max(float(img_area), 1e-9)
        gt_small_flags.append(area_ratio <= float(SMALL_OBJECT_AREA_RATIO))

    matched_gt = [False] * len(gt_boxes)
    matched_pred = [False] * len(pred_boxes)
    matched_ious = [0.0] * len(gt_boxes)

    for cls_name in class_names:
        gt_indices = [idx for idx, gt in enumerate(gt_boxes) if str(gt.get("class", "")) == cls_name]
        pred_indices = [
            idx
            for idx, pred in enumerate(pred_boxes)
            if str(pred.get("class", "")) == cls_name
        ]
        pred_indices.sort(key=lambda idx: float(pred_boxes[idx].get("confidence", 0.0)), reverse=True)

        for pred_idx in pred_indices:
            pred_bbox = pred_boxes[pred_idx].get("bbox", [0, 0, 0, 0])
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx in gt_indices:
                if matched_gt[gt_idx]:
                    continue
                iou = box_iou(pred_bbox, gt_boxes[gt_idx].get("bbox", [0, 0, 0, 0]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= float(iou_thresh):
                matched_gt[best_gt_idx] = True
                matched_pred[pred_idx] = True
                matched_ious[best_gt_idx] = best_iou
                per_class[cls_name]["tp"] += 1
            else:
                per_class[cls_name]["fp"] += 1

        per_class[cls_name]["fn"] += sum(1 for gt_idx in gt_indices if not matched_gt[gt_idx])

    matched_count = sum(1 for flag in matched_gt if flag)
    matched_iou_sum = float(sum(iou for iou in matched_ious if iou > 0.0))
    small_gt_count = sum(1 for flag in gt_small_flags if flag)
    small_tp_count = sum(1 for gt_idx, flag in enumerate(gt_small_flags) if flag and matched_gt[gt_idx])
    key_iou_sum = float(
        sum(
            matched_ious[idx]
            for idx, gt in enumerate(gt_boxes)
            if matched_gt[idx] and str(gt.get("class", "")) in key_classes
        )
    )
    key_iou_count = sum(
        1
        for idx, gt in enumerate(gt_boxes)
        if matched_gt[idx] and str(gt.get("class", "")) in key_classes
    )

    dog_cat_gt_count = sum(1 for gt in gt_boxes if str(gt.get("class", "")) in {"dog", "cat"})
    dog_cat_confusions = 0
    for pred_idx, pred in enumerate(pred_boxes):
        if matched_pred[pred_idx]:
            continue
        pred_cls = str(pred.get("class", ""))
        if pred_cls not in {"dog", "cat"}:
            continue
        opposite = "cat" if pred_cls == "dog" else "dog"
        for gt in gt_boxes:
            if str(gt.get("class", "")) != opposite:
                continue
            if box_iou(pred.get("bbox", [0, 0, 0, 0]), gt.get("bbox", [0, 0, 0, 0])) >= 0.30:
                dog_cat_confusions += 1
                break

    total_tp = sum(stats["tp"] for stats in per_class.values())
    total_fp = sum(stats["fp"] for stats in per_class.values())
    total_fn = sum(stats["fn"] for stats in per_class.values())
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "matched_count": matched_count,
        "matched_iou_sum": matched_iou_sum,
        "small_gt_count": small_gt_count,
        "small_tp_count": small_tp_count,
        "key_iou_sum": key_iou_sum,
        "key_iou_count": key_iou_count,
        "dog_cat_gt_count": dog_cat_gt_count,
        "dog_cat_confusions": dog_cat_confusions,
        "per_class": per_class,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    class_names: Optional[List[str]] = None,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_IOU_THRESH,
    center_combine: str = INFER_CENTER_COMBINE,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0
    metric_classes = list(class_names) if class_names is not None else []
    per_class_totals = {
        cls_name: {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
        for cls_name in metric_classes
    }
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "matched_count": 0,
        "matched_iou_sum": 0.0,
        "small_gt_count": 0,
        "small_tp_count": 0,
        "key_iou_sum": 0.0,
        "key_iou_count": 0,
        "dog_cat_gt_count": 0,
        "dog_cat_confusions": 0,
    }

    for images, targets in loader:
        image_ids = [t["image_id"] for t in targets]
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

        if metric_classes:
            batch_predictions = postprocess_batch(
                outputs=outputs,
                image_ids=image_ids,
                metas=None,
                class_names=metric_classes,
                num_classes=len(metric_classes),
                img_size=int(img_size),
                conf_thresh=float(conf_thresh),
                nms_thresh=float(nms_thresh),
                reg_decode="auto",
                center_combine=str(center_combine),
            )
            for pred, target in zip(batch_predictions, targets):
                gt_entries = []
                gt_boxes_np = target["boxes"].detach().cpu().numpy()
                gt_labels_np = target["labels"].detach().cpu().numpy()
                for box, label in zip(gt_boxes_np.tolist(), gt_labels_np.tolist()):
                    cls_idx = int(label)
                    if 0 <= cls_idx < len(metric_classes):
                        gt_entries.append({"class": metric_classes[cls_idx], "bbox": [float(v) for v in box]})

                image_metrics = evaluate_image_predictions(
                    gt_boxes=gt_entries,
                    pred_boxes=list(pred.get("boxes", [])),
                    class_names=metric_classes,
                    iou_thresh=0.5,
                    img_area=float(max(int(img_size) * int(img_size), 1)),
                )
                for key in totals:
                    totals[key] += image_metrics[key]
                per_class_metrics = image_metrics["per_class"]
                for cls_name, stats in per_class_metrics.items():
                    for stat_name, value in stats.items():
                        per_class_totals[cls_name][stat_name] += int(value)

    if steps == 0:
        return {k: float("inf") for k in running}

    metrics = {k: v / steps for k, v in running.items()}
    if not metric_classes:
        return metrics

    total_tp = int(totals["tp"])
    total_fp = int(totals["fp"])
    total_fn = int(totals["fn"])
    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_tp + total_fn, 1)

    per_class_recalls: List[float] = []
    key_recalls: List[float] = []
    for cls_name in metric_classes:
        gt_count = per_class_totals[cls_name]["gt"]
        recall = per_class_totals[cls_name]["tp"] / max(gt_count, 1) if gt_count > 0 else 0.0
        metrics[f"recall_{cls_name}"] = float(recall)
        if gt_count > 0:
            per_class_recalls.append(recall)
        if cls_name in {"car", "dog", "cat"} and gt_count > 0:
            key_recalls.append(recall)

    macro_recall = float(np.mean(per_class_recalls)) if per_class_recalls else float(micro_recall)
    key_recall = float(np.mean(key_recalls)) if key_recalls else float(micro_recall)
    small_gt_count = int(totals["small_gt_count"])
    small_object_recall = (
        float(totals["small_tp_count"]) / small_gt_count
        if small_gt_count > 0
        else float(micro_recall)
    )
    mean_matched_iou = float(totals["matched_iou_sum"]) / max(int(totals["matched_count"]), 1)
    key_mean_iou = (
        float(totals["key_iou_sum"]) / max(int(totals["key_iou_count"]), 1)
        if int(totals["key_iou_count"]) > 0
        else float(mean_matched_iou)
    )
    dog_cat_confusion_rate = float(totals["dog_cat_confusions"]) / max(int(totals["dog_cat_gt_count"]), 1)
    selection_score = (
        0.35 * key_recall
        + 0.20 * small_object_recall
        + 0.20 * key_mean_iou
        + 0.15 * micro_recall
        + 0.10 * micro_precision
        - 0.20 * dog_cat_confusion_rate
    )

    metrics.update(
        {
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "macro_recall": float(macro_recall),
            "key_recall": float(key_recall),
            "small_object_recall": float(small_object_recall),
            "mean_matched_iou": float(mean_matched_iou),
            "key_mean_iou": float(key_mean_iou),
            "dog_cat_confusion_rate": float(dog_cat_confusion_rate),
            "selection_score": float(selection_score),
        }
    )
    return metrics


def build_optimizer(model: AnchorFreeDetector, lr_backbone: float, lr_head: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone_fpn.parameters())
    head_params = (
        list(model.head_s4.parameters())
        + list(model.head_s8.parameters())
        + list(model.head_s16.parameters())
        + list(model.head_s32.parameters())
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
    best_metric_score: Optional[float] = None,
    selection_metric: str = "val_loss",
    extra_metrics: Optional[Dict[str, float]] = None,
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
            "best_metric_score": best_metric_score,
            "selection_metric": selection_metric,
            "metrics": extra_metrics or {},
        },
        str(path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor-free detector (ResNet34 + 4-level FPN).")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--val_image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models"))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_backbone", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--center_radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--detail_focus_prob", type=float, default=DETAIL_FOCUS_PROB)
    parser.add_argument("--metric_conf_thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--metric_nms_thresh", type=float, default=NMS_IOU_THRESH)
    parser.add_argument(
        "--metric_center_combine",
        type=str,
        default=str(INFER_CENTER_COMBINE),
        choices=["cls", "soft", "sqrt", "mul"],
    )
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    resolved_workers, max_safe_workers = resolve_num_workers(int(args.num_workers))
    pin_memory = should_pin_memory(device)

    train_samples, classes = parse_samples(args.train_data, args.image_dir, class_names=None)
    val_samples, _ = parse_samples(args.val_data, args.val_image_dir, class_names=classes)
    num_classes = len(classes)

    train_ds = DetectionDataset(
        train_samples,
        transforms=get_train_transforms(args.img_size),
        detail_focus_prob=float(args.detail_focus_prob),
    )
    val_ds = DetectionDataset(
        val_samples,
        transforms=get_val_transforms(args.img_size),
        detail_focus_prob=0.0,
    )

    class_weights = None if args.no_class_weights else compute_class_weights(
        num_classes=num_classes,
        train_samples=train_samples,
        val_samples=val_samples,
    )
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
        num_workers=resolved_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=resolved_workers,
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
    best_metric_score = float("-inf")
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(str(args.resume), map_location=device)
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Resume checkpoint partially loaded ({exc}).")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as exc:
                print(f"Skipped optimizer state from resume checkpoint: {exc}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except ValueError as exc:
                print(f"Skipped scheduler state from resume checkpoint: {exc}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_metric_score = float(ckpt.get("best_metric_score", float("-inf")))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pth"
    best_loss_path = args.checkpoint_dir / "best_loss.pth"
    last_path = args.checkpoint_dir / "last.pth"

    print(f"Device: {device}, AMP: {amp_enabled}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
    print(f"Balanced sampling: {not args.no_balanced_sampling}")
    print(f"Class weights enabled: {not args.no_class_weights}")
    print(f"Detail focus crop probability: {float(args.detail_focus_prob):.2f}")
    print(f"Num workers: requested={args.num_workers}, resolved={resolved_workers}, max_safe={max_safe_workers}")
    print(f"Pin memory: {pin_memory}")
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
            class_names=classes,
            img_size=args.img_size,
            conf_thresh=float(args.metric_conf_thresh),
            nms_thresh=float(args.metric_nms_thresh),
            center_combine=str(args.metric_center_combine),
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(cls={train_metrics['loss_cls']:.4f}, reg={train_metrics['loss_reg']:.4f}, ctr={train_metrics['loss_ctr']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"score={val_metrics.get('selection_score', 0.0):.4f} | "
            f"recall={val_metrics.get('micro_recall', 0.0):.4f} | "
            f"key_recall={val_metrics.get('key_recall', 0.0):.4f} | "
            f"small_recall={val_metrics.get('small_object_recall', 0.0):.4f} | "
            f"dog_cat_conf={val_metrics.get('dog_cat_confusion_rate', 0.0):.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                path=best_loss_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                classes=classes,
                img_size=args.img_size,
                best_metric_score=best_metric_score,
                selection_metric="val_loss",
                extra_metrics=val_metrics,
            )
            print(f"Saved best loss checkpoint: {best_loss_path} (val_loss={best_val_loss:.4f})")

        if float(val_metrics.get("selection_score", float("-inf"))) > best_metric_score:
            best_metric_score = float(val_metrics["selection_score"])
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                classes=classes,
                img_size=args.img_size,
                best_metric_score=best_metric_score,
                selection_metric="selection_score",
                extra_metrics=val_metrics,
            )
            print(
                f"Saved best detection checkpoint: {best_path} "
                f"(score={best_metric_score:.4f}, key_recall={val_metrics.get('key_recall', 0.0):.4f})"
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
            best_metric_score=best_metric_score,
            selection_metric="last",
            extra_metrics=val_metrics,
        )

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
            best_metric_score=best_metric_score,
            selection_metric="selection_score",
        )
    if not best_loss_path.exists():
        save_checkpoint(
            path=best_loss_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            best_val_loss=best_val_loss,
            classes=classes,
            img_size=args.img_size,
            best_metric_score=best_metric_score,
            selection_metric="val_loss",
        )

    print(f"Training done. Best detection model: {best_path}")
    print(f"Best loss model: {best_loss_path}")


if __name__ == "__main__":
    main()
