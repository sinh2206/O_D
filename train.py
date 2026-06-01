from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.config import (
    CENTER_RADIUS,
    CLASS_FREQ_PRIOR_TRAIN,
    CLASS_FREQ_PRIOR_VAL,
    CLASS_LOSS_WEIGHTS,
    CLASS_NAMES,
    CLASS_SAMPLER_WEIGHTS,
    DEFAULT_SCHEDULER,
    DEFAULT_SEED,
    EARLY_STOP_DELTA,
    EARLY_STOP_PATIENCE,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LOW_LIGHT_CLAHE_CLIP,
    LOW_LIGHT_GAMMA,
    LOW_LIGHT_MEAN_THRESH,
    MAP_CONF_THRESH,
    MAP_EVAL_INTERVAL,
    MAP_NMS_THRESH,
    MEAN,
    MIXUP_PROB,
    MOSAIC_PROB,
    NUM_CLASSES,
    PLATEAU_FACTOR,
    PLATEAU_MIN_LR,
    PLATEAU_PATIENCE,
    SMALL_OBJECT_AREA_RATIO,
    SMALL_OBJECT_BONUS,
    STD,
    STRIDES,
)
from utils.loss import DetectionLoss
from utils.model import AnchorFreeDetector
from utils.nms import postprocess_batch
from utils.runtime import EarlyStopping, create_grad_scaler, get_scheduler, resolve_num_workers, set_global_seed, should_pin_memory

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: List[List[float]]
    labels: List[int]


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
    set_global_seed(seed=int(seed), deterministic=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    def __init__(
        self,
        samples: List[Sample],
        transforms: A.Compose,
        img_size: int,
        train_mode: bool = False,
        mosaic_prob: float = 0.0,
        mixup_prob: float = 0.0,
    ):
        self.samples = samples
        self.transforms = transforms
        self.img_size = int(img_size)
        self.train_mode = bool(train_mode)
        self.mosaic_prob = max(0.0, min(1.0, float(mosaic_prob)))
        self.mixup_prob = max(0.0, min(1.0, float(mixup_prob)))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample_rgb(self, sample: Sample) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = imread_unicode(sample.image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")
        image = enhance_low_light_bgr(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = np.asarray(sample.boxes, dtype=np.float32) if sample.boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.asarray(sample.labels, dtype=np.int64) if sample.labels else np.zeros((0,), dtype=np.int64)
        return image, boxes, labels

    def _clip_boxes(self, boxes: np.ndarray, labels: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        b = boxes.astype(np.float32).copy()
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(w))
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(h))
        bw = b[:, 2] - b[:, 0]
        bh = b[:, 3] - b[:, 1]
        keep = (bw > 1.0) & (bh > 1.0)
        if not np.any(keep):
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return b[keep], labels.astype(np.int64)[keep]

    def _resize_with_letterbox(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h0, w0 = image.shape[:2]
        scale = min(float(self.img_size) / max(w0, 1), float(self.img_size) / max(h0, 1))
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        dx = (self.img_size - new_w) // 2
        dy = (self.img_size - new_h) // 2
        canvas[dy : dy + new_h, dx : dx + new_w] = resized

        if boxes.size == 0:
            return canvas, boxes
        b = boxes.copy().astype(np.float32)
        b[:, [0, 2]] = b[:, [0, 2]] * float(scale) + float(dx)
        b[:, [1, 3]] = b[:, [1, 3]] * float(scale) + float(dy)
        return canvas, b

    def _sample_index(self) -> int:
        return random.randrange(len(self.samples))

    def _load_for_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample = self.samples[idx]
        image, boxes, labels = self._load_sample_rgb(sample)
        image, boxes = self._resize_with_letterbox(image, boxes)
        boxes, labels = self._clip_boxes(boxes, labels, w=self.img_size, h=self.img_size)
        return image, boxes, labels

    def _build_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.img_size
        mosaic = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        indices = [idx] + [self._sample_index() for _ in range(3)]
        xc = random.randint(s // 2, (3 * s) // 2)
        yc = random.randint(s // 2, (3 * s) // 2)

        all_boxes: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for i, sample_idx in enumerate(indices):
            img, boxes, labels = self._load_for_mosaic(sample_idx)
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if boxes.size > 0:
                b = boxes.copy()
                b[:, [0, 2]] += float(x1a - x1b)
                b[:, [1, 3]] += float(y1a - y1b)
                all_boxes.append(b)
                all_labels.append(labels)

        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
            merged_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
            merged_boxes, merged_labels = self._clip_boxes(merged_boxes, merged_labels, w=2 * s, h=2 * s)
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int64)

        return mosaic, merged_boxes, merged_labels

    def _maybe_mixup(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mixup_prob <= 0.0 or random.random() >= self.mixup_prob:
            return image, boxes, labels
        mix_image, mix_boxes, mix_labels = self._build_mosaic(self._sample_index())
        lam = float(np.random.beta(1.5, 1.5))
        blended = (lam * image.astype(np.float32) + (1.0 - lam) * mix_image.astype(np.float32)).clip(0, 255).astype(np.uint8)

        if boxes.size == 0:
            merged_boxes, merged_labels = mix_boxes, mix_labels
        elif mix_boxes.size == 0:
            merged_boxes, merged_labels = boxes, labels
        else:
            merged_boxes = np.concatenate([boxes, mix_boxes], axis=0)
            merged_labels = np.concatenate([labels, mix_labels], axis=0)
        merged_boxes, merged_labels = self._clip_boxes(merged_boxes, merged_labels, w=image.shape[1], h=image.shape[0])
        return blended, merged_boxes, merged_labels

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.train_mode and self.mosaic_prob > 0.0 and random.random() < self.mosaic_prob:
            image, boxes, labels = self._build_mosaic(idx)
            image, boxes, labels = self._maybe_mixup(image, boxes, labels)
            class_labels = labels.tolist()
            bboxes = boxes.tolist()
        else:
            image, boxes, labels = self._load_sample_rgb(sample)
            class_labels = labels.tolist()
            bboxes = boxes.tolist()

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
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
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
        worker_init_fn=seed_worker,
        generator=generator,
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


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def compute_ap_from_scores(
    scores: np.ndarray,
    matches: np.ndarray,
    num_gt: int,
) -> float:
    if num_gt <= 0:
        return 0.0
    if scores.size == 0:
        return 0.0

    order = np.argsort(-scores)
    tp = matches[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / max(float(num_gt), 1e-12)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


@torch.no_grad()
def evaluate_map50(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    img_size: int,
    conf_thresh: float,
    nms_thresh: float,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model.eval()
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    per_class_scores: Dict[int, List[float]] = {i: [] for i in range(len(class_names))}
    per_class_match: Dict[int, List[int]] = {i: [] for i in range(len(class_names))}
    per_class_gt: Dict[int, int] = {i: 0 for i in range(len(class_names))}

    for images, targets in loader:
        image_ids = [str(t.get("image_id", f"img_{i}")) for i, t in enumerate(targets)]
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)

        batch_preds = postprocess_batch(
            outputs=outputs,
            image_ids=image_ids,
            metas=None,
            class_names=class_names,
            num_classes=len(class_names),
            img_size=img_size,
            conf_thresh=float(conf_thresh),
            nms_thresh=float(nms_thresh),
            reg_decode="auto",
            min_box_size=1.0,
        )

        for pred_item, tgt in zip(batch_preds, targets):
            gt_boxes_t = tgt["boxes"].detach().cpu().numpy() if isinstance(tgt.get("boxes"), torch.Tensor) else np.zeros((0, 4), dtype=np.float32)
            gt_labels_t = tgt["labels"].detach().cpu().numpy() if isinstance(tgt.get("labels"), torch.Tensor) else np.zeros((0,), dtype=np.int64)
            gt_by_class: Dict[int, List[List[float]]] = {i: [] for i in range(len(class_names))}
            for b, l in zip(gt_boxes_t.tolist(), gt_labels_t.tolist()):
                li = int(l)
                if 0 <= li < len(class_names):
                    gt_by_class[li].append([float(v) for v in b])
                    per_class_gt[li] += 1

            pred_by_class: Dict[int, List[dict]] = {i: [] for i in range(len(class_names))}
            for pb in pred_item.get("boxes", []):
                cls_name = str(pb.get("class", ""))
                if cls_name not in cls_to_idx:
                    continue
                cls_idx = cls_to_idx[cls_name]
                pred_by_class[cls_idx].append(pb)

            for cls_idx in range(len(class_names)):
                preds_cls = sorted(pred_by_class[cls_idx], key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                gts_cls = gt_by_class[cls_idx]
                used_gt: set[int] = set()
                for pb in preds_cls:
                    score = float(pb.get("confidence", 0.0))
                    bbox = [float(v) for v in pb.get("bbox", [0.0, 0.0, 0.0, 0.0])]
                    best_iou = 0.0
                    best_idx = -1
                    for gi, gb in enumerate(gts_cls):
                        if gi in used_gt:
                            continue
                        iou = box_iou_xyxy(bbox, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gi
                    matched = 0
                    if best_idx >= 0 and best_iou >= 0.5:
                        used_gt.add(best_idx)
                        matched = 1
                    per_class_scores[cls_idx].append(score)
                    per_class_match[cls_idx].append(matched)

    ap_by_class: Dict[str, float] = {}
    ap_values: List[float] = []
    for cls_idx, cls_name in enumerate(class_names):
        scores = np.asarray(per_class_scores[cls_idx], dtype=np.float64)
        matches = np.asarray(per_class_match[cls_idx], dtype=np.int64)
        ap = compute_ap_from_scores(scores=scores, matches=matches, num_gt=per_class_gt[cls_idx])
        ap_by_class[cls_name] = float(ap)
        ap_values.append(float(ap))

    map50 = float(np.mean(ap_values)) if ap_values else 0.0
    return {
        "map50": map50,
        "ap_by_class": ap_by_class,
    }


def build_optimizer(model: AnchorFreeDetector, lr_backbone: float, lr_head: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone_fpn.parameters())
    head_params = (
        list(model.head_s8.parameters())
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
    scheduler: Optional[Any],
    epoch: int,
    best_val_loss: float,
    classes: List[str],
    img_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "classes": classes,
        "img_size": img_size,
        "strides": STRIDES,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, str(path))


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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER, choices=["plateau", "cosine"])
    parser.add_argument("--plateau_factor", type=float, default=PLATEAU_FACTOR)
    parser.add_argument("--plateau_patience", type=int, default=PLATEAU_PATIENCE)
    parser.add_argument("--plateau_min_lr", type=float, default=PLATEAU_MIN_LR)
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--early_stop_delta", type=float, default=EARLY_STOP_DELTA)
    parser.add_argument("--mosaic_prob", type=float, default=MOSAIC_PROB)
    parser.add_argument("--mixup_prob", type=float, default=MIXUP_PROB)
    parser.add_argument("--map_eval_interval", type=int, default=MAP_EVAL_INTERVAL)
    parser.add_argument("--map_conf_thresh", type=float, default=MAP_CONF_THRESH)
    parser.add_argument("--map_nms_thresh", type=float, default=MAP_NMS_THRESH)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--center_radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    num_workers, max_workers = resolve_num_workers(int(args.num_workers))
    pin_memory = should_pin_memory(device)

    train_samples, classes = parse_samples(args.train_data, args.image_dir, class_names=None)
    val_samples, _ = parse_samples(args.val_data, args.val_image_dir, class_names=classes)
    num_classes = len(classes)

    train_ds = DetectionDataset(
        train_samples,
        transforms=get_train_transforms(args.img_size),
        img_size=args.img_size,
        train_mode=True,
        mosaic_prob=float(args.mosaic_prob),
        mixup_prob=float(args.mixup_prob),
    )
    val_ds = DetectionDataset(
        val_samples,
        transforms=get_val_transforms(args.img_size),
        img_size=args.img_size,
        train_mode=False,
        mosaic_prob=0.0,
        mixup_prob=0.0,
    )

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
        sampler=train_sampler,
        pin_memory=pin_memory,
        seed=args.seed,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=args.seed + 1000,
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
    scheduler: Any = get_scheduler(
        optimizer=optimizer,
        epochs=int(args.epochs),
        mode=str(args.scheduler),
        plateau_factor=float(args.plateau_factor),
        plateau_patience=int(args.plateau_patience),
        plateau_min_lr=float(args.plateau_min_lr),
    )
    scaler = create_grad_scaler(device=device, enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    best_map50 = 0.0
    early_stopper = EarlyStopping(
        patience=max(1, int(args.early_stop_patience)),
        min_delta=float(args.early_stop_delta),
        mode="min",
    )
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
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if math.isfinite(best_val_loss):
            early_stopper.best = float(best_val_loss)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"

    print(f"Device: {device}, AMP: {amp_enabled}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
    print(f"Balanced sampling: {not args.no_balanced_sampling}")
    print(f"Class weights enabled: {not args.no_class_weights}")
    print(f"Seed: {args.seed}")
    print(f"Workers: requested={args.num_workers}, resolved={num_workers}, max_safe={max_workers}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Early stopping patience: {args.early_stop_patience}")
    print(f"Mosaic prob: {float(args.mosaic_prob):.2f}, MixUp prob: {float(args.mixup_prob):.2f}")
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
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step()

        map_out: Optional[Dict[str, Any]] = None
        eval_interval = max(1, int(args.map_eval_interval))
        if (epoch % eval_interval == 0) or (epoch == start_epoch) or (epoch == args.epochs):
            map_out = evaluate_map50(
                model=model,
                loader=val_loader,
                device=device,
                class_names=classes,
                img_size=args.img_size,
                conf_thresh=float(args.map_conf_thresh),
                nms_thresh=float(args.map_nms_thresh),
                amp_enabled=amp_enabled,
            )
            best_map50 = max(best_map50, float(map_out["map50"]))

        lr_str = ",".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        msg = (
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(cls={train_metrics['loss_cls']:.4f}, reg={train_metrics['loss_reg']:.4f}, ctr={train_metrics['loss_ctr']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f} | lr=[{lr_str}]"
        )
        if map_out is not None:
            msg += f" | mAP@0.5={float(map_out['map50']):.4f}"
        print(msg)

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

        improved, should_stop = early_stopper.update(val_metrics["loss"])
        if improved:
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
        else:
            print(f"No val_loss improvement for {early_stopper.bad_epochs} epoch(s).")

        if should_stop:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val_loss improvement > {float(args.early_stop_delta)} for "
                f"{int(args.early_stop_patience)} consecutive epoch(s)."
            )
            break

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
    print(f"Best val_loss: {best_val_loss:.4f}, best mAP@0.5 observed: {best_map50:.4f}")


if __name__ == "__main__":
    main()
