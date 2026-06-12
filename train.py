from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    MIN_BOX_SIZE,
    NUM_CLASSES,
    SMALL_OBJECT_AREA_RATIO,
    SMALL_OBJECT_BONUS,
    STD,
    STRIDES,
)
from utils.loss import DetectionLoss
from utils.model import AnchorFreeDetector
from utils.nms import postprocess_batch
from utils.runtime import configure_reproducibility, create_grad_scaler, get_scheduler, resolve_num_workers, seed_worker, should_pin_memory

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


def get_train_transforms(img_size: int, use_stochastic_aug: bool = False) -> A.Compose:
    if not bool(use_stochastic_aug):
        return get_val_transforms(img_size)

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
        crop_aug = A.RandomSizedBBoxSafeCrop(height=img_size, width=img_size, erosion_rate=0.0, p=0.24)
    except TypeError:
        crop_aug = A.NoOp(p=1.0)

    return A.Compose(
        [
            crop_aug,
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),
            A.RandomRotate90(p=0.12),
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


def box_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)

    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def integrate_pr_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map50(
    gt_by_image: Dict[str, List[dict]],
    pred_by_image: Dict[str, List[dict]],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> Tuple[float, List[float]]:
    ap_per_class: List[float] = []

    for class_id in range(int(num_classes)):
        gt_for_class: Dict[str, List[List[float]]] = {}
        matched: Dict[str, np.ndarray] = {}
        total_gt = 0

        for image_id, items in gt_by_image.items():
            cls_boxes = [item["bbox"] for item in items if int(item["label"]) == class_id]
            gt_for_class[image_id] = cls_boxes
            matched[image_id] = np.zeros((len(cls_boxes),), dtype=bool)
            total_gt += len(cls_boxes)

        if total_gt == 0:
            ap_per_class.append(float("nan"))
            continue

        preds_for_class: List[Tuple[float, str, List[float]]] = []
        for image_id, items in pred_by_image.items():
            for item in items:
                if int(item["label"]) != class_id:
                    continue
                preds_for_class.append((float(item["score"]), image_id, list(item["bbox"])))

        preds_for_class.sort(key=lambda item: item[0], reverse=True)
        if not preds_for_class:
            ap_per_class.append(0.0)
            continue

        tp = np.zeros((len(preds_for_class),), dtype=np.float32)
        fp = np.zeros((len(preds_for_class),), dtype=np.float32)

        for pred_idx, (_, image_id, pred_box) in enumerate(preds_for_class):
            gt_boxes = gt_for_class.get(image_id, [])
            if not gt_boxes:
                fp[pred_idx] = 1.0
                continue

            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if matched[image_id][gt_idx]:
                    continue
                iou = box_iou_xyxy(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= float(iou_thresh):
                matched[image_id][best_gt_idx] = True
                tp[pred_idx] = 1.0
            else:
                fp[pred_idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(float(total_gt), 1.0)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap_per_class.append(integrate_pr_curve(recalls, precisions))

    valid_aps = [ap for ap in ap_per_class if np.isfinite(ap)]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0
    return mean_ap, [0.0 if not np.isfinite(ap) else float(ap) for ap in ap_per_class]


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: Optional[WeightedRandomSampler] = None,
    pin_memory: bool = False,
    generator: Optional[torch.Generator] = None,
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
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
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


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    class_names: Sequence[str],
    img_size: int,
    eval_conf_thresh: float,
    eval_iou_thresh: float,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_reg": 0.0, "loss_ctr": 0.0}
    steps = 0
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    gt_by_image: Dict[str, List[dict]] = {}
    pred_by_image: Dict[str, List[dict]] = {}

    for images, targets in loader:
        image_ids = [str(t["image_id"]) for t in targets]
        for target in targets:
            boxes = target["boxes"].detach().cpu().numpy() if isinstance(target["boxes"], torch.Tensor) else np.zeros((0, 4), dtype=np.float32)
            labels = target["labels"].detach().cpu().numpy() if isinstance(target["labels"], torch.Tensor) else np.zeros((0,), dtype=np.int64)
            gt_items = []
            for box, label in zip(boxes.tolist(), labels.tolist()):
                gt_items.append({"label": int(label), "bbox": [float(v) for v in box]})
            gt_by_image[str(target["image_id"])] = gt_items

        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        batch_preds = postprocess_batch(
            outputs=outputs,
            image_ids=image_ids,
            metas=None,
            class_names=class_names,
            num_classes=len(class_names),
            img_size=img_size,
            conf_thresh=float(eval_conf_thresh),
            nms_thresh=float(eval_iou_thresh),
            reg_decode="auto",
            center_combine="sqrt",
            min_box_size=MIN_BOX_SIZE,
        )
        for batch_pred in batch_preds:
            pred_items = []
            for pred_box in batch_pred.get("boxes", []):
                class_name = str(pred_box.get("class", ""))
                if class_name not in class_to_idx:
                    continue
                pred_items.append(
                    {
                        "label": int(class_to_idx[class_name]),
                        "score": float(pred_box.get("confidence", 0.0)),
                        "bbox": [float(v) for v in pred_box.get("bbox", [0.0, 0.0, 0.0, 0.0])],
                    }
                )
            pred_by_image[str(batch_pred.get("image_id", ""))] = pred_items

        running["loss"] += float(loss.detach().item())
        running["loss_cls"] += float(loss_dict["loss_cls"].item())
        running["loss_reg"] += float(loss_dict["loss_reg"].item())
        running["loss_ctr"] += float(loss_dict["loss_ctr"].item())
        steps += 1

    if steps == 0:
        metrics = {k: float("inf") for k in running}
        metrics["map50"] = 0.0
        return metrics

    map50, _ = compute_map50(
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        num_classes=len(class_names),
        iou_thresh=0.5,
    )
    metrics = {k: v / steps for k, v in running.items()}
    metrics["map50"] = float(map50)
    return metrics


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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    best_val_loss: float,
    best_map50: float,
    classes: List[str],
    img_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_map50": best_map50,
        "classes": classes,
        "img_size": img_size,
        "strides": STRIDES,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor-free detector with reproducible defaults.")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--val_image_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models"))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr_backbone", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--center_radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--no_scale_ranges", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic CUDA/data-loading defaults for repeatable training runs.",
    )
    parser.add_argument(
        "--use_train_augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to stochastic train augmentations. Disabled by default for stable ablations.",
    )
    parser.add_argument(
        "--fixed_lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep backbone/head learning rates constant across epochs instead of cosine decay.",
    )
    parser.add_argument("--eval_conf_thresh", type=float, default=0.05)
    parser.add_argument("--eval_nms_thresh", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility(seed=int(args.seed), deterministic=bool(args.deterministic))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp) and (not args.deterministic)
    resolved_workers, max_safe_workers = resolve_num_workers(int(args.num_workers), deterministic=bool(args.deterministic))
    pin_memory = should_pin_memory(device)
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args.seed))
    val_generator = torch.Generator()
    val_generator.manual_seed(int(args.seed))

    train_samples, classes = parse_samples(args.train_data, args.image_dir, class_names=None)
    val_samples, _ = parse_samples(args.val_data, args.val_image_dir, class_names=classes)
    num_classes = len(classes)

    train_ds = DetectionDataset(train_samples, transforms=get_train_transforms(args.img_size, use_stochastic_aug=bool(args.use_train_augment)))
    val_ds = DetectionDataset(val_samples, transforms=get_val_transforms(args.img_size))

    class_weights = None if args.no_class_weights else compute_class_weights(
        num_classes=num_classes,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    train_sampler = None
    if (not args.no_balanced_sampling) and (not args.deterministic) and class_weights is not None:
        sample_weights = build_sample_weights(train_samples, class_weights)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.deterministic,
        num_workers=resolved_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=resolved_workers,
        pin_memory=pin_memory,
        generator=val_generator,
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
    scheduler = get_scheduler(
        optimizer,
        epochs=int(args.epochs),
        warmup_epochs=0,
        min_lr_ratio=0.05,
        schedule="constant" if bool(args.fixed_lr) else "cosine",
    )
    scaler = create_grad_scaler(device=device, enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    best_map50 = float("-inf")
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
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_map50 = float(ckpt.get("best_map50", float("-inf")))

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"

    print(f"Device: {device}, AMP: {amp_enabled}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {classes}")
    print(f"Deterministic mode: {bool(args.deterministic)}")
    print(f"Train augmentation: {bool(args.use_train_augment)}")
    print(f"Balanced sampling: {(not args.no_balanced_sampling) and (not args.deterministic)}")
    print(f"Class weights enabled: {not args.no_class_weights}")
    print(f"Fixed LR schedule: {bool(args.fixed_lr)}")
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
            img_size=int(args.img_size),
            eval_conf_thresh=float(args.eval_conf_thresh),
            eval_iou_thresh=float(args.eval_nms_thresh),
        )
        if scheduler is not None:
            scheduler.step()

        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(cls={train_metrics['loss_cls']:.4f}, reg={train_metrics['loss_reg']:.4f}, ctr={train_metrics['loss_ctr']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_map50={val_metrics['map50']:.4f} | "
            f"lr={[round(lr, 8) for lr in current_lrs]}"
        )

        is_better_map = float(val_metrics["map50"]) > float(best_map50) + 1e-9
        is_same_map_better_loss = abs(float(val_metrics["map50"]) - float(best_map50)) <= 1e-9 and float(val_metrics["loss"]) < float(best_val_loss)
        if is_better_map or is_same_map_better_loss:
            best_val_loss = val_metrics["loss"]
            best_map50 = val_metrics["map50"]
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_map50=best_map50,
                classes=classes,
                img_size=args.img_size,
            )
            print(f"Saved best checkpoint: {best_path} (val_map50={best_map50:.4f}, val_loss={best_val_loss:.4f})")

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_map50=best_map50,
            classes=classes,
            img_size=args.img_size,
        )

    if not best_path.exists():
        save_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            best_val_loss=best_val_loss,
            best_map50=best_map50,
            classes=classes,
            img_size=args.img_size,
        )

    print(f"Training done. Best model: {best_path}")


if __name__ == "__main__":
    main()
