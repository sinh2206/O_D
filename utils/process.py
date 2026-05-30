from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLASS_NAMES, IMG_SIZE
from .image_ops import augment_train, augment_val, enhance_low_light_bgr, imread_unicode

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DetectionSample:
    image_id: str
    file_name: str
    boxes: List[List[float]]
    labels: List[int]


def load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def clamp_boxes(boxes: np.ndarray, width: int, height: int, min_area: float = 1.0) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(width))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(height))
    keep = []
    for i, b in enumerate(out):
        x1, y1, x2, y2 = b.tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < float(min_area):
            continue
        keep.append(i)
    if not keep:
        return np.zeros((0, 4), dtype=np.float32)
    return out[keep].astype(np.float32)


class DetectionDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        image_dir: Path,
        img_size: int = IMG_SIZE,
        augment: bool = False,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.class_names = list(class_names) if class_names is not None else list(CLASS_NAMES)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transforms = augment_train(self.img_size) if self.augment else augment_val(self.img_size)

        data = load_annotation(annotation_path)
        self._records: List[DetectionSample] = []
        ann_map: Dict[str, List[dict]] = {}
        for ann in data.get("annotations", []):
            image_id = str(ann.get("image_id", ""))
            ann_map.setdefault(image_id, []).append(ann)

        for image in data.get("images", []):
            image_id = str(image.get("id", ""))
            file_name = Path(str(image.get("file_name", image_id))).name
            image_path = image_dir / file_name
            if not image_path.exists():
                fallback = image_dir / image_id
                if fallback.exists():
                    image_path = fallback
                else:
                    continue

            boxes: List[List[float]] = []
            labels: List[int] = []
            for ann in ann_map.get(image_id, []):
                cls_name = str(ann.get("class", ""))
                if cls_name not in self.class_to_idx:
                    continue
                bbox = ann.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[cls_name])

            self._records.append(
                DetectionSample(
                    image_id=image_id,
                    file_name=file_name,
                    boxes=boxes,
                    labels=labels,
                )
            )

        if not self._records:
            raise ValueError(f"No valid samples found in {annotation_path} for image_dir={image_dir}")

    @property
    def records(self) -> List[DetectionSample]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[DetectionSample]:
        return list(self._records)

    def __getitem__(self, idx: int):
        sample = self._records[idx]
        image_path = self.image_dir / sample.file_name
        image = imread_unicode(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = enhance_low_light_bgr(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = [list(b) for b in sample.boxes]
        labels = [int(v) for v in sample.labels]
        transformed = self.transforms(image=image_rgb, bboxes=bboxes, class_labels=labels)

        image_t = transformed["image"].float()
        out_boxes = np.asarray(transformed["bboxes"], dtype=np.float32) if len(transformed["bboxes"]) else np.zeros((0, 4), dtype=np.float32)
        out_boxes = clamp_boxes(out_boxes, width=self.img_size, height=self.img_size, min_area=1.0)
        out_labels = np.asarray(transformed["class_labels"], dtype=np.int64)
        if out_boxes.shape[0] != out_labels.shape[0]:
            out_labels = out_labels[: out_boxes.shape[0]]

        target = {
            "boxes": torch.as_tensor(out_boxes, dtype=torch.float32),
            "labels": torch.as_tensor(out_labels, dtype=torch.long),
            "image_id": sample.image_id,
            "file_name": sample.file_name,
        }
        return image_t, target


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
