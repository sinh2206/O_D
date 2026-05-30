from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from .config import IMG_SIZE, MEAN, STD

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LetterboxMeta:
    scale: float
    dx: float
    dy: float
    orig_w: int
    orig_h: int
    new_w: int
    new_h: int


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image_bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in VALID_EXTS:
        ext = ".jpg"
        path = path.with_suffix(ext)
    ok, enc = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    enc.tofile(str(path))
    return True


def enhance_low_light_bgr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) >= 82.0:
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    lut = np.array([((i / 255.0) ** 0.82) * 255.0 for i in range(256)], dtype=np.float32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(out, lut)


def letterbox_resize(
    image_bgr: np.ndarray,
    img_size: int = IMG_SIZE,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxMeta]:
    h, w = image_bgr.shape[:2]
    scale = min(float(img_size) / max(w, 1), float(img_size) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((img_size, img_size, 3), color, dtype=np.uint8)
    dx = (img_size - new_w) // 2
    dy = (img_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    meta = LetterboxMeta(
        scale=scale,
        dx=float(dx),
        dy=float(dy),
        orig_w=int(w),
        orig_h=int(h),
        new_w=int(new_w),
        new_h=int(new_h),
    )
    return canvas, meta


def preprocess_image(image_bgr: np.ndarray, img_size: int = IMG_SIZE) -> Tuple[torch.Tensor, LetterboxMeta]:
    image_bgr = enhance_low_light_bgr(image_bgr)
    canvas, meta = letterbox_resize(image_bgr, img_size=img_size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return tensor, meta


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_area=1.0,
        min_visibility=0.0,
        clip=True,
    )


def augment_train(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.90, 1.10), translate_percent=(-0.06, 0.06), rotate=(-6, 6), shear=(-2, 2), border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.35),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=8, p=0.25),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.15),
            A.MotionBlur(blur_limit=3, p=0.10),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def augment_val(img_size: int = IMG_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def letterbox_boxes_to_meta(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * float(meta.scale) + float(meta.dx)
    out[:, [1, 3]] = out[:, [1, 3]] * float(meta.scale) + float(meta.dy)
    return out


def meta_to_unletterbox(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - float(meta.dx)) / max(float(meta.scale), 1e-12)
    out[:, [1, 3]] = (out[:, [1, 3]] - float(meta.dy)) / max(float(meta.scale), 1e-12)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(meta.orig_w))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(meta.orig_h))
    return out
