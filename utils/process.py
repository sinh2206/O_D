from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from .config import MEAN, STD

PathLike = Union[str, Path]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LetterboxMeta:
    scale: float
    dx: float
    dy: float
    orig_w: int
    orig_h: int


def imread_unicode(path: PathLike) -> Optional[np.ndarray]:
    p = Path(path)
    if not p.exists():
        return None
    arr = np.fromfile(str(p), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path: PathLike, image_bgr: np.ndarray) -> bool:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() if p.suffix.lower() in VALID_EXTS else ".jpg"
    out_path = p if p.suffix.lower() in VALID_EXTS else p.with_suffix(ext)
    ok, enc = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    enc.tofile(str(out_path))
    return True


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
    return cv2.LUT(out, lut)


def letterbox_preprocess(image_bgr: np.ndarray, img_size: int) -> Tuple[torch.Tensor, LetterboxMeta]:
    image_bgr = enhance_low_light_bgr(image_bgr)
    h, w = image_bgr.shape[:2]

    scale = min(float(img_size) / max(w, 1), float(img_size) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    dx = (img_size - new_w) // 2
    dy = (img_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    meta = LetterboxMeta(scale=scale, dx=float(dx), dy=float(dy), orig_w=int(w), orig_h=int(h))
    return tensor, meta


def letterbox_boxes_xyxy(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy.reshape(0, 4).astype(np.float32)
    out = boxes_xyxy.astype(np.float32).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * float(meta.scale) + float(meta.dx)
    out[:, [1, 3]] = out[:, [1, 3]] * float(meta.scale) + float(meta.dy)
    return out


def draw_prediction(image_bgr: np.ndarray, boxes: Sequence[Dict[str, Any]], class_names: Sequence[str]) -> np.ndarray:
    out = image_bgr.copy()
    cls_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for obj in boxes:
        cls_name = str(obj.get("class", "unknown"))
        score = float(obj.get("confidence", obj.get("score", 0.0)))
        x1, y1, x2, y2 = [int(round(v)) for v in obj.get("bbox", [0, 0, 0, 0])]

        idx = cls_to_idx.get(cls_name, 0)
        color = (
            int((53 * (idx + 1)) % 255),
            int((97 * (idx + 1)) % 255),
            int((193 * (idx + 1)) % 255),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}:{score:.2f}"
        cv2.putText(out, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
