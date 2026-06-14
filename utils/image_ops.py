from __future__ import annotations

"""Unicode-safe image IO plus simple detection visualization utilities."""

from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from .config import VALID_EXTS


def imread_unicode(path: Path) -> np.ndarray | None:
    """Read an image from disk while supporting Unicode file paths."""

    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image_bgr: np.ndarray) -> bool:
    """Write one image to disk with Unicode-safe path handling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    ext = suffix if suffix in VALID_EXTS else ".jpg"
    target = path if suffix in VALID_EXTS else path.with_suffix(ext)
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    encoded.tofile(str(target))
    return True


def class_color(index: int) -> Tuple[int, int, int]:
    """Generate a stable pseudo-random color from a class index."""

    return (
        int((53 * (index + 1)) % 255),
        int((97 * (index + 1)) % 255),
        int((193 * (index + 1)) % 255),
    )


def draw_boxes(image_bgr: np.ndarray, boxes: Sequence[Dict[str, object]], class_names: Sequence[str]) -> np.ndarray:
    """Render exported JSON-format detection boxes on an image."""

    out = image_bgr.copy()
    class_to_idx = {str(name): idx for idx, name in enumerate(class_names)}

    for item in boxes:
        bbox = item.get("bbox", [])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        cls_name = str(item.get("class", "object"))
        score = float(item.get("confidence", 0.0))
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        color = class_color(class_to_idx.get(cls_name, 0))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{cls_name}:{score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top = max(0, y1 - th - base - 6)
        bottom = max(th + base + 4, y1)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, bottom), color, -1)
        cv2.putText(out, label, (x1 + 3, bottom - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out
