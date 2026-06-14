from __future__ import annotations

"""Result conversion helpers for YOLOv8 predictions."""

from typing import Any, Dict, List, Sequence

import numpy as np


def _clip_box(box: Sequence[float], image_w: int, image_h: int) -> List[int] | None:
    """Clamp one xyxy box to image bounds and convert it to integer coordinates."""

    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(image_w), x1))
    y1 = max(0.0, min(float(image_h), y1))
    x2 = max(0.0, min(float(image_w), x2))
    y2 = max(0.0, min(float(image_h), y2))

    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = int(round(x2))
    iy2 = int(round(y2))

    ix1 = max(0, min(image_w - 1, ix1)) if image_w > 0 else 0
    iy1 = max(0, min(image_h - 1, iy1)) if image_h > 0 else 0
    ix2 = max(0, min(image_w, ix2)) if image_w > 0 else 0
    iy2 = max(0, min(image_h, iy2)) if image_h > 0 else 0

    if ix2 <= ix1:
        ix2 = min(image_w, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(image_h, iy1 + 1)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return [ix1, iy1, ix2, iy2]


def result_to_export_boxes(result: Any, class_names: Sequence[str], max_det: int) -> List[Dict[str, object]]:
    """Convert one Ultralytics `Results` object into the repo's JSON box format."""

    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    xyxy = boxes_obj.xyxy.detach().cpu().numpy()
    conf = boxes_obj.conf.detach().cpu().numpy()
    cls_ids = boxes_obj.cls.detach().cpu().numpy().astype(np.int64)
    orig_h, orig_w = result.orig_shape

    order = np.argsort(-conf)
    exported: List[Dict[str, object]] = []
    for idx in order[: max(0, int(max_det))]:
        cls_id = int(cls_ids[idx])
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        bbox = _clip_box(xyxy[idx].tolist(), image_w=int(orig_w), image_h=int(orig_h))
        if bbox is None:
            continue
        exported.append(
            {
                "class": cls_name,
                "confidence": float(max(0.0, min(1.0, float(conf[idx])))),
                "bbox": bbox,
            }
        )
    return exported


def results_to_prediction_json(results: Sequence[Any], image_ids: Sequence[str], class_names: Sequence[str], max_det: int) -> List[Dict[str, object]]:
    """Convert a batch of Ultralytics results to the JSON array expected by the repo."""

    if len(results) != len(image_ids):
        raise ValueError("The number of YOLO results must match the number of image ids.")

    payload: List[Dict[str, object]] = []
    for image_id, result in zip(image_ids, results):
        payload.append(
            {
                "image_id": str(image_id),
                "boxes": result_to_export_boxes(result, class_names=class_names, max_det=max_det),
            }
        )
    return payload
