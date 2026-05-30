from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import CONF_THRESH, IMG_SIZE, NMS_IOU_THRESH, NUM_CLASSES
from .image_ops import imread_unicode, preprocess_image
from .nms import LetterboxMeta, postprocess_batch

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_predictions_json(predictions: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_class_thresholds(predictions: List[dict], class_conf_thresh: Sequence[float], class_names: Sequence[str]) -> List[dict]:
    th_map = {c: float(class_conf_thresh[i]) for i, c in enumerate(class_names) if i < len(class_conf_thresh)}
    out: List[dict] = []
    for pred in predictions:
        keep = []
        for box in pred.get("boxes", []):
            cls_name = str(box.get("class", ""))
            score = float(box.get("confidence", 0.0))
            if score >= th_map.get(cls_name, 0.5):
                keep.append(box)
        out.append({"image_id": pred.get("image_id"), "boxes": keep})
    return out


@torch.no_grad()
def predict_images(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    device: torch.device,
    batch_size: int = 8,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_IOU_THRESH,
    class_names: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
) -> List[dict]:
    model.eval()
    amp_enabled = device.type == "cuda"
    predictions: List[dict] = []

    for start in range(0, len(image_paths), max(1, int(batch_size))):
        batch_paths = list(image_paths[start : start + batch_size])
        tensors: List[torch.Tensor] = []
        metas: List[LetterboxMeta] = []
        image_ids: List[str] = []

        for path in batch_paths:
            image = imread_unicode(path)
            if image is None:
                continue
            tensor, meta = preprocess_image(image, img_size=img_size)
            tensors.append(tensor)
            metas.append(meta)
            image_ids.append(path.name)

        if not tensors:
            continue

        images = torch.stack(tensors, dim=0).to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)

        batch_predictions = postprocess_batch(
            outputs=outputs,
            image_ids=image_ids,
            metas=metas,
            class_names=class_names,
            num_classes=len(class_names) if class_names is not None else NUM_CLASSES,
            img_size=img_size,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
        )
        predictions.extend(batch_predictions)

    if output_path is not None:
        save_predictions_json(predictions, output_path)
    return predictions
