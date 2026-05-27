from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .config import (
    ANCHOR_SIZES,
    CLASS_NAMES,
    CONF_THRESH,
    IMG_SIZE,
    MAX_OBJECTS_PER_IMAGE,
    NMS_CLASS_AGNOSTIC,
    NMS_IOU_THRESH,
    STRIDE,
)
from .nms import postprocess_batch
from .process import LetterboxMeta, letterbox_preprocess


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    device: torch.device,
    batch_size: int = 16,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_IOU_THRESH,
    class_names: Sequence[str] = CLASS_NAMES,
    stride: int = STRIDE,
    max_objects_per_image: int = MAX_OBJECTS_PER_IMAGE,
    class_agnostic_nms: bool = NMS_CLASS_AGNOSTIC,
) -> List[Dict[str, object]]:
    from .process import imread_unicode

    model.eval()
    results: List[Dict[str, object]] = []
    amp_enabled = device.type == "cuda"

    for start in range(0, len(image_paths), max(1, int(batch_size))):
        batch_paths = image_paths[start : start + max(1, int(batch_size))]
        tensors: List[torch.Tensor] = []
        metas: List[LetterboxMeta] = []
        image_ids: List[str] = []

        for p in batch_paths:
            image = imread_unicode(p)
            if image is None:
                continue
            tensor, meta = letterbox_preprocess(image, img_size=img_size)
            tensors.append(tensor)
            metas.append(meta)
            image_ids.append(p.name)

        if not tensors:
            continue

        images = torch.stack(tensors, dim=0).to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)

        batch_results = postprocess_batch(
            outputs=outputs,
            image_ids=image_ids,
            metas=metas,
            class_names=class_names,
            conf_thresh=float(conf_thresh),
            nms_thresh=float(nms_thresh),
            img_size=int(img_size),
            stride=int(stride),
            anchor_sizes=ANCHOR_SIZES,
            max_objects_per_image=int(max_objects_per_image),
            class_agnostic_nms=bool(class_agnostic_nms),
        )
        results.extend(batch_results)
    return results


def save_predictions_json(predictions: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def apply_class_thresholds(
    predictions: List[Dict[str, object]],
    class_conf_thresh: Sequence[float],
    class_names: Sequence[str] = CLASS_NAMES,
) -> List[Dict[str, object]]:
    thresh_map = {
        class_names[i]: float(class_conf_thresh[i])
        for i in range(min(len(class_names), len(class_conf_thresh)))
    }

    out: List[Dict[str, object]] = []
    for pred in predictions:
        keep = []
        for box in pred.get("boxes", []):
            cls_name = str(box.get("class", ""))
            score = float(box.get("confidence", box.get("score", 0.0)))
            if score >= thresh_map.get(cls_name, 0.0):
                keep.append(box)
        out.append({"image_id": pred.get("image_id"), "boxes": keep})
    return out


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    model_cls: torch.nn.Module,
) -> tuple[torch.nn.Module, List[str], int]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    classes = list(ckpt.get("classes", CLASS_NAMES))
    img_size = int(ckpt.get("img_size", IMG_SIZE))

    model = model_cls(num_classes=len(classes), num_anchors=len(ANCHOR_SIZES), pretrained=False).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, classes, img_size
