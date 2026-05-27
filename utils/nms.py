from __future__ import annotations

"""
Post-processing (decode + confidence filtering + class-wise NMS) for anchor-free detector.

This module is synchronized with `utils/model.py` output format:
{
  "cls_logits": [Tensor(B,C,H,W), Tensor(B,C,H,W)],
  "reg_preds": [Tensor(B,4,H,W), Tensor(B,4,H,W)],   # (t,l,b,r)
  "center_logits": [Tensor(B,1,H,W), Tensor(B,1,H,W)] optional,
  "strides": [16, 32]
}

Pipeline:
1) Decode multi-level predictions to xyxy boxes on letterbox image.
2) Compute confidence score from classification (+ centerness if available).
3) Threshold by confidence.
4) Apply class-wise NMS.
5) Remap boxes from letterbox space back to original image space.
6) Return JSON-ready results.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:
    from .config import (
        CLASS_NAMES,
        CONF_THRESH,
        IMG_SIZE,
        NMS_IOU_THRESH,
        NMS_IOU_THRESH_PER_CLASS,
        NUM_CLASSES,
        STRIDES,
    )
except Exception:
    # Safe fallbacks when config.py is not present.
    CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
    CONF_THRESH = 0.5
    NMS_IOU_THRESH = 0.35
    NMS_IOU_THRESH_PER_CLASS = [0.35, 0.35, 0.35, 0.35, 0.35]
    IMG_SIZE = 320
    NUM_CLASSES = 5
    STRIDES = [16, 32]


@dataclass
class LetterboxMeta:
    """
    Mapping info from original image to letterbox image.

    scale: scaling factor used before padding
    dx,dy: left/top padding offsets in letterbox image
    orig_w,orig_h: original image size
    """

    scale: float
    dx: float
    dy: float
    orig_w: int
    orig_h: int


def _make_grid(h: int, w: int, stride: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    try:
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        gy, gx = torch.meshgrid(ys, xs)
    return gx, gy


def _apply_regression_activation(reg_raw: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """
    Convert raw regression outputs to positive distances.

    mode:
    - "exp":      distances = exp(raw)
    - "relu":     distances = relu(raw)
    - "identity": distances = raw (assume already non-negative)
    - "auto":     if tensor has negative values -> exp, otherwise identity

    Note: `utils/model.py` currently outputs `relu(reg_out)`, so default "auto"
    behaves as identity for that model (synchronized behavior).
    """
    mode = mode.lower()

    if mode == "exp":
        return torch.exp(reg_raw).clamp(max=1e4)
    if mode == "relu":
        return torch.relu(reg_raw)
    if mode == "identity":
        return reg_raw
    if mode == "auto":
        if torch.any(reg_raw < 0):
            return torch.exp(reg_raw).clamp(max=1e4)
        return reg_raw
    raise ValueError(f"Unsupported reg_decode mode: {mode}")


def _classification_scores(
    cls_logits: torch.Tensor,
    num_classes: int,
    background_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute class score and class id per location.

    Returns:
    - cls_score: (H,W)
    - cls_id:    (H,W) in [0..num_classes-1]

    Cases:
    - logits channels == num_classes: softmax over C channels.
    - logits channels == num_classes+1: treat one channel as background.
    """
    c, _, _ = cls_logits.shape

    if c == num_classes:
        prob = torch.softmax(cls_logits, dim=0)
        cls_score, cls_id = prob.max(dim=0)
        return cls_score, cls_id

    if c == num_classes + 1:
        prob = torch.softmax(cls_logits, dim=0)

        if background_index is None:
            background_index = num_classes
        if background_index < 0 or background_index >= c:
            raise ValueError("background_index out of range for cls logits")

        fg_indices = [i for i in range(c) if i != background_index]
        fg_prob = prob[fg_indices, :, :]

        fg_max, fg_id_local = fg_prob.max(dim=0)
        fg_score = 1.0 - prob[background_index, :, :]

        # Combine explicit foreground probability with class peak.
        cls_score = fg_max * fg_score

        fg_ids_tensor = torch.tensor(fg_indices, device=cls_logits.device, dtype=torch.long)
        cls_id = fg_ids_tensor[fg_id_local]
        return cls_score, cls_id

    raise ValueError(
        f"Unexpected cls channels={c}. Expected num_classes ({num_classes}) "
        f"or num_classes+1 ({num_classes + 1})."
    )


def decode_level(
    cls_logits: torch.Tensor,
    reg_preds: torch.Tensor,
    center_logits: Optional[torch.Tensor],
    stride: int,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    num_classes: int = NUM_CLASSES,
    reg_decode: str = "auto",
    center_combine: str = "mul",
    background_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode one FPN level for a single image.

    Inputs:
    - cls_logits:   (C,H,W)
    - reg_preds:    (4,H,W) raw (t,l,b,r)
    - center_logits:(1,H,W) or None

    Returns filtered tensors on letterbox image:
    - boxes_xyxy: (N,4)
    - scores:     (N,)
    - class_ids:  (N,)
    """
    if cls_logits.dim() != 3:
        raise ValueError("cls_logits must be (C,H,W)")
    if reg_preds.dim() != 3 or reg_preds.shape[0] != 4:
        raise ValueError("reg_preds must be (4,H,W)")

    _, h, w = cls_logits.shape
    device = cls_logits.device

    cls_score, cls_id = _classification_scores(
        cls_logits=cls_logits,
        num_classes=num_classes,
        background_index=background_index,
    )

    if center_logits is not None:
        if center_logits.dim() != 3 or center_logits.shape[0] != 1:
            raise ValueError("center_logits must be (1,H,W) when provided")
        center = torch.sigmoid(center_logits[0])
        center_combine = center_combine.lower()
        if center_combine == "sqrt":
            conf = torch.sqrt((cls_score * center).clamp(min=1e-12))
        else:
            conf = cls_score * center
    else:
        conf = cls_score

    keep = conf > float(conf_thresh)
    if not torch.any(keep):
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    reg = _apply_regression_activation(reg_preds, mode=reg_decode)

    gx, gy = _make_grid(h, w, stride=float(stride), device=device)

    t = reg[0]
    l = reg[1]
    b = reg[2]
    r = reg[3]

    x1 = (gx - l).clamp(0.0, float(img_size))
    y1 = (gy - t).clamp(0.0, float(img_size))
    x2 = (gx + r).clamp(0.0, float(img_size))
    y2 = (gy + b).clamp(0.0, float(img_size))

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    ys, xs = torch.where(keep)
    boxes_keep = boxes[ys, xs]
    scores_keep = conf[ys, xs]
    cls_keep = cls_id[ys, xs].long()

    # Keep only valid geometry.
    valid = (boxes_keep[:, 2] > boxes_keep[:, 0]) & (boxes_keep[:, 3] > boxes_keep[:, 1])
    boxes_keep = boxes_keep[valid]
    scores_keep = scores_keep[valid]
    cls_keep = cls_keep[valid]

    return boxes_keep, scores_keep, cls_keep


def decode_multilevel(
    outputs: Dict[str, Any],
    image_index: int = 0,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    num_classes: int = NUM_CLASSES,
    strides: Optional[Sequence[int]] = None,
    reg_decode: str = "auto",
    center_combine: str = "mul",
    background_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode and confidence-filter all FPN levels for one image in batch.

    Returns:
    - boxes_xyxy_letterbox: (N,4)
    - scores: (N,)
    - class_ids: (N,)
    """
    cls_levels = outputs["cls_logits"]
    reg_levels = outputs["reg_preds"]
    ctr_levels = outputs.get("center_logits", None)

    if strides is None:
        strides = outputs.get("strides", STRIDES)

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []
    all_classes: List[torch.Tensor] = []

    for lvl, (cls_l, reg_l, stride) in enumerate(zip(cls_levels, reg_levels, strides)):
        cls_i = cls_l[image_index] if cls_l.dim() == 4 else cls_l
        reg_i = reg_l[image_index] if reg_l.dim() == 4 else reg_l

        ctr_i = None
        if ctr_levels is not None:
            ctr_l = ctr_levels[lvl]
            ctr_i = ctr_l[image_index] if ctr_l.dim() == 4 else ctr_l

        b, s, c = decode_level(
            cls_logits=cls_i,
            reg_preds=reg_i,
            center_logits=ctr_i,
            stride=int(stride),
            img_size=int(img_size),
            conf_thresh=float(conf_thresh),
            num_classes=int(num_classes),
            reg_decode=reg_decode,
            center_combine=center_combine,
            background_index=background_index,
        )

        if b.numel() > 0:
            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)

    if not all_boxes:
        device = cls_levels[0].device
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    return (
        torch.cat(all_boxes, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_classes, dim=0),
    )


def _box_iou_xyxy(one: torch.Tensor, many: torch.Tensor) -> torch.Tensor:
    """IoU between one box (4,) and many boxes (N,4)."""
    xx1 = torch.maximum(one[0], many[:, 0])
    yy1 = torch.maximum(one[1], many[:, 1])
    xx2 = torch.minimum(one[2], many[:, 2])
    yy2 = torch.minimum(one[3], many[:, 3])

    inter_w = (xx2 - xx1).clamp(min=0)
    inter_h = (yy2 - yy1).clamp(min=0)
    inter = inter_w * inter_h

    area_one = (one[2] - one[0]).clamp(min=0) * (one[3] - one[1]).clamp(min=0)
    area_many = (many[:, 2] - many[:, 0]).clamp(min=0) * (many[:, 3] - many[:, 1]).clamp(min=0)

    union = area_one + area_many - inter + 1e-6
    return inter / union


def nms_single_class(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """
    Pure PyTorch NMS for one class.

    Returns indices relative to input tensors.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = _box_iou_xyxy(boxes[i], boxes[rest])
        rest = rest[ious <= float(iou_thresh)]
        order = rest

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _resolve_class_nms_thresh(
    nms_thresh: Union[float, Sequence[float]],
    class_index: int,
) -> float:
    if isinstance(nms_thresh, (list, tuple)):
        if class_index < 0 or class_index >= len(nms_thresh):
            return float(NMS_IOU_THRESH)
        return float(nms_thresh[class_index])
    return float(nms_thresh)


def class_wise_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    nms_thresh: Union[float, Sequence[float]] = NMS_IOU_THRESH,
) -> torch.Tensor:
    """
    Apply NMS independently for each class id.

    Returns global kept indices.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    keep_global: List[torch.Tensor] = []
    unique_cls = class_ids.unique(sorted=True)

    for cls in unique_cls:
        cls_mask = class_ids == cls
        idx = torch.where(cls_mask)[0]
        if idx.numel() == 0:
            continue

        cls_i = int(cls.item())
        cls_thresh = _resolve_class_nms_thresh(nms_thresh=nms_thresh, class_index=cls_i)
        cls_keep_rel = nms_single_class(boxes[idx], scores[idx], iou_thresh=cls_thresh)
        keep_global.append(idx[cls_keep_rel])

    if not keep_global:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    keep = torch.cat(keep_global, dim=0)
    # Return in global confidence order.
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return keep


def remap_boxes_to_original(boxes: torch.Tensor, meta: LetterboxMeta) -> torch.Tensor:
    """
    Convert boxes from letterbox (IMG_SIZE space) to original image coordinates.
    """
    if boxes.numel() == 0:
        return boxes

    scale = float(meta.scale)
    dx = float(meta.dx)
    dy = float(meta.dy)

    out = boxes.clone().float()
    out[:, 0] = (out[:, 0] - dx) / max(scale, 1e-12)
    out[:, 1] = (out[:, 1] - dy) / max(scale, 1e-12)
    out[:, 2] = (out[:, 2] - dx) / max(scale, 1e-12)
    out[:, 3] = (out[:, 3] - dy) / max(scale, 1e-12)

    out[:, 0] = out[:, 0].clamp(0.0, float(meta.orig_w))
    out[:, 1] = out[:, 1].clamp(0.0, float(meta.orig_h))
    out[:, 2] = out[:, 2].clamp(0.0, float(meta.orig_w))
    out[:, 3] = out[:, 3].clamp(0.0, float(meta.orig_h))

    return out


def filter_small_boxes(boxes: torch.Tensor, min_size: float = 2.0) -> torch.Tensor:
    """Return boolean mask keeping boxes with width/height >= min_size."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes.device)

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return (w >= float(min_size)) & (h >= float(min_size))


def _is_fully_inside(one: torch.Tensor, other: torch.Tensor) -> bool:
    return bool(
        one[0] >= other[0]
        and one[1] >= other[1]
        and one[2] <= other[2]
        and one[3] <= other[3]
    )


def suppress_same_class_contained(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    kept: List[int] = []

    for idx_t in order:
        idx = int(idx_t.item())
        cls = int(class_ids[idx].item())
        box = boxes[idx]
        drop = False
        for k in kept:
            if int(class_ids[k].item()) != cls:
                continue
            keep_box = boxes[k]
            if _is_fully_inside(box, keep_box) or _is_fully_inside(keep_box, box):
                drop = True
                break
        if not drop:
            kept.append(idx)

    if not kept:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    return torch.as_tensor(kept, dtype=torch.long, device=boxes.device)


def postprocess_single_image(
    outputs: Dict[str, Any],
    image_id: str,
    letterbox_meta: Optional[LetterboxMeta] = None,
    image_index: int = 0,
    class_names: Optional[Sequence[str]] = None,
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: Union[float, Sequence[float]] = NMS_IOU_THRESH_PER_CLASS,
    reg_decode: str = "auto",
    center_combine: str = "mul",
    background_index: Optional[int] = None,
    min_box_size: float = 2.0,
) -> Dict[str, Any]:
    """
    Full decode + NMS + remap pipeline for one image.

    Output format:
    {
      "image_id": "xxx.jpg",
      "boxes": [
         {"class": "person", "confidence": 0.91, "bbox": [x1,y1,x2,y2]},
         ...
      ]
    }
    """
    if class_names is None:
        class_names = CLASS_NAMES

    boxes, scores, cls_ids = decode_multilevel(
        outputs=outputs,
        image_index=image_index,
        img_size=img_size,
        conf_thresh=conf_thresh,
        num_classes=num_classes,
        strides=outputs.get("strides", STRIDES),
        reg_decode=reg_decode,
        center_combine=center_combine,
        background_index=background_index,
    )

    if boxes.numel() == 0:
        return {"image_id": image_id, "boxes": []}

    keep = class_wise_nms(
        boxes=boxes,
        scores=scores,
        class_ids=cls_ids,
        nms_thresh=nms_thresh,
    )

    boxes = boxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    if letterbox_meta is not None:
        boxes = remap_boxes_to_original(boxes, letterbox_meta)

    valid = filter_small_boxes(boxes, min_size=min_box_size)
    boxes = boxes[valid]
    scores = scores[valid]
    cls_ids = cls_ids[valid]

    keep_contained = suppress_same_class_contained(boxes=boxes, scores=scores, class_ids=cls_ids)
    boxes = boxes[keep_contained]
    scores = scores[keep_contained]
    cls_ids = cls_ids[keep_contained]

    pred_boxes: List[Dict[str, Any]] = []
    for b, s, c in zip(boxes.tolist(), scores.tolist(), cls_ids.tolist()):
        cls_idx = int(c)
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)

        pred_boxes.append(
            {
                "class": cls_name,
                "confidence": float(s),
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            }
        )

    return {
        "image_id": image_id,
        "boxes": pred_boxes,
    }


def postprocess_batch(
    outputs: Dict[str, Any],
    image_ids: Sequence[str],
    metas: Optional[Sequence[Optional[LetterboxMeta]]] = None,
    class_names: Optional[Sequence[str]] = None,
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: Union[float, Sequence[float]] = NMS_IOU_THRESH_PER_CLASS,
    reg_decode: str = "auto",
    center_combine: str = "mul",
    background_index: Optional[int] = None,
    min_box_size: float = 2.0,
) -> List[Dict[str, Any]]:
    """Batch wrapper for JSON-ready predictions."""
    bsz = outputs["cls_logits"][0].shape[0]
    if len(image_ids) != bsz:
        raise ValueError(f"image_ids length ({len(image_ids)}) must equal batch size ({bsz}).")

    if metas is None:
        metas = [None] * bsz
    if len(metas) != bsz:
        raise ValueError(f"metas length ({len(metas)}) must equal batch size ({bsz}).")

    results: List[Dict[str, Any]] = []
    for i in range(bsz):
        results.append(
            postprocess_single_image(
                outputs=outputs,
                image_id=image_ids[i],
                letterbox_meta=metas[i],
                image_index=i,
                class_names=class_names,
                num_classes=num_classes,
                img_size=img_size,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                reg_decode=reg_decode,
                center_combine=center_combine,
                background_index=background_index,
                min_box_size=min_box_size,
            )
        )
    return results
