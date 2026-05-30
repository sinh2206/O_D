from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import ANCHORS, ANCHOR_MASKS, CLASS_NAMES, CONF_THRESH, IMG_SIZE, MAX_OBJECTS_PER_IMAGE, NMS_IOU_THRESH, NUM_CLASSES, STRIDES
from .image_ops import LetterboxMeta, meta_to_unletterbox


def _make_grid(h: int, w: int, stride: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    try:
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        gy, gx = torch.meshgrid(ys, xs)
    return gx, gy


def _box_iou_xyxy(one: torch.Tensor, many: torch.Tensor) -> torch.Tensor:
    xx1 = torch.maximum(one[0], many[:, 0])
    yy1 = torch.maximum(one[1], many[:, 1])
    xx2 = torch.minimum(one[2], many[:, 2])
    yy2 = torch.minimum(one[3], many[:, 3])
    inter_w = (xx2 - xx1).clamp(min=0)
    inter_h = (yy2 - yy1).clamp(min=0)
    inter = inter_w * inter_h
    area_one = (one[2] - one[0]).clamp(min=0) * (one[3] - one[1]).clamp(min=0)
    area_many = (many[:, 2] - many[:, 0]).clamp(min=0) * (many[:, 3] - many[:, 1]).clamp(min=0)
    return inter / (area_one + area_many - inter + 1e-9)


def nms_per_class(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = NMS_IOU_THRESH) -> torch.Tensor:
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


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: Optional[torch.Tensor] = None,
    iou_thresh: float = NMS_IOU_THRESH,
    class_agnostic: bool = False,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    if class_agnostic or class_ids is None:
        return nms_per_class(boxes, scores, iou_thresh=iou_thresh)

    keep_all: List[torch.Tensor] = []
    for cls in class_ids.unique(sorted=True):
        idx = torch.where(class_ids == cls)[0]
        if idx.numel() == 0:
            continue
        keep_rel = nms_per_class(boxes[idx], scores[idx], iou_thresh=iou_thresh)
        keep_all.append(idx[keep_rel])

    if not keep_all:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    keep = torch.cat(keep_all, dim=0)
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return keep


def decode_predictions(
    pred: torch.Tensor,
    stride: int,
    img_size: int,
    anchor_sizes: Sequence[Tuple[int, int]],
    conf_thresh: float = CONF_THRESH,
    num_classes: int = NUM_CLASSES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.dim() == 4:
        pred = pred[0]
    if pred.dim() != 3:
        raise ValueError("decode_predictions expects a single-image tensor shaped (C,H,W) or (A*(5+C),H,W)")

    num_anchors = len(anchor_sizes)
    expected = num_anchors * (5 + num_classes)
    if pred.shape[0] != expected:
        raise ValueError(f"Expected {expected} channels for decode, got {pred.shape[0]}")

    _, h, w = pred.shape
    pred = pred.view(num_anchors, 5 + num_classes, h, w).permute(0, 2, 3, 1).contiguous()  # A,H,W,5+C

    tx = torch.sigmoid(pred[..., 0])
    ty = torch.sigmoid(pred[..., 1])
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = torch.sigmoid(pred[..., 4])
    cls_prob = torch.sigmoid(pred[..., 5:])
    cls_score, cls_id = cls_prob.max(dim=-1)
    score = obj * cls_score

    keep = score >= float(conf_thresh)
    if not torch.any(keep):
        device = pred.device
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    gx, gy = _make_grid(h, w, stride=stride, device=pred.device)
    gx = gx.unsqueeze(0)
    gy = gy.unsqueeze(0)
    anchors = torch.as_tensor(anchor_sizes, dtype=torch.float32, device=pred.device).view(num_anchors, 1, 1, 2)

    cx = (tx + gx) * float(stride)
    cy = (ty + gy) * float(stride)
    bw = torch.exp(tw).clamp(max=1e4) * anchors[..., 0]
    bh = torch.exp(th).clamp(max=1e4) * anchors[..., 1]

    x1 = (cx - bw / 2.0).clamp(0.0, float(img_size))
    y1 = (cy - bh / 2.0).clamp(0.0, float(img_size))
    x2 = (cx + bw / 2.0).clamp(0.0, float(img_size))
    y2 = (cy + bh / 2.0).clamp(0.0, float(img_size))

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)[keep]
    scores = score[keep]
    cls_ids = cls_id[keep].long()

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    cls_ids = cls_ids[valid]

    return boxes, scores, cls_ids


def _normalize_outputs(outputs: Any) -> List[torch.Tensor]:
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    if isinstance(outputs, dict):
        if "predictions" in outputs:
            return list(outputs["predictions"])
        if "raw_outputs" in outputs:
            return list(outputs["raw_outputs"])
        if "outputs" in outputs:
            return list(outputs["outputs"])
    raise TypeError(f"Unsupported outputs type: {type(outputs)!r}")


def postprocess_batch(
    outputs: Any,
    image_ids: Sequence[str],
    metas: Optional[Sequence[Optional[LetterboxMeta]]] = None,
    class_names: Optional[Sequence[str]] = None,
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_IOU_THRESH,
    reg_decode: str = "auto",
    center_combine: str = "mul",
    min_box_size: float = 2.0,
    class_agnostic_nms: bool = False,
) -> List[Dict[str, Any]]:
    del reg_decode, center_combine
    if class_names is None:
        class_names = CLASS_NAMES

    pred_levels = _normalize_outputs(outputs)
    if len(pred_levels) != len(STRIDES):
        raise ValueError(f"Expected {len(STRIDES)} prediction levels, got {len(pred_levels)}")

    batch_size = pred_levels[0].shape[0]
    if len(image_ids) != batch_size:
        raise ValueError(f"image_ids length {len(image_ids)} does not match batch size {batch_size}")

    if metas is None:
        metas = [None] * batch_size

    results: List[Dict[str, Any]] = []
    for bi, image_id in enumerate(image_ids):
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_cls: List[torch.Tensor] = []

        for lvl, stride in enumerate(STRIDES):
            level_pred = pred_levels[lvl][bi]
            boxes, scores, cls_ids = decode_predictions(
                pred=level_pred,
                stride=int(stride),
                img_size=img_size,
                anchor_sizes=ANCHORS[lvl],
                conf_thresh=conf_thresh,
                num_classes=num_classes,
            )
            if boxes.numel() == 0:
                continue
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_cls.append(cls_ids)

        if not all_boxes:
            results.append({"image_id": image_id, "boxes": []})
            continue

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        cls_ids = torch.cat(all_cls, dim=0)

        keep = nms(boxes, scores, class_ids=cls_ids, iou_thresh=nms_thresh, class_agnostic=class_agnostic_nms)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        if metas[bi] is not None:
            boxes_np = meta_to_unletterbox(boxes.detach().cpu().numpy(), metas[bi])
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32, device=boxes.device)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        cls_ids = cls_ids[valid]

        if boxes.shape[0] > int(MAX_OBJECTS_PER_IMAGE):
            topk = torch.argsort(scores, descending=True)[: int(MAX_OBJECTS_PER_IMAGE)]
            boxes = boxes[topk]
            scores = scores[topk]
            cls_ids = cls_ids[topk]

        pred_boxes: List[Dict[str, Any]] = []
        for box, score, cls_id in zip(boxes.tolist(), scores.tolist(), cls_ids.tolist()):
            cls_idx = int(cls_id)
            cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
            pred_boxes.append(
                {
                    "class": cls_name,
                    "confidence": float(score),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                }
            )

        results.append({"image_id": image_id, "boxes": pred_boxes})

    return results
