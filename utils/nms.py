from __future__ import annotations

"""
Post-processing (decode + confidence filtering + class-wise NMS) for anchor-free detector.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from .config import (
        CLASS_NAMES,
        CLASS_SCORE_SCALES,
        CONF_THRESH,
        CONTAINED_BOX_REPLACE_MARGIN,
        DECODE_CANDIDATE_CONF,
        DECODE_TOPK,
        IMG_SIZE,
        INFER_CENTER_COMBINE,
        MAX_OBJECTS_PER_IMAGE,
        MIN_BOX_SIZE,
        NMS_IOU_THRESH,
        NUM_CLASSES,
        PRE_NMS_MAX_CANDIDATES,
        STRIDES,
    )
except Exception:
    CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
    CLASS_SCORE_SCALES = [1.0 for _ in CLASS_NAMES]
    INFER_CENTER_COMBINE = "cls"
    CONF_THRESH = 0.50
    CONTAINED_BOX_REPLACE_MARGIN = 0.12
    DECODE_CANDIDATE_CONF = 0.20
    DECODE_TOPK = 2
    PRE_NMS_MAX_CANDIDATES = 3500
    NMS_IOU_THRESH = 0.50
    IMG_SIZE = 320
    NUM_CLASSES = 5
    STRIDES = [4, 8, 16, 32]
    MAX_OBJECTS_PER_IMAGE = 20
    MIN_BOX_SIZE = 1.0


@dataclass
class LetterboxMeta:
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


def _combine_center_scores(
    class_scores: torch.Tensor,
    center_logits: Optional[torch.Tensor],
    center_combine: str,
) -> torch.Tensor:
    if center_logits is None:
        return class_scores
    if center_logits.dim() != 3 or center_logits.shape[0] != 1:
        raise ValueError("center_logits must be (1,H,W) when provided")

    center = torch.sigmoid(center_logits[0]).unsqueeze(0)
    mode = center_combine.lower()
    if mode == "sqrt":
        return torch.sqrt((class_scores * center).clamp(min=1e-12))
    if mode == "soft":
        return class_scores * (0.5 + 0.5 * center)
    if mode == "cls":
        return class_scores
    return class_scores * center


def decode_level(
    cls_logits: torch.Tensor,
    reg_preds: torch.Tensor,
    center_logits: Optional[torch.Tensor],
    stride: int,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    num_classes: int = NUM_CLASSES,
    reg_decode: str = "auto",
    center_combine: str = INFER_CENTER_COMBINE,
    background_index: Optional[int] = None,
    topk: int = DECODE_TOPK,
    pre_nms_max_candidates: int = PRE_NMS_MAX_CANDIDATES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cls_logits.dim() != 3:
        raise ValueError("cls_logits must be (C,H,W)")
    if reg_preds.dim() != 3 or reg_preds.shape[0] != 4:
        raise ValueError("reg_preds must be (4,H,W)")

    c, h, w = cls_logits.shape
    device = cls_logits.device
    topk = max(1, int(topk))
    conf_thresh = float(min(conf_thresh, DECODE_CANDIDATE_CONF if c == num_classes else conf_thresh))

    if c == num_classes:
        class_scores = torch.sigmoid(cls_logits)
        if len(CLASS_SCORE_SCALES) == int(num_classes):
            scale = torch.as_tensor(CLASS_SCORE_SCALES, dtype=class_scores.dtype, device=device).view(-1, 1, 1)
            class_scores = torch.clamp(class_scores * scale, min=0.0, max=1.0)
        class_scores = _combine_center_scores(class_scores, center_logits=center_logits, center_combine=center_combine)
        k = min(topk, int(num_classes))
        conf_k, cls_id_k = torch.topk(class_scores, k=k, dim=0)
        keep = conf_k > conf_thresh
    elif c == num_classes + 1:
        prob = torch.softmax(cls_logits, dim=0)
        if background_index is None:
            background_index = num_classes
        if background_index < 0 or background_index >= c:
            raise ValueError("background_index out of range for cls logits")

        fg_indices = [i for i in range(c) if i != background_index]
        class_scores = prob[fg_indices, :, :]
        if len(CLASS_SCORE_SCALES) == int(num_classes):
            scale = torch.as_tensor(CLASS_SCORE_SCALES, dtype=class_scores.dtype, device=device).view(-1, 1, 1)
            class_scores = torch.clamp(class_scores * scale, min=0.0, max=1.0)
        class_scores = class_scores * (1.0 - prob[background_index, :, :]).unsqueeze(0)
        class_scores = _combine_center_scores(class_scores, center_logits=center_logits, center_combine=center_combine)
        conf_k, cls_id_local = torch.topk(class_scores, k=1, dim=0)
        fg_ids_tensor = torch.tensor(fg_indices, device=device, dtype=torch.long)
        cls_id_k = fg_ids_tensor[cls_id_local]
        keep = conf_k > conf_thresh
    else:
        raise ValueError(
            f"Unexpected cls channels={c}. Expected num_classes ({num_classes}) "
            f"or num_classes+1 ({num_classes + 1})."
        )

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

    rank_idx, ys, xs = torch.where(keep)
    boxes_keep = boxes[ys, xs]
    scores_keep = conf_k[rank_idx, ys, xs]
    cls_keep = cls_id_k[rank_idx, ys, xs].long()

    valid = (boxes_keep[:, 2] > boxes_keep[:, 0]) & (boxes_keep[:, 3] > boxes_keep[:, 1])
    boxes_keep = boxes_keep[valid]
    scores_keep = scores_keep[valid]
    cls_keep = cls_keep[valid]

    if boxes_keep.shape[0] > int(pre_nms_max_candidates):
        top_keep = torch.argsort(scores_keep, descending=True)[: int(pre_nms_max_candidates)]
        boxes_keep = boxes_keep[top_keep]
        scores_keep = scores_keep[top_keep]
        cls_keep = cls_keep[top_keep]

    return boxes_keep, scores_keep, cls_keep


def decode_multilevel(
    outputs: Dict[str, Any],
    image_index: int = 0,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    num_classes: int = NUM_CLASSES,
    strides: Optional[Sequence[int]] = None,
    reg_decode: str = "auto",
    center_combine: str = INFER_CENTER_COMBINE,
    background_index: Optional[int] = None,
    topk: int = DECODE_TOPK,
    pre_nms_max_candidates: int = PRE_NMS_MAX_CANDIDATES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            topk=topk,
            pre_nms_max_candidates=pre_nms_max_candidates,
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

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    class_ids = torch.cat(all_classes, dim=0)

    if boxes.shape[0] > int(pre_nms_max_candidates):
        keep = torch.argsort(scores, descending=True)[: int(pre_nms_max_candidates)]
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
    return boxes, scores, class_ids


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
    union = area_one + area_many - inter + 1e-6
    return inter / union


def _is_fully_inside_xyxy(inner: torch.Tensor, outer: torch.Tensor) -> bool:
    return bool(
        float(inner[0]) >= float(outer[0])
        and float(inner[1]) >= float(outer[1])
        and float(inner[2]) <= float(outer[2])
        and float(inner[3]) <= float(outer[3])
    )


def _box_area_xyxy(box: torch.Tensor) -> float:
    return float(max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1])))


def _suppress_same_class_contained(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, class_ids

    order = torch.argsort(scores, descending=True)
    keep: List[int] = []

    for idx in order.tolist():
        cls = int(class_ids[idx].item())
        candidate = boxes[idx]
        candidate_score = float(scores[idx].item())
        candidate_area = _box_area_xyxy(candidate)
        drop = False
        replace_slot = -1

        for slot, kept_idx in enumerate(keep):
            if int(class_ids[kept_idx].item()) != cls:
                continue
            kept_box = boxes[kept_idx]
            kept_score = float(scores[kept_idx].item())
            kept_area = _box_area_xyxy(kept_box)

            cand_inside_kept = _is_fully_inside_xyxy(candidate, kept_box)
            kept_inside_cand = _is_fully_inside_xyxy(kept_box, candidate)
            if not (cand_inside_kept or kept_inside_cand):
                continue

            if (
                kept_inside_cand
                and candidate_area > kept_area
                and candidate_score + float(CONTAINED_BOX_REPLACE_MARGIN) >= kept_score
            ):
                replace_slot = slot
                break

            drop = True
            break

        if replace_slot >= 0:
            keep[replace_slot] = idx
            continue

        if not drop:
            keep.append(idx)

    if not keep:
        device = boxes.device
        return (
            torch.zeros((0, 4), dtype=boxes.dtype, device=device),
            torch.zeros((0,), dtype=scores.dtype, device=device),
            torch.zeros((0,), dtype=class_ids.dtype, device=device),
        )

    keep_idx = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[keep_idx], scores[keep_idx], class_ids[keep_idx]


def nms_single_class(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
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
        order = rest[ious <= float(iou_thresh)]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def class_wise_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    nms_thresh: float = NMS_IOU_THRESH,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    keep_global: List[torch.Tensor] = []
    for cls in class_ids.unique(sorted=True):
        idx = torch.where(class_ids == cls)[0]
        if idx.numel() == 0:
            continue
        cls_keep_rel = nms_single_class(boxes[idx], scores[idx], iou_thresh=float(nms_thresh))
        keep_global.append(idx[cls_keep_rel])

    if not keep_global:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    keep = torch.cat(keep_global, dim=0)
    return keep[torch.argsort(scores[keep], descending=True)]


def remap_boxes_to_original(boxes: torch.Tensor, meta: LetterboxMeta) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone().float()
    scale = float(meta.scale)
    out[:, 0] = (out[:, 0] - float(meta.dx)) / max(scale, 1e-12)
    out[:, 1] = (out[:, 1] - float(meta.dy)) / max(scale, 1e-12)
    out[:, 2] = (out[:, 2] - float(meta.dx)) / max(scale, 1e-12)
    out[:, 3] = (out[:, 3] - float(meta.dy)) / max(scale, 1e-12)
    out[:, 0] = out[:, 0].clamp(0.0, float(meta.orig_w))
    out[:, 1] = out[:, 1].clamp(0.0, float(meta.orig_h))
    out[:, 2] = out[:, 2].clamp(0.0, float(meta.orig_w))
    out[:, 3] = out[:, 3].clamp(0.0, float(meta.orig_h))
    return out


def filter_small_boxes(boxes: torch.Tensor, min_size: float = MIN_BOX_SIZE) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes.device)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return (w >= float(min_size)) & (h >= float(min_size))


def postprocess_single_image(
    outputs: Dict[str, Any],
    image_id: str,
    letterbox_meta: Optional[LetterboxMeta] = None,
    image_index: int = 0,
    class_names: Optional[Sequence[str]] = None,
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_IOU_THRESH,
    reg_decode: str = "auto",
    center_combine: str = INFER_CENTER_COMBINE,
    background_index: Optional[int] = None,
    min_box_size: float = MIN_BOX_SIZE,
    topk: int = DECODE_TOPK,
    pre_nms_max_candidates: int = PRE_NMS_MAX_CANDIDATES,
) -> Dict[str, Any]:
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
        topk=topk,
        pre_nms_max_candidates=pre_nms_max_candidates,
    )
    if boxes.numel() == 0:
        return {"image_id": image_id, "boxes": []}

    keep = class_wise_nms(boxes=boxes, scores=scores, class_ids=cls_ids, nms_thresh=nms_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    if letterbox_meta is not None:
        boxes = remap_boxes_to_original(boxes, letterbox_meta)

    valid = filter_small_boxes(boxes, min_size=min_box_size)
    boxes = boxes[valid]
    scores = scores[valid]
    cls_ids = cls_ids[valid]
    boxes, scores, cls_ids = _suppress_same_class_contained(boxes, scores, cls_ids)

    if boxes.shape[0] > int(MAX_OBJECTS_PER_IMAGE):
        keep_top = torch.argsort(scores, descending=True)[: int(MAX_OBJECTS_PER_IMAGE)]
        boxes = boxes[keep_top]
        scores = scores[keep_top]
        cls_ids = cls_ids[keep_top]

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
    return {"image_id": image_id, "boxes": pred_boxes}


def postprocess_batch(
    outputs: Dict[str, Any],
    image_ids: Sequence[str],
    metas: Optional[Sequence[Optional[LetterboxMeta]]] = None,
    class_names: Optional[Sequence[str]] = None,
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_IOU_THRESH,
    reg_decode: str = "auto",
    center_combine: str = INFER_CENTER_COMBINE,
    background_index: Optional[int] = None,
    min_box_size: float = MIN_BOX_SIZE,
    topk: int = DECODE_TOPK,
    pre_nms_max_candidates: int = PRE_NMS_MAX_CANDIDATES,
) -> List[Dict[str, Any]]:
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
                topk=topk,
                pre_nms_max_candidates=pre_nms_max_candidates,
            )
        )
    return results
