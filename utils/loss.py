from __future__ import annotations

"""
Loss utilities for anchor-free detection (FCOS/YOLOX-simplified style).

Total loss:
    L = lambda_cls * L_cls + lambda_reg * L_reg + lambda_ctr * L_ctr

Where:
- L_cls: focal loss for classification.
- L_reg: GIoU + normalized SmoothL1 for box regression.
- L_ctr: BCE-with-logits for centerness (optional if branch exists).

Target assignment:
- Center sampling per FPN level.
- A location is positive if it is inside GT box, inside center region, and
  inside that level's scale range.
- If multiple GT boxes match one location, pick the GT with smallest area.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import (
        CENTER_RADIUS,
        FOCAL_ALPHA,
        FOCAL_GAMMA,
        LABEL_SMOOTHING,
        LAMBDA_CLS,
        LAMBDA_CTR,
        LAMBDA_REG,
        LAMBDA_REG_IOU,
        LAMBDA_REG_L1,
        NEGATIVE_FOCAL_WEIGHT,
        NUM_CLASSES,
        REG_SMOOTH_L1_BETA,
        STRIDES,
        TINY_ASSIGN_EXPAND_STRIDE,
        TINY_OBJECT_MAX_SIDE_FACTOR,
    )
except Exception:
    # Safe fallbacks so this module still works if config.py is not available.
    NUM_CLASSES = 5
    STRIDES = [8, 16, 32]
    CENTER_RADIUS = 2.0
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25
    LABEL_SMOOTHING = 0.01
    NEGATIVE_FOCAL_WEIGHT = 0.5
    LAMBDA_CLS = 1.0
    LAMBDA_REG = 1.0
    LAMBDA_CTR = 0.5
    LAMBDA_REG_IOU = 1.0
    LAMBDA_REG_L1 = 0.25
    REG_SMOOTH_L1_BETA = 0.10
    TINY_OBJECT_MAX_SIDE_FACTOR = 2.0
    TINY_ASSIGN_EXPAND_STRIDE = 0.75

EPS = 1e-8
DEFAULT_CENTER_RADIUS = float(CENTER_RADIUS)


def _meshgrid_xy(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create x/y coordinate grids in feature-map space."""

    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    try:
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        gy, gx = torch.meshgrid(ys, xs)
    return gx, gy


def build_level_points(height: int, width: int, stride: float, device: torch.device) -> torch.Tensor:
    """Build (x, y) point centers for one feature map level in input-image pixels."""
    gx, gy = _meshgrid_xy(height, width, device)
    px = (gx + 0.5) * float(stride)
    py = (gy + 0.5) * float(stride)
    return torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1)


def tlbr_to_xyxy(points_xy: torch.Tensor, tlbr: torch.Tensor) -> torch.Tensor:
    """
    Convert distances (t, l, b, r) to box xyxy at each point.

    Args:
        points_xy: (N, 2) with columns [x, y]
        tlbr:      (N, 4) with columns [t, l, b, r]
    """
    t = tlbr[:, 0]
    l = tlbr[:, 1]
    b = tlbr[:, 2]
    r = tlbr[:, 3]

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute area for a batch of xyxy boxes."""

    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Generalized IoU loss between two sets of boxes (N,4) in xyxy."""
    if pred_boxes.numel() == 0:
        return pred_boxes.new_tensor(0.0)

    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = _box_area(pred_boxes)
    area_t = _box_area(target_boxes)
    union = area_p + area_t - inter + EPS
    iou = inter / union

    ex1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    ey1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    ex2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    ey2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enc_w = (ex2 - ex1).clamp(min=0)
    enc_h = (ey2 - ey1).clamp(min=0)
    enc_area = enc_w * enc_h + EPS

    giou = iou - (enc_area - union) / enc_area
    loss = 1.0 - giou

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def normalized_box_l1_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    beta: float = REG_SMOOTH_L1_BETA,
    reduction: str = "mean",
) -> torch.Tensor:
    """SmoothL1 on xyxy edges after normalizing by target box width/height."""

    if pred_boxes.numel() == 0:
        return pred_boxes.new_tensor(0.0)

    tgt_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1.0)
    tgt_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1.0)
    scale = torch.stack([tgt_w, tgt_h, tgt_w, tgt_h], dim=-1)

    diff = (pred_boxes - target_boxes) / scale
    loss = F.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        beta=float(beta),
        reduction="none",
    ).sum(dim=-1)

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def _centerness_from_tlbr(tlbr: torch.Tensor) -> torch.Tensor:
    """Centerness target from GT distances (t, l, b, r)."""
    t = tlbr[:, 0].clamp(min=EPS)
    l = tlbr[:, 1].clamp(min=EPS)
    b = tlbr[:, 2].clamp(min=EPS)
    r = tlbr[:, 3].clamp(min=EPS)

    lr = torch.min(l, r) / torch.max(l, r)
    tb = torch.min(t, b) / torch.max(t, b)
    return torch.sqrt((lr * tb).clamp(min=0.0, max=1.0))


def focal_sigmoid_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
    negative_weight: float = NEGATIVE_FOCAL_WEIGHT,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss for multi-label classification."""
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    if negative_weight != 1.0:
        neg_mask = targets <= 0.5
        loss = torch.where(neg_mask, loss * float(negative_weight), loss)
    if class_weights is not None:
        w = class_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        # Weight positives more strongly, but do not amplify the vast number of
        # negative locations for a class, which tends to suppress recall.
        pos_mask = targets > 0.5
        loss = torch.where(pos_mask, loss * w, loss)

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def focal_softmax_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    bg_index: int,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Softmax focal loss when logits include explicit background class.

    logits:     (N, C+1)
    target_idx: (N,) in [0..C] where C is bg_index
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    valid = target_idx >= 0
    if not valid.any():
        return logits.new_tensor(0.0)

    idx = target_idx[valid].long()
    lp = log_probs[valid, :].gather(1, idx.unsqueeze(1)).squeeze(1)
    p = probs[valid, :].gather(1, idx.unsqueeze(1)).squeeze(1)

    alpha_t = torch.full_like(p, float(alpha))
    alpha_t[idx == int(bg_index)] = float(1.0 - alpha)

    loss = -alpha_t * (1.0 - p).pow(gamma) * lp

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def _normalize_targets(
    targets: Any,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """
    Normalize targets to list[{"boxes":(N,4), "labels":(N,)}] for each image.

    Supported formats:
    - list of dicts: [{"boxes":..., "labels":...}, ...]
    - dict of batched tensors: {"boxes":(B,N,4), "labels":(B,N)}
    - tuple/list of (boxes, labels) batched tensors.
    """

    def _clean_one(boxes: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Normalize one image target by removing padding and invalid boxes."""

        if boxes.numel() == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            }

        boxes = boxes.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        if boxes.dim() == 1:
            boxes = boxes.view(-1, 4)
        if labels.dim() > 1:
            labels = labels.view(-1)

        # Remove padding labels and invalid boxes.
        valid = labels >= 0
        if valid.numel() == boxes.shape[0]:
            boxes = boxes[valid]
            labels = labels[valid]

        if boxes.numel() == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            }

        wh = boxes[:, 2:] - boxes[:, :2]
        box_valid = (wh[:, 0] > 1.0) & (wh[:, 1] > 1.0)
        boxes = boxes[box_valid]
        labels = labels[box_valid]

        return {
            "boxes": boxes,
            "labels": labels,
        }

    out: List[Dict[str, torch.Tensor]] = []

    if isinstance(targets, list):
        for i in range(batch_size):
            if i >= len(targets):
                out.append(_clean_one(torch.zeros((0, 4), device=device), torch.zeros((0,), device=device)))
                continue
            item = targets[i]
            if isinstance(item, dict):
                out.append(_clean_one(item.get("boxes", torch.zeros((0, 4))), item.get("labels", torch.zeros((0,)))))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                out.append(_clean_one(item[0], item[1]))
            else:
                out.append(_clean_one(torch.zeros((0, 4)), torch.zeros((0,))))
        return out

    if isinstance(targets, dict):
        b_boxes = targets.get("boxes", None)
        b_labels = targets.get("labels", None)
        if b_boxes is None or b_labels is None:
            raise ValueError("Targets dict must contain 'boxes' and 'labels'.")

        if b_boxes.dim() == 2:
            b_boxes = b_boxes.unsqueeze(0)
            b_labels = b_labels.unsqueeze(0)

        for i in range(batch_size):
            out.append(_clean_one(b_boxes[i], b_labels[i]))
        return out

    if isinstance(targets, (tuple, list)) and len(targets) >= 2:
        b_boxes, b_labels = targets[0], targets[1]
        if b_boxes.dim() == 2:
            b_boxes = b_boxes.unsqueeze(0)
            b_labels = b_labels.unsqueeze(0)
        for i in range(batch_size):
            out.append(_clean_one(b_boxes[i], b_labels[i]))
        return out

    raise ValueError("Unsupported target format for compute_loss/build_targets.")


def _default_scale_ranges(strides: Sequence[int]) -> List[Tuple[float, float]]:
    """
    FCOS-like object size ranges (max(l,t,r,b) in pixels) per level.

    The ranges are intentionally slightly overlapping to keep small and
    partially occluded objects from disappearing at level boundaries.
    """
    ranges: List[Tuple[float, float]] = []
    for i, s in enumerate(strides):
        if i == 0:
            lower = 0.0
        else:
            lower = float(2.5 * strides[i - 1])

        if i == len(strides) - 1:
            upper = float("inf")
        else:
            upper = float(8.0 * s)

        ranges.append((lower, upper))
    return ranges


def _assign_level_targets(
    points_xy: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    stride: int,
    num_classes: int,
    use_softmax_bg: bool,
    scale_range: Optional[Tuple[float, float]],
    center_radius: float,
    tiny_object_max_side_factor: float,
    tiny_assign_expand_stride: float,
) -> Dict[str, torch.Tensor]:
    """
    Assign targets for one image and one FPN level.

    Returns flat tensors over points (P):
      - cls_idx: (P,)      class id or background id
      - cls_onehot: (P,C)  for sigmoid focal mode
      - reg_tlbr: (P,4)
      - ctr: (P,)
      - pos_mask: (P,)
    """
    device = points_xy.device
    p = points_xy.shape[0]

    bg_index = num_classes
    cls_idx = torch.full((p,), fill_value=bg_index if use_softmax_bg else -1, device=device, dtype=torch.long)
    cls_onehot = torch.zeros((p, num_classes), device=device, dtype=torch.float32)
    reg_tlbr = torch.zeros((p, 4), device=device, dtype=torch.float32)
    ctr = torch.zeros((p,), device=device, dtype=torch.float32)
    pos_mask = torch.zeros((p,), device=device, dtype=torch.bool)

    if gt_boxes.numel() == 0:
        if not use_softmax_bg:
            # In sigmoid mode negatives are all-zero one-hot.
            cls_idx.fill_(-1)
        return {
            "cls_idx": cls_idx,
            "cls_onehot": cls_onehot,
            "reg_tlbr": reg_tlbr,
            "ctr": ctr,
            "pos_mask": pos_mask,
        }

    # Distances point -> GT edges for every (point, gt): shape (P, N).
    x = points_xy[:, 0:1]
    y = points_xy[:, 1:2]

    x1 = gt_boxes[:, 0].view(1, -1)
    y1 = gt_boxes[:, 1].view(1, -1)
    x2 = gt_boxes[:, 2].view(1, -1)
    y2 = gt_boxes[:, 3].view(1, -1)

    l = x - x1
    t = y - y1
    r = x2 - x
    b = y2 - y

    ltrb = torch.stack([l, t, r, b], dim=-1)  # (P, N, 4) order l,t,r,b
    tlbr = torch.stack([t, l, b, r], dim=-1)  # (P, N, 4) order t,l,b,r

    # Keep boundary points as valid for extremely small/quantized boxes.
    inside_box = (ltrb.min(dim=-1).values >= 0.0)

    # Center sampling region.
    gx = 0.5 * (x1 + x2)
    gy = 0.5 * (y1 + y2)
    radius = float(center_radius) * float(stride)

    cx1 = gx - radius
    cy1 = gy - radius
    cx2 = gx + radius
    cy2 = gy + radius

    inside_center = (x >= cx1) & (x <= cx2) & (y >= cy1) & (y <= cy2)
    inside_center = inside_center.to(dtype=torch.bool)

    candidate = inside_box & inside_center

    if scale_range is not None:
        max_dist = ltrb.max(dim=-1).values
        lo, hi = float(scale_range[0]), float(scale_range[1])
        if math.isfinite(hi):
            in_range = (max_dist >= lo) & (max_dist < hi)
        else:
            in_range = max_dist >= lo
        candidate = candidate & in_range

    # Tiny-object fallback: if a tiny GT gets no positive location at this
    # level, force the nearest point around its center as positive.
    gt_w = (x2 - x1).squeeze(0).clamp(min=0.0)
    gt_h = (y2 - y1).squeeze(0).clamp(min=0.0)
    gt_max_side = torch.maximum(gt_w, gt_h)
    tiny_limit = float(tiny_object_max_side_factor) * float(stride)
    tiny_mask = gt_max_side <= tiny_limit

    if tiny_mask.any():
        point_x = points_xy[:, 0]
        point_y = points_xy[:, 1]
        expand = float(tiny_assign_expand_stride) * float(stride)
        gx_1d = gx.squeeze(0)
        gy_1d = gy.squeeze(0)

        for gt_idx in torch.where(tiny_mask)[0].tolist():
            if bool(candidate[:, gt_idx].any()):
                continue

            ex1 = float(x1[0, gt_idx] - expand)
            ey1 = float(y1[0, gt_idx] - expand)
            ex2 = float(x2[0, gt_idx] + expand)
            ey2 = float(y2[0, gt_idx] + expand)

            in_expanded = (point_x >= ex1) & (point_x <= ex2) & (point_y >= ey1) & (point_y <= ey2)
            pool = torch.where(in_expanded)[0]
            if pool.numel() == 0:
                pool = torch.arange(points_xy.shape[0], device=device)

            dx = point_x[pool] - gx_1d[gt_idx]
            dy = point_y[pool] - gy_1d[gt_idx]
            best_local = torch.argmin(dx * dx + dy * dy)
            best_point = int(pool[best_local].item())
            candidate[best_point, gt_idx] = True

    # Resolve overlaps by smallest GT area.
    gt_area = _box_area(gt_boxes).view(1, -1).expand(p, -1)
    inf = torch.full_like(gt_area, 1e12)
    candidate_area = torch.where(candidate, gt_area, inf)

    min_area, min_ids = candidate_area.min(dim=1)
    pos = min_area < 1e11

    if pos.any():
        pos_idx = torch.where(pos)[0]
        gt_ids = min_ids[pos_idx]

        assigned_labels = gt_labels[gt_ids].long().clamp(min=0, max=max(num_classes - 1, 0))
        assigned_tlbr = tlbr[pos_idx, gt_ids, :]  # (Npos, 4)

        pos_mask[pos_idx] = True
        reg_tlbr[pos_idx] = assigned_tlbr
        ctr[pos_idx] = _centerness_from_tlbr(assigned_tlbr)

        cls_onehot[pos_idx, assigned_labels] = 1.0

        if use_softmax_bg:
            cls_idx[pos_idx] = assigned_labels
        else:
            cls_idx[pos_idx] = assigned_labels

    if use_softmax_bg:
        # Negatives stay as background class.
        cls_idx[~pos] = bg_index
    else:
        # Negatives stay one-hot all zeros (sigmoid focal).
        cls_idx[~pos] = -1

    return {
        "cls_idx": cls_idx,
        "cls_onehot": cls_onehot,
        "reg_tlbr": reg_tlbr,
        "ctr": ctr,
        "pos_mask": pos_mask,
    }


def build_targets(
    outputs: Dict[str, Any],
    targets: Any,
    num_classes: int = NUM_CLASSES,
    strides: Optional[Sequence[int]] = None,
    center_radius: float = DEFAULT_CENTER_RADIUS,
    use_scale_ranges: bool = True,
    tiny_object_max_side_factor: float = TINY_OBJECT_MAX_SIDE_FACTOR,
    tiny_assign_expand_stride: float = TINY_ASSIGN_EXPAND_STRIDE,
) -> Dict[str, Any]:
    """
    Build multi-level targets from model outputs and raw GT targets.

    Returns:
      {
        "cls_targets":  list of tensors
          - softmax mode: (B,H,W) class index in [0..C] (C is background)
          - sigmoid mode: (B,H,W,C) one-hot foreground labels
        "reg_targets":  list[(B,H,W,4)]  in order (t,l,b,r)
        "ctr_targets":  list[(B,H,W)]
        "pos_masks":    list[(B,H,W)] bool
        "points":       list[(H*W,2)]
        "use_softmax_bg": bool,
        "bg_index": int,
      }
    """
    cls_levels = outputs["cls_logits"]
    if strides is None:
        strides = outputs.get("strides", STRIDES)
    strides = list(strides)

    batch_size = cls_levels[0].shape[0]
    device = cls_levels[0].device

    # If class channels are C+1, use softmax focal with explicit background.
    c_out = cls_levels[0].shape[1]
    use_softmax_bg = (c_out == int(num_classes) + 1)
    bg_index = int(num_classes)

    targets_list = _normalize_targets(targets, batch_size=batch_size, device=device)

    if use_scale_ranges:
        scale_ranges = _default_scale_ranges(strides)
    else:
        scale_ranges = [None for _ in strides]

    cls_targets: List[torch.Tensor] = []
    reg_targets: List[torch.Tensor] = []
    ctr_targets: List[torch.Tensor] = []
    pos_masks: List[torch.Tensor] = []
    points_all: List[torch.Tensor] = []

    for lvl, (cls_t, stride, srange) in enumerate(zip(cls_levels, strides, scale_ranges)):
        _, _, h, w = cls_t.shape
        points = build_level_points(h, w, float(stride), device=device)
        points_all.append(points)

        if use_softmax_bg:
            cls_lvl = torch.full((batch_size, h * w), fill_value=bg_index, device=device, dtype=torch.long)
        else:
            cls_lvl = torch.zeros((batch_size, h * w, num_classes), device=device, dtype=torch.float32)

        reg_lvl = torch.zeros((batch_size, h * w, 4), device=device, dtype=torch.float32)
        ctr_lvl = torch.zeros((batch_size, h * w), device=device, dtype=torch.float32)
        pos_lvl = torch.zeros((batch_size, h * w), device=device, dtype=torch.bool)

        for b in range(batch_size):
            gt = targets_list[b]
            assigned = _assign_level_targets(
                points_xy=points,
                gt_boxes=gt["boxes"],
                gt_labels=gt["labels"],
                stride=int(stride),
                num_classes=int(num_classes),
                use_softmax_bg=use_softmax_bg,
                scale_range=srange,
                center_radius=float(center_radius),
                tiny_object_max_side_factor=float(tiny_object_max_side_factor),
                tiny_assign_expand_stride=float(tiny_assign_expand_stride),
            )

            if use_softmax_bg:
                cls_lvl[b] = assigned["cls_idx"]
            else:
                cls_lvl[b] = assigned["cls_onehot"]
            reg_lvl[b] = assigned["reg_tlbr"]
            ctr_lvl[b] = assigned["ctr"]
            pos_lvl[b] = assigned["pos_mask"]

        if use_softmax_bg:
            cls_targets.append(cls_lvl.view(batch_size, h, w))
        else:
            cls_targets.append(cls_lvl.view(batch_size, h, w, num_classes))

        reg_targets.append(reg_lvl.view(batch_size, h, w, 4))
        ctr_targets.append(ctr_lvl.view(batch_size, h, w))
        pos_masks.append(pos_lvl.view(batch_size, h, w))

    return {
        "cls_targets": cls_targets,
        "reg_targets": reg_targets,
        "ctr_targets": ctr_targets,
        "pos_masks": pos_masks,
        "points": points_all,
        "use_softmax_bg": use_softmax_bg,
        "bg_index": bg_index,
    }


def compute_loss(
    outputs: Dict[str, Any],
    targets: Any,
    num_classes: int = NUM_CLASSES,
    strides: Optional[Sequence[int]] = None,
    lambda_cls: float = LAMBDA_CLS,
    lambda_reg: float = LAMBDA_REG,
    lambda_ctr: float = LAMBDA_CTR,
    lambda_reg_iou: float = LAMBDA_REG_IOU,
    lambda_reg_l1: float = LAMBDA_REG_L1,
    focal_alpha: float = FOCAL_ALPHA,
    focal_gamma: float = FOCAL_GAMMA,
    negative_focal_weight: float = NEGATIVE_FOCAL_WEIGHT,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = LABEL_SMOOTHING,
    reg_smooth_l1_beta: float = REG_SMOOTH_L1_BETA,
    center_radius: float = DEFAULT_CENTER_RADIUS,
    use_scale_ranges: bool = True,
    tiny_object_max_side_factor: float = TINY_OBJECT_MAX_SIDE_FACTOR,
    tiny_assign_expand_stride: float = TINY_ASSIGN_EXPAND_STRIDE,
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-level anchor-free loss.

    Expects model outputs like:
      {
        "cls_logits": [B,C,H,W, ...],
        "reg_preds": [B,4,H,W, ...],   # (t,l,b,r) non-negative preferred
        "center_logits": [B,1,H,W, ...] (optional),
        "strides": [16,32,...]
      }
    """
    cls_levels = outputs["cls_logits"]
    reg_levels = outputs["reg_preds"]
    ctr_levels = outputs.get("center_logits", None)

    target_pack = build_targets(
        outputs=outputs,
        targets=targets,
        num_classes=num_classes,
        strides=strides,
        center_radius=center_radius,
        use_scale_ranges=use_scale_ranges,
        tiny_object_max_side_factor=tiny_object_max_side_factor,
        tiny_assign_expand_stride=tiny_assign_expand_stride,
    )

    use_softmax_bg = bool(target_pack["use_softmax_bg"])
    bg_index = int(target_pack["bg_index"])

    total_cls = cls_levels[0].new_tensor(0.0)
    total_reg_iou = cls_levels[0].new_tensor(0.0)
    total_reg_l1 = cls_levels[0].new_tensor(0.0)
    total_ctr = cls_levels[0].new_tensor(0.0)
    total_pos = cls_levels[0].new_tensor(0.0)

    for lvl, (cls_t, reg_t) in enumerate(zip(cls_levels, reg_levels)):
        bsz, c, h, w = cls_t.shape
        points = target_pack["points"][lvl]  # (H*W,2)

        cls_logits = cls_t.permute(0, 2, 3, 1).reshape(-1, c)
        reg_pred = reg_t.permute(0, 2, 3, 1).reshape(-1, 4)

        pos_mask = target_pack["pos_masks"][lvl].reshape(-1)
        reg_tgt = target_pack["reg_targets"][lvl].reshape(-1, 4)
        ctr_tgt = target_pack["ctr_targets"][lvl].reshape(-1)

        # Classification loss.
        if use_softmax_bg:
            cls_tgt = target_pack["cls_targets"][lvl].reshape(-1)
            cls_loss = focal_softmax_loss(
                logits=cls_logits,
                target_idx=cls_tgt,
                bg_index=bg_index,
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction="sum",
            )
        else:
            cls_tgt = target_pack["cls_targets"][lvl].reshape(-1, int(num_classes))
            cls_loss = focal_sigmoid_loss(
                logits=cls_logits,
                targets=cls_tgt,
                alpha=focal_alpha,
                gamma=focal_gamma,
                negative_weight=negative_focal_weight,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                reduction="sum",
            )
        total_cls = total_cls + cls_loss

        # Regression and centerness only on positives.
        n_pos = pos_mask.sum().float()
        total_pos = total_pos + n_pos

        if pos_mask.any():
            points_rep = points.unsqueeze(0).expand(bsz, -1, -1).reshape(-1, 2)

            reg_pos = reg_pred[pos_mask]
            tgt_pos = reg_tgt[pos_mask]
            pts_pos = points_rep[pos_mask]

            pred_boxes = tlbr_to_xyxy(pts_pos, reg_pos)
            tgt_boxes = tlbr_to_xyxy(pts_pos, tgt_pos)
            total_reg_iou = total_reg_iou + giou_loss(pred_boxes, tgt_boxes, reduction="sum")
            total_reg_l1 = total_reg_l1 + normalized_box_l1_loss(
                pred_boxes,
                tgt_boxes,
                beta=float(reg_smooth_l1_beta),
                reduction="sum",
            )

            if ctr_levels is not None:
                ctr_logits = ctr_levels[lvl].permute(0, 2, 3, 1).reshape(-1)
                ctr_pred = ctr_logits[pos_mask]
                ctr_gt = ctr_tgt[pos_mask]
                total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                    ctr_pred,
                    ctr_gt,
                    reduction="sum",
                )

    normalizer = torch.clamp(total_pos, min=1.0)

    loss_cls = total_cls / normalizer
    loss_reg_iou = total_reg_iou / normalizer
    loss_reg_l1 = total_reg_l1 / normalizer
    loss_reg = float(lambda_reg_iou) * loss_reg_iou + float(lambda_reg_l1) * loss_reg_l1
    loss_ctr = total_ctr / normalizer if ctr_levels is not None else total_cls.new_tensor(0.0)

    total = (
        float(lambda_cls) * loss_cls
        + float(lambda_reg) * loss_reg
        + float(lambda_ctr) * loss_ctr
    )

    return {
        "loss": total,
        "loss_cls": loss_cls.detach(),
        "loss_reg": loss_reg.detach(),
        "loss_ctr": loss_ctr.detach(),
        "num_pos": total_pos.detach(),
    }


class DetectionLoss(nn.Module):
    """
    nn.Module wrapper for anchor-free detection loss.

    Usage:
        criterion = DetectionLoss(num_classes=5)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss"]
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        strides: Optional[Sequence[int]] = None,
        lambda_cls: float = LAMBDA_CLS,
        lambda_reg: float = LAMBDA_REG,
        lambda_ctr: float = LAMBDA_CTR,
        lambda_reg_iou: float = LAMBDA_REG_IOU,
        lambda_reg_l1: float = LAMBDA_REG_L1,
        focal_alpha: float = FOCAL_ALPHA,
        focal_gamma: float = FOCAL_GAMMA,
        negative_focal_weight: float = NEGATIVE_FOCAL_WEIGHT,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = LABEL_SMOOTHING,
        reg_smooth_l1_beta: float = REG_SMOOTH_L1_BETA,
        center_radius: float = DEFAULT_CENTER_RADIUS,
        use_scale_ranges: bool = True,
        tiny_object_max_side_factor: float = TINY_OBJECT_MAX_SIDE_FACTOR,
        tiny_assign_expand_stride: float = TINY_ASSIGN_EXPAND_STRIDE,
    ):
        """Store loss hyperparameters and optional class weights as buffers."""

        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = list(strides) if strides is not None else list(STRIDES)
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.lambda_ctr = float(lambda_ctr)
        self.lambda_reg_iou = float(lambda_reg_iou)
        self.lambda_reg_l1 = float(lambda_reg_l1)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.negative_focal_weight = float(negative_focal_weight)
        if class_weights is None:
            self.register_buffer("class_weights", torch.empty(0), persistent=False)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32), persistent=False)
        self.label_smoothing = float(label_smoothing)
        self.reg_smooth_l1_beta = float(reg_smooth_l1_beta)
        self.center_radius = float(center_radius)
        self.use_scale_ranges = bool(use_scale_ranges)
        self.tiny_object_max_side_factor = float(tiny_object_max_side_factor)
        self.tiny_assign_expand_stride = float(tiny_assign_expand_stride)

    def forward(self, outputs: Dict[str, Any], targets: Any) -> Dict[str, torch.Tensor]:
        """Compute the full multi-branch detection loss for one batch."""

        return compute_loss(
            outputs=outputs,
            targets=targets,
            num_classes=self.num_classes,
            strides=self.strides,
            lambda_cls=self.lambda_cls,
            lambda_reg=self.lambda_reg,
            lambda_ctr=self.lambda_ctr,
            lambda_reg_iou=self.lambda_reg_iou,
            lambda_reg_l1=self.lambda_reg_l1,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            negative_focal_weight=self.negative_focal_weight,
            class_weights=self.class_weights if self.class_weights.numel() > 0 else None,
            label_smoothing=self.label_smoothing,
            reg_smooth_l1_beta=self.reg_smooth_l1_beta,
            center_radius=self.center_radius,
            use_scale_ranges=self.use_scale_ranges,
            tiny_object_max_side_factor=self.tiny_object_max_side_factor,
            tiny_assign_expand_stride=self.tiny_assign_expand_stride,
        )
