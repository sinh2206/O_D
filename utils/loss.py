from __future__ import annotations

"""
Loss utilities for anchor-free detection (FCOS/YOLOX-simplified style).

Total loss:
    L = lambda_cls * L_cls + lambda_reg * L_reg + lambda_ctr * L_ctr

Where:
- L_cls: focal loss for classification.
- L_reg: GIoU loss for box regression (decoded from t,l,b,r distances).
- L_ctr: BCE-with-logits for centerness (optional if branch exists).

Target assignment:
- Center sampling per FPN level.
- A location is positive if it is inside GT box, inside center region, and
  inside that level's scale range.
- If multiple GT boxes match one location, pick the GT with smallest area.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import (
        FOCAL_ALPHA,
        FOCAL_GAMMA,
        LAMBDA_CLS,
        LAMBDA_CTR,
        LAMBDA_REG,
        LABEL_SMOOTHING,
        NUM_CLASSES,
        STRIDES,
    )
except Exception:
    # Safe fallbacks so this module still works if config.py is not available.
    NUM_CLASSES = 5
    STRIDES = [16, 32]
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25
    LAMBDA_CLS = 1.0
    LAMBDA_REG = 1.0
    LAMBDA_CTR = 0.5
    LABEL_SMOOTHING = 0.05

EPS = 1e-8
DEFAULT_CENTER_RADIUS = 1.5


def _meshgrid_xy(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if class_weights is not None:
        cw = class_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        loss = loss * cw

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

    For strides [16, 32] -> [(0, 64), (64, inf)]
    """
    ranges: List[Tuple[float, float]] = []
    for i, s in enumerate(strides):
        if i == 0:
            lower = 0.0
        else:
            lower = float(4.0 * strides[i - 1])

        if i == len(strides) - 1:
            upper = 1e8
        else:
            upper = float(4.0 * s)

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

    inside_box = (ltrb.min(dim=-1).values > 0.0)

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
        in_range = (max_dist >= lo) & (max_dist <= hi)
        candidate = candidate & in_range

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
    focal_alpha: float = FOCAL_ALPHA,
    focal_gamma: float = FOCAL_GAMMA,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = LABEL_SMOOTHING,
    center_radius: float = DEFAULT_CENTER_RADIUS,
    use_scale_ranges: bool = True,
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
    )

    use_softmax_bg = bool(target_pack["use_softmax_bg"])
    bg_index = int(target_pack["bg_index"])

    total_cls = cls_levels[0].new_tensor(0.0)
    total_reg = cls_levels[0].new_tensor(0.0)
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
            total_reg = total_reg + giou_loss(pred_boxes, tgt_boxes, reduction="sum")

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
    loss_reg = total_reg / normalizer
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
        focal_alpha: float = FOCAL_ALPHA,
        focal_gamma: float = FOCAL_GAMMA,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = LABEL_SMOOTHING,
        center_radius: float = DEFAULT_CENTER_RADIUS,
        use_scale_ranges: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = list(strides) if strides is not None else list(STRIDES)
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.lambda_ctr = float(lambda_ctr)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        if class_weights is None:
            self.register_buffer("class_weights", torch.empty(0), persistent=False)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32), persistent=False)
        self.label_smoothing = float(label_smoothing)
        self.center_radius = float(center_radius)
        self.use_scale_ranges = bool(use_scale_ranges)

    def forward(self, outputs: Dict[str, Any], targets: Any) -> Dict[str, torch.Tensor]:
        return compute_loss(
            outputs=outputs,
            targets=targets,
            num_classes=self.num_classes,
            strides=self.strides,
            lambda_cls=self.lambda_cls,
            lambda_reg=self.lambda_reg,
            lambda_ctr=self.lambda_ctr,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            class_weights=self.class_weights if self.class_weights.numel() > 0 else None,
            label_smoothing=self.label_smoothing,
            center_radius=self.center_radius,
            use_scale_ranges=self.use_scale_ranges,
        )
