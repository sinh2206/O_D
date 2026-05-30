from __future__ import annotations

"""
Forecast head and decoding utilities for anchor-free detection.

Design (per FPN level):
- Input feature: (B, 128, H, W)
- 3x [Conv3x3 -> BN -> LeakyReLU(0.1)] shared stem
- Classification branch: Conv1x1 -> (B, C, H, W)
- Regression branch: Conv1x1 -> (B, 4, H, W), decoded as (t, l, b, r)

Confidence (no objectness branch):
- class_prob = sigmoid(cls_logits)
- confidence = max(class_prob)

Box decode at location (i, j):
- cx = (j + 0.5) * stride
- cy = (i + 0.5) * stride
- x1 = cx - l, y1 = cy - t, x2 = cx + r, y2 = cy + b
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CONF_THRESH, FPN_CHANNELS, IMG_SIZE, NMS_IOU_THRESH, NUM_CLASSES


class ConvBNLeaky(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AnchorFreeForecastHead(nn.Module):
    """Decoupled prediction head for one FPN level."""

    def __init__(self, in_ch: int = FPN_CHANNELS, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNLeaky(in_ch, in_ch, k=3, s=1, p=1),
            ConvBNLeaky(in_ch, in_ch, k=3, s=1, p=1),
            ConvBNLeaky(in_ch, in_ch, k=3, s=1, p=1),
        )

        self.cls_pred = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
        self.reg_pred = nn.Conv2d(in_ch, 4, kernel_size=1, bias=True)

        self._init_params()

    def _init_params(self) -> None:
        # Initialize low class confidence at start.
        nn.init.normal_(self.cls_pred.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_pred.bias, -4.0)

        nn.init.normal_(self.reg_pred.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 1.0)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(feat)
        cls_logits = self.cls_pred(x)
        reg_tlbr = F.relu(self.reg_pred(x))
        return {
            "cls_logits": cls_logits,
            "reg_preds": reg_tlbr,
        }


class MultiScaleForecast(nn.Module):
    """Two independent heads for stride16 and stride32 feature maps."""

    def __init__(self, in_ch: int = FPN_CHANNELS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head_s16 = AnchorFreeForecastHead(in_ch=in_ch, num_classes=num_classes)
        self.head_s32 = AnchorFreeForecastHead(in_ch=in_ch, num_classes=num_classes)

    def forward(self, p3_out: torch.Tensor, p4_out: torch.Tensor) -> Dict[str, Any]:
        out16 = self.head_s16(p3_out)
        out32 = self.head_s32(p4_out)
        return {
            "cls_logits": [out16["cls_logits"], out32["cls_logits"]],
            "reg_preds": [out16["reg_preds"], out32["reg_preds"]],
            "strides": [16, 32],
        }


def _build_grid(h: int, w: int, stride: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gx, gy


def decode_level(
    cls_logits: torch.Tensor,
    reg_preds: torch.Tensor,
    stride: float,
    conf_thresh: float = CONF_THRESH,
    img_size: Optional[int] = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode one level output to boxes + scores + class ids for a single image.

    Inputs:
    - cls_logits: (C, H, W)
    - reg_preds:  (4, H, W), order (t, l, b, r)
    """
    if cls_logits.dim() != 3 or reg_preds.dim() != 3:
        raise ValueError("decode_level expects cls_logits (C,H,W) and reg_preds (4,H,W).")

    c, h, w = cls_logits.shape
    if reg_preds.shape[0] != 4:
        raise ValueError("reg_preds first dim must be 4 for (t,l,b,r).")

    cls_prob = torch.sigmoid(cls_logits)
    best_score, best_cls = cls_prob.max(dim=0)  # (H, W)

    mask = best_score >= conf_thresh
    if not mask.any():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    ys, xs = torch.where(mask)
    scores = best_score[ys, xs]
    cls_ids = best_cls[ys, xs]

    gx, gy = _build_grid(h, w, stride=float(stride), device=cls_logits.device)
    cx = gx[ys, xs]
    cy = gy[ys, xs]

    t = reg_preds[0, ys, xs]
    l = reg_preds[1, ys, xs]
    b = reg_preds[2, ys, xs]
    r = reg_preds[3, ys, xs]

    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b

    if img_size is not None:
        x1 = x1.clamp(0, float(img_size))
        y1 = y1.clamp(0, float(img_size))
        x2 = x2.clamp(0, float(img_size))
        y2 = y2.clamp(0, float(img_size))

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return (
        boxes.detach().cpu().numpy().astype(np.float32),
        scores.detach().cpu().numpy().astype(np.float32),
        cls_ids.detach().cpu().numpy().astype(np.int64),
    )


def decode_multilevel(
    outputs: Dict[str, Any],
    conf_thresh: float = CONF_THRESH,
    img_size: Optional[int] = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode model outputs from `utils/model.py` or `MultiScaleForecast`.

    Expected keys:
    - outputs['cls_logits']: List[(B,C,H,W)] or List[(C,H,W)]
    - outputs['reg_preds']:  List[(B,4,H,W)] or List[(4,H,W)]
    - outputs['strides']:    List[int]

    Returns concatenated (boxes, scores, cls_ids) for batch index 0.
    """
    cls_levels = outputs["cls_logits"]
    reg_levels = outputs["reg_preds"]
    strides = outputs.get("strides", [16, 32])

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_cls: List[np.ndarray] = []

    for cls_t, reg_t, stride in zip(cls_levels, reg_levels, strides):
        if cls_t.dim() == 4:
            cls_i = cls_t[0]
        else:
            cls_i = cls_t

        if reg_t.dim() == 4:
            reg_i = reg_t[0]
        else:
            reg_i = reg_t

        boxes, scores, cls_ids = decode_level(
            cls_logits=cls_i,
            reg_preds=reg_i,
            stride=float(stride),
            conf_thresh=conf_thresh,
            img_size=img_size,
        )
        if boxes.shape[0] == 0:
            continue

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_cls.append(cls_ids)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.concatenate(all_boxes, axis=0),
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_cls, axis=0),
    )


def nms_per_class_numpy(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou_thresh: float = NMS_IOU_THRESH):
    """Simple per-class NMS (numpy) to keep high-confidence boxes (e.g. 0.95)."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    keep_global: List[int] = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        if idx.size == 0:
            continue
        b = boxes[idx]
        s = scores[idx]

        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = s.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep_global.append(int(idx[i]))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < float(iou_thresh)]

    if not keep_global:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(sorted(set(keep_global)), dtype=np.int64)
