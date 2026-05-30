from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ANCHORS,
    ANCHOR_MASKS,
    CONF_THRESH,
    IOU_IGNORE_THRESH,
    IOU_POS_THRESH,
    LAMBDA_BOX,
    LAMBDA_CLS,
    LAMBDA_NOOBJ,
    LAMBDA_OBJ,
    LABEL_SMOOTHING,
    NUM_CLASSES,
    STRIDES,
)

EPS = 1e-9


def _prepare_targets(targets: Any, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    if isinstance(targets, list):
        out: List[Dict[str, torch.Tensor]] = []
        for t in targets:
            boxes = t.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
            labels = t.get("labels", torch.zeros((0,), dtype=torch.long))
            out.append(
                {
                    "boxes": boxes.to(device=device, dtype=torch.float32),
                    "labels": labels.to(device=device, dtype=torch.long),
                }
            )
        return out

    if isinstance(targets, dict):
        boxes = targets.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).to(device=device, dtype=torch.float32)
        labels = targets.get("labels", torch.zeros((0,), dtype=torch.long)).to(device=device, dtype=torch.long)
        return [{"boxes": boxes, "labels": labels}]

    raise TypeError(f"Unsupported targets type: {type(targets)!r}")


def _wh_iou(gt_wh: torch.Tensor, anchors_wh: torch.Tensor) -> torch.Tensor:
    gt = gt_wh.view(-1, 2)[0]
    inter = torch.minimum(gt[0], anchors_wh[:, 0]) * torch.minimum(gt[1], anchors_wh[:, 1])
    union = (gt[0] * gt[1]) + (anchors_wh[:, 0] * anchors_wh[:, 1]) - inter + EPS
    return inter / union


def _make_grid(height: int, width: int, stride: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * float(stride)
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * float(stride)
    try:
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        gy, gx = torch.meshgrid(ys, xs)
    return gx, gy


def build_targets(
    targets_list: List[Dict[str, torch.Tensor]],
    stride: int,
    img_size: int,
    device: torch.device,
    anchors: Sequence[Sequence[Tuple[int, int]]] = ANCHORS,
    anchor_mask: Optional[Sequence[int]] = None,
    num_classes: int = NUM_CLASSES,
    iou_ignore_thresh: float = IOU_IGNORE_THRESH,
    label_smoothing: float = LABEL_SMOOTHING,
) -> Dict[str, torch.Tensor]:
    if anchor_mask is None:
        anchor_mask = list(range(len(anchors)))

    anchor_mask = list(anchor_mask)
    flat_anchors = torch.as_tensor([a for level in anchors for a in level], dtype=torch.float32, device=device)
    anchor_wh = torch.as_tensor([flat_anchors[i].tolist() for i in anchor_mask], dtype=torch.float32, device=device)
    grid = max(1, int(round(float(img_size) / float(stride))))
    batch_size = len(targets_list)

    obj_mask = torch.zeros((batch_size, len(anchor_mask), grid, grid), dtype=torch.bool, device=device)
    noobj_mask = torch.ones((batch_size, len(anchor_mask), grid, grid), dtype=torch.bool, device=device)
    tx = torch.zeros((batch_size, len(anchor_mask), grid, grid), dtype=torch.float32, device=device)
    ty = torch.zeros_like(tx)
    tw = torch.zeros_like(tx)
    th = torch.zeros_like(tx)
    tcls = torch.zeros((batch_size, len(anchor_mask), grid, grid, num_classes), dtype=torch.float32, device=device)
    tbox = torch.zeros((batch_size, len(anchor_mask), grid, grid, 4), dtype=torch.float32, device=device)

    for bi, target in enumerate(targets_list):
        boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32, device=device))
        labels = target.get("labels", torch.zeros((0,), dtype=torch.long, device=device))
        if boxes.numel() == 0:
            continue

        boxes = boxes.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        for box, cls_idx in zip(boxes, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw = max(1e-6, x2 - x1)
            bh = max(1e-6, y2 - y1)
            gx = 0.5 * (x1 + x2)
            gy = 0.5 * (y1 + y2)
            gi = int(gx / float(stride))
            gj = int(gy / float(stride))
            if gi < 0 or gj < 0 or gi >= grid or gj >= grid:
                continue

            gt_wh = torch.tensor([[bw, bh]], dtype=torch.float32, device=device)
            ious = _wh_iou(gt_wh, flat_anchors)
            best_global_anchor = int(torch.argmax(ious).item())

            # Ignore anchors that overlap this GT strongly, even if they are not positive.
            for local_idx, global_idx in enumerate(anchor_mask):
                anc_w, anc_h = flat_anchors[global_idx].tolist()
                anchor_iou = float(_wh_iou(gt_wh, torch.tensor([[float(anc_w), float(anc_h)]], device=device)).item())
                if anchor_iou >= float(iou_ignore_thresh):
                    noobj_mask[bi, local_idx, gj, gi] = False

            if best_global_anchor not in anchor_mask:
                continue

            local_idx = anchor_mask.index(best_global_anchor)
            obj_mask[bi, local_idx, gj, gi] = True
            noobj_mask[bi, local_idx, gj, gi] = False

            tx[bi, local_idx, gj, gi] = gx / float(stride) - gi
            ty[bi, local_idx, gj, gi] = gy / float(stride) - gj
            anc_w, anc_h = flat_anchors[best_global_anchor].tolist()
            tw[bi, local_idx, gj, gi] = torch.log(torch.tensor(bw / max(float(anc_w), EPS), device=device))
            th[bi, local_idx, gj, gi] = torch.log(torch.tensor(bh / max(float(anc_h), EPS), device=device))
            tbox[bi, local_idx, gj, gi] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
            tcls[bi, local_idx, gj, gi, int(cls_idx.item())] = 1.0

    if label_smoothing > 0.0:
        tcls = tcls * (1.0 - float(label_smoothing)) + float(label_smoothing) / max(num_classes, 1)

    return {
        "obj_mask": obj_mask,
        "noobj_mask": noobj_mask,
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "tcls": tcls,
        "tbox": tbox,
        "anchor_wh": anchor_wh,
        "stride": torch.tensor(float(stride), device=device),
        "grid": torch.tensor(grid, device=device),
    }


def _decode_boxes_from_raw(
    pred: torch.Tensor,
    stride: int,
    anchor_wh: torch.Tensor,
) -> torch.Tensor:
    bsz, num_anchors, h, w, _ = pred.shape
    gx, gy = _make_grid(h, w, stride=stride, device=pred.device)
    gx = gx.view(1, 1, h, w)
    gy = gy.view(1, 1, h, w)

    tx = torch.sigmoid(pred[..., 0])
    ty = torch.sigmoid(pred[..., 1])
    tw = pred[..., 2]
    th = pred[..., 3]

    anc = anchor_wh.view(1, num_anchors, 1, 1, 2)
    px = (tx + gx.unsqueeze(-1)[..., 0]) * float(stride)
    py = (ty + gy.unsqueeze(-1)[..., 0]) * float(stride)
    pw = torch.exp(tw).clamp(max=1e4) * anc[..., 0]
    ph = torch.exp(th).clamp(max=1e4) * anc[..., 1]

    x1 = px - pw / 2.0
    y1 = py - ph / 2.0
    x2 = px + pw / 2.0
    y2 = py + ph / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.numel() == 0:
        return pred_boxes.new_tensor(0.0)

    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_t = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)
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
    return (1.0 - giou).mean()


def compute_loss(
    predictions: Sequence[torch.Tensor],
    targets: Any,
    device: torch.device,
    *,
    img_size: int = 320,
    strides: Sequence[int] = STRIDES,
    anchors: Sequence[Sequence[Tuple[int, int]]] = ANCHORS,
    anchor_masks: Sequence[Sequence[int]] = ANCHOR_MASKS,
    num_classes: int = NUM_CLASSES,
    class_weights: Optional[torch.Tensor] = None,
    lambda_obj: float = LAMBDA_OBJ,
    lambda_noobj: float = LAMBDA_NOOBJ,
    lambda_box: float = LAMBDA_BOX,
    lambda_cls: float = LAMBDA_CLS,
    label_smoothing: float = LABEL_SMOOTHING,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    targets_list = _prepare_targets(targets, device=device)
    if len(predictions) != len(strides):
        raise ValueError(f"Expected {len(strides)} prediction scales, got {len(predictions)}")

    total_loss = torch.tensor(0.0, device=device)
    loss_obj_total = torch.tensor(0.0, device=device)
    loss_noobj_total = torch.tensor(0.0, device=device)
    loss_box_total = torch.tensor(0.0, device=device)
    loss_cls_total = torch.tensor(0.0, device=device)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    for scale_idx, (pred, stride, anchor_mask) in enumerate(zip(predictions, strides, anchor_masks)):
        if pred.dim() != 4:
            raise ValueError("Each prediction tensor must be (B, A*(5+C), H, W)")

        bsz, ch, h, w = pred.shape
        expected = len(anchor_mask) * (5 + num_classes)
        if ch != expected:
            raise ValueError(f"Scale {scale_idx}: expected {expected} channels, got {ch}")

        pred = pred.view(bsz, len(anchor_mask), 5 + num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
        target_maps = build_targets(
            targets_list=targets_list,
            stride=int(stride),
            img_size=img_size,
            device=device,
            anchors=anchors,
            anchor_mask=anchor_mask,
            num_classes=num_classes,
            iou_ignore_thresh=IOU_IGNORE_THRESH,
            label_smoothing=label_smoothing,
        )

        obj_mask = target_maps["obj_mask"]
        noobj_mask = target_maps["noobj_mask"]
        tx = target_maps["tx"]
        ty = target_maps["ty"]
        tw = target_maps["tw"]
        th = target_maps["th"]
        tcls = target_maps["tcls"]
        tbox = target_maps["tbox"]
        anc = target_maps["anchor_wh"]

        obj_logits = pred[..., 4]
        cls_logits = pred[..., 5:]

        obj_target = obj_mask.float()
        noobj_target = torch.zeros_like(obj_target)
        obj_loss = bce(obj_logits[obj_mask], torch.ones_like(obj_logits[obj_mask])) if obj_mask.any() else obj_logits.new_tensor(0.0)
        noobj_loss = bce(obj_logits[noobj_mask], torch.zeros_like(obj_logits[noobj_mask])) if noobj_mask.any() else obj_logits.new_tensor(0.0)
        obj_loss = obj_loss.mean() if obj_loss.numel() > 0 else obj_logits.new_tensor(0.0)
        noobj_loss = noobj_loss.mean() if noobj_loss.numel() > 0 else obj_logits.new_tensor(0.0)

        if obj_mask.any():
            px = torch.sigmoid(pred[..., 0][obj_mask])
            py = torch.sigmoid(pred[..., 1][obj_mask])
            pw = pred[..., 2][obj_mask]
            ph = pred[..., 3][obj_mask]
            box_loss = (
                mse(px, tx[obj_mask]).mean()
                + mse(py, ty[obj_mask]).mean()
                + mse(pw, tw[obj_mask]).mean()
                + mse(ph, th[obj_mask]).mean()
            )

            cls_target = tcls[obj_mask]
            cls_pred = cls_logits[obj_mask]
            cls_loss = bce(cls_pred, cls_target)
            if class_weights is not None and cls_loss.numel() > 0:
                w = class_weights.to(device=device, dtype=cls_loss.dtype).view(1, -1)
                cls_loss = cls_loss * w
            cls_loss = cls_loss.mean() if cls_loss.numel() > 0 else cls_logits.new_tensor(0.0)
        else:
            box_loss = pred.new_tensor(0.0)
            cls_loss = pred.new_tensor(0.0)

        loss_scale = lambda_obj * obj_loss + lambda_noobj * noobj_loss + lambda_box * box_loss + lambda_cls * cls_loss

        loss_obj_total = loss_obj_total + obj_loss
        loss_noobj_total = loss_noobj_total + noobj_loss
        loss_box_total = loss_box_total + box_loss
        loss_cls_total = loss_cls_total + cls_loss
        total_loss = total_loss + loss_scale

    loss_dict = {
        "loss": total_loss,
        "loss_obj": loss_obj_total,
        "loss_noobj": loss_noobj_total,
        "loss_reg": loss_box_total,
        "loss_cls": loss_cls_total,
        "loss_ctr": torch.tensor(0.0, device=device),
    }
    return total_loss, loss_dict


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        strides: Sequence[int] = STRIDES,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = LABEL_SMOOTHING,
        center_radius: float = 1.5,
        use_scale_ranges: bool = True,
        anchors: Sequence[Sequence[Tuple[int, int]]] = ANCHORS,
        anchor_masks: Sequence[Sequence[int]] = ANCHOR_MASKS,
        lambda_obj: float = LAMBDA_OBJ,
        lambda_noobj: float = LAMBDA_NOOBJ,
        lambda_box: float = LAMBDA_BOX,
        lambda_cls: float = LAMBDA_CLS,
        img_size: int = 320,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = list(strides)
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)
        self.center_radius = float(center_radius)
        self.use_scale_ranges = bool(use_scale_ranges)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.lambda_obj = float(lambda_obj)
        self.lambda_noobj = float(lambda_noobj)
        self.lambda_box = float(lambda_box)
        self.lambda_cls = float(lambda_cls)
        self.img_size = int(img_size)

    def forward(self, predictions: Sequence[torch.Tensor], targets: Any) -> Dict[str, torch.Tensor]:
        total, loss_dict = compute_loss(
            predictions=predictions,
            targets=targets,
            device=predictions[0].device,
            img_size=self.img_size,
            strides=self.strides,
            anchors=self.anchors,
            anchor_masks=self.anchor_masks,
            num_classes=self.num_classes,
            class_weights=self.class_weights,
            lambda_obj=self.lambda_obj,
            lambda_noobj=self.lambda_noobj,
            lambda_box=self.lambda_box,
            lambda_cls=self.lambda_cls,
            label_smoothing=self.label_smoothing,
        )
        loss_dict["loss"] = total
        return loss_dict
