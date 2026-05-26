from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import (
    ANCHOR_SIZES,
    IOU_IGNORE_THRESH,
    IOU_POS_THRESH,
    LAMBDA_BOX,
    LAMBDA_CLS,
    LAMBDA_NOOBJ,
    LAMBDA_OBJ,
    NUM_ANCHORS,
    NUM_CLASSES,
)


def _wh_iou(one_wh: torch.Tensor, many_wh: torch.Tensor) -> torch.Tensor:
    one = one_wh.view(1, 2)
    many = many_wh.view(-1, 2)
    inter = torch.minimum(one, many).prod(dim=1)
    union = one.prod(dim=1) + many.prod(dim=1) - inter
    return inter / torch.clamp(union, min=1e-9)


def _aligned_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((0,), device=boxes1.device, dtype=torch.float32)
    if boxes1.shape != boxes2.shape:
        raise ValueError(f"Aligned IoU expects same shape, got {boxes1.shape} vs {boxes2.shape}")

    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / torch.clamp(union, min=1e-9)


def build_targets(
    targets_list: List[Dict[str, Any]],
    stride: int,
    img_size: int,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> Tuple[torch.Tensor, ...]:
    batch_size = len(targets_list)
    grid_h = int(img_size // stride)
    grid_w = int(img_size // stride)

    anchors = torch.as_tensor(ANCHOR_SIZES, dtype=torch.float32, device=device)  # (A,2) in pixels

    obj_mask = torch.zeros((batch_size, grid_h, grid_w, NUM_ANCHORS), dtype=torch.bool, device=device)
    noobj_mask = torch.ones((batch_size, grid_h, grid_w, NUM_ANCHORS), dtype=torch.bool, device=device)
    tx = torch.zeros((batch_size, grid_h, grid_w, NUM_ANCHORS), dtype=torch.float32, device=device)
    ty = torch.zeros((batch_size, grid_h, grid_w, NUM_ANCHORS), dtype=torch.float32, device=device)
    tw = torch.zeros((batch_size, grid_h, grid_w, NUM_ANCHORS), dtype=torch.float32, device=device)
    th = torch.zeros((batch_size, grid_h, grid_w, NUM_ANCHORS), dtype=torch.float32, device=device)
    tcls = torch.zeros((batch_size, grid_h, grid_w, NUM_ANCHORS), dtype=torch.long, device=device)
    tbox_xyxy = torch.zeros((batch_size, grid_h, grid_w, NUM_ANCHORS, 4), dtype=torch.float32, device=device)

    for b in range(batch_size):
        gt_boxes = targets_list[b].get("boxes")
        gt_labels = targets_list[b].get("labels")
        if gt_boxes is None or gt_labels is None:
            continue
        if gt_boxes.numel() == 0:
            continue

        gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
        gt_labels = gt_labels.to(device=device, dtype=torch.long).clamp(min=0, max=max(int(num_classes) - 1, 0))

        gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
        gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
        gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
        gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)

        gi = torch.clamp((gx / float(stride)).long(), min=0, max=grid_w - 1)
        gj = torch.clamp((gy / float(stride)).long(), min=0, max=grid_h - 1)

        gt_wh = torch.stack([gw, gh], dim=1)
        for i in range(gt_boxes.shape[0]):
            wh_iou = _wh_iou(gt_wh[i], anchors)  # (A,)
            best_anchor = int(torch.argmax(wh_iou).item())

            pos_ids = torch.nonzero(wh_iou >= float(IOU_POS_THRESH), as_tuple=False).view(-1)
            if pos_ids.numel() == 0:
                pos_ids = torch.tensor([best_anchor], device=device, dtype=torch.long)
            elif best_anchor not in pos_ids.tolist():
                pos_ids = torch.cat([pos_ids, torch.tensor([best_anchor], device=device, dtype=torch.long)], dim=0)

            ignore_ids = torch.nonzero(wh_iou >= float(IOU_IGNORE_THRESH), as_tuple=False).view(-1)
            if ignore_ids.numel() > 0:
                noobj_mask[b, gj[i], gi[i], ignore_ids] = False

            for aid_t in pos_ids:
                aid = int(aid_t.item())
                obj_mask[b, gj[i], gi[i], aid] = True
                noobj_mask[b, gj[i], gi[i], aid] = False

                tx[b, gj[i], gi[i], aid] = gx[i] / float(stride) - gi[i].float()
                ty[b, gj[i], gi[i], aid] = gy[i] / float(stride) - gj[i].float()
                tw[b, gj[i], gi[i], aid] = torch.log(gw[i] / anchors[aid, 0] + 1e-9)
                th[b, gj[i], gi[i], aid] = torch.log(gh[i] / anchors[aid, 1] + 1e-9)
                tcls[b, gj[i], gi[i], aid] = gt_labels[i]
                tbox_xyxy[b, gj[i], gi[i], aid] = gt_boxes[i]

    return obj_mask, noobj_mask, tx, ty, tw, th, tcls, tbox_xyxy


def _decode_xyxy_from_raw(
    predictions: torch.Tensor,
    stride: int,
    anchors: torch.Tensor,
) -> torch.Tensor:
    b, gh, gw, a, _ = predictions.shape
    device = predictions.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(gh, device=device, dtype=torch.float32),
        torch.arange(gw, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = grid_x.view(1, gh, gw, 1).expand(b, gh, gw, a)
    grid_y = grid_y.view(1, gh, gw, 1).expand(b, gh, gw, a)

    tx_raw = predictions[..., 0]
    ty_raw = predictions[..., 1]
    tw_raw = predictions[..., 2]
    th_raw = predictions[..., 3]

    cx = (torch.sigmoid(tx_raw) + grid_x) * float(stride)
    cy = (torch.sigmoid(ty_raw) + grid_y) * float(stride)

    aw = anchors[:, 0].view(1, 1, 1, a)
    ah = anchors[:, 1].view(1, 1, 1, a)
    bw = torch.exp(torch.clamp(tw_raw, min=-10.0, max=10.0)) * aw
    bh = torch.exp(torch.clamp(th_raw, min=-10.0, max=10.0)) * ah

    x1 = cx - bw * 0.5
    y1 = cy - bh * 0.5
    x2 = cx + bw * 0.5
    y2 = cy + bh * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_loss(
    predictions: torch.Tensor,
    targets_list: List[Dict[str, Any]],
    device: torch.device,
    stride: int,
    img_size: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    predictions = predictions.to(device)
    num_classes = int(max(1, predictions.shape[-1] - 5))
    obj_mask, noobj_mask, tx, ty, tw, th, tcls, tbox_xyxy = build_targets(
        targets_list=targets_list,
        stride=stride,
        img_size=img_size,
        device=device,
        num_classes=num_classes,
    )

    pred_tx = predictions[..., 0]
    pred_ty = predictions[..., 1]
    pred_tw = predictions[..., 2]
    pred_th = predictions[..., 3]
    pred_obj = predictions[..., 4]
    pred_cls = predictions[..., 5:]

    pos_count = int(obj_mask.sum().item())
    neg_count = int(noobj_mask.sum().item())
    pos_norm = max(pos_count, 1)
    neg_norm = max(neg_count, 1)

    if pos_count > 0:
        loss_x = F.smooth_l1_loss(torch.sigmoid(pred_tx[obj_mask]), tx[obj_mask], reduction="sum") / pos_norm
        loss_y = F.smooth_l1_loss(torch.sigmoid(pred_ty[obj_mask]), ty[obj_mask], reduction="sum") / pos_norm
        loss_w = F.smooth_l1_loss(pred_tw[obj_mask], tw[obj_mask], reduction="sum") / pos_norm
        loss_h = F.smooth_l1_loss(pred_th[obj_mask], th[obj_mask], reduction="sum") / pos_norm
        regression_loss = loss_x + loss_y + loss_w + loss_h
    else:
        regression_loss = torch.zeros((), device=device)

    obj_target = torch.zeros_like(pred_obj)
    obj_target[obj_mask] = 1.0

    if pos_count > 0:
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj[obj_mask], obj_target[obj_mask], reduction="sum") / pos_norm
    else:
        obj_loss = torch.zeros((), device=device)

    if neg_count > 0:
        noobj_loss = F.binary_cross_entropy_with_logits(
            pred_obj[noobj_mask],
            obj_target[noobj_mask],
            reduction="sum",
        ) / neg_norm
    else:
        noobj_loss = torch.zeros((), device=device)

    if pos_count > 0:
        classification_loss = F.cross_entropy(pred_cls[obj_mask], tcls[obj_mask], reduction="sum") / pos_norm
    else:
        classification_loss = torch.zeros((), device=device)

    anchors = torch.as_tensor(ANCHOR_SIZES, dtype=torch.float32, device=device)
    pred_boxes = _decode_xyxy_from_raw(predictions, stride=stride, anchors=anchors)
    if pos_count > 0:
        ious = _aligned_iou_xyxy(pred_boxes[obj_mask], tbox_xyxy[obj_mask])
        mean_iou = float(ious.mean().item()) if ious.numel() > 0 else 0.0
    else:
        mean_iou = 0.0

    total_loss = (
        float(LAMBDA_BOX) * regression_loss
        + float(LAMBDA_OBJ) * obj_loss
        + float(LAMBDA_NOOBJ) * noobj_loss
        + float(LAMBDA_CLS) * classification_loss
    )

    stats = {
        "loss": float(total_loss.detach().item()),
        "loss_obj": float(obj_loss.detach().item()),
        "loss_noobj": float(noobj_loss.detach().item()),
        "loss_box": float(regression_loss.detach().item()),
        "loss_cls": float(classification_loss.detach().item()),
        "pos": float(pos_count),
        "neg": float(neg_count),
        "mean_iou_pos": mean_iou,
    }
    return total_loss, stats
