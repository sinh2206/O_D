from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.ops as ops

from .config import ANCHOR_SIZES, CONF_THRESH, NMS_IOU_THRESH, NUM_ANCHORS, STRIDE
from .process import LetterboxMeta


def decode_predictions(
    pred: torch.Tensor,
    stride: int,
    img_size: int,
    anchor_sizes: Sequence[Tuple[float, float]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, gh, gw, a, _ = pred.shape
    if a != NUM_ANCHORS:
        raise ValueError(f"Prediction anchor dim mismatch: got {a}, expected {NUM_ANCHORS}")

    device = pred.device
    anchors = torch.as_tensor(anchor_sizes, dtype=torch.float32, device=device)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(gh, device=device, dtype=torch.float32),
        torch.arange(gw, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = grid_x.view(1, gh, gw, 1).expand(b, gh, gw, a)
    grid_y = grid_y.view(1, gh, gw, 1).expand(b, gh, gw, a)

    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = torch.sigmoid(pred[..., 4])
    cls_prob = F.softmax(pred[..., 5:], dim=-1)

    cx = (torch.sigmoid(tx) + grid_x) * float(stride)
    cy = (torch.sigmoid(ty) + grid_y) * float(stride)

    aw = anchors[:, 0].view(1, 1, 1, a)
    ah = anchors[:, 1].view(1, 1, 1, a)
    bw = torch.exp(torch.clamp(tw, min=-10.0, max=10.0)) * aw
    bh = torch.exp(torch.clamp(th, min=-10.0, max=10.0)) * ah

    x1 = (cx - bw * 0.5).clamp(0.0, float(img_size))
    y1 = (cy - bh * 0.5).clamp(0.0, float(img_size))
    x2 = (cx + bw * 0.5).clamp(0.0, float(img_size))
    y2 = (cy + bh * 0.5).clamp(0.0, float(img_size))
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(b, -1, 4)

    cls_score, cls_idx = torch.max(cls_prob, dim=-1)
    scores = (obj * cls_score).view(b, -1)
    labels = cls_idx.view(b, -1)
    return boxes, scores, labels


def _map_boxes_back_to_original(boxes: torch.Tensor, meta: LetterboxMeta) -> torch.Tensor:
    out = boxes.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - float(meta.dx)) / float(meta.scale)
    out[:, [1, 3]] = (out[:, [1, 3]] - float(meta.dy)) / float(meta.scale)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(meta.orig_w))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(meta.orig_h))
    return out


def postprocess_batch(
    outputs: torch.Tensor,
    image_ids: Sequence[str],
    metas: Sequence[LetterboxMeta],
    class_names: Sequence[str],
    conf_thresh: float = CONF_THRESH,
    nms_thresh: float = NMS_IOU_THRESH,
    img_size: int = 320,
    stride: int = STRIDE,
    anchor_sizes: Sequence[Tuple[float, float]] = ANCHOR_SIZES,
) -> List[Dict[str, object]]:
    boxes, scores, labels = decode_predictions(
        pred=outputs,
        stride=stride,
        img_size=img_size,
        anchor_sizes=anchor_sizes,
    )

    results: List[Dict[str, object]] = []
    batch_size = boxes.shape[0]
    for i in range(batch_size):
        b = boxes[i]
        s = scores[i]
        c = labels[i]

        keep = s >= float(conf_thresh)
        b = b[keep]
        s = s[keep]
        c = c[keep]

        if b.numel() == 0:
            results.append({"image_id": image_ids[i], "boxes": []})
            continue

        nms_keep = ops.batched_nms(b, s, c, float(nms_thresh))
        b = b[nms_keep]
        s = s[nms_keep]
        c = c[nms_keep]

        b = _map_boxes_back_to_original(b, metas[i])
        out_boxes: List[Dict[str, object]] = []
        for j in range(b.shape[0]):
            cls_id = int(c[j].item())
            cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            conf = float(s[j].item())
            out_boxes.append(
                {
                    "bbox": [float(v) for v in b[j].tolist()],
                    "class": cls_name,
                    "confidence": conf,
                    "score": conf,
                }
            )

        results.append({"image_id": image_ids[i], "boxes": out_boxes})
    return results
