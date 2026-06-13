from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from utils.config import (
    CHAIR_SUPPRESS_WITH_PERSON_IOU,
    CHAIR_SUPPRESS_MAX_AREA_RATIO,
    CLASS_CONF_THRESH,
    CLASS_NAMES,
    CONF_THRESH,
    LOW_LIGHT_CLAHE_CLIP,
    LOW_LIGHT_GAMMA,
    LOW_LIGHT_MEAN_THRESH,
    IMG_SIZE,
    INFER_CENTER_COMBINE,
    MAX_OBJECTS_PER_IMAGE,
    MIN_BOX_SIZE,
    MIN_EXPORT_CONF,
    MEAN,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    STD,
)
from utils.model import AnchorFreeDetector
from utils.nms import LetterboxMeta, postprocess_batch

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_VAL_IMAGE_DIR = Path("public/val/images")
DEFAULT_VAL_ANNOTATION = Path("public/annotations/val.json")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_JSON = Path("val_predictions.json")


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image_bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix.lower() in VALID_EXTS else ".jpg"
    out_path = path if path.suffix.lower() in VALID_EXTS else path.with_suffix(ext)
    ok, enc = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    enc.tofile(str(out_path))
    return True


def enhance_low_light_bgr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) >= float(LOW_LIGHT_MEAN_THRESH):
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(LOW_LIGHT_CLAHE_CLIP), tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** float(LOW_LIGHT_GAMMA)) * 255.0 for i in range(256)], dtype=np.float32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(out, lut)


def letterbox_preprocess(image_bgr: np.ndarray, img_size: int) -> Tuple[torch.Tensor, LetterboxMeta]:
    image_bgr = enhance_low_light_bgr(image_bgr)
    h, w = image_bgr.shape[:2]
    scale = min(float(img_size) / max(w, 1), float(img_size) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    dx = (img_size - new_w) // 2
    dy = (img_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    meta = LetterboxMeta(scale=scale, dx=float(dx), dy=float(dy), orig_w=int(w), orig_h=int(h))
    return tensor, meta


def collect_images(image_dir: Path) -> List[Path]:
    imgs = [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return imgs


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _normalize_bbox_for_draw(bbox: Sequence[float], image_shape: Tuple[int, int, int]) -> Optional[List[int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        return None

    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))

    ix1 = max(0, min(w - 1, _round_half_up(x1)))
    iy1 = max(0, min(h - 1, _round_half_up(y1)))
    ix2 = max(0, min(w - 1, _round_half_up(x2)))
    iy2 = max(0, min(h - 1, _round_half_up(y2)))

    if ix2 <= ix1:
        if ix1 < w - 1:
            ix2 = ix1 + 1
        else:
            return None
    if iy2 <= iy1:
        if iy1 < h - 1:
            iy2 = iy1 + 1
        else:
            return None

    return [int(ix1), int(iy1), int(ix2), int(iy2)]


def _is_fully_inside(inner: Sequence[int], outer: Sequence[int]) -> bool:
    return bool(
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _suppress_same_class_contained_int(boxes: List[dict]) -> List[dict]:
    ordered = sorted(boxes, key=lambda b: float(b.get("confidence", 0.0)), reverse=True)
    kept: List[dict] = []

    for box in ordered:
        cls = str(box.get("class", ""))
        bbox = box.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bbox_i = [int(v) for v in bbox]

        drop = False
        replace_idx = -1
        area = max(0, bbox_i[2] - bbox_i[0]) * max(0, bbox_i[3] - bbox_i[1])
        score = float(box.get("confidence", 0.0))

        for k_idx, kept_box in enumerate(kept):
            if str(kept_box.get("class", "")) != cls:
                continue
            kept_bbox = [int(v) for v in kept_box["bbox"]]
            kept_area = max(0, kept_bbox[2] - kept_bbox[0]) * max(0, kept_bbox[3] - kept_bbox[1])
            kept_score = float(kept_box.get("confidence", 0.0))

            cand_inside_kept = _is_fully_inside(bbox_i, kept_bbox)
            kept_inside_cand = _is_fully_inside(kept_bbox, bbox_i)
            if not (cand_inside_kept or kept_inside_cand):
                continue

            if kept_inside_cand and area > kept_area and score + 0.03 >= kept_score:
                replace_idx = k_idx
                break

            drop = True
            break

        if replace_idx >= 0:
            new_box = dict(box)
            new_box["bbox"] = bbox_i
            kept[replace_idx] = new_box
            continue

        if not drop:
            new_box = dict(box)
            new_box["bbox"] = bbox_i
            kept.append(new_box)

    return kept


def _draw_labeled_box(
    image_bgr: np.ndarray,
    bbox: Sequence[float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    norm = _normalize_bbox_for_draw(bbox, image_bgr.shape)
    if norm is None:
        return

    x1, y1, x2, y2 = norm
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    text_x = x1
    text_y = y1 - 4
    if text_y - th - baseline < 0:
        text_y = min(image_bgr.shape[0] - 4, y2 + th + baseline + 4)

    top = max(0, text_y - th - baseline - 2)
    bottom = min(image_bgr.shape[0] - 1, text_y + baseline + 2)
    right = min(image_bgr.shape[1] - 1, text_x + tw + 4)

    cv2.rectangle(image_bgr, (text_x, top), (right, bottom), color, -1)
    cv2.putText(
        image_bgr,
        label,
        (text_x + 2, bottom - baseline - 1),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def sanitize_predictions_for_export(
    predictions: List[dict],
    image_dir: Path,
    class_names: Sequence[str],
    max_objects: int = MAX_OBJECTS_PER_IMAGE,
) -> List[dict]:
    valid_classes = set(class_names)
    out: List[dict] = []

    for pred in predictions:
        image_id = str(pred.get("image_id", ""))
        if not image_id:
            continue

        image = imread_unicode(image_dir / image_id)
        if image is None:
            continue
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            continue

        cleaned: List[dict] = []
        for box in pred.get("boxes", []):
            cls_name = str(box.get("class", ""))
            if cls_name not in valid_classes:
                continue

            conf = float(box.get("confidence", 0.0))
            if not math.isfinite(conf):
                continue
            conf = max(0.0, min(1.0, conf))
            if conf < float(MIN_EXPORT_CONF):
                continue

            bbox = box.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue

            x1 = max(0.0, min(float(w), x1))
            x2 = max(0.0, min(float(w), x2))
            y1 = max(0.0, min(float(h), y1))
            y2 = max(0.0, min(float(h), y2))

            ix1 = _round_half_up(x1)
            iy1 = _round_half_up(y1)
            ix2 = _round_half_up(x2)
            iy2 = _round_half_up(y2)

            ix1 = max(0, min(w - 1, ix1))
            iy1 = max(0, min(h - 1, iy1))
            ix2 = max(0, min(w, ix2))
            iy2 = max(0, min(h, iy2))

            if ix2 <= ix1:
                if ix1 < w:
                    ix2 = ix1 + 1
                else:
                    continue
            if iy2 <= iy1:
                if iy1 < h:
                    iy2 = iy1 + 1
                else:
                    continue

            cleaned.append(
                {
                    "class": cls_name,
                    "confidence": float(conf),
                    "bbox": [int(ix1), int(iy1), int(ix2), int(iy2)],
                }
            )

        cleaned = _suppress_same_class_contained_int(cleaned)
        cleaned = sorted(cleaned, key=lambda b: float(b.get("confidence", 0.0)), reverse=True)[: max(0, int(max_objects))]
        out.append({"image_id": image_id, "boxes": cleaned})

    return out


def clean_results_dir(results_dir: Path) -> None:
    if not results_dir.exists():
        return
    for p in results_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == "hardcase_summary.json":
            continue
        if p.name.startswith("hardcase_") and p.suffix.lower() in VALID_EXTS:
            p.unlink()


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _inverse_hflip_boxes(boxes: Sequence[dict], width: int) -> List[dict]:
    out: List[dict] = []
    for box in boxes:
        bbox = box.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        nx1 = float(width) - x2
        nx2 = float(width) - x1
        new_box = dict(box)
        new_box["bbox"] = [nx1, y1, nx2, y2]
        out.append(new_box)
    return out


def _inverse_rot180_boxes(boxes: Sequence[dict], width: int, height: int) -> List[dict]:
    out: List[dict] = []
    for box in boxes:
        bbox = box.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        nx1 = float(width) - x2
        ny1 = float(height) - y2
        nx2 = float(width) - x1
        ny2 = float(height) - y1
        new_box = dict(box)
        new_box["bbox"] = [nx1, ny1, nx2, ny2]
        out.append(new_box)
    return out


def _merge_tta_boxes(
    all_boxes: Sequence[dict],
    iou_thresh: float = 0.45,
    min_votes: int = 2,
) -> List[dict]:
    if not all_boxes:
        return []

    ordered = sorted(all_boxes, key=lambda b: float(b.get("confidence", 0.0)), reverse=True)
    used = [False] * len(ordered)
    merged: List[dict] = []

    for i, base in enumerate(ordered):
        if used[i]:
            continue
        cls_name = str(base.get("class", ""))
        base_bbox = base.get("bbox", [0.0, 0.0, 0.0, 0.0])
        cluster = [i]
        used[i] = True

        for j in range(i + 1, len(ordered)):
            if used[j]:
                continue
            cand = ordered[j]
            if str(cand.get("class", "")) != cls_name:
                continue
            if box_iou(base_bbox, cand.get("bbox", [0.0, 0.0, 0.0, 0.0])) >= float(iou_thresh):
                used[j] = True
                cluster.append(j)

        votes = len(cluster)
        confs = [float(ordered[k].get("confidence", 0.0)) for k in cluster]
        max_conf = max(confs)

        if votes < int(min_votes) and max_conf < 0.60:
            continue

        denom = max(sum(confs), 1e-9)
        x1 = sum(float(ordered[k]["bbox"][0]) * confs[idx] for idx, k in enumerate(cluster)) / denom
        y1 = sum(float(ordered[k]["bbox"][1]) * confs[idx] for idx, k in enumerate(cluster)) / denom
        x2 = sum(float(ordered[k]["bbox"][2]) * confs[idx] for idx, k in enumerate(cluster)) / denom
        y2 = sum(float(ordered[k]["bbox"][3]) * confs[idx] for idx, k in enumerate(cluster)) / denom

        merged.append(
            {
                "class": cls_name,
                "confidence": float(max_conf),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return merged


@torch.no_grad()
def _run_tta_fallback_single(
    model: AnchorFreeDetector,
    image_bgr: np.ndarray,
    image_id: str,
    device: torch.device,
    img_size: int,
    conf_thresh: float,
    nms_thresh: float,
    class_names: Sequence[str],
    center_combine: str,
    tta_min_votes: int,
) -> List[dict]:
    h, w = image_bgr.shape[:2]
    fallback_conf = max(0.20, min(float(conf_thresh) * 0.75, 0.28))

    variants = [
        ("orig", image_bgr, lambda boxes: list(boxes)),
        ("hflip", cv2.flip(image_bgr, 1), lambda boxes: _inverse_hflip_boxes(boxes, width=w)),
        ("rot180", cv2.rotate(image_bgr, cv2.ROTATE_180), lambda boxes: _inverse_rot180_boxes(boxes, width=w, height=h)),
    ]

    all_boxes: List[dict] = []
    amp_enabled = device.type == "cuda"

    for _, variant_img, inv_fn in variants:
        tensor, meta = letterbox_preprocess(variant_img, img_size=img_size)
        images = torch.stack([tensor], dim=0).to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
        pred = postprocess_batch(
            outputs=outputs,
            image_ids=[image_id],
            metas=[meta],
            class_names=class_names,
            num_classes=len(class_names),
            img_size=img_size,
            conf_thresh=fallback_conf,
            nms_thresh=nms_thresh,
            reg_decode="auto",
            center_combine=str(center_combine),
            min_box_size=MIN_BOX_SIZE,
        )[0]["boxes"]
        all_boxes.extend(inv_fn(pred))

    merged = _merge_tta_boxes(all_boxes, iou_thresh=0.45, min_votes=max(1, int(tta_min_votes)))
    merged = _suppress_same_class_contained_int(merged)
    merged = sorted(merged, key=lambda b: float(b.get("confidence", 0.0)), reverse=True)
    return merged[: int(MAX_OBJECTS_PER_IMAGE)]


def load_ground_truth(annotation_path: Path, class_names: Sequence[str]) -> Dict[str, List[dict]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    valid_classes = set(class_names)

    gt: Dict[str, List[dict]] = {}
    for image in data.get("images", []):
        image_id = str(image.get("id", ""))
        if image_id:
            gt[image_id] = []

    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id", ""))
        cls_name = str(ann.get("class", ""))
        bbox = ann.get("bbox", [])
        if image_id not in gt or cls_name not in valid_classes:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        gt[image_id].append({"class": cls_name, "bbox": [x1, y1, x2, y2]})

    return gt


def score_image_error(
    gt_boxes: Sequence[dict],
    pred_boxes: Sequence[dict],
    class_names: Sequence[str],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    fn = 0
    fp = 0
    tp = 0
    loc_penalty = 0.0

    for cls_name in class_names:
        gt_cls = [b for b in gt_boxes if str(b.get("class", "")) == cls_name]
        pred_cls = sorted(
            [b for b in pred_boxes if str(b.get("class", "")) == cls_name],
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )

        matched_gt: set[int] = set()
        for pred in pred_cls:
            pb = pred.get("bbox", [0, 0, 0, 0])
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_cls):
                if idx in matched_gt:
                    continue
                iou = box_iou(pb, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= float(iou_thresh):
                matched_gt.add(best_idx)
                tp += 1
                loc_penalty += 1.0 - best_iou
            else:
                fp += 1

        fn += max(0, len(gt_cls) - len(matched_gt))

    error_score = 3.0 * fn + 1.5 * fp + loc_penalty
    total_boxes = len(gt_boxes) + len(pred_boxes)
    error_ratio = error_score / max(1.0, float(total_boxes))

    return {
        "error_score": float(error_score),
        "error_ratio": float(error_ratio),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def match_predictions_to_ground_truth(
    gt_boxes: Sequence[dict],
    pred_boxes: Sequence[dict],
    class_names: Sequence[str],
    iou_thresh: float = 0.5,
) -> Tuple[List[bool], List[Optional[int]], List[bool]]:
    gt_by_class: Dict[str, List[Tuple[int, dict]]] = {}
    for idx, gt in enumerate(gt_boxes):
        cls_name = str(gt.get("class", ""))
        if cls_name not in class_names:
            continue
        gt_by_class.setdefault(cls_name, []).append((idx, gt))

    pred_sorted = sorted(
        [(idx, pred) for idx, pred in enumerate(pred_boxes)],
        key=lambda item: float(item[1].get("confidence", 0.0)),
        reverse=True,
    )

    pred_is_correct = [False] * len(pred_boxes)
    pred_matched_gt: List[Optional[int]] = [None] * len(pred_boxes)
    gt_is_matched = [False] * len(gt_boxes)

    matched_gt_indices: Dict[str, set[int]] = {cls: set() for cls in class_names}

    for pred_idx, pred in pred_sorted:
        cls_name = str(pred.get("class", ""))
        if cls_name not in gt_by_class:
            continue

        pb = pred.get("bbox", [0, 0, 0, 0])
        best_iou = 0.0
        best_gt_idx: Optional[int] = None

        for gt_idx, gt in gt_by_class[cls_name]:
            if gt_idx in matched_gt_indices[cls_name]:
                continue
            iou = box_iou(pb, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= float(iou_thresh):
            matched_gt_indices[cls_name].add(best_gt_idx)
            pred_is_correct[pred_idx] = True
            pred_matched_gt[pred_idx] = best_gt_idx
            gt_is_matched[best_gt_idx] = True

    return pred_is_correct, pred_matched_gt, gt_is_matched


def draw_hardcase(
    image_bgr: np.ndarray,
    gt_boxes: Sequence[dict],
    pred_boxes: Sequence[dict],
    class_names: Sequence[str],
    iou_thresh: float = 0.5,
) -> np.ndarray:
    out = image_bgr.copy()
    pred_is_correct, _, gt_is_matched = match_predictions_to_ground_truth(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        class_names=class_names,
        iou_thresh=iou_thresh,
    )

    for gt_idx, gt in enumerate(gt_boxes):
        cls_name = str(gt.get("class", ""))
        color = (80, 180, 255) if gt_is_matched[gt_idx] else (140, 140, 140)
        _draw_labeled_box(out, gt.get("bbox", [0, 0, 0, 0]), f"GT:{cls_name}", color, thickness=2)

    for pred_idx, pred in enumerate(pred_boxes):
        cls_name = str(pred.get("class", ""))
        conf = float(pred.get("confidence", 0.0))
        color = (40, 220, 70) if pred_is_correct[pred_idx] else (30, 30, 255)
        _draw_labeled_box(out, pred.get("bbox", [0, 0, 0, 0]), f"PD:{cls_name}:{conf:.2f}", color, thickness=2)

    return out


def export_hardcase_summary(
    predictions: List[dict],
    image_dir: Path,
    annotation_path: Path,
    class_names: Sequence[str],
    results_dir: Path,
    top_k: int = 100,
    iou_thresh: float = 0.5,
) -> Tuple[int, Path]:
    gt = load_ground_truth(annotation_path, class_names)
    pred_map = {str(p.get("image_id", "")): list(p.get("boxes", [])) for p in predictions}

    scored: List[dict] = []
    for image_id, gt_boxes in gt.items():
        pred_boxes = pred_map.get(image_id, [])
        metrics = score_image_error(gt_boxes, pred_boxes, class_names, iou_thresh=iou_thresh)
        scored.append(
            {
                "image_id": image_id,
                "error_score": metrics["error_score"],
                "error_ratio": metrics["error_ratio"],
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
            }
        )

    scored.sort(key=lambda x: (x["error_ratio"], x["error_score"], x["fn"], x["fp"]), reverse=True)
    top_items = scored[: max(0, int(top_k))]

    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "hardcase_summary.json"
    summary_path.write_text(json.dumps(top_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return len(top_items), summary_path


def load_hardcase_summary(summary_path: Path) -> List[dict]:
    if not summary_path.exists():
        return []
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id", ""))
        if not image_id:
            continue
        out.append(item)
    return out


def draw_prediction(image_bgr: np.ndarray, boxes: Sequence[dict], class_names: Sequence[str]) -> np.ndarray:
    out = image_bgr.copy()
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    for obj in boxes:
        cls_name = str(obj["class"])
        score = float(obj["confidence"])
        idx = cls_to_idx.get(cls_name, 0)
        color = (
            int((53 * (idx + 1)) % 255),
            int((97 * (idx + 1)) % 255),
            int((193 * (idx + 1)) % 255),
        )
        label = f"{cls_name}:{score:.2f}"
        _draw_labeled_box(out, obj.get("bbox", [0, 0, 0, 0]), label, color, thickness=2)
    return out


def apply_class_thresholds(
    predictions: List[dict],
    class_names: Sequence[str],
    class_conf_thresh: Sequence[float],
) -> List[dict]:
    th_map = {c: float(class_conf_thresh[i]) for i, c in enumerate(class_names) if i < len(class_conf_thresh)}
    out: List[dict] = []
    for pred in predictions:
        keep = []
        for box in pred.get("boxes", []):
            c = str(box.get("class", ""))
            s = float(box.get("confidence", 0.0))
            thr = max(th_map.get(c, 0.5), float(MIN_EXPORT_CONF))
            if s >= thr:
                keep.append(box)
        out.append({"image_id": pred.get("image_id"), "boxes": keep})
    return out


def suppress_chair_inside_person(predictions: List[dict], iou_thresh: float) -> List[dict]:
    def _box_area(bbox: Sequence[float]) -> float:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    out: List[dict] = []
    for pred in predictions:
        boxes = pred.get("boxes", [])
        persons = [b for b in boxes if str(b.get("class")) == "person"]
        keep: List[dict] = []
        for b in boxes:
            cls = str(b.get("class", ""))
            if cls != "chair":
                keep.append(b)
                continue
            chair_box = b.get("bbox", [0, 0, 0, 0])
            remove = False
            chair_area = _box_area(chair_box)
            for p in persons:
                person_box = p.get("bbox", [0, 0, 0, 0])
                if box_iou(chair_box, person_box) < float(iou_thresh):
                    continue
                if not _is_fully_inside(chair_box, person_box):
                    continue
                person_area = _box_area(person_box)
                if chair_area <= float(CHAIR_SUPPRESS_MAX_AREA_RATIO) * max(person_area, 1.0):
                    remove = True
                    break
            if not remove:
                keep.append(b)
        out.append({"image_id": pred.get("image_id"), "boxes": keep})
    return out


@torch.no_grad()
def run_inference(
    model: AnchorFreeDetector,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int,
    img_size: int,
    conf_thresh: float,
    nms_thresh: float,
    class_names: Sequence[str],
    center_combine: str = INFER_CENTER_COMBINE,
    enable_tta_fallback: bool = False,
    tta_min_votes: int = 2,
) -> List[dict]:
    results: List[dict] = []
    amp_enabled = device.type == "cuda"

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors: List[torch.Tensor] = []
        metas: List[LetterboxMeta] = []
        image_ids: List[str] = []
        originals: Dict[str, np.ndarray] = {}

        for p in batch_paths:
            image = imread_unicode(p)
            if image is None:
                continue
            tensor, meta = letterbox_preprocess(image, img_size=img_size)
            tensors.append(tensor)
            metas.append(meta)
            image_ids.append(p.name)
            originals[p.name] = image

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
            num_classes=len(class_names),
            img_size=img_size,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            reg_decode="auto",
            center_combine=str(center_combine),
            min_box_size=MIN_BOX_SIZE,
        )

        if enable_tta_fallback:
            for item in batch_results:
                if len(item.get("boxes", [])) > 0:
                    continue
                image_id = str(item.get("image_id", ""))
                image_bgr = originals.get(image_id)
                if image_bgr is None:
                    continue
                tta_boxes = _run_tta_fallback_single(
                    model=model,
                    image_bgr=image_bgr,
                    image_id=image_id,
                    device=device,
                    img_size=img_size,
                    conf_thresh=conf_thresh,
                    nms_thresh=nms_thresh,
                    class_names=class_names,
                    center_combine=center_combine,
                    tta_min_votes=tta_min_votes,
                )
                if tta_boxes:
                    item["boxes"] = tta_boxes

        results.extend(batch_results)

    return results


def save_preview_images(predictions: List[dict], image_dir: Path, preview_dir: Path, limit: int, class_names: Sequence[str]) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for pred in predictions[:limit]:
        image_id = pred["image_id"]
        img_path = image_dir / image_id
        image = imread_unicode(img_path)
        if image is None:
            continue

        vis = draw_prediction(image, pred.get("boxes", []), class_names=class_names)
        out_path = preview_dir / image_id
        if imwrite_unicode(out_path, vis):
            saved += 1
    return saved


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Tuple[AnchorFreeDetector, List[str], int]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    classes = ckpt.get("classes", CLASS_NAMES)
    img_size = int(ckpt.get("img_size", IMG_SIZE))
    num_classes = len(classes)

    model = AnchorFreeDetector(num_classes=num_classes, pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Checkpoint partially loaded ({exc}).")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    model.eval()

    return model, list(classes), img_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict val set, export integer bbox JSON, and save hardcase summary.")
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_VAL_IMAGE_DIR)
    parser.add_argument("--val_annotation", type=Path, default=DEFAULT_VAL_ANNOTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--checkpoint",
        "--model_path",
        dest="checkpoint",
        type=Path,
        default=Path("models/best.pth"),
        help="Path to trained model checkpoint (.pth). '--model_path' is kept as a backward-compatible alias.",
    )
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--nms_thresh", type=float, default=NMS_IOU_THRESH)
    parser.add_argument(
        "--center_combine",
        type=str,
        default=str(INFER_CENTER_COMBINE),
        choices=["cls", "soft", "sqrt", "mul"],
        help="How to combine class score and centerness at inference.",
    )
    parser.add_argument(
        "--class_conf",
        type=str,
        default=",".join(str(x) for x in CLASS_CONF_THRESH),
        help="Per-class thresholds in CLASS_NAMES order, e.g. '0.38,0.40,0.40,0.40,0.72'",
    )
    parser.add_argument("--chair_suppress_iou", type=float, default=CHAIR_SUPPRESS_WITH_PERSON_IOU)
    parser.add_argument("--hardcase_topk", type=int, default=50)
    parser.add_argument("--hardcase_iou", type=float, default=0.5)
    parser.add_argument("--tta_fallback", action="store_true", help="Enable TTA fallback on images with zero detections.")
    parser.add_argument("--tta_min_votes", type=int, default=2, help="Minimum TTA consensus votes to keep a fallback box.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.val_annotation.exists():
        raise FileNotFoundError(f"Validation annotation not found: {args.val_annotation}")

    if args.image_dir.resolve() != DEFAULT_VAL_IMAGE_DIR.resolve():
        raise ValueError(f"--image_dir must be '{DEFAULT_VAL_IMAGE_DIR}'. Got: {args.image_dir}")
    if args.val_annotation.resolve() != DEFAULT_VAL_ANNOTATION.resolve():
        raise ValueError(f"--val_annotation must be '{DEFAULT_VAL_ANNOTATION}'. Got: {args.val_annotation}")
    if args.results_dir.resolve() != DEFAULT_RESULTS_DIR.resolve():
        raise ValueError(f"--results_dir must be '{DEFAULT_RESULTS_DIR}'. Got: {args.results_dir}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model, ckpt_classes, ckpt_img_size = load_checkpoint_model(args.checkpoint, device=device)
    model = model.to(memory_format=torch.channels_last)
    class_names = ckpt_classes if ckpt_classes else CLASS_NAMES
    img_size = args.img_size if args.img_size > 0 else ckpt_img_size

    class_conf = [float(x.strip()) for x in str(args.class_conf).split(",") if x.strip()]
    if len(class_conf) != len(class_names):
        raise ValueError(f"--class_conf must have {len(class_names)} values (got {len(class_conf)}).")

    image_paths = collect_images(args.image_dir)
    if not image_paths:
        raise ValueError(f"No images found in: {args.image_dir}")

    predictions = run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=max(1, args.batch_size),
        img_size=img_size,
        conf_thresh=float(args.conf_thresh),
        nms_thresh=float(args.nms_thresh),
        class_names=class_names,
        center_combine=str(args.center_combine),
        enable_tta_fallback=bool(args.tta_fallback),
        tta_min_votes=max(1, int(args.tta_min_votes)),
    )
    predictions = apply_class_thresholds(predictions, class_names=class_names, class_conf_thresh=class_conf)
    predictions = suppress_chair_inside_person(predictions, iou_thresh=float(args.chair_suppress_iou))

    predictions = sanitize_predictions_for_export(
        predictions=predictions,
        image_dir=args.image_dir,
        class_names=class_names,
        max_objects=MAX_OBJECTS_PER_IMAGE,
    )
    clean_results_dir(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    hardcase_count, summary_path = export_hardcase_summary(
        predictions=predictions,
        image_dir=args.image_dir,
        annotation_path=args.val_annotation,
        class_names=class_names,
        results_dir=args.results_dir,
        top_k=max(0, int(args.hardcase_topk)),
        iou_thresh=float(args.hardcase_iou),
    )

    print(f"Device: {device}")
    ckpt_meta = torch.load(str(args.checkpoint), map_location="cpu")
    print(
        "Checkpoint meta: "
        f"epoch={ckpt_meta.get('epoch', 'NA')}, "
        f"best_val_loss={ckpt_meta.get('best_val_loss', 'NA')}, "
        f"img_size={ckpt_meta.get('img_size', 'NA')}"
    )
    print(f"Predicted images: {len(predictions)}")
    print(f"Saved JSON: {args.output}")
    print(f"TTA fallback: {bool(args.tta_fallback)} (min_votes={max(1, int(args.tta_min_votes))})")
    print(f"Hardcase items: {hardcase_count}")
    print(f"Hardcase summary: {summary_path}")


if __name__ == "__main__":
    main()
