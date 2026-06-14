from __future__ import annotations

"""Optional public-set analysis helpers for exported YOLOv8 predictions."""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .runtime import load_json, write_json


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU between two xyxy boxes."""

    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def load_ground_truth(annotation_path: Path) -> Dict[str, List[dict]]:
    """Load the repo's JSON annotations into a per-image dictionary."""

    data = load_json(annotation_path)
    gt: Dict[str, List[dict]] = {str(im.get("id", "")): [] for im in data.get("images", [])}
    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id", ""))
        bbox = ann.get("bbox", [])
        if image_id not in gt or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        gt[image_id].append({"class": str(ann.get("class", "")), "bbox": [int(v) for v in bbox]})
    return gt


def evaluate_predictions(gt_map: Dict[str, List[dict]], predictions: Sequence[dict], iou_thresh: float = 0.5) -> Dict[str, float]:
    """Compute simple micro precision/recall style metrics for quick inspection."""

    pred_map = {str(item.get("image_id", "")): list(item.get("boxes", [])) for item in predictions}
    tp = 0
    fp = 0
    fn = 0
    iou_sum = 0.0

    for image_id, gt_boxes in gt_map.items():
        pred_boxes = pred_map.get(image_id, [])
        used_gt: set[int] = set()

        for pred in pred_boxes:
            pred_cls = str(pred.get("class", ""))
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in used_gt or str(gt.get("class", "")) != pred_cls:
                    continue
                current_iou = box_iou(pred.get("bbox", [0, 0, 0, 0]), gt.get("bbox", [0, 0, 0, 0]))
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= float(iou_thresh):
                used_gt.add(best_idx)
                tp += 1
                iou_sum += best_iou
            else:
                fp += 1

        fn += max(0, len(gt_boxes) - len(used_gt))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    mean_iou = iou_sum / max(tp, 1)
    return {
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "mean_iou": float(mean_iou),
    }


def build_hardcase_summary(gt_map: Dict[str, List[dict]], predictions: Sequence[dict], top_k: int = 50, iou_thresh: float = 0.5) -> List[dict]:
    """Rank the hardest public images by missed detections and false positives."""

    pred_map = {str(item.get("image_id", "")): list(item.get("boxes", [])) for item in predictions}
    items: List[dict] = []

    for image_id, gt_boxes in gt_map.items():
        pred_boxes = pred_map.get(image_id, [])
        matched_gt: set[int] = set()
        tp = 0
        fp = 0
        iou_penalty = 0.0

        for pred in pred_boxes:
            pred_cls = str(pred.get("class", ""))
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt or str(gt.get("class", "")) != pred_cls:
                    continue
                current_iou = box_iou(pred.get("bbox", [0, 0, 0, 0]), gt.get("bbox", [0, 0, 0, 0]))
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= float(iou_thresh):
                matched_gt.add(best_idx)
                tp += 1
                iou_penalty += 1.0 - best_iou
            else:
                fp += 1

        fn = max(0, len(gt_boxes) - len(matched_gt))
        error_score = 3.0 * fn + 1.5 * fp + iou_penalty
        items.append(
            {
                "image_id": image_id,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "error_score": float(error_score),
            }
        )

    items.sort(key=lambda item: (item["error_score"], item["fn"], item["fp"]), reverse=True)
    return items[: max(0, int(top_k))]


def save_analysis_outputs(predictions: Sequence[dict], annotation_path: Path, results_dir: Path, top_k: int = 50, iou_thresh: float = 0.5) -> Tuple[Path, Path]:
    """Write a simple metrics summary and a hardcase ranking JSON."""

    results_dir.mkdir(parents=True, exist_ok=True)
    gt_map = load_ground_truth(annotation_path)
    metrics = evaluate_predictions(gt_map, predictions, iou_thresh=iou_thresh)
    hardcases = build_hardcase_summary(gt_map, predictions, top_k=top_k, iou_thresh=iou_thresh)

    metrics_path = write_json(results_dir / "analysis_summary.json", metrics)
    hardcase_path = write_json(results_dir / "hardcase_summary.json", hardcases)
    return metrics_path, hardcase_path
