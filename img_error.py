from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2

from predict import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_VAL_ANNOTATION,
    DEFAULT_VAL_IMAGE_DIR,
    imread_unicode,
    imwrite_unicode,
    load_ground_truth,
    load_hardcase_summary,
    match_predictions_to_ground_truth,
)
from utils.config import CLASS_NAMES

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SUMMARY = DEFAULT_RESULTS_DIR / "hardcase_summary.json"
DEFAULT_PREDICTIONS = Path("val_predictions.json")


def load_predictions(predictions_path: Path) -> Dict[str, List[dict]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    data = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Predictions file must contain a list: {predictions_path}")
    pred_map: Dict[str, List[dict]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id", ""))
        if not image_id:
            continue
        pred_map[image_id] = list(item.get("boxes", []))
    return pred_map


def clean_hardcase_images(results_dir: Path) -> None:
    if not results_dir.exists():
        return
    for path in results_dir.iterdir():
        if not path.is_file():
            continue
        if path.name == "hardcase_summary.json":
            continue
        if path.name.startswith("hardcase_") and path.suffix.lower() in VALID_EXTS:
            path.unlink()


def draw_header(image, text: str) -> None:
    height, width = image.shape[:2]
    bar_h = 28
    cv2.rectangle(image, (0, 0), (width, bar_h), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _round_half_up(value: float) -> int:
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def _normalize_bbox_for_draw(bbox, image_shape):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
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


def _draw_labeled_box(image_bgr, bbox, label: str, color, thickness: int = 2) -> None:
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


def draw_prediction_only(
    image_bgr,
    gt_boxes: Sequence[dict],
    pred_boxes: Sequence[dict],
    class_names: Sequence[str],
    iou_thresh: float = 0.5,
):
    out = image_bgr.copy()
    pred_is_correct, _, _ = match_predictions_to_ground_truth(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        class_names=class_names,
        iou_thresh=iou_thresh,
    )

    for idx, pred in enumerate(pred_boxes):
        cls_name = str(pred.get("class", ""))
        conf = float(pred.get("confidence", 0.0))
        color = (40, 220, 70) if pred_is_correct[idx] else (30, 30, 255)
        _draw_labeled_box(out, pred.get("bbox", [0, 0, 0, 0]), f"PD:{cls_name}:{conf:.2f}", color, thickness=2)

    return out


def render_hardcases(
    summary_path: Path,
    predictions_path: Path,
    image_dir: Path,
    annotation_path: Path,
    results_dir: Path,
    limit: int = 0,
) -> int:
    summary_items = load_hardcase_summary(summary_path)
    if not summary_items:
        raise ValueError(f"No hardcase entries found in: {summary_path}")

    pred_map = load_predictions(predictions_path)
    gt_map = load_ground_truth(annotation_path, CLASS_NAMES)

    results_dir.mkdir(parents=True, exist_ok=True)
    clean_hardcase_images(results_dir)

    if limit > 0:
        summary_items = summary_items[:limit]

    saved = 0
    for rank, item in enumerate(summary_items, start=1):
        image_id = str(item.get("image_id", ""))
        if not image_id:
            continue

        image = imread_unicode(image_dir / image_id)
        if image is None:
            continue

        vis = draw_prediction_only(
            image_bgr=image,
            gt_boxes=gt_map.get(image_id, []),
            pred_boxes=pred_map.get(image_id, []),
            class_names=CLASS_NAMES,
            iou_thresh=0.5,
        )
        draw_header(
            vis,
            f"rank={rank} err={float(item.get('error_score', 0.0)):.3f} "
            f"ratio={float(item.get('error_ratio', 0.0)):.3f} "
            f"tp={int(item.get('tp', 0))} fp={int(item.get('fp', 0))} fn={int(item.get('fn', 0))}",
        )

        out_name = f"hardcase_{rank:03d}_{image_id}"
        if imwrite_unicode(results_dir / out_name, vis):
            saved += 1

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render hardcase images from results/hardcase_summary.json.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_VAL_IMAGE_DIR)
    parser.add_argument("--val_annotation", type=Path, default=DEFAULT_VAL_ANNOTATION)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N hardcase entries. 0 = all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(f"Summary not found: {args.summary}")
    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions not found: {args.predictions}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.val_annotation.exists():
        raise FileNotFoundError(f"Validation annotation not found: {args.val_annotation}")

    if args.summary.resolve() != DEFAULT_SUMMARY.resolve():
        raise ValueError(f"--summary must be '{DEFAULT_SUMMARY}'. Got: {args.summary}")
    if args.results_dir.resolve() != DEFAULT_RESULTS_DIR.resolve():
        raise ValueError(f"--results_dir must be '{DEFAULT_RESULTS_DIR}'. Got: {args.results_dir}")

    saved = render_hardcases(
        summary_path=args.summary,
        predictions_path=args.predictions,
        image_dir=args.image_dir,
        annotation_path=args.val_annotation,
        results_dir=args.results_dir,
        limit=max(0, int(args.limit)),
    )

    print(f"Rendered hardcase images: {saved}")
    print(f"Source summary: {args.summary}")
    print(f"Predictions: {args.predictions}")
    print(f"Output dir: {args.results_dir}")


if __name__ == "__main__":
    main()
