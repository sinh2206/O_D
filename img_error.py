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
    draw_hardcase,
    imread_unicode,
    imwrite_unicode,
    load_ground_truth,
    load_hardcase_summary,
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

        vis = draw_hardcase(
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
