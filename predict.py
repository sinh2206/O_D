from __future__ import annotations

"""Run YOLOv8 inference and export detections in the repo's JSON format."""

import argparse
from pathlib import Path
from typing import List

from utils.config import (
    DEFAULT_CONF_THRESH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESH,
    DEFAULT_MAX_DET,
    DEFAULT_RESULTS_DIR,
)
from utils.forecast import save_analysis_outputs
from utils.image_ops import draw_boxes, imread_unicode, imwrite_unicode
from utils.model import get_model_class_names, load_yolo_model
from utils.nms import results_to_prediction_json
from utils.runtime import ensure_dir, list_images, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    """Define the CLI used for YOLOv8 prediction export."""

    parser = argparse.ArgumentParser(description="Predict bounding boxes with YOLOv8 and export them as JSON.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing images to infer.")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"), help="Output predictions JSON path.")
    parser.add_argument("--checkpoint", "--model_path", dest="checkpoint", type=Path, default=Path("models/best.pt"), help="Path to a trained YOLOv8 .pt checkpoint.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Inference image size.")
    parser.add_argument("--conf_thresh", type=float, default=DEFAULT_CONF_THRESH, help="Confidence threshold.")
    parser.add_argument("--iou_thresh", type=float, default=DEFAULT_IOU_THRESH, help="NMS IoU threshold inside YOLOv8.")
    parser.add_argument("--max_det", type=int, default=DEFAULT_MAX_DET, help="Maximum detections per image.")
    parser.add_argument("--device", type=str, default="auto", help="Inference device, e.g. auto, cpu, 0.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for optional analysis/preview outputs.")
    parser.add_argument("--annotation", type=Path, default=None, help="Optional ground-truth JSON for public-set analysis.")
    parser.add_argument("--save_vis", action="store_true", help="Save predicted preview images under results_dir/previews.")
    parser.add_argument("--hardcase_topk", type=int, default=50, help="Number of hardcase rows to export when annotation is provided.")
    return parser.parse_args()


def save_preview_images(predictions: List[dict], image_dir: Path, preview_dir: Path, class_names: List[str]) -> int:
    """Render exported detections on images and save them to disk."""

    preview_dir = ensure_dir(preview_dir)
    saved = 0
    for pred in predictions:
        image_id = str(pred.get("image_id", ""))
        if not image_id:
            continue
        image = imread_unicode(image_dir / image_id)
        if image is None:
            continue
        vis = draw_boxes(image, pred.get("boxes", []), class_names=class_names)
        if imwrite_unicode(preview_dir / image_id, vis):
            saved += 1
    return saved


def main() -> None:
    """Load a YOLOv8 checkpoint, run inference and export predictions JSON."""

    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {args.image_dir}")

    model = load_yolo_model(args.checkpoint)
    class_names = get_model_class_names(model)

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=int(args.img_size),
        conf=float(args.conf_thresh),
        iou=float(args.iou_thresh),
        max_det=int(args.max_det),
        device=resolve_device(args.device),
        verbose=False,
        save=False,
        stream=False,
    )

    predictions = results_to_prediction_json(
        results=results,
        image_ids=[path.name for path in image_paths],
        class_names=class_names,
        max_det=int(args.max_det),
    )
    write_json(args.output, predictions)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images predicted: {len(predictions)}")
    print(f"Output JSON: {args.output}")

    if args.save_vis:
        preview_dir = ensure_dir(args.results_dir / "previews")
        saved = save_preview_images(predictions, image_dir=args.image_dir, preview_dir=preview_dir, class_names=class_names)
        print(f"Saved preview images: {saved} -> {preview_dir}")

    if args.annotation is not None:
        metrics_path, hardcase_path = save_analysis_outputs(
            predictions=predictions,
            annotation_path=args.annotation,
            results_dir=args.results_dir,
            top_k=int(args.hardcase_topk),
            iou_thresh=float(args.iou_thresh),
        )
        print(f"Analysis summary: {metrics_path}")
        print(f"Hardcase summary: {hardcase_path}")


if __name__ == "__main__":
    main()
