import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_ANNOTATION = Path("public/annotations/val.json")
DEFAULT_IMAGE_DIR = Path("public/val/images")
DEFAULT_HARDCASE = Path("results/hardcase_summary.json")
DEFAULT_PREDICTIONS = Path("val_predictions.json")
DEFAULT_OUTPUT_DIR = Path("small_obj")
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "small_obj.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_area_xyxy(bbox: Sequence[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = bbox_area_xyxy(box_a)
    area_b = bbox_area_xyxy(box_b)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def match_predictions_to_ground_truth(
    gt_boxes: Sequence[dict],
    pred_boxes: Sequence[dict],
    iou_thresh: float,
) -> List[bool]:
    gt_by_class: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    for gt_idx, gt in enumerate(gt_boxes):
        gt_by_class[str(gt.get("class", ""))].append((gt_idx, gt))

    matched_gt = {cls_name: set() for cls_name in gt_by_class}
    gt_is_matched = [False] * len(gt_boxes)

    pred_sorted = sorted(
        pred_boxes,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )

    for pred in pred_sorted:
        cls_name = str(pred.get("class", ""))
        if cls_name not in gt_by_class:
            continue

        best_iou = 0.0
        best_gt_idx: Optional[int] = None
        for gt_idx, gt in gt_by_class[cls_name]:
            if gt_idx in matched_gt[cls_name]:
                continue
            iou = box_iou(pred.get("bbox", [0, 0, 0, 0]), gt.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= float(iou_thresh):
            matched_gt[cls_name].add(best_gt_idx)
            gt_is_matched[best_gt_idx] = True

    return gt_is_matched


def build_small_object_report(
    annotation_path: Path,
    image_dir: Path,
    hardcase_path: Path,
    output_dir: Path,
    area_threshold: float,
    predictions_path: Optional[Path],
    iou_thresh: float,
) -> dict:
    annotation_data = load_json(annotation_path)
    images = annotation_data.get("images", [])
    annotations = annotation_data.get("annotations", [])

    image_map = {str(image["id"]): image for image in images}
    ann_map: Dict[str, List[dict]] = defaultdict(list)
    for ann in annotations:
        ann_map[str(ann.get("image_id", ""))].append(ann)

    hardcase_items = load_json(hardcase_path)
    hardcase_map = {str(item.get("image_id", "")): item for item in hardcase_items}

    prediction_map: Dict[str, List[dict]] = {}
    if predictions_path and predictions_path.exists():
        prediction_items = load_json(predictions_path)
        prediction_map = {
            str(item.get("image_id", "")): list(item.get("boxes", []))
            for item in prediction_items
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    report_images: List[dict] = []
    total_small_objects = 0
    total_missed_small_objects = 0
    images_with_missed_small_objects = 0
    hardcase_overlap_count = 0
    hardcase_overlap_with_fn_count = 0
    hardcase_overlap_with_missed_small_objects = 0

    for image_id, meta in image_map.items():
        image_annotations = ann_map.get(image_id, [])
        small_annotations: List[dict] = []
        for ann_idx, ann in enumerate(image_annotations):
            bbox = ann.get("bbox", [0, 0, 0, 0])
            area = bbox_area_xyxy(bbox)
            if area < float(area_threshold):
                small_annotations.append(
                    {
                        "ann_index": ann_idx,
                        "class": str(ann.get("class", "")),
                        "bbox": [float(v) for v in bbox],
                        "area": area,
                    }
                )

        if not small_annotations:
            continue

        total_small_objects += len(small_annotations)

        file_name = Path(str(meta.get("file_name", image_id))).name
        src_image = image_dir / file_name
        dst_image = output_dir / file_name
        copied = False
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
            copied = True

        hardcase_entry = hardcase_map.get(image_id)
        in_hardcase = hardcase_entry is not None
        if in_hardcase:
            hardcase_overlap_count += 1
            if int(hardcase_entry.get("fn", 0)) > 0:
                hardcase_overlap_with_fn_count += 1

        missed_small_object_count = 0
        if prediction_map:
            gt_match_flags = match_predictions_to_ground_truth(
                gt_boxes=image_annotations,
                pred_boxes=prediction_map.get(image_id, []),
                iou_thresh=iou_thresh,
            )
            for small_ann in small_annotations:
                ann_idx = int(small_ann["ann_index"])
                matched = ann_idx < len(gt_match_flags) and gt_match_flags[ann_idx]
                small_ann["matched"] = bool(matched)
                if not matched:
                    missed_small_object_count += 1

        for small_ann in small_annotations:
            small_ann.pop("ann_index", None)

        total_missed_small_objects += missed_small_object_count
        if missed_small_object_count > 0:
            images_with_missed_small_objects += 1
            if in_hardcase:
                hardcase_overlap_with_missed_small_objects += 1

        report_images.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "copied_to": str(dst_image.as_posix()) if copied else None,
                "image_size": {
                    "width": int(meta.get("width", 0)),
                    "height": int(meta.get("height", 0)),
                },
                "total_object_count": len(image_annotations),
                "small_object_count": len(small_annotations),
                "missed_small_object_count": missed_small_object_count,
                "in_hardcase_summary": in_hardcase,
                "hardcase": hardcase_entry,
                "small_objects": small_annotations,
            }
        )

    report_images.sort(
        key=lambda item: (
            item["missed_small_object_count"],
            item["small_object_count"],
            item["image_id"],
        ),
        reverse=True,
    )

    return {
        "area_threshold": float(area_threshold),
        "iou_threshold": float(iou_thresh),
        "source_annotation": str(annotation_path.as_posix()),
        "source_image_dir": str(image_dir.as_posix()),
        "source_hardcase_summary": str(hardcase_path.as_posix()),
        "source_predictions": str(predictions_path.as_posix()) if prediction_map else None,
        "summary": {
            "total_images_in_annotation": len(images),
            "total_annotations": len(annotations),
            "images_with_small_objects": len(report_images),
            "small_objects_total": total_small_objects,
            "hardcase_images_total": len(hardcase_items),
            "small_object_images_in_hardcase": hardcase_overlap_count,
            "small_object_hardcase_images_with_fn": hardcase_overlap_with_fn_count,
            "images_with_missed_small_objects": images_with_missed_small_objects,
            "missed_small_objects_total": total_missed_small_objects,
            "missed_small_object_images_in_hardcase": hardcase_overlap_with_missed_small_objects,
        },
        "images": report_images,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract images that contain small objects and compare them with hardcase results."
    )
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--hardcase", type=Path, default=DEFAULT_HARDCASE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--area-threshold", type=float, default=1000.0)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_small_object_report(
        annotation_path=args.annotation,
        image_dir=args.image_dir,
        hardcase_path=args.hardcase,
        output_dir=args.output_dir,
        area_threshold=args.area_threshold,
        predictions_path=args.predictions,
        iou_thresh=args.iou_thresh,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    print(f"Saved JSON: {args.output_json}")
    print(f"Images with small objects: {summary['images_with_small_objects']}")
    print(f"Total small objects: {summary['small_objects_total']}")
    print(f"Small-object images in hardcase summary: {summary['small_object_images_in_hardcase']}")
    print(f"Small-object hardcase images with FN > 0: {summary['small_object_hardcase_images_with_fn']}")
    if report["source_predictions"]:
        print(f"Images with missed small objects: {summary['images_with_missed_small_objects']}")
        print(f"Missed small objects total: {summary['missed_small_objects_total']}")
        print(
            "Missed small-object images that also appear in hardcase summary: "
            f"{summary['missed_small_object_images_in_hardcase']}"
        )


if __name__ == "__main__":
    main()
