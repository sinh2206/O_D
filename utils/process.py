from __future__ import annotations

"""Dataset conversion from the project's JSON format to Ultralytics YOLO format."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from .config import FALLBACK_CLASS_NAMES
from .image_ops import imread_unicode
from .runtime import ensure_dir, hardlink_or_copy, load_json, reset_dir


@dataclass
class DetectionRecord:
    """One image plus its object boxes in xyxy format."""

    image_id: str
    file_name: str
    image_path: Path
    width: int
    height: int
    boxes: List[List[float]]
    labels: List[int]


def load_detection_records(annotation_path: Path, image_dir: Path, class_names: Sequence[str] | None = None) -> Tuple[List[DetectionRecord], List[str]]:
    """Parse the project annotations into normalized image-level records."""

    data = load_json(annotation_path)
    names = list(class_names) if class_names else list(data.get("classes", []) or FALLBACK_CLASS_NAMES)
    class_to_idx = {str(name): idx for idx, name in enumerate(names)}

    annotations_by_image: Dict[str, List[dict]] = {}
    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id", ""))
        annotations_by_image.setdefault(image_id, []).append(ann)

    records: List[DetectionRecord] = []
    for image_info in data.get("images", []):
        image_id = str(image_info.get("id", ""))
        file_name = Path(str(image_info.get("file_name", image_id))).name
        image_path = image_dir / file_name
        if not image_path.exists():
            fallback = image_dir / image_id
            if fallback.exists():
                image_path = fallback
            else:
                continue

        width = int(image_info.get("width", 0) or 0)
        height = int(image_info.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            image = imread_unicode(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]

        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in annotations_by_image.get(image_id, []):
            cls_name = str(ann.get("class", ""))
            if cls_name not in class_to_idx:
                continue
            bbox = ann.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(class_to_idx[cls_name])

        records.append(
            DetectionRecord(
                image_id=image_id,
                file_name=file_name,
                image_path=image_path,
                width=width,
                height=height,
                boxes=boxes,
                labels=labels,
            )
        )

    if not records:
        raise ValueError(f"No valid records were found in {annotation_path}")
    return records, names


def xyxy_to_yolo(box: Sequence[float], image_w: int, image_h: int) -> Tuple[float, float, float, float]:
    """Convert one xyxy box to normalized YOLO `(cx, cy, w, h)` format."""

    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(image_w), x1))
    y1 = max(0.0, min(float(image_h), y1))
    x2 = max(0.0, min(float(image_w), x2))
    y2 = max(0.0, min(float(image_h), y2))

    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / float(image_w), cy / float(image_h), bw / float(image_w), bh / float(image_h)


def write_yolo_label_file(path: Path, labels: Sequence[int], boxes: Sequence[Sequence[float]], image_w: int, image_h: int) -> Path:
    """Write one YOLO label file from class ids and xyxy boxes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for cls_id, box in zip(labels, boxes):
        cx, cy, bw, bh = xyxy_to_yolo(box, image_w=image_w, image_h=image_h)
        lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def prepare_split(records: Sequence[DetectionRecord], split_name: str, output_dir: Path) -> None:
    """Create YOLO-style `images/<split>` and `labels/<split>` folders."""

    image_out_dir = ensure_dir(output_dir / "images" / split_name)
    label_out_dir = ensure_dir(output_dir / "labels" / split_name)

    for record in records:
        linked_image = image_out_dir / record.file_name
        hardlink_or_copy(record.image_path, linked_image)
        label_path = label_out_dir / f"{Path(record.file_name).stem}.txt"
        write_yolo_label_file(label_path, record.labels, record.boxes, image_w=record.width, image_h=record.height)


def build_yolo_dataset(
    train_data: Path,
    val_data: Path,
    train_image_dir: Path,
    val_image_dir: Path,
    output_dir: Path,
) -> Tuple[Path, List[str]]:
    """Generate a full Ultralytics YOLO detection dataset tree plus `data.yaml`."""

    output_dir = reset_dir(output_dir)
    train_records, class_names = load_detection_records(train_data, train_image_dir)
    val_records, _ = load_detection_records(val_data, val_image_dir, class_names=class_names)

    prepare_split(train_records, "train", output_dir)
    prepare_split(val_records, "val", output_dir)

    yaml_path = output_dir / "data.yaml"
    yaml_payload = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path, class_names
