from __future__ import annotations

import argparse
import json
import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Record:
    image_id: str
    file_name: str
    boxes: List[List[float]]
    labels: List[str]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path: Path, image_bgr: np.ndarray) -> bool:
    ext = path.suffix.lower()
    if ext not in VALID_EXTS:
        ext = ".jpg"
        path = path.with_suffix(ext)
    ok, enc = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    enc.tofile(str(path))
    return True


def make_pad_if_needed(img_size: int):
    params = inspect.signature(A.PadIfNeeded.__init__).parameters
    kwargs = {
        "min_height": img_size,
        "min_width": img_size,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 1.0,
    }
    if "fill" in params:
        kwargs["fill"] = (114, 114, 114)
        if "fill_mask" in params:
            kwargs["fill_mask"] = 0
    elif "value" in params:
        kwargs["value"] = (114, 114, 114)
        if "mask_value" in params:
            kwargs["mask_value"] = 0
    return A.PadIfNeeded(**kwargs)


def collect_records(annotation_path: Path, image_dir: Path) -> Tuple[List[Record], List[str]]:
    data = load_json(annotation_path)
    classes = data.get("classes", [])

    img_map = {im["id"]: im for im in data.get("images", [])}
    ann_map: Dict[str, List[dict]] = {iid: [] for iid in img_map.keys()}
    for ann in data.get("annotations", []):
        iid = ann.get("image_id")
        if iid in ann_map:
            ann_map[iid].append(ann)

    records: List[Record] = []
    for iid, meta in img_map.items():
        fname = Path(meta.get("file_name", iid)).name
        img_path = image_dir / fname
        if not img_path.exists():
            continue

        boxes: List[List[float]] = []
        labels: List[str] = []
        for ann in ann_map.get(iid, []):
            if ann.get("class") not in classes:
                continue
            x1, y1, x2, y2 = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(str(ann["class"]))

        records.append(Record(image_id=iid, file_name=fname, boxes=boxes, labels=labels))

    if not records:
        raise ValueError("No records found. Check annotation path and image directory.")
    return records, classes


def scale_boxes_xyxy(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] *= float(sx)
    out[:, [1, 3]] *= float(sy)
    return out


def clip_filter_boxes(boxes: np.ndarray, labels: List[str], w: int, h: int, min_area: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    if boxes.size == 0:
        return boxes.reshape(0, 4), []

    b = boxes.copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(w))
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(h))

    keep = []
    for i in range(b.shape[0]):
        x1, y1, x2, y2 = b[i].tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        keep.append(i)

    if not keep:
        return np.zeros((0, 4), dtype=np.float32), []

    return b[keep].astype(np.float32), [labels[i] for i in keep]


def draw_boxes(image_bgr: np.ndarray, boxes: np.ndarray, labels: List[str], classes: List[str]) -> np.ndarray:
    out = image_bgr.copy()
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        idx = cls_to_idx.get(label, 0)
        color = (
            int((53 * (idx + 1)) % 255),
            int((97 * (idx + 1)) % 255),
            int((193 * (idx + 1)) % 255),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def load_resized_record(record: Record, image_dir: Path, target_size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    img_path = image_dir / record.file_name
    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h0, w0 = img.shape[:2]
    scale = min(float(target_size) / max(w0, 1), float(target_size) / max(h0, 1))
    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_r = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    img_r[dy : dy + new_h, dx : dx + new_w] = resized

    if len(record.boxes) == 0:
        return img_r, np.zeros((0, 4), dtype=np.float32), []

    boxes = np.asarray(record.boxes, dtype=np.float32)
    boxes = scale_boxes_xyxy(boxes, sx=scale, sy=scale)
    boxes[:, [0, 2]] += float(dx)
    boxes[:, [1, 3]] += float(dy)
    boxes, labels = clip_filter_boxes(boxes, list(record.labels), w=target_size, h=target_size, min_area=1.0)
    return img_r, boxes, labels


def build_mosaic(records: List[Record], image_dir: Path, img_size: int, primary_idx: int, rng: random.Random):
    s = img_size
    mosaic = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)

    indices = [primary_idx] + [rng.randrange(len(records)) for _ in range(3)]
    xc = rng.randint(s // 2, (3 * s) // 2)
    yc = rng.randint(s // 2, (3 * s) // 2)

    all_boxes: List[np.ndarray] = []
    all_labels: List[str] = []

    for i, idx in enumerate(indices):
        rec = records[idx]
        img, boxes, labels = load_resized_record(rec, image_dir=image_dir, target_size=s)
        h, w = img.shape[:2]

        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

        mosaic[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        if boxes.size > 0:
            b = boxes.copy()
            b[:, [0, 2]] += (x1a - x1b)
            b[:, [1, 3]] += (y1a - y1b)
            all_boxes.append(b)
            all_labels.extend(labels)

    if all_boxes:
        boxes_cat = np.concatenate(all_boxes, axis=0).astype(np.float32)
        boxes_cat, labels_out = clip_filter_boxes(boxes_cat, all_labels, w=2 * s, h=2 * s, min_area=1.0)
    else:
        boxes_cat = np.zeros((0, 4), dtype=np.float32)
        labels_out = []

    return mosaic, boxes_cat, labels_out


def maybe_mixup(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    records: List[Record],
    image_dir: Path,
    img_size: int,
    rng: random.Random,
    p: float = 0.5,
):
    if rng.random() >= p:
        return image, boxes, labels

    mix_idx = rng.randrange(len(records))
    mix_image, mix_boxes, mix_labels = build_mosaic(
        records=records,
        image_dir=image_dir,
        img_size=img_size,
        primary_idx=mix_idx,
        rng=rng,
    )

    lam = float(np.random.beta(1.5, 1.5))
    blended = (lam * image.astype(np.float32) + (1.0 - lam) * mix_image.astype(np.float32)).clip(0, 255).astype(np.uint8)

    if boxes.size == 0:
        new_boxes = mix_boxes
        new_labels = list(mix_labels)
    elif mix_boxes.size == 0:
        new_boxes = boxes
        new_labels = list(labels)
    else:
        new_boxes = np.concatenate([boxes, mix_boxes], axis=0)
        new_labels = list(labels) + list(mix_labels)

    new_boxes, new_labels = clip_filter_boxes(new_boxes, new_labels, w=2 * img_size, h=2 * img_size, min_area=1.0)
    return blended, new_boxes, new_labels


def build_train_transform(img_size: int) -> A.Compose:
    try:
        affine = A.Affine(
            scale=(0.85, 1.25),
            translate_percent=(-0.04, 0.04),
            rotate=(-4, 4),
            shear=(-1.0, 1.0),
            border_mode=cv2.BORDER_CONSTANT,
            fill=114,
            p=0.25,
        )
    except TypeError:
        affine = A.Affine(
            scale=(0.85, 1.25),
            translate_percent=(-0.04, 0.04),
            rotate=(-4, 4),
            shear=(-1.0, 1.0),
            mode=cv2.BORDER_CONSTANT,
            cval=114,
            p=0.25,
        )

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            make_pad_if_needed(img_size),
            A.HorizontalFlip(p=0.5),
            affine,
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.2,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
            clip=True,
        ),
    )


def denormalize_to_bgr(tensor_chw: np.ndarray) -> np.ndarray:
    img = tensor_chw.transpose(1, 2, 0)
    img = img * np.array(STD, dtype=np.float32) + np.array(MEAN, dtype=np.float32)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def export_augmented_samples(
    records: List[Record],
    classes: List[str],
    image_dir: Path,
    output_dir: Path,
    img_size: int,
    num_samples: int,
    include_image: Optional[str],
    mixup_prob: float,
    seed: int,
) -> int:
    rng = random.Random(seed)
    np.random.seed(seed)

    tfm = build_train_transform(img_size=img_size)

    out_ori = output_dir / "original"
    out_aug = output_dir / "augmented"
    out_cmp = output_dir / "compare"
    out_ori.mkdir(parents=True, exist_ok=True)
    out_aug.mkdir(parents=True, exist_ok=True)
    out_cmp.mkdir(parents=True, exist_ok=True)

    idx_order = list(range(len(records)))
    rng.shuffle(idx_order)

    selected: List[int] = []
    if include_image:
        for i, rec in enumerate(records):
            if rec.file_name == include_image or rec.image_id == include_image:
                selected.append(i)
                break

    for i in idx_order:
        if len(selected) >= num_samples:
            break
        if i not in selected:
            selected.append(i)

    saved = 0
    for rank, idx in enumerate(selected[:num_samples], start=1):
        rec = records[idx]

        ori_img, ori_boxes, ori_labels = load_resized_record(rec, image_dir=image_dir, target_size=img_size)
        ori_vis = draw_boxes(ori_img, ori_boxes, ori_labels, classes)

        mosaic_img, boxes, labels = build_mosaic(records, image_dir=image_dir, img_size=img_size, primary_idx=idx, rng=rng)
        mosaic_img, boxes, labels = maybe_mixup(
            image=mosaic_img,
            boxes=boxes,
            labels=labels,
            records=records,
            image_dir=image_dir,
            img_size=img_size,
            rng=rng,
            p=mixup_prob,
        )

        transformed = tfm(image=mosaic_img, bboxes=boxes.tolist(), class_labels=labels)
        aug_tensor = transformed["image"].detach().cpu().numpy()
        aug_boxes = np.asarray(transformed["bboxes"], dtype=np.float32) if len(transformed["bboxes"]) > 0 else np.zeros((0, 4), dtype=np.float32)
        aug_labels = [str(x) for x in transformed["class_labels"]]

        aug_img = denormalize_to_bgr(aug_tensor)
        aug_vis = draw_boxes(aug_img, aug_boxes, aug_labels, classes)

        stem = f"{rank:03d}_{Path(rec.file_name).stem}"
        ori_path = out_ori / f"{stem}.jpg"
        aug_path = out_aug / f"{stem}.jpg"
        cmp_path = out_cmp / f"{stem}.jpg"

        ok1 = imwrite_unicode(ori_path, ori_vis)
        ok2 = imwrite_unicode(aug_path, aug_vis)
        comp = np.concatenate([ori_vis, aug_vis], axis=1)
        ok3 = imwrite_unicode(cmp_path, comp)

        if ok1 and ok2 and ok3:
            saved += 1

    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Anchor-Free augmentation previews: Mosaic -> Affine -> HSV -> Flip.")
    p.add_argument("--annotation", type=Path, default=Path("public/annotations/val.json"))
    p.add_argument("--image_dir", type=Path, default=Path("public/val/images"))
    p.add_argument("--output_dir", type=Path, default=Path("results"))
    p.add_argument("--img_size", type=int, default=320)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--include", type=str, default="img_326a06a3c024.jpg")
    p.add_argument("--mixup_prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records, classes = collect_records(args.annotation, args.image_dir)
    saved = export_augmented_samples(
        records=records,
        classes=classes,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        num_samples=args.num_samples,
        include_image=args.include,
        mixup_prob=float(args.mixup_prob),
        seed=int(args.seed),
    )

    print(f"Saved {saved} sample pairs to {args.output_dir}")
    print(f"  - original:  {args.output_dir / 'original'}")
    print(f"  - augmented: {args.output_dir / 'augmented'}")
    print(f"  - compare:   {args.output_dir / 'compare'}")


if __name__ == "__main__":
    main()
