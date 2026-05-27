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
    CLASS_CONF_THRESH,
    CLASS_NAMES,
    CONF_THRESH,
    IMG_SIZE,
    MEAN,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    STD,
)
from utils.model import AnchorFreeDetector
from utils.nms import LetterboxMeta, postprocess_batch

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_VAL_IMAGE_DIR = Path("public/val/images")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_VAL_ANNOTATION = Path("public/annotations/val.json")


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
    if float(gray.mean()) >= 82.0:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** 0.82) * 255.0 for i in range(256)], dtype=np.float32)
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


def _is_fully_inside(inner: List[int], outer: List[int]) -> bool:
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
        bb = box.get("bbox", [])
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        bb = [int(v) for v in bb]
        drop = False
        for k in kept:
            if str(k.get("class", "")) != cls:
                continue
            kb = [int(v) for v in k["bbox"]]
            if _is_fully_inside(bb, kb) or _is_fully_inside(kb, bb):
                drop = True
                break
        if not drop:
            keep_box = dict(box)
            keep_box["bbox"] = bb
            kept.append(keep_box)
    return kept


def sanitize_predictions_for_export(
    predictions: List[dict],
    image_dir: Path,
    class_names: Sequence[str],
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

            ix1 = int(math.floor(x1))
            iy1 = int(math.floor(y1))
            ix2 = int(math.ceil(x2))
            iy2 = int(math.ceil(y2))

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

            if ix2 > w:
                ix2 = w
            if iy2 > h:
                iy2 = h
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            cleaned.append(
                {
                    "class": cls_name,
                    "confidence": float(conf),
                    "bbox": [int(ix1), int(iy1), int(ix2), int(iy2)],
                }
            )

        cleaned = _suppress_same_class_contained_int(cleaned)
        out.append({"image_id": image_id, "boxes": cleaned})

    return out


def clean_preview_dir(preview_dir: Path, safe_root: Optional[Path] = None) -> None:
    if not preview_dir.exists():
        return
    resolved_preview = preview_dir.resolve()
    if safe_root is not None:
        resolved_safe_root = safe_root.resolve()
        if resolved_preview != resolved_safe_root:
            raise ValueError(
                f"Refusing to clean preview_dir outside safe root. preview_dir={preview_dir}, safe_root={safe_root}"
            )
    for p in resolved_preview.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            p.unlink()


def draw_prediction(image_bgr: np.ndarray, boxes: Sequence[dict], class_names: Sequence[str]) -> np.ndarray:
    out = image_bgr.copy()
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    for obj in boxes:
        cls_name = str(obj["class"])
        score = float(obj["confidence"])
        x1, y1, x2, y2 = [int(round(v)) for v in obj["bbox"]]
        idx = cls_to_idx.get(cls_name, 0)
        color = (
            int((53 * (idx + 1)) % 255),
            int((97 * (idx + 1)) % 255),
            int((193 * (idx + 1)) % 255),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}:{score:.2f}"
        cv2.putText(out, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


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
            thr = th_map.get(c, 0.5)
            if s >= thr:
                keep.append(box)
        out.append({"image_id": pred.get("image_id"), "boxes": keep})
    return out


def suppress_chair_inside_person(predictions: List[dict], iou_thresh: float) -> List[dict]:
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
            for p in persons:
                if box_iou(chair_box, p.get("bbox", [0, 0, 0, 0])) >= float(iou_thresh):
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
) -> List[dict]:
    results: List[dict] = []
    amp_enabled = device.type == "cuda"

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors: List[torch.Tensor] = []
        metas: List[LetterboxMeta] = []
        image_ids: List[str] = []

        for p in batch_paths:
            image = imread_unicode(p)
            if image is None:
                continue
            tensor, meta = letterbox_preprocess(image, img_size=img_size)
            tensors.append(tensor)
            metas.append(meta)
            image_ids.append(p.name)

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
            center_combine="mul",
            min_box_size=2.0,
        )
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


def load_ground_truth_map(annotation_path: Path, class_names: Sequence[str]) -> Dict[str, List[dict]]:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Validation annotation not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    valid_classes = set(class_names)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    gt_by_id: Dict[str, List[dict]] = {}
    file_name_to_id: Dict[str, str] = {}

    for im in images:
        iid = str(im.get("id", "")).strip()
        if not iid:
            continue
        gt_by_id.setdefault(iid, [])
        fname = Path(str(im.get("file_name", ""))).name
        if fname:
            file_name_to_id[fname] = iid

    for ann in annotations:
        iid = str(ann.get("image_id", "")).strip()
        cls_name = str(ann.get("class", "")).strip()
        bbox = ann.get("bbox", [])
        if not iid or cls_name not in valid_classes:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        gt_by_id.setdefault(iid, []).append({"class": cls_name, "bbox": [x1, y1, x2, y2]})

    out: Dict[str, List[dict]] = {}
    for iid, items in gt_by_id.items():
        out[iid] = list(items)
    for fname, iid in file_name_to_id.items():
        out[fname] = list(gt_by_id.get(iid, []))

    return out


def _match_class_greedy(pred_cls: List[dict], gt_cls: List[dict], iou_thresh: float) -> Tuple[int, int, int]:
    preds = sorted(pred_cls, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    matched = [False] * len(gt_cls)
    tp = 0
    fp = 0

    for p in preds:
        pb = p.get("bbox", [0, 0, 0, 0])
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_cls):
            if matched[j]:
                continue
            iou = box_iou(pb, g.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= float(iou_thresh):
            matched[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = len(gt_cls) - tp
    return tp, fp, fn


def compute_image_error_stats(
    pred_boxes: List[dict],
    gt_boxes: List[dict],
    class_names: Sequence[str],
    iou_thresh: float,
) -> dict:
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    for cls_name in class_names:
        pred_cls = [b for b in pred_boxes if str(b.get("class", "")) == cls_name]
        gt_cls = [b for b in gt_boxes if str(b.get("class", "")) == cls_name]
        tp, fp, fn = _match_class_greedy(pred_cls=pred_cls, gt_cls=gt_cls, iou_thresh=iou_thresh)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    denom = max(1, len(gt_boxes) + len(pred_boxes))
    error_ratio = float(fp_sum + fn_sum) / float(denom)
    return {
        "tp": int(tp_sum),
        "fp": int(fp_sum),
        "fn": int(fn_sum),
        "num_gt": int(len(gt_boxes)),
        "num_pred": int(len(pred_boxes)),
        "error_ratio": float(error_ratio),
    }


def draw_gt_and_pred(
    image_bgr: np.ndarray,
    gt_boxes: List[dict],
    pred_boxes: List[dict],
    stats: dict,
) -> np.ndarray:
    out = image_bgr.copy()

    for g in gt_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in g.get("bbox", [0, 0, 0, 0])]
        cls_name = str(g.get("class", ""))
        cv2.rectangle(out, (x1, y1), (x2, y2), (20, 20, 235), 2)
        cv2.putText(out, f"GT:{cls_name}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 235), 1, cv2.LINE_AA)

    for p in pred_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in p.get("bbox", [0, 0, 0, 0])]
        cls_name = str(p.get("class", ""))
        conf = float(p.get("confidence", 0.0))
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 210, 40), 2)
        cv2.putText(
            out,
            f"PR:{cls_name}:{conf:.2f}",
            (x1, min(out.shape[0] - 6, y2 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 210, 40),
            1,
            cv2.LINE_AA,
        )

    summary = (
        f"err={stats['error_ratio']:.3f} "
        f"tp={stats['tp']} fp={stats['fp']} fn={stats['fn']} "
        f"gt={stats['num_gt']} pred={stats['num_pred']}"
    )
    cv2.rectangle(out, (0, 0), (min(out.shape[1] - 1, 460), 22), (0, 0, 0), -1)
    cv2.putText(out, summary, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def clean_image_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            p.unlink()


def save_top_error_images(
    predictions: List[dict],
    image_dir: Path,
    class_names: Sequence[str],
    val_annotation: Path,
    output_dir: Path,
    top_k: int = 50,
    iou_thresh: float = 0.5,
) -> int:
    gt_map = load_ground_truth_map(annotation_path=val_annotation, class_names=class_names)

    ranked: List[dict] = []
    for pred in predictions:
        image_id = str(pred.get("image_id", ""))
        if not image_id:
            continue
        gt_boxes = gt_map.get(image_id, [])
        pred_boxes = list(pred.get("boxes", []))
        stats = compute_image_error_stats(
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            class_names=class_names,
            iou_thresh=float(iou_thresh),
        )
        ranked.append({"image_id": image_id, "gt_boxes": gt_boxes, "pred_boxes": pred_boxes, "stats": stats})

    ranked.sort(
        key=lambda x: (
            -float(x["stats"]["error_ratio"]),
            -int(x["stats"]["fn"]),
            -int(x["stats"]["fp"]),
            str(x["image_id"]),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_image_dir(output_dir)

    saved = 0
    for idx, item in enumerate(ranked[: max(0, int(top_k))], start=1):
        image_id = item["image_id"]
        image = imread_unicode(image_dir / image_id)
        if image is None:
            continue
        vis = draw_gt_and_pred(
            image_bgr=image,
            gt_boxes=item["gt_boxes"],
            pred_boxes=item["pred_boxes"],
            stats=item["stats"],
        )
        out_path = output_dir / f"worst_{idx:03d}_{Path(image_id).name}"
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
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, list(classes), img_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with anchor-free detector and export JSON.")
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_VAL_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "val_predictions.json")
    parser.add_argument(
        "--checkpoint",
        "--model_path",
        dest="checkpoint",
        type=Path,
        default=Path("models/best.pth"),
        help="Path to trained model checkpoint (.pth). '--model_path' is kept as a backward-compatible alias.",
    )
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--nms_thresh", type=float, default=NMS_IOU_THRESH)
    parser.add_argument(
        "--class_conf",
        type=str,
        default=",".join(str(x) for x in CLASS_CONF_THRESH),
        help="Per-class thresholds in CLASS_NAMES order, e.g. '0.38,0.40,0.40,0.40,0.72'",
    )
    parser.add_argument("--chair_suppress_iou", type=float, default=CHAIR_SUPPRESS_WITH_PERSON_IOU)
    parser.add_argument("--preview_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--preview_count", type=int, default=50)
    parser.add_argument("--val_annotation", type=Path, default=DEFAULT_VAL_ANNOTATION)
    parser.add_argument("--worst_dir", type=Path, default=DEFAULT_RESULTS_DIR / "worst50")
    parser.add_argument("--worst_count", type=int, default=50)
    parser.add_argument("--worst_iou_thresh", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow_non_val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    expected_val_dir = DEFAULT_VAL_IMAGE_DIR.resolve()
    expected_results_dir = DEFAULT_RESULTS_DIR.resolve()
    actual_image_dir = args.image_dir.resolve()
    if (not args.allow_non_val) and (actual_image_dir != expected_val_dir):
        raise ValueError(
            f"--image_dir must be '{DEFAULT_VAL_IMAGE_DIR}' to export val results only. "
            f"Got: {args.image_dir}"
        )
    output_parent = args.output.resolve().parent
    preview_dir_resolved = args.preview_dir.resolve()
    if (not args.allow_non_val) and (output_parent != expected_results_dir):
        raise ValueError(
            f"--output must be inside '{DEFAULT_RESULTS_DIR}' to export val results only. "
            f"Got: {args.output}"
        )
    if (not args.allow_non_val) and (preview_dir_resolved != expected_results_dir):
        raise ValueError(
            f"--preview_dir must be '{DEFAULT_RESULTS_DIR}' to export val preview images only. "
            f"Got: {args.preview_dir}"
        )
    worst_dir_resolved = args.worst_dir.resolve()
    if (not args.allow_non_val) and (expected_results_dir not in worst_dir_resolved.parents) and (worst_dir_resolved != expected_results_dir):
        raise ValueError(
            f"--worst_dir must be inside '{DEFAULT_RESULTS_DIR}' to export val error images only. "
            f"Got: {args.worst_dir}"
        )

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
    )
    predictions = apply_class_thresholds(predictions, class_names=class_names, class_conf_thresh=class_conf)
    predictions = suppress_chair_inside_person(predictions, iou_thresh=float(args.chair_suppress_iou))
    predictions = sanitize_predictions_for_export(
        predictions=predictions,
        image_dir=args.image_dir,
        class_names=class_names,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    clean_preview_dir(args.preview_dir, safe_root=expected_results_dir if not args.allow_non_val else None)
    saved = save_preview_images(
        predictions=predictions,
        image_dir=args.image_dir,
        preview_dir=args.preview_dir,
        limit=max(0, args.preview_count),
        class_names=class_names,
    )

    worst_saved = save_top_error_images(
        predictions=predictions,
        image_dir=args.image_dir,
        class_names=class_names,
        val_annotation=args.val_annotation,
        output_dir=args.worst_dir,
        top_k=max(0, args.worst_count),
        iou_thresh=float(args.worst_iou_thresh),
    )

    print(f"Device: {device}")
    print(f"Predicted images: {len(predictions)}")
    print(f"Saved JSON: {args.output}")
    print(f"Saved preview images: {saved} -> {args.preview_dir}")
    print(f"Saved worst-error images: {worst_saved} -> {args.worst_dir}")


if __name__ == "__main__":
    main()
