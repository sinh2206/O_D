from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from utils.config import (
    AGNOSTIC_NMS_IOU_THRESH,
    CLASS_NAMES,
    CONF_THRESH,
    CROSS_CLASS_CONTAIN_THRESH,
    CROSS_CLASS_IOU_THRESH,
    IMG_SIZE,
    MEAN,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    SAME_CLASS_CONTAIN_THRESH,
    STD,
)
from utils.image_ops import enhance_low_light_bgr
from utils.model import AnchorFreeDetector
from utils.nms import LetterboxMeta, postprocess_batch

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def load_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_topk_images_by_objects(annotation_path: Path, image_dir: Path, top_k: int) -> List[Path]:
    data = load_annotation(annotation_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    count_by_image: Dict[str, int] = defaultdict(int)
    for ann in annotations:
        image_id = str(ann.get("image_id", ""))
        if image_id:
            count_by_image[image_id] += 1

    rows: List[Tuple[int, str, Path]] = []
    for im in images:
        image_id = str(im.get("id", ""))
        file_name = Path(str(im.get("file_name", image_id))).name
        count = int(count_by_image.get(image_id, 0))
        p = image_dir / file_name
        if not p.exists():
            p = image_dir / image_id
        if p.exists() and p.suffix.lower() in VALID_EXTS:
            rows.append((count, image_id, p))

    rows.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in rows[: max(0, top_k)]]


def build_gt_from_annotation(annotation_path: Path, image_dir: Path) -> Tuple[Dict[str, List[dict]], Dict[str, Path]]:
    data = load_annotation(annotation_path)
    classes = set(data.get("classes", CLASS_NAMES))
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    gt_map: Dict[str, List[dict]] = defaultdict(list)
    path_map: Dict[str, Path] = {}

    for im in images:
        image_id = str(im.get("id", ""))
        file_name = Path(str(im.get("file_name", image_id))).name
        p = image_dir / file_name
        if not p.exists():
            p = image_dir / image_id
        if p.exists() and p.suffix.lower() in VALID_EXTS:
            path_map[image_id] = p

    for ann in annotations:
        image_id = str(ann.get("image_id", ""))
        cls_name = str(ann.get("class", ""))
        if image_id not in path_map:
            continue
        if cls_name not in classes:
            continue
        b = ann.get("bbox", [0, 0, 0, 0])
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        if x2 <= x1 or y2 <= y1:
            continue
        gt_map[image_id].append({"class": cls_name, "bbox": [x1, y1, x2, y2]})

    return gt_map, path_map


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    iw = max(0.0, xx2 - xx1)
    ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter + 1e-9
    return float(inter / denom)


def compute_image_error(
    pred_boxes: List[dict],
    gt_boxes: List[dict],
    iou_thr: float = 0.5,
) -> Dict[str, float]:
    pred = sorted(pred_boxes, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    gt_used = [False] * len(gt_boxes)

    tp = 0
    fp = 0
    loc_pen = 0.0
    cls_miss = 0

    for p in pred:
        p_cls = str(p.get("class", ""))
        p_box = p.get("bbox", [0, 0, 0, 0])

        best_iou = 0.0
        best_idx = -1
        for i, g in enumerate(gt_boxes):
            if gt_used[i]:
                continue
            iou = box_iou(p_box, g.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx < 0:
            fp += 1
            continue

        g_cls = str(gt_boxes[best_idx].get("class", ""))
        if best_iou >= iou_thr and p_cls == g_cls:
            gt_used[best_idx] = True
            tp += 1
            loc_pen += max(0.0, 1.0 - best_iou)
        elif best_iou >= iou_thr and p_cls != g_cls:
            # strong overlap but wrong class: count as heavy mistake
            gt_used[best_idx] = True
            fp += 1
            cls_miss += 1
        else:
            fp += 1

    fn = sum(1 for x in gt_used if not x)
    denom = max(1.0, float(len(gt_boxes)))
    err = (fp + fn + 0.5 * loc_pen + 1.5 * cls_miss) / denom
    return {
        "error_score": float(err),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "cls_miss": float(cls_miss),
    }


def select_worst_predictions(
    predictions: List[dict],
    gt_map: Dict[str, List[dict]],
    top_k: int,
    iou_thr: float = 0.5,
) -> List[dict]:
    rows: List[Tuple[float, dict]] = []
    for pred in predictions:
        image_id = str(pred.get("image_id", ""))
        gt_boxes = gt_map.get(image_id, [])
        stat = compute_image_error(pred.get("boxes", []), gt_boxes, iou_thr=iou_thr)
        item = dict(pred)
        item["error"] = stat
        rows.append((float(stat["error_score"]), item))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in rows[: max(0, top_k)]]


def draw_prediction(
    image_bgr: np.ndarray,
    boxes: Sequence[dict],
    class_names: Sequence[str],
    gt_boxes: Optional[Sequence[dict]] = None,
    error_info: Optional[dict] = None,
) -> np.ndarray:
    out = image_bgr.copy()
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    if gt_boxes is not None:
        for g in gt_boxes:
            cls_name = str(g.get("class", "gt"))
            x1, y1, x2, y2 = [int(round(v)) for v in g.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), 2)
            cv2.putText(out, f"GT:{cls_name}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 220, 40), 1, cv2.LINE_AA)

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

    if error_info is not None:
        t = f"err={error_info.get('error_score', 0):.2f} fp={int(error_info.get('fp', 0))} fn={int(error_info.get('fn', 0))} cm={int(error_info.get('cls_miss', 0))}"
        cv2.putText(out, t, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
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
    agnostic_nms_thresh: float,
    cross_class_iou_thresh: float,
    cross_class_contain_thresh: float,
    same_class_contain_thresh: float,
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

        images = torch.stack(tensors, dim=0).to(device, non_blocking=True)
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
            agnostic_nms_thresh=agnostic_nms_thresh,
            same_class_contain_thresh=same_class_contain_thresh,
            cross_class_iou_thresh=cross_class_iou_thresh,
            cross_class_contain_thresh=cross_class_contain_thresh,
        )
        results.extend(batch_results)

    return results


def save_preview_images(
    predictions: List[dict],
    image_path_map: Dict[str, Path],
    preview_dir: Path,
    limit: int,
    class_names: Sequence[str],
    gt_map: Optional[Dict[str, List[dict]]] = None,
) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for pred in predictions[:limit]:
        image_id = pred["image_id"]
        img_path = image_path_map.get(image_id)
        if img_path is None:
            continue
        image = imread_unicode(img_path)
        if image is None:
            continue

        vis = draw_prediction(
            image,
            pred.get("boxes", []),
            class_names=class_names,
            gt_boxes=(gt_map.get(image_id, []) if gt_map is not None else None),
            error_info=pred.get("error"),
        )
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
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, list(classes), img_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with anchor-free detector and export JSON.")
    parser.add_argument("--image_dir", type=Path, default=None, help="Predict all images in this folder.")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument(
        "--checkpoint",
        "--model_path",
        dest="checkpoint",
        type=Path,
        default=Path("models/best.pth"),
        help="Path to trained model checkpoint (.pth). '--model_path' is kept as a backward-compatible alias.",
    )
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--nms_thresh", type=float, default=NMS_IOU_THRESH)
    parser.add_argument("--agnostic_nms_thresh", type=float, default=AGNOSTIC_NMS_IOU_THRESH)
    parser.add_argument("--same_class_contain_thresh", type=float, default=SAME_CLASS_CONTAIN_THRESH)
    parser.add_argument("--cross_class_iou_thresh", type=float, default=CROSS_CLASS_IOU_THRESH)
    parser.add_argument("--cross_class_contain_thresh", type=float, default=CROSS_CLASS_CONTAIN_THRESH)
    parser.add_argument("--preview_dir", type=Path, default=Path("results"))
    parser.add_argument("--preview_count", type=int, default=50)
    parser.add_argument(
        "--error_top_k",
        type=int,
        default=0,
        help="If >0, rank images by prediction-vs-GT error and export top-K worst images.",
    )
    parser.add_argument("--error_iou", type=float, default=0.5, help="IoU threshold used for error ranking.")
    parser.add_argument(
        "--top_k_objects",
        type=int,
        default=0,
        help="If >0, select top-K images with most objects from train/val annotations (split ~50/50).",
    )
    parser.add_argument("--train_data", type=Path, default=Path("public/annotations/train.json"))
    parser.add_argument("--val_data", type=Path, default=Path("public/annotations/val.json"))
    parser.add_argument("--train_image_dir", type=Path, default=Path("public/train/images"))
    parser.add_argument("--val_image_dir", type=Path, default=Path("public/val/images"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.image_dir is None and int(args.top_k_objects) <= 0 and int(args.error_top_k) <= 0:
        args.error_top_k = 50

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, ckpt_classes, ckpt_img_size = load_checkpoint_model(args.checkpoint, device=device)
    class_names = ckpt_classes if ckpt_classes else CLASS_NAMES
    img_size = args.img_size if args.img_size > 0 else ckpt_img_size

    image_paths: List[Path] = []
    gt_map_all: Dict[str, List[dict]] = {}
    image_path_map: Dict[str, Path] = {}
    use_error_ranking = int(args.error_top_k) > 0

    if use_error_ranking:
        if not args.train_data.exists() or not args.val_data.exists():
            raise FileNotFoundError("When --error_top_k > 0, --train_data and --val_data must exist.")
        if not args.train_image_dir.exists() or not args.val_image_dir.exists():
            raise FileNotFoundError("When --error_top_k > 0, --train_image_dir and --val_image_dir must exist.")

        gt_train, path_train = build_gt_from_annotation(args.train_data, args.train_image_dir)
        gt_val, path_val = build_gt_from_annotation(args.val_data, args.val_image_dir)

        gt_map_all.update(gt_train)
        gt_map_all.update(gt_val)
        image_path_map.update(path_train)
        image_path_map.update(path_val)
        image_paths = list(image_path_map.values())

    elif int(args.top_k_objects) > 0:
        if not args.train_data.exists() or not args.val_data.exists():
            raise FileNotFoundError("When --top_k_objects > 0, --train_data and --val_data must exist.")
        if not args.train_image_dir.exists() or not args.val_image_dir.exists():
            raise FileNotFoundError("When --top_k_objects > 0, --train_image_dir and --val_image_dir must exist.")

        top_k = int(args.top_k_objects)
        top_k_train = top_k // 2
        top_k_val = top_k - top_k_train
        train_paths = select_topk_images_by_objects(args.train_data, args.train_image_dir, top_k=top_k_train)
        val_paths = select_topk_images_by_objects(args.val_data, args.val_image_dir, top_k=top_k_val)
        image_paths = train_paths + val_paths
    else:
        if args.image_dir is None:
            raise ValueError("Please provide --image_dir, or set --top_k_objects > 0.")
        if not args.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        image_paths = collect_images(args.image_dir)

    if not image_paths:
        raise ValueError("No images selected for inference.")

    # Keep unique image names while preserving first occurrence.
    unique_by_name: Dict[str, Path] = {}
    for p in image_paths:
        unique_by_name.setdefault(p.name, p)
    image_paths = list(unique_by_name.values())
    if not image_path_map:
        image_path_map = dict(unique_by_name)

    predictions = run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=max(1, args.batch_size),
        img_size=img_size,
        conf_thresh=float(args.conf_thresh),
        nms_thresh=float(args.nms_thresh),
        class_names=class_names,
        agnostic_nms_thresh=float(args.agnostic_nms_thresh),
        same_class_contain_thresh=float(args.same_class_contain_thresh),
        cross_class_iou_thresh=float(args.cross_class_iou_thresh),
        cross_class_contain_thresh=float(args.cross_class_contain_thresh),
    )

    if use_error_ranking:
        predictions = select_worst_predictions(
            predictions=predictions,
            gt_map=gt_map_all,
            top_k=int(args.error_top_k),
            iou_thr=float(args.error_iou),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    saved = save_preview_images(
        predictions=predictions,
        image_path_map=image_path_map,
        preview_dir=args.preview_dir,
        limit=max(0, int(args.error_top_k) if use_error_ranking else args.preview_count),
        class_names=class_names,
        gt_map=gt_map_all if use_error_ranking else None,
    )

    print(f"Device: {device}")
    print(f"Predicted images: {len(predictions)}")
    print(f"Saved JSON: {args.output}")
    print(f"Saved preview images: {saved} -> {args.preview_dir}")


if __name__ == "__main__":
    main()
