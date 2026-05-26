from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from utils.config import CLASS_CONF_THRESH, CLASS_NAMES, CONF_THRESH, IMG_SIZE, NMS_IOU_THRESH
from utils.forecast import apply_class_thresholds, load_checkpoint_model, run_inference, save_predictions_json
from utils.model import YOLOv2Detector
from utils.process import draw_prediction, imread_unicode, imwrite_unicode
from utils.runtime import device_summary, resolve_device

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def save_preview_images(predictions: List[dict], image_dir: Path, preview_dir: Path, limit: int, class_names: List[str]) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for pred in predictions[: max(0, int(limit))]:
        image_id = str(pred.get("image_id", ""))
        if not image_id:
            continue
        image = imread_unicode(image_dir / image_id)
        if image is None:
            continue
        vis = draw_prediction(image, pred.get("boxes", []), class_names=class_names)
        if imwrite_unicode(preview_dir / image_id, vis):
            saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with YOLOv2-style detector and export JSON.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument(
        "--checkpoint",
        "--model_path",
        dest="checkpoint",
        type=Path,
        default=Path("models/best.pth"),
        help="Path to trained checkpoint (.pth). '--model_path' is supported as alias.",
    )
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Set <=0 to use img_size from checkpoint.")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH)
    parser.add_argument("--nms_thresh", type=float, default=NMS_IOU_THRESH)
    parser.add_argument(
        "--class_conf",
        type=str,
        default=",".join(str(x) for x in CLASS_CONF_THRESH),
        help="Per-class thresholds in CLASS_NAMES order, e.g. '0.30,0.30,0.30,0.30,0.30'.",
    )
    parser.add_argument("--preview_dir", type=Path, default=Path("results"))
    parser.add_argument("--preview_count", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model, ckpt_classes, ckpt_img_size = load_checkpoint_model(
        checkpoint_path=args.checkpoint,
        device=device,
        model_cls=YOLOv2Detector,
    )
    model = model.to(memory_format=torch.channels_last)

    class_names = ckpt_classes if len(ckpt_classes) > 0 else list(CLASS_NAMES)
    img_size = int(args.img_size) if int(args.img_size) > 0 else int(ckpt_img_size)

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
        batch_size=max(1, int(args.batch_size)),
        img_size=img_size,
        conf_thresh=float(args.conf_thresh),
        nms_thresh=float(args.nms_thresh),
        class_names=class_names,
    )
    predictions = apply_class_thresholds(
        predictions=predictions,
        class_conf_thresh=class_conf,
        class_names=class_names,
    )

    save_predictions_json(predictions=predictions, output_path=args.output)
    saved = save_preview_images(
        predictions=predictions,
        image_dir=args.image_dir,
        preview_dir=args.preview_dir,
        limit=max(0, int(args.preview_count)),
        class_names=class_names,
    )

    print(f"Device: {device_summary(device)}")
    print(f"Predicted images: {len(predictions)}")
    print(f"Saved JSON: {args.output}")
    print(f"Saved preview images: {saved} -> {args.preview_dir}")


if __name__ == "__main__":
    main()
