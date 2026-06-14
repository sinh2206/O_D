from __future__ import annotations

"""Train a YOLOv8 detector on the repo's custom JSON annotations."""

import argparse
from pathlib import Path

from utils.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_PATIENCE,
    DEFAULT_PROJECT_NAME,
    GENERATED_DATASET_DIRNAME,
)
from utils.loss import summarize_training_run
from utils.model import load_yolo_model
from utils.process import build_yolo_dataset
from utils.runtime import copy_if_exists, ensure_dir, resolve_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    """Define the CLI for YOLOv8 training on this project."""

    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector on the project's JSON annotations.")
    parser.add_argument("--train_data", type=Path, required=True, help="Path to train JSON annotations.")
    parser.add_argument("--val_data", type=Path, required=True, help="Path to validation JSON annotations.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Path to training images directory.")
    parser.add_argument("--val_image_dir", type=Path, required=True, help="Path to validation images directory.")
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Directory used for YOLO runs and final weights.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ultralytics model name or local .pt/.yaml file.")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Training image size.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers for Ultralytics training.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Training device, e.g. auto, cpu, 0, 0,1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--run_name", type=str, default=DEFAULT_PROJECT_NAME, help="Subdirectory name inside checkpoint_dir for this YOLO run.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional local weights (.pt) to continue fine-tuning from.")
    parser.add_argument("--cache", action="store_true", help="Cache images in memory for faster training.")
    parser.add_argument("--optimizer", type=str, default="auto", help="Ultralytics optimizer setting, e.g. auto, SGD, AdamW.")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic this many epochs before training ends.")
    return parser.parse_args()


def main() -> None:
    """Convert the dataset to YOLO format and launch Ultralytics training."""

    args = parse_args()
    seed_everything(int(args.seed))

    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    generated_dataset_dir = checkpoint_dir / GENERATED_DATASET_DIRNAME

    dataset_yaml, class_names = build_yolo_dataset(
        train_data=args.train_data,
        val_data=args.val_data,
        train_image_dir=args.image_dir,
        val_image_dir=args.val_image_dir,
        output_dir=generated_dataset_dir,
    )

    model_source = args.resume if args.resume is not None else args.model
    model = load_yolo_model(model_source)

    train_kwargs = {
        "data": str(dataset_yaml),
        "imgsz": int(args.img_size),
        "epochs": int(args.epochs),
        "batch": int(args.batch_size),
        "workers": int(args.workers),
        "device": resolve_device(args.device),
        "project": str(checkpoint_dir),
        "name": str(args.run_name),
        "exist_ok": True,
        "patience": int(args.patience),
        "seed": int(args.seed),
        "cache": bool(args.cache),
        "optimizer": str(args.optimizer),
        "close_mosaic": int(args.close_mosaic),
        "verbose": True,
        "plots": True,
    }

    print(f"YOLO dataset YAML: {dataset_yaml}")
    print(f"Classes: {class_names}")
    print(f"Model source: {model_source}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    model.train(**train_kwargs)

    run_dir = checkpoint_dir / args.run_name
    weights_dir = run_dir / "weights"
    best_weight = copy_if_exists(weights_dir / "best.pt", checkpoint_dir / "best.pt")
    last_weight = copy_if_exists(weights_dir / "last.pt", checkpoint_dir / "last.pt")

    train_summary = summarize_training_run(run_dir)
    train_summary.update(
        {
            "dataset_yaml": str(dataset_yaml),
            "class_names": class_names,
            "best_weight": str(best_weight) if best_weight else None,
            "last_weight": str(last_weight) if last_weight else None,
        }
    )
    write_json(checkpoint_dir / "train_summary.json", train_summary)

    print(f"Training finished. Best weight: {best_weight}")
    print(f"Last weight: {last_weight}")
    print(f"Summary JSON: {checkpoint_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
