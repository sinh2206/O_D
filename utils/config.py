from __future__ import annotations

"""Shared defaults for the YOLOv8-based training and inference pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

FALLBACK_CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]

DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 2
DEFAULT_PATIENCE = 30
DEFAULT_DEVICE = "auto"
DEFAULT_PROJECT_NAME = "yolov8_train"

DEFAULT_CONF_THRESH = 0.25
DEFAULT_IOU_THRESH = 0.45
DEFAULT_MAX_DET = 100

GENERATED_DATASET_DIRNAME = "_yolo_dataset"
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "models"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
