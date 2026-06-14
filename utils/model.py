from __future__ import annotations

"""Thin wrapper helpers around the Ultralytics YOLO model class."""

from pathlib import Path
from typing import List

from ultralytics import YOLO

from .config import DEFAULT_MODEL
from .runtime import normalize_class_names


def load_yolo_model(weights: str | Path = DEFAULT_MODEL) -> YOLO:
    """Load a YOLOv8 model from a pretrained name or a local checkpoint path."""

    return YOLO(str(weights))


def get_model_class_names(model: YOLO) -> List[str]:
    """Extract class names from a YOLO model in a stable list order."""

    return normalize_class_names(getattr(model, "names", {}))
