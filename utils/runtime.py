from __future__ import annotations

"""Runtime helpers shared by the YOLOv8 wrappers."""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import torch

from .config import VALID_EXTS


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for more reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str = "auto") -> str:
    """Convert a user-friendly device string into an Ultralytics-compatible value."""

    pref = str(preferred).strip().lower()
    if pref == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if pref in {"cpu", "mps"}:
        return pref
    if pref.startswith("cuda:"):
        return pref.split(":", 1)[1]
    return pref


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    """Delete and recreate a directory tree."""

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def hardlink_or_copy(src: Path, dst: Path) -> Path:
    """Create a hardlink when possible and fall back to a normal copy."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))
    return dst


def copy_if_exists(src: Path, dst: Path) -> Path | None:
    """Copy a file only when the source exists."""

    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return dst


def load_json(path: Path) -> Any:
    """Read one UTF-8 JSON file from disk."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> Path:
    """Write a JSON payload using UTF-8 and pretty indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def to_jsonable(value: Any) -> Any:
    """Recursively convert tensors, arrays and paths into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def normalize_class_names(names_like: Any) -> List[str]:
    """Normalize class-name containers from JSON or Ultralytics into a list."""

    if isinstance(names_like, dict):
        items = sorted(names_like.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _, v in items]
    if isinstance(names_like, (list, tuple)):
        return [str(v) for v in names_like]
    return []


def list_images(image_dir: Path) -> List[Path]:
    """Collect supported image files from one directory in sorted order."""

    return [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]
