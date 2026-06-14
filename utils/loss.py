from __future__ import annotations

"""Training-summary helpers for Ultralytics YOLO runs.

The actual optimization losses are handled internally by Ultralytics YOLOv8.
This module only reads the generated `results.csv` file and extracts the most
useful public metrics for downstream reporting.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List


def _read_results_csv(csv_path: Path) -> List[Dict[str, float]]:
    """Read one Ultralytics `results.csv` file as numeric dictionaries."""

    if not csv_path.exists():
        return []

    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: Dict[str, float] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                key = key.strip()
                if value is None or str(value).strip() == "":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            if row:
                rows.append(row)
    return rows


def _pick_metric(row: Dict[str, float], *keys: str) -> float | None:
    """Return the first metric present in a CSV row from a list of candidates."""

    for key in keys:
        if key in row:
            return float(row[key])
    return None


def summarize_training_run(run_dir: Path) -> Dict[str, Any]:
    """Extract compact best/final metrics from a YOLO training run directory."""

    rows = _read_results_csv(run_dir / "results.csv")
    if not rows:
        return {"run_dir": str(run_dir), "message": "results.csv not found"}

    def map50(row: Dict[str, float]) -> float:
        return float(_pick_metric(row, "metrics/mAP50(B)", "metrics/mAP50") or -1.0)

    best_row = max(rows, key=map50)
    final_row = rows[-1]
    best_epoch = int(best_row.get("epoch", rows.index(best_row))) + 1

    return {
        "run_dir": str(run_dir),
        "epochs_logged": len(rows),
        "best_epoch": best_epoch,
        "best_map50": _pick_metric(best_row, "metrics/mAP50(B)", "metrics/mAP50"),
        "best_map50_95": _pick_metric(best_row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "best_precision": _pick_metric(best_row, "metrics/precision(B)", "metrics/precision"),
        "best_recall": _pick_metric(best_row, "metrics/recall(B)", "metrics/recall"),
        "final_train_box_loss": _pick_metric(final_row, "train/box_loss"),
        "final_train_cls_loss": _pick_metric(final_row, "train/cls_loss"),
        "final_train_dfl_loss": _pick_metric(final_row, "train/dfl_loss"),
        "final_val_box_loss": _pick_metric(final_row, "val/box_loss"),
        "final_val_cls_loss": _pick_metric(final_row, "val/cls_loss"),
        "final_val_dfl_loss": _pick_metric(final_row, "val/dfl_loss"),
    }
