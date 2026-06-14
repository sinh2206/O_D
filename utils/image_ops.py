from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, DARK_LUMA_THRESHOLD, LOWLIGHT_GAMMA


def _gamma_lut(gamma: float) -> np.ndarray:
    """Precompute a lookup table for gamma correction on uint8 images."""

    inv = max(float(gamma), 1e-6)
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(x, inv) * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)


def enhance_low_light_bgr(
    image_bgr: np.ndarray,
    luma_threshold: float = DARK_LUMA_THRESHOLD,
    gamma: float = LOWLIGHT_GAMMA,
    clahe_clip: float = CLAHE_CLIP_LIMIT,
    tile_grid: int = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Enhance low-light images with conditional CLAHE + gamma correction.

    - If image is bright enough, return unchanged image.
    - If dark, apply CLAHE on L channel (LAB) then gentle gamma lift.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    luma = float(gray.mean())
    if luma >= float(luma_threshold):
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(tile_grid), int(tile_grid)))
    l = clahe.apply(l)
    out = cv2.merge([l, a, b])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    lut = _gamma_lut(float(gamma))
    out = cv2.LUT(out, lut)
    return out
