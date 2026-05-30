from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Keep the project conservative on VRAM usage. Change to 416 if you want the
# canonical YOLOv3 input size and can afford the extra compute.
IMG_SIZE = 320

# Canonical COCO anchors, scaled to the active IMG_SIZE.
_BASE_ANCHORS_416 = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]


def _scale_anchor_sets(anchor_sets: Sequence[Sequence[Tuple[float, float]]], scale: float) -> List[List[Tuple[int, int]]]:
    out: List[List[Tuple[int, int]]] = []
    for level in anchor_sets:
        scaled_level: List[Tuple[int, int]] = []
        for w, h in level:
            scaled_level.append((max(1, int(round(float(w) * scale))), max(1, int(round(float(h) * scale)))))
        out.append(scaled_level)
    return out


ANCHORS = _scale_anchor_sets(_BASE_ANCHORS_416, IMG_SIZE / 416.0)
ANCHOR_MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
STRIDES = [8, 16, 32]

# Backward-compatible alias used by older modules.
FPN_CHANNELS = 256
YOLO_HEAD_CHANNELS = 256

CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.45
IOU_POS_THRESH = 0.50
IOU_IGNORE_THRESH = 0.50
MAX_OBJECTS_PER_IMAGE = 15

LAMBDA_OBJ = 1.0
LAMBDA_NOOBJ = 1.0
LAMBDA_BOX = 5.0
LAMBDA_CLS = 1.0

# Loss settings retained for compatibility with older code paths.
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LAMBDA_CTR = 0.0
LAMBDA_REG = LAMBDA_BOX
LABEL_SMOOTHING = 0.03

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEFAULT_DATA_DIR = Path("public")
DEFAULT_TRAIN_ANN = Path("public/annotations/train.json")
DEFAULT_VAL_ANN = Path("public/annotations/val.json")
DEFAULT_RESULTS_DIR = Path("results")

CLASS_CONF_THRESH = [0.30, 0.30, 0.30, 0.30, 0.30]
CHAIR_SUPPRESS_WITH_PERSON_IOU = 0.55

# Legacy class-frequency priors preserved so older training scripts still import.
CLASS_FREQ_PRIOR_TRAIN = [0.5477, 0.1258, 0.0966, 0.0783, 0.1516]
CLASS_FREQ_PRIOR_VAL = [0.5314, 0.1400, 0.1019, 0.0871, 0.1395]
CLASS_LOSS_WEIGHTS = [1.15, 1.20, 1.12, 1.10, 0.70]
CLASS_SAMPLER_WEIGHTS = [1.10, 1.18, 1.10, 1.08, 0.72]

# Standard torchvision/ImageNet normalization aliases used by older code.
IMAGENET_MEAN = MEAN
IMAGENET_STD = STD
