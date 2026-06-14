from __future__ import annotations

"""
Shared configuration for the anchor-free object detection project.
"""

IMG_SIZE = 512

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Dataset class frequencies (train/val original splits).
CLASS_FREQ_PRIOR_TRAIN = [0.5477, 0.1258, 0.0966, 0.0783, 0.1516]
CLASS_FREQ_PRIOR_VAL = [0.5314, 0.1400, 0.1019, 0.0871, 0.1395]

# Class emphasis factors used by train.py to derive final weights from the
# observed class frequencies.
CLASS_LOSS_WEIGHTS = [0.50, 1.10, 1.25, 1.35, 1.55]
CLASS_SAMPLER_WEIGHTS = [0.65, 1.12, 1.28, 1.30, 1.45]

# Speed-oriented default preset for Kaggle T4:
# keep the stride-4 branch for tiny details, but shrink the feature width and
# image size so epoch time stays practical.
STRIDES = [4, 8, 16, 32]
FPN_CHANNELS = 96
HEAD_NUM_CONVS = 1

TRAIN_BATCH_SIZE = 16
TRAIN_NUM_WORKERS = -1
TRAIN_PREFETCH_FACTOR = 2
TRAIN_ENABLE_LOW_LIGHT = False
VAL_ENABLE_LOW_LIGHT = False
MAX_VAL_BATCHES = 32
METRIC_EVAL_INTERVAL = 3
METRIC_EVAL_BATCHES = 32
ENABLE_TORCH_COMPILE = True

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONF_THRESH = 0.35
NMS_IOU_THRESH = 0.50
MAX_OBJECTS_PER_IMAGE = 20

# Per-class confidence thresholds used after decode/NMS:
# person, car, dog, cat, chair
CLASS_CONF_THRESH = [0.35, 0.30, 0.29, 0.30, 0.35]
MIN_EXPORT_CONF = 0.25
CLASS_SCORE_SCALES = [0.72, 1.05, 1.12, 1.14, 1.12]

# Keep more than one class hypothesis per location to reduce dog/cat confusion.
DECODE_CANDIDATE_CONF = 0.20
DECODE_TOPK = 2
PRE_NMS_MAX_CANDIDATES = 3500
CONTAINED_BOX_REPLACE_MARGIN = 0.12

CHAIR_SUPPRESS_WITH_PERSON_IOU = 0.92
CHAIR_SUPPRESS_MAX_AREA_RATIO = 0.25
MIN_BOX_SIZE = 1.0
NEGATIVE_FOCAL_WEIGHT = 0.35
CENTER_RADIUS = 2.75
INFER_CENTER_COMBINE = "sqrt"

SMALL_OBJECT_AREA_RATIO = 0.012
SMALL_OBJECT_BONUS = 0.70
LOW_LIGHT_MEAN_THRESH = 70.0
LOW_LIGHT_CLAHE_CLIP = 1.8
LOW_LIGHT_GAMMA = 0.90

CLAHE_CLIP_LIMIT = LOW_LIGHT_CLAHE_CLIP
CLAHE_TILE_GRID = 8
DARK_LUMA_THRESHOLD = LOW_LIGHT_MEAN_THRESH
LOWLIGHT_GAMMA = LOW_LIGHT_GAMMA

# Object-centric augmentation for hard cases.
DETAIL_FOCUS_PROB = 0.22
DETAIL_FOCUS_CONTEXT_RANGE = (1.35, 2.60)
DETAIL_FOCUS_JITTER = 0.18
DETAIL_FOCUS_MIN_VISIBLE = 0.45
DETAIL_FOCUS_AREA_RATIO = 0.10
PARTIAL_OCCLUSION_PROB = 0.10

TINY_OBJECT_MAX_SIDE_FACTOR = 2.5
TINY_ASSIGN_EXPAND_STRIDE = 1.0

FOCAL_ALPHA = 0.35
FOCAL_GAMMA = 1.5
LAMBDA_CLS = 1.25
LAMBDA_REG = 1.20
LAMBDA_REG_L1 = 0.20
LAMBDA_CTR = 0.30
LABEL_SMOOTHING = 0.0
