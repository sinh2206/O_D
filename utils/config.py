from __future__ import annotations

"""
Shared configuration for the anchor-free object detection project.
"""

IMG_SIZE = 640

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Dataset class frequencies (train/val original splits).
CLASS_FREQ_PRIOR_TRAIN = [0.5477, 0.1258, 0.0966, 0.0783, 0.1516]
CLASS_FREQ_PRIOR_VAL = [0.5314, 0.1400, 0.1019, 0.0871, 0.1395]

# Class emphasis factors used by train.py to derive final weights from the
# observed class frequencies.
CLASS_LOSS_WEIGHTS = [0.60, 1.25, 1.35, 1.45, 1.20]
CLASS_SAMPLER_WEIGHTS = [0.70, 1.20, 1.30, 1.30, 1.15]

STRIDES = [8, 16, 32]
FPN_CHANNELS = 128

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONF_THRESH = 0.35
NMS_IOU_THRESH = 0.50
MAX_OBJECTS_PER_IMAGE = 15
# Per-class confidence thresholds used after decode/NMS:
# person, car, dog, cat, chair
CLASS_CONF_THRESH = [0.35, 0.35, 0.35, 0.35, 0.35]
MIN_EXPORT_CONF = 0.3
CLASS_SCORE_SCALES = [0.80, 1.10, 1.20, 1.20, 1.05]
CHAIR_SUPPRESS_WITH_PERSON_IOU = 0.92
CHAIR_SUPPRESS_MAX_AREA_RATIO = 0.25
MIN_BOX_SIZE = 1.0
NEGATIVE_FOCAL_WEIGHT = 0.20
CENTER_RADIUS = 4.0
INFER_CENTER_COMBINE = "sqrt"

SMALL_OBJECT_AREA_RATIO = 0.012
SMALL_OBJECT_BONUS = 0.70
LOW_LIGHT_MEAN_THRESH = 70.0
LOW_LIGHT_CLAHE_CLIP = 1.8
LOW_LIGHT_GAMMA = 0.90

FOCAL_ALPHA = 0.35
FOCAL_GAMMA = 1.5
LAMBDA_CLS = 1.25
LAMBDA_REG = 1.0
LAMBDA_CTR = 0.35
LABEL_SMOOTHING = 0.0

# Reproducibility defaults.
DEFAULT_SEED = 42

# Training policy defaults.
DEFAULT_SCHEDULER = "plateau"  # "plateau" or "cosine"
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 2
PLATEAU_MIN_LR = 1e-6

EARLY_STOP_PATIENCE = 5
EARLY_STOP_DELTA = 1e-4

# Strong augmentation defaults.
MOSAIC_PROB = 0.35
MIXUP_PROB = 0.20

# Online validation mAP settings.
MAP_EVAL_INTERVAL = 2
MAP_CONF_THRESH = 0.35
MAP_NMS_THRESH = 0.50

# Training speed profile.
TRAIN_SPEED_MODE = "fast"  # "quality", "balanced", "fast"
FAST_IMG_SIZE = 512
FAST_VAL_INTERVAL = 2
FAST_MAP_EVAL_INTERVAL = 5
FAST_MAP_MAX_BATCHES = 40
FAST_MOSAIC_PROB = 0.20
FAST_MIXUP_PROB = 0.10
FAST_PREFETCH_FACTOR = 2
FAST_CACHE_IMAGES = True
FAST_CACHE_MAX_IMAGES = 1200
FAST_DETERMINISTIC = False
FAST_ENABLE_TF32 = True
