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

# Fixed class weights in CLASS_NAMES order:
# person, car, dog, cat, chair
CLASS_LOSS_WEIGHTS = [1.0, 1.0, 1.1, 1.1, 1.2]
CLASS_SAMPLER_WEIGHTS = [1.0, 1.0, 1.1, 1.1, 1.2]

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
CLASS_SCORE_SCALES = [1.0, 1.0, 1.1, 1.1, 1.2]
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
