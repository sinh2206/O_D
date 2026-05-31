from __future__ import annotations

"""
Shared configuration for the anchor-free object detection project.
"""

IMG_SIZE = 416

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Dataset class frequencies (train/val original splits).
CLASS_FREQ_PRIOR_TRAIN = [0.5477, 0.1258, 0.0966, 0.0783, 0.1516]
CLASS_FREQ_PRIOR_VAL = [0.5314, 0.1400, 0.1019, 0.0871, 0.1395]

# Training priorities to boost recall on crowded scenes without over-penalizing positives.
CLASS_LOSS_WEIGHTS = [1.25, 1.05, 1.00, 1.00, 1.30]
CLASS_SAMPLER_WEIGHTS = [1.20, 1.05, 1.00, 1.00, 1.20]

STRIDES = [8, 16, 32]
FPN_CHANNELS = 128

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONF_THRESH = 0.50
NMS_IOU_THRESH = 0.50
MAX_OBJECTS_PER_IMAGE = 15
# Per-class confidence thresholds used after decode/NMS:
# person, car, dog, cat, chair
CLASS_CONF_THRESH = [0.50, 0.50, 0.50, 0.50, 0.50]
MIN_EXPORT_CONF = 0.50
CHAIR_SUPPRESS_WITH_PERSON_IOU = 0.92
CHAIR_SUPPRESS_MAX_AREA_RATIO = 0.25
MIN_BOX_SIZE = 1.0
NEGATIVE_FOCAL_WEIGHT = 0.5
CENTER_RADIUS = 2.0

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LAMBDA_CLS = 1.0
LAMBDA_REG = 1.0
LAMBDA_CTR = 0.5
LABEL_SMOOTHING = 0.01
