from __future__ import annotations

"""
Shared configuration for the anchor-free object detection project.
"""

IMG_SIZE = 320

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Dataset class frequencies (train/val original splits).
CLASS_FREQ_PRIOR_TRAIN = [0.5477, 0.1258, 0.0966, 0.0783, 0.1516]
CLASS_FREQ_PRIOR_VAL = [0.5314, 0.1400, 0.1019, 0.0871, 0.1395]

# Training priorities to boost recall for person/car/dog/cat and suppress chair.
# Tuned for harder crowded-person and low-frequency dog samples.
CLASS_LOSS_WEIGHTS = [1.25, 1.28, 1.36, 1.12, 0.62]
CLASS_SAMPLER_WEIGHTS = [1.18, 1.24, 1.34, 1.10, 0.66]

STRIDES = [16, 32]
FPN_CHANNELS = 128

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONF_THRESH = 0.35
NMS_IOU_THRESH = 0.35
NMS_IOU_THRESH_BY_CLASS = [0.50, 0.28, 0.36, 0.36, 0.35]
# Per-class confidence thresholds used after decode/NMS:
# person, car, dog, cat, chair
CLASS_CONF_THRESH = [0.34, 0.38, 0.34, 0.38, 0.72]
CHAIR_SUPPRESS_WITH_PERSON_IOU = 0.55

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LAMBDA_CLS = 1.0
LAMBDA_REG = 1.0
LAMBDA_CTR = 0.7
LAMBDA_REG_L1 = 0.3
LABEL_SMOOTHING = 0.03
