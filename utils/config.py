from __future__ import annotations

"""
Shared configuration for the anchor-free object detection project.
"""

IMG_SIZE = 320

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Class frequency priors from original datasets (order: person, car, dog, cat, chair).
CLASS_FREQ_PRIOR_TRAIN = [0.5477, 0.1258, 0.0966, 0.0783, 0.1516]
CLASS_FREQ_PRIOR_VAL = [0.5314, 0.1400, 0.1019, 0.0871, 0.1395]

STRIDES = [16, 32]
FPN_CHANNELS = 128

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONF_THRESH = 0.6
NMS_IOU_THRESH = 0.28
AGNOSTIC_NMS_IOU_THRESH = 0.55
CROSS_CLASS_IOU_THRESH = 0.82
CROSS_CLASS_CONTAIN_THRESH = 0.88
SAME_CLASS_CONTAIN_THRESH = 0.78

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LAMBDA_CLS = 1.0
LAMBDA_REG = 1.0
LAMBDA_CTR = 0.5
LABEL_SMOOTHING = 0.03

DARK_LUMA_THRESHOLD = 82.0
LOWLIGHT_GAMMA = 0.82
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = 8
