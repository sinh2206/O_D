import numpy as np

# Image configuration
IMG_SIZE = 320

# Class configuration
CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Grid and Stride configuration
STRIDE = 32
GRID_SIZE = IMG_SIZE // STRIDE  # Should be 10 for 320/32

# Anchor configuration (width, height) in pixels
# Example: 3 common sizes
ANCHOR_SIZES = [(48, 48), (96, 96), (192, 192)]
NUM_ANCHORS = len(ANCHOR_SIZES)

# Loss weights and thresholds
IOU_POS_THRESH = 0.5
IOU_IGNORE_THRESH = 0.5
LAMBDA_OBJ = 5.0
LAMBDA_NOOBJ = 0.5
LAMBDA_BOX = 5.0
LAMBDA_CLS = 1.0

# Normalization constants (ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Inference defaults
CONF_THRESH = 0.3
NMS_IOU_THRESH = 0.45
