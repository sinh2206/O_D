import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any, List
from .config import MEAN, STD

def imread_unicode(path: str) -> np.ndarray:
    """Read image using OpenCV with Unicode path support."""
    try:
        with open(path, "rb") as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(chunk, cv2.IMREAD_COLOR)
    except Exception:
        return None

def letterbox_preprocess(image: np.ndarray, img_size: int = 320) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Resize image to img_size while maintaining aspect ratio using padding.
    Returns:
        tensor: (3, img_size, img_size) normalized
        meta: dict containing scale and offsets for mapping back
    """
    h, w = image.shape[:2]
    scale = min(img_size / h, img_size / w)
    nh, nw = int(h * scale), int(w * scale)
    
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    dx = (img_size - nw) // 2
    dy = (img_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw, :] = image_resized
    
    # BGR to RGB
    img_rgb = canvas[:, :, ::-1].transpose(2, 0, 1) # (3, H, W)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    
    # Normalize
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    meta = {
        "scale": scale,
        "dx": dx,
        "dy": dy,
        "orig_w": w,
        "orig_h": h
    }
    return img_tensor, meta

def draw_prediction(image: np.ndarray, boxes: List[Dict[str, Any]], class_names: List[str]) -> np.ndarray:
    """
    Draw boxes on image. boxes is a list of dicts: {'bbox': [x1, y1, x2, y2], 'class': str, 'conf': float}
    """
    img = image.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b['bbox'])
        label = b['class']
        conf = b['conf']
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def enhance_low_light_bgr(image: np.ndarray) -> np.ndarray:
    """Improve low light using CLAHE on L channel of LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
