import torch
import torchvision.ops as ops
import torch.nn.functional as F
from typing import List, Dict, Any
from .config import STRIDE, ANCHOR_SIZES, CONF_THRESH, NMS_IOU_THRESH, CLASS_NAMES

def decode_predictions(pred, device):
    """
    pred: (B, H, W, A, 5 + C)
    Returns:
        boxes: (B, H*W*A, 4) in absolute pixels of the letterbox image [x1, y1, x2, y2]
        scores: (B, H*W*A)
        classes: (B, H*W*A)
    """
    b, h, w, a, _ = pred.shape
    anchors = torch.tensor(ANCHOR_SIZES, device=device).float() # (A, 2)
    
    # Create grid
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    grid_x = grid_x.view(1, h, w, 1).expand(b, h, w, a)
    grid_y = grid_y.view(1, h, w, 1).expand(b, h, w, a)
    
    # Anchor wh
    anchor_w = anchors[:, 0].view(1, 1, 1, a).expand(b, h, w, a)
    anchor_h = anchors[:, 1].view(1, 1, 1, a).expand(b, h, w, a)
    
    # Decoding
    tx = torch.sigmoid(pred[..., 0])
    ty = torch.sigmoid(pred[..., 1])
    tw = pred[..., 2]
    th = pred[..., 3]
    obj_conf = torch.sigmoid(pred[..., 4])
    cls_scores = F.softmax(pred[..., 5:], dim=-1)
    
    # Coordinates in grid units
    bx = tx + grid_x
    by = ty + grid_y
    bw = torch.exp(tw) * (anchor_w / STRIDE)
    bh = torch.exp(th) * (anchor_h / STRIDE)
    
    # To absolute pixels
    bx *= STRIDE
    by *= STRIDE
    bw *= STRIDE
    bh *= STRIDE
    
    # Convert to x1, y1, x2, y2
    x1 = bx - bw / 2
    y1 = by - bh / 2
    x2 = bx + bw / 2
    y2 = by + bh / 2
    
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(b, -1, 4)
    
    # Class confidence
    # max_score (B, H, W, A), max_class (B, H, W, A)
    max_cls_score, max_cls_idx = torch.max(cls_scores, dim=-1)
    scores = (obj_conf * max_cls_score).view(b, -1)
    classes = max_cls_idx.view(b, -1)
    
    return boxes, scores, classes

def postprocess_batch(outputs, metas, conf_thresh=CONF_THRESH, nms_thresh=NMS_IOU_THRESH):
    """
    outputs: (B, H, W, A, 5 + C)
    metas: list of metadata dicts from letterbox_preprocess
    """
    device = outputs.device
    boxes, scores, classes = decode_predictions(outputs, device)
    
    batch_results = []
    for i in range(len(metas)):
        b = boxes[i]
        s = scores[i]
        c = classes[i]
        meta = metas[i]
        
        # Filter by threshold
        mask = s > conf_thresh
        b, s, c = b[mask], s[mask], c[mask]
        
        if b.shape[0] == 0:
            batch_results.append([])
            continue
            
        # NMS (batched NMS to handle classes separately)
        keep = ops.batched_nms(b, s, c, nms_thresh)
        b, s, c = b[keep], s[keep], c[keep]
        
        # Map back to original image
        scale = meta['scale']
        dx, dy = meta['dx'], meta['dy']
        
        b[:, [0, 2]] = (b[:, [0, 2]] - dx) / scale
        b[:, [1, 3]] = (b[:, [1, 3]] - dy) / scale
        
        # Clip to original image boundaries
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, meta['orig_w'])
        b[:, [1, 3]] = b[:, [1, 3]].clamp(0, meta['orig_h'])
        
        img_results = []
        for j in range(b.shape[0]):
            img_results.append({
                'bbox': b[j].tolist(),
                'score': float(s[j]),
                'class': CLASS_NAMES[int(c[j])],
                'conf': float(s[j]) # duplicate for convenience
            })
        batch_results.append(img_results)
        
    return batch_results
