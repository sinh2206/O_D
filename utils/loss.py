import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from .config import STRIDE, ANCHOR_SIZES, NUM_CLASSES, NUM_ANCHORS, IMG_SIZE, \
    IOU_IGNORE_THRESH, LAMBDA_OBJ, LAMBDA_NOOBJ, LAMBDA_BOX, LAMBDA_CLS

def box_iou(box1, box2):
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-16)
    return iou

def build_targets(targets_list: List[Dict[str, Any]], grid_h: int, grid_w: int, device: torch.device):
    """
    targets_list: list of dicts with 'boxes' (N, 4) in absolute pixels [x1, y1, x2, y2] and 'labels' (N,)
    Returns masks and target values for the loss.
    """
    batch_size = len(targets_list)
    anchors = torch.tensor(ANCHOR_SIZES, device=device).float() # (A, 2)
    
    obj_mask = torch.zeros(batch_size, grid_h, grid_w, NUM_ANCHORS, device=device, dtype=torch.bool)
    noobj_mask = torch.ones(batch_size, grid_h, grid_w, NUM_ANCHORS, device=device, dtype=torch.bool)
    
    tx = torch.zeros(batch_size, grid_h, grid_w, NUM_ANCHORS, device=device)
    ty = torch.zeros(batch_size, grid_h, grid_w, NUM_ANCHORS, device=device)
    tw = torch.zeros(batch_size, grid_h, grid_w, NUM_ANCHORS, device=device)
    th = torch.zeros(batch_size, grid_h, grid_w, NUM_ANCHORS, device=device)
    tcls = torch.zeros(batch_size, grid_h, grid_w, NUM_ANCHORS, device=device, dtype=torch.long)
    
    for b in range(batch_size):
        gt_boxes = targets_list[b]['boxes'] # (N, 4)
        gt_labels = targets_list[b]['labels'] # (N,)
        
        if gt_boxes.shape[0] == 0:
            continue
            
        # Convert absolute to grid coordinates
        # gx, gy are center of box in grid units
        gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / (2.0 * STRIDE)
        gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / (2.0 * STRIDE)
        gw = (gt_boxes[:, 2] - gt_boxes[:, 0]) / STRIDE
        gh = (gt_boxes[:, 3] - gt_boxes[:, 1]) / STRIDE
        
        # For each GT box, find the best anchor
        # Use IoU of widths/heights starting at 0,0
        # (This is the standard YOLOv2 anchor matching)
        gt_wh = torch.stack([gw, gh], dim=1) # (N, 2)
        
        # Calculate IoU of wh with anchors
        # wh_iou(N, A)
        gt_wh_expanded = gt_wh.unsqueeze(1) # (N, 1, 2)
        anchors_expanded = anchors.unsqueeze(0) # (1, A, 2)
        
        inter = torch.min(gt_wh_expanded, anchors_expanded).prod(2) # (N, A)
        union = gt_wh_expanded.prod(2) + anchors_expanded.prod(2) - inter
        wh_ious = inter / union
        
        best_n = wh_ious.argmax(1) # (N,)
        
        # Assignment
        gi = gx.long().clamp(0, grid_w - 1)
        gj = gy.long().clamp(0, grid_h - 1)
        
        for i in range(gt_boxes.shape[0]):
            n = best_n[i]
            x, y = gi[i], gj[i]
            
            obj_mask[b, y, x, n] = True
            noobj_mask[b, y, x, n] = False
            
            tx[b, y, x, n] = gx[i] - x
            ty[b, y, x, n] = gy[i] - y
            tw[b, y, x, n] = torch.log(gw[i] / anchors[n, 0] * STRIDE + 1e-16)
            th[b, y, x, n] = torch.log(gh[i] / anchors[n, 1] * STRIDE + 1e-16)
            tcls[b, y, x, n] = gt_labels[i]
            
        # Optional: handle ignore mask
        # Anchors with high IoU but not best should not contribute to noobj loss
        # (Simplified for now, can be added if needed)
        
    return obj_mask, noobj_mask, tx, ty, tw, th, tcls

def compute_loss(predictions, targets_list, device):
    """
    predictions: (B, H, W, A, 5 + C)
    targets_list: list of dicts
    """
    b, h, w, a, _ = predictions.shape
    obj_mask, noobj_mask, tx, ty, tw, th, tcls = build_targets(targets_list, h, w, device)
    
    # Extract prediction components
    pred_tx = torch.sigmoid(predictions[..., 0])
    pred_ty = torch.sigmoid(predictions[..., 1])
    pred_tw = predictions[..., 2]
    pred_th = predictions[..., 3]
    pred_conf = torch.sigmoid(predictions[..., 4])
    pred_cls = predictions[..., 5:] # Logits
    
    # 1. Regression Loss (only for objects)
    loss_x = F.mse_loss(pred_tx[obj_mask], tx[obj_mask])
    loss_y = F.mse_loss(pred_ty[obj_mask], ty[obj_mask])
    loss_w = F.mse_loss(pred_tw[obj_mask], tw[obj_mask])
    loss_h = F.mse_loss(pred_th[obj_mask], th[obj_mask])
    loss_reg = loss_x + loss_y + loss_w + loss_h
    
    # 2. Objectness Loss (BCE)
    # obj_mask values are 1, noobj_mask targets are 0
    t_obj = obj_mask.float()
    loss_obj = F.binary_cross_entropy(pred_conf[obj_mask], t_obj[obj_mask])
    loss_noobj = F.binary_cross_entropy(pred_conf[noobj_mask], t_obj[noobj_mask])
    
    # 3. Classification Loss (CE, only for objects)
    if obj_mask.any():
        loss_cls = F.cross_entropy(pred_cls[obj_mask], tcls[obj_mask])
    else:
        loss_cls = torch.tensor(0.0, device=device)
    
    total_loss = (LAMBDA_BOX * loss_reg) + \
                 (LAMBDA_OBJ * loss_obj) + \
                 (LAMBDA_NOOBJ * loss_noobj) + \
                 (LAMBDA_CLS * loss_cls)
    
    return total_loss, {
        "loss": total_loss.item(),
        "reg": loss_reg.item(),
        "obj": loss_obj.item(),
        "noobj": loss_noobj.item(),
        "cls": loss_cls.item()
    }
