import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path

def resolve_device(preferred="cuda"):
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def device_summary(device):
    if device.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"

def load_checkpoint(path, model, optimizer=None):
    if not Path(path).exists():
        return 0, float('inf')
    
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
    return ckpt.get('epoch', 0), ckpt.get('best_loss', float('inf'))

def save_checkpoint(path, epoch, model, optimizer, best_loss):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }, path)

def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    # Differentiate learning rates for backbone and head if desired
    # Here we just use a single LR for simplicity or filter by name
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params, 'lr': lr}
    ], weight_decay=weight_decay)
    
    return optimizer

def get_scheduler(optimizer, epochs, warmup_epochs=3):
    # Cosine annealing with warmup is often good for YOLO
    # Simple cosine annealing for now
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
