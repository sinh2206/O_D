import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm

from utils.config import IMG_SIZE, NUM_CLASSES, MEAN, STD, NUM_ANCHORS
from utils.model import YOLOv2Detector
from utils.loss import compute_loss
from utils.runtime import resolve_device, get_optimizer, get_scheduler, save_checkpoint, load_checkpoint
from utils.process import imread_unicode, enhance_low_light_bgr

class DetectionDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        import json
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        self.image_info = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.data.get('categories', [])}
        
        # Build image_id to annotations map
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_id = img_info['id']
        img_path = self.image_dir / img_info['file_name']
        
        image = imread_unicode(str(img_path))
        if image is None:
            # Fallback or skip
            return self.__getitem__((idx + 1) % len(self))
            
        image = enhance_low_light_bgr(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        anns = self.img_to_anns.get(img_id, [])
        bboxes = []
        labels = []
        for ann in anns:
            # Assuming bbox is [x1, y1, x2, y2] or [x, y, w, h] - need to be careful
            # Let's assume [x1, y1, x2, y2] as per common project structure here
            bbox = ann['bbox']
            if len(bbox) == 4:
                bboxes.append(bbox)
                labels.append(ann['category_id'] if 'category_id' in ann else 0)
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, category_ids=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['category_ids']

        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': img_id
        }
        
        return image, target

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    total_loss = 0
    
    for images, targets in pbar:
        images = images.to(device)
        # targets is a list of dicts, we'll move them to device in loss.py or here
        targets = [{'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets]
        
        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = compute_loss(predictions, targets, device)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), obj=loss_dict['obj'], reg=loss_dict['reg'])
        
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ann', type=str, required=True)
    parser.add_argument('--train_img', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output', type=str, default='models/best.pth')
    args = parser.parse_args()
    
    device = resolve_device()
    
    transform = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=114),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    
    dataset = DetectionDataset(args.train_ann, args.train_img, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    model = YOLOv2Detector(pretrained=True).to(device)
    optimizer = get_optimizer(model, lr=args.lr)
    scheduler = get_scheduler(optimizer, args.epochs)
    
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer)
        
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(args.output, epoch, model, optimizer, best_loss)
            print(f"Saved best model with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()
