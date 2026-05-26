import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .config import NUM_CLASSES, NUM_ANCHORS

class YOLOv2Detector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, num_anchors: int = NUM_ANCHORS, pretrained: bool = True):
        super(YOLOv2Detector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone: ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        
        # Remove avgpool and fc
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1, # stride 4
            backbone.layer2, # stride 8
            backbone.layer3, # stride 16
            backbone.layer4  # stride 32
        )
        
        # Neck: reduce channels if needed
        # Layer 4 has 512 channels
        self.neck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Head: 3x3 convs
        self.head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Final 1x1 conv to output predictions
            nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), kernel_size=1)
        )

    def forward(self, x):
        """
        Input: (B, 3, H, W)
        Output: (B, grid_h, grid_w, num_anchors, 5 + num_classes)
        """
        batch_size = x.size(0)
        
        features = self.backbone(x) # (B, 512, H/32, W/32)
        features = self.neck(features) # (B, 256, H/32, W/32)
        out = self.head(features) # (B, A*(5+C), H/32, W/32)
        
        # Reshape to (B, H/32, W/32, A, 5 + C)
        # Permute from (B, A*(5+C), H, W) -> (B, H, W, A*(5+C)) -> (B, H, W, A, 5+C)
        h, w = out.shape[2], out.shape[3]
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(batch_size, h, w, self.num_anchors, 5 + self.num_classes)
        
        return out
