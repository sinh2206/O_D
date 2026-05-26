from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .config import NUM_ANCHORS, NUM_CLASSES


def _conv_bn_lrelu(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class YOLOv2Detector(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_anchors: int = NUM_ANCHORS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )  # stride 32, channels 512

        self.neck = _conv_bn_lrelu(512, 256, k=1, s=1, p=0)
        self.head = nn.Sequential(
            _conv_bn_lrelu(256, 512, k=3, s=1, p=1),
            _conv_bn_lrelu(512, 512, k=3, s=1, p=1),
            _conv_bn_lrelu(512, 512, k=3, s=1, p=1),
            nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.neck(feat)
        raw = self.head(feat)  # (B, A*(5+C), Gh, Gw)

        b, _, gh, gw = raw.shape
        out = raw.permute(0, 2, 3, 1).contiguous()
        out = out.view(b, gh, gw, self.num_anchors, 5 + self.num_classes)
        return out
