from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .config import ANCHOR_MASKS, NUM_CLASSES, STRIDES, YOLO_HEAD_CHANNELS


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.conv1 = ConvBlock(channels, hidden, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(hidden, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.stage1 = ResidualStage(32, 64, num_blocks=1)
        self.stage2 = ResidualStage(64, 128, num_blocks=2)
        self.stage3 = ResidualStage(128, 256, num_blocks=8)
        self.stage4 = ResidualStage(256, 512, num_blocks=8)
        self.stage5 = ResidualStage(512, 1024, num_blocks=4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feat_s8 = self.stage3(x)
        feat_s16 = self.stage4(feat_s8)
        feat_s32 = self.stage5(feat_s16)
        return feat_s8, feat_s16, feat_s32


class YOLOHeadBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(mid_channels * 2, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = ConvBlock(mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(mid_channels * 2, mid_channels, kernel_size=1, stride=1, padding=0)
        self.pred = nn.Conv2d(mid_channels, num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.pred.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        pred = self.pred(x)
        return pred, x


class YOLOv3(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False, num_anchors: int = 3):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        self.strides = list(STRIDES)
        self.anchor_masks = [list(m) for m in ANCHOR_MASKS]

        self.backbone = Darknet53()

        self.det_s32 = YOLOHeadBlock(1024, 512, num_classes=self.num_classes, num_anchors=self.num_anchors)
        self.reduce_s32 = ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
        self.det_s16 = YOLOHeadBlock(256 + 512, 256, num_classes=self.num_classes, num_anchors=self.num_anchors)
        self.reduce_s16 = ConvBlock(256, 128, kernel_size=1, stride=1, padding=0)
        self.det_s8 = YOLOHeadBlock(128 + 256, 128, num_classes=self.num_classes, num_anchors=self.num_anchors)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Backward-compatible optimizer groups used by the older train.py.
        self.backbone_fpn = nn.ModuleList([self.backbone, self.reduce_s32, self.reduce_s16])
        self.head_s16 = nn.ModuleList([self.det_s8, self.det_s16])
        self.head_s32 = nn.ModuleList([self.det_s32])

        if pretrained:
            # Darknet53 pretraining is not bundled; keep the flag for API compatibility.
            self._init_biases()

    def _init_biases(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feat_s8, feat_s16, feat_s32 = self.backbone(x)

        pred_s32, route = self.det_s32(feat_s32)
        up_s16 = self.upsample(self.reduce_s32(route))
        pred_s16, route = self.det_s16(torch.cat([up_s16, feat_s16], dim=1))
        up_s8 = self.upsample(self.reduce_s16(route))
        pred_s8, _ = self.det_s8(torch.cat([up_s8, feat_s8], dim=1))

        return [pred_s8, pred_s16, pred_s32]


# Compatibility alias kept for older code paths.
AnchorFreeDetector = YOLOv3
