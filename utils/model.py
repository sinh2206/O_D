from __future__ import annotations

"""
Anchor-Free feature extractor and detection head.

Architecture summary (input IMG_SIZE x IMG_SIZE):
- Backbone: ResNet-34 pretrained (no avgpool/fc)
  stem(stride 4) -> layer1(C1, stride 4, 64ch) -> layer2(C2, stride 8, 128ch)
  -> layer3(C3, stride 16, 256ch) -> layer4(C4, stride 32, 512ch)
- Neck: 4-level FPN
  lateral1: 1x1 conv (64  -> FPN_CHANNELS)
  lateral2: 1x1 conv (128 -> FPN_CHANNELS)
  lateral3: 1x1 conv (256 -> FPN_CHANNELS)
  lateral4: 1x1 conv (512 -> FPN_CHANNELS)
  top-down: upsample(P4) + P3, upsample(P3) + P2, upsample(P2) + P1
  smooth: conv3x3 + BN + LeakyReLU(0.1) on all levels
- Outputs:
  P1_out: (B, FPN_CHANNELS, H/4,  W/4),  stride 4
  P2_out: (B, FPN_CHANNELS, H/8,  W/8),  stride 8
  P3_out: (B, FPN_CHANNELS, H/16, W/16), stride 16
  P4_out: (B, FPN_CHANNELS, H/32, W/32), stride 32
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import FPN_CHANNELS, HEAD_NUM_CONVS, NUM_CLASSES, STRIDES


class ConvBNLeaky(nn.Module):
    """Conv2d + BatchNorm2d + LeakyReLU(0.1)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet34FPN3L(nn.Module):
    """Backbone + 4-level FPN feature extractor (stride 4/8/16/32)."""

    def __init__(self, fpn_channels: int = FPN_CHANNELS, pretrained: bool = True):
        super().__init__()

        backbone = self._build_resnet34(pretrained=pretrained)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # C1: stride 4, 64ch
        self.layer2 = backbone.layer2  # C2: stride 8, 128ch
        self.layer3 = backbone.layer3  # C3: stride 16, 256ch
        self.layer4 = backbone.layer4  # C4: stride 32, 512ch

        self.lateral1 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(512, fpn_channels, kernel_size=1)

        self.fpn_out1 = ConvBNLeaky(fpn_channels, fpn_channels, k=3, s=1, p=1)
        self.fpn_out2 = ConvBNLeaky(fpn_channels, fpn_channels, k=3, s=1, p=1)
        self.fpn_out3 = ConvBNLeaky(fpn_channels, fpn_channels, k=3, s=1, p=1)
        self.fpn_out4 = ConvBNLeaky(fpn_channels, fpn_channels, k=3, s=1, p=1)

    @staticmethod
    def _build_resnet34(pretrained: bool):
        if not pretrained:
            return models.resnet34(weights=None)

        try:
            return models.resnet34(weights="IMAGENET1K_V1")
        except Exception:
            try:
                from torchvision.models import ResNet34_Weights

                return models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                return models.resnet34(weights=None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p1 = self.lateral1(c1)
        p2 = self.lateral2(c2)
        p3 = self.lateral3(c3)
        p4 = self.lateral4(c4)

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = p3 + p4_up
        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        p2 = p2 + p3_up
        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="nearest")
        p1 = p1 + p2_up

        p1_out = self.fpn_out1(p1)
        p2_out = self.fpn_out2(p2)
        p3_out = self.fpn_out3(p3)
        p4_out = self.fpn_out4(p4)
        return p1_out, p2_out, p3_out, p4_out


class AnchorFreeHead(nn.Module):
    """
    Anchor-free head for one feature level.

    Outputs per level:
    - cls_logits:   (B, C, H, W)
    - reg_preds:    (B, 4, H, W), non-negative via ReLU
    - center_logits:(B, 1, H, W)
    """

    def __init__(self, in_ch: int, num_classes: int, num_convs: int = 2):
        super().__init__()

        cls_layers: List[nn.Module] = []
        reg_layers: List[nn.Module] = []
        ch = in_ch
        for _ in range(num_convs):
            cls_layers.append(ConvBNLeaky(ch, in_ch, k=3, s=1, p=1))
            reg_layers.append(ConvBNLeaky(ch, in_ch, k=3, s=1, p=1))
            ch = in_ch

        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)

        self.cls_out = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        self.reg_out = nn.Conv2d(in_ch, 4, kernel_size=1)
        self.center_out = nn.Conv2d(in_ch, 1, kernel_size=1)
        self._init_params()

    def _init_params(self) -> None:
        nn.init.normal_(self.cls_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_out.bias, -1.2)

        nn.init.normal_(self.reg_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reg_out.bias, 1.0)

        nn.init.normal_(self.center_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.center_out.bias, -1.0)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)
        return {
            "cls_logits": self.cls_out(cls_feat),
            "reg_preds": F.relu(self.reg_out(reg_feat)),
            "center_logits": self.center_out(reg_feat),
        }


class AnchorFreeDetector(nn.Module):
    """
    Full detector: ResNet34 + FPN(4 levels) + per-level anchor-free heads.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        feat_channels: int = FPN_CHANNELS,
        pretrained: bool = True,
        legacy_single_output: bool = False,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.legacy_single_output = bool(legacy_single_output)

        self.backbone_fpn = ResNet34FPN3L(fpn_channels=feat_channels, pretrained=pretrained)
        self.head_s4 = AnchorFreeHead(in_ch=feat_channels, num_classes=num_classes, num_convs=HEAD_NUM_CONVS)
        self.head_s8 = AnchorFreeHead(in_ch=feat_channels, num_classes=num_classes, num_convs=HEAD_NUM_CONVS)
        self.head_s16 = AnchorFreeHead(in_ch=feat_channels, num_classes=num_classes, num_convs=HEAD_NUM_CONVS)
        self.head_s32 = AnchorFreeHead(in_ch=feat_channels, num_classes=num_classes, num_convs=HEAD_NUM_CONVS)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backbone_fpn(x)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        p1_out, p2_out, p3_out, p4_out = self.extract_features(x)

        out4 = self.head_s4(p1_out)
        out8 = self.head_s8(p2_out)
        out16 = self.head_s16(p3_out)
        out32 = self.head_s32(p4_out)

        outputs: Dict[str, object] = {
            "features": {
                "stride4": p1_out,
                "stride8": p2_out,
                "stride16": p3_out,
                "stride32": p4_out,
            },
            "cls_logits": [
                out4["cls_logits"],
                out8["cls_logits"],
                out16["cls_logits"],
                out32["cls_logits"],
            ],
            "reg_preds": [
                out4["reg_preds"],
                out8["reg_preds"],
                out16["reg_preds"],
                out32["reg_preds"],
            ],
            "center_logits": [
                out4["center_logits"],
                out8["center_logits"],
                out16["center_logits"],
                out32["center_logits"],
            ],
            "strides": list(STRIDES),
        }

        if self.legacy_single_output:
            size4 = out4["cls_logits"].shape[-2:]
            cls8_up = F.interpolate(out8["cls_logits"], size=size4, mode="nearest")
            cls16_up = F.interpolate(out16["cls_logits"], size=size4, mode="nearest")
            cls32_up = F.interpolate(out32["cls_logits"], size=size4, mode="nearest")
            reg8_up = F.interpolate(out8["reg_preds"], size=size4, mode="nearest")
            reg16_up = F.interpolate(out16["reg_preds"], size=size4, mode="nearest")
            reg32_up = F.interpolate(out32["reg_preds"], size=size4, mode="nearest")
            ctr8_up = F.interpolate(out8["center_logits"], size=size4, mode="nearest")
            ctr16_up = F.interpolate(out16["center_logits"], size=size4, mode="nearest")
            ctr32_up = F.interpolate(out32["center_logits"], size=size4, mode="nearest")

            outputs["cls_logits_legacy"] = (out4["cls_logits"] + cls8_up + cls16_up + cls32_up) / 4.0
            outputs["reg_preds_legacy"] = (out4["reg_preds"] + reg8_up + reg16_up + reg32_up) / 4.0
            outputs["center_logits_legacy"] = (out4["center_logits"] + ctr8_up + ctr16_up + ctr32_up) / 4.0

        return outputs


# Backward-compatible alias for older imports.
ResNet18FPN2L = ResNet34FPN3L
