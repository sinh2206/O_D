from __future__ import annotations

"""
Anchor-Free feature extractor and detection head.

Architecture summary (input 320x320):
- Backbone: ResNet-18 pretrained (no avgpool/fc)
  stem -> layer1 -> layer2 -> layer3(C3, stride 16, 256ch) -> layer4(C4, stride 32, 512ch)
- Neck: 2-level FPN
  lateral3: 1x1 conv (256 -> 128)
  lateral4: 1x1 conv (512 -> 128)
  top-down: upsample(P4) + P3
  smooth: conv3x3 + BN + LeakyReLU(0.1) on both levels
- Outputs:
  P3_out: (B, 128, H/16, W/16), stride 16  (for 320 -> 20x20)
  P4_out: (B, 128, H/32, W/32), stride 32  (for 320 -> 10x10)

Compatibility with preprocess.py:
- Model expects normalized RGB tensors of shape (B, 3, H, W).
- H, W should be divisible by 32 for exact stride-16/32 grid sizes.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import FPN_CHANNELS, NUM_CLASSES


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


class ResNet18FPN2L(nn.Module):
    """Backbone + 2-level FPN feature extractor (stride 16/32)."""

    def __init__(self, fpn_channels: int = FPN_CHANNELS, pretrained: bool = True):
        super().__init__()

        backbone = self._build_resnet18(pretrained=pretrained)

        # Backbone stages.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # C3: stride 16, 256ch
        self.layer4 = backbone.layer4  # C4: stride 32, 512ch

        # Lateral connections.
        self.lateral3 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(512, fpn_channels, kernel_size=1)

        # Smooth layers.
        self.fpn_out3 = ConvBNLeaky(fpn_channels, fpn_channels, k=3, s=1, p=1)
        self.fpn_out4 = ConvBNLeaky(fpn_channels, fpn_channels, k=3, s=1, p=1)

    @staticmethod
    def _build_resnet18(pretrained: bool):
        if not pretrained:
            return models.resnet18(weights=None)

        try:
            return models.resnet18(weights="IMAGENET1K_V1")
        except Exception:
            try:
                from torchvision.models import ResNet18_Weights

                return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                return models.resnet18(weights=None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        c3 = self.layer3(x)
        c4 = self.layer4(c3)

        p3 = self.lateral3(c3)
        p4 = self.lateral4(c4)

        # Top-down fusion: upsample stride32 -> stride16 and add.
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = p3 + p4_up

        # Smooth.
        p3_out = self.fpn_out3(p3)
        p4_out = self.fpn_out4(p4)

        return p3_out, p4_out


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
        # Low prior for positives at init.
        nn.init.normal_(self.cls_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_out.bias, -2.2)

        nn.init.normal_(self.reg_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reg_out.bias, 1.0)

        nn.init.normal_(self.center_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.center_out.bias, -2.2)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)

        cls_logits = self.cls_out(cls_feat)
        reg_preds = F.relu(self.reg_out(reg_feat))
        center_logits = self.center_out(reg_feat)

        return {
            "cls_logits": cls_logits,
            "reg_preds": reg_preds,
            "center_logits": center_logits,
        }


class AnchorFreeDetector(nn.Module):
    """
    Full detector: ResNet18 + FPN(2 levels) + per-level anchor-free heads.

    Multi-scale forward output:
    {
      "features": {"stride16": P3_out, "stride32": P4_out},
      "cls_logits": [cls_s16, cls_s32],
      "reg_preds": [reg_s16, reg_s32],
      "center_logits": [ctr_s16, ctr_s32],
      "strides": [16, 32]
    }

    If `legacy_single_output=True`, additionally returns merged single-scale tensors in
    keys: cls_logits_legacy, reg_preds_legacy, center_logits_legacy (stride16 space).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        feat_channels: int = FPN_CHANNELS,
        pretrained: bool = True,
        legacy_single_output: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.legacy_single_output = legacy_single_output

        self.backbone_fpn = ResNet18FPN2L(fpn_channels=feat_channels, pretrained=pretrained)
        self.head_s16 = AnchorFreeHead(in_ch=feat_channels, num_classes=num_classes, num_convs=2)
        self.head_s32 = AnchorFreeHead(in_ch=feat_channels, num_classes=num_classes, num_convs=2)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone_fpn(x)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        p3_out, p4_out = self.extract_features(x)

        out16 = self.head_s16(p3_out)
        out32 = self.head_s32(p4_out)

        outputs: Dict[str, object] = {
            "features": {"stride16": p3_out, "stride32": p4_out},
            "cls_logits": [out16["cls_logits"], out32["cls_logits"]],
            "reg_preds": [out16["reg_preds"], out32["reg_preds"]],
            "center_logits": [out16["center_logits"], out32["center_logits"]],
            "strides": [16, 32],
        }

        if self.legacy_single_output:
            # Compatibility path: merge stride32 predictions to stride16 resolution.
            size16 = out16["cls_logits"].shape[-2:]
            cls32_up = F.interpolate(out32["cls_logits"], size=size16, mode="nearest")
            reg32_up = F.interpolate(out32["reg_preds"], size=size16, mode="nearest")
            ctr32_up = F.interpolate(out32["center_logits"], size=size16, mode="nearest")

            outputs["cls_logits_legacy"] = 0.5 * (out16["cls_logits"] + cls32_up)
            outputs["reg_preds_legacy"] = 0.5 * (out16["reg_preds"] + reg32_up)
            outputs["center_logits_legacy"] = 0.5 * (out16["center_logits"] + ctr32_up)

        return outputs
