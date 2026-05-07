from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3GRUClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        backbone = self._build_backbone(pretrained=pretrained)
        self.feature_extractor = backbone
        self.feature_dim = 576

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.temporal = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), num_classes),
        )

    @staticmethod
    def _build_backbone(pretrained: bool) -> nn.Module:
        if hasattr(models, "MobileNet_V3_Small_Weights"):
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
        else:
            model = models.mobilenet_v3_small(pretrained=pretrained)

        return nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = frames.shape
        flat = frames.view(batch_size * time_steps, channels, height, width)
        features = self.feature_extractor(flat)
        features = features.view(batch_size, time_steps, self.feature_dim)
        temporal_out, _ = self.temporal(features)
        logits = self.head(temporal_out[:, -1, :])
        return logits
