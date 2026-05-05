from __future__ import annotations

import torch
import torch.nn as nn

from core.model.mobilenetv3_classifier import _resolve_mobilenet_builder


class PoseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.net(x)


class MobileNetV3PoseFusion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pose_input_dim: int,
        pretrained: bool = True,
        image_dropout: float = 0.2,
        pose_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        pose_dropout: float = 0.1,
    ):
        super().__init__()
        builder, weights_enum = _resolve_mobilenet_builder()

        if weights_enum is not None:
            weights = weights_enum.DEFAULT if pretrained else None
            backbone = builder(weights=weights)
        else:
            backbone = builder(pretrained=pretrained)

        self.image_features = backbone.features
        self.image_pool = backbone.avgpool
        self.image_input_dim = backbone.classifier[0].in_features
        self.image_embedding_dim = backbone.classifier[0].out_features
        self.image_projector = nn.Sequential(
            nn.Linear(self.image_input_dim, self.image_embedding_dim),
            nn.Hardswish(inplace=False),
            nn.Dropout(p=image_dropout, inplace=False),
        )

        self.pose_encoder = PoseMLP(
            input_dim=pose_input_dim,
            hidden_dim=pose_hidden_dim,
            dropout=pose_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.image_embedding_dim + pose_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=image_dropout, inplace=False),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, image, pose_feature):
        image_feature = self.image_features(image)
        image_feature = self.image_pool(image_feature)
        image_feature = torch.flatten(image_feature, 1)
        image_feature = self.image_projector(image_feature)

        pose_feature = self.pose_encoder(pose_feature)
        fused = torch.cat([image_feature, pose_feature], dim=1)
        return self.classifier(fused)
