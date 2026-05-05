from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _resolve_mobilenet_builder():
    weights_enum = None
    if hasattr(models, "MobileNet_V3_Small_Weights"):
        weights_enum = models.MobileNet_V3_Small_Weights
    return models.mobilenet_v3_small, weights_enum


def build_mobilenetv3_small(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    builder, weights_enum = _resolve_mobilenet_builder()

    if weights_enum is not None:
        weights = weights_enum.DEFAULT if pretrained else None
        model = builder(weights=weights)
    else:
        model = builder(pretrained=pretrained)

    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("Unexpected MobileNetV3 classifier structure.")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if len(model.classifier) >= 4 and isinstance(model.classifier[2], nn.Dropout):
        model.classifier[2] = nn.Dropout(p=dropout, inplace=True)

    return model
