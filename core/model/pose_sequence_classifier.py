from __future__ import annotations

import torch
import torch.nn as nn


class PoseGRUClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pose_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.temporal = nn.GRU(
            input_size=int(pose_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), num_classes),
        )

    def forward(self, pose_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.temporal(pose_seq)
        logits = self.head(out[:, -1, :])
        return logits

