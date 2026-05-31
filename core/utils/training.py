from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / max(total, 1)


def build_class_weights(class_counts: dict[str, int], class_names: list[str]) -> torch.Tensor:
    counts = np.array([class_counts[name] for name in class_names], dtype=np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def save_json(payload: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
