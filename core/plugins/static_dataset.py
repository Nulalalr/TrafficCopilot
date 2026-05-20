from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.system.contracts import Sample


@dataclass
class StaticDataset:
    class_names: list[str]
    dataset_root: Path | None = None

    def samples(self, split: str) -> list[Sample]:
        return []

