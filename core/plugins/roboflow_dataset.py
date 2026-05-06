from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from core.system.contracts import Sample


@dataclass
class RoboflowDataset:
    dataset_root: Path
    class_csv_name: str = "_classes.csv"

    def __post_init__(self):
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        self._class_names: list[str] = []
        self._cache: dict[str, list[Sample]] = {}

    @property
    def class_names(self) -> list[str]:
        if not self._class_names:
            _ = self.samples("train")
        return list(self._class_names)

    def samples(self, split: str) -> list[Sample]:
        split = self._normalize_split(split)
        if split in self._cache:
            return list(self._cache[split])

        csv_path = self.dataset_root / split / self.class_csv_name
        if not csv_path.exists():
            raise FileNotFoundError(str(csv_path))

        samples: list[Sample] = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            class_names = reader.fieldnames[1:] if reader.fieldnames else []
            if not class_names:
                raise ValueError(f"Invalid classes csv header: {csv_path}")
            if not self._class_names:
                self._class_names = list(class_names)
            elif list(class_names) != list(self._class_names):
                raise ValueError(f"Class names mismatch for split={split}: {csv_path}")

            for row in reader:
                filename = row.get("filename")
                if not filename:
                    continue
                label = max(self._class_names, key=lambda name: int(row.get(name, 0)))
                samples.append(Sample(path=self.dataset_root / split / filename, label=label, split=split))

        self._cache[split] = list(samples)
        return list(samples)

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = (split or "").strip().lower()
        if split in {"val", "valid", "validation"}:
            return "valid"
        if split in {"train", "test"}:
            return split
        raise ValueError(f"Invalid split: {split}")
