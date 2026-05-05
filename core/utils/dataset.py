from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GestureSample:
    image_path: Path
    label_name: str
    label_index: int
    split: str


class TrafficGestureDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata
        self.samples = []
        self.class_names = []
        self.class_to_index = {}
        self._load()

    def _load(self):
        csv_path = self.dataset_root / self.split / "_classes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Label file not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.class_names = reader.fieldnames[1:]
            self.class_to_index = {name: index for index, name in enumerate(self.class_names)}

            for row in reader:
                label_name = max(self.class_names, key=lambda name: int(row[name]))
                self.samples.append(
                    GestureSample(
                        image_path=self.dataset_root / self.split / row["filename"],
                        label_name=label_name,
                        label_index=self.class_to_index[label_name],
                        split=self.split,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_metadata:
            return image, sample.label_index, str(sample.image_path), sample.label_name
        return image, sample.label_index

    def class_counts(self):
        counts = {name: 0 for name in self.class_names}
        for sample in self.samples:
            counts[sample.label_name] += 1
        return counts
