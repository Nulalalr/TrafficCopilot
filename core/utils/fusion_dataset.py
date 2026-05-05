from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from core.utils.dataset import TrafficGestureDataset
from core.utils.pose_features import build_pose_lookup, extract_pose_feature_vector, load_pose_json


class TrafficGestureFusionDataset(TrafficGestureDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        pose_json_path: str | Path,
        transform: Optional[Callable] = None,
    ):
        super().__init__(dataset_root=dataset_root, split=split, transform=transform, return_metadata=False)
        self.pose_json_path = Path(pose_json_path)
        self.pose_records = build_pose_lookup(load_pose_json(self.pose_json_path))

    def __getitem__(self, index: int):
        image, label_index = super().__getitem__(index)
        sample = self.samples[index]
        relative_path = sample.image_path.relative_to(Path.cwd()).as_posix()
        alt_relative_path = sample.image_path.relative_to(self.dataset_root.parent).as_posix()

        record = self.pose_records.get(relative_path) or self.pose_records.get(alt_relative_path)
        if record is None:
            raise KeyError(f"Pose record not found for {sample.image_path}")

        pose_feature = torch.tensor(extract_pose_feature_vector(record), dtype=torch.float32)
        return image, pose_feature, label_index
