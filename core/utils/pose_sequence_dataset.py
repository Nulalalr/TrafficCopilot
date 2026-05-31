from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = [
    "Stop",
    "Forward",
    "Left Turn",
    "Left Turn Waiting",
    "Right Turn",
    "Lane Changing",
    "Slow Down",
    "Pull Over",
]


@dataclass(frozen=True)
class PoseVideoSample:
    video_path: Path
    csv_path: Path
    split: str


class PoliceGesturePoseSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        clip_len: int = 16,
        stride: int = 8,
        include_background: bool = False,
        pose_root: str | Path | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.include_background = bool(include_background)
        self.pose_root = Path(pose_root) if pose_root is not None else None

        self.class_names = list(CLASS_NAMES)
        self.samples = self._discover_samples()
        self.sequences = self._build_sequences()

    def _discover_samples(self) -> list[PoseVideoSample]:
        split_dir = self.dataset_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(str(split_dir))
        samples: list[PoseVideoSample] = []
        for video_path in sorted(split_dir.glob("*.mp4")):
            csv_path = video_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            samples.append(PoseVideoSample(video_path=video_path, csv_path=csv_path, split=self.split))
        if not samples:
            raise FileNotFoundError(f"No mp4/csv pairs found under {split_dir}")
        return samples

    @staticmethod
    def _load_labels(csv_path: Path) -> list[int]:
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            row = next(reader)
        return [int(x) for x in row]

    def _pose_cache_path(self, video_path: Path) -> Path:
        if self.pose_root is None:
            raise ValueError("pose_root is required for pose sequence dataset")
        split_dir = self.pose_root / self.split
        return split_dir / f"{video_path.stem}.npz"

    def _build_sequences(self) -> list[dict]:
        sequences: list[dict] = []
        for sample in self.samples:
            labels = self._load_labels(sample.csv_path)
            total_frames = len(labels)
            if total_frames < self.clip_len:
                continue

            pose_path = self._pose_cache_path(sample.video_path)
            if not pose_path.exists():
                continue

            for start in range(0, total_frames - self.clip_len + 1, self.stride):
                clip_labels = labels[start : start + self.clip_len]
                positive_labels = [x for x in clip_labels if x > 0]
                if not positive_labels and not self.include_background:
                    continue

                target_id = max(set(clip_labels), key=clip_labels.count)
                if target_id == 0 and not self.include_background:
                    continue
                if target_id > 0:
                    target_index = target_id - 1
                else:
                    target_index = -1

                if target_index < 0 or target_index >= len(self.class_names):
                    continue

                sequences.append(
                    {
                        "video_path": sample.video_path,
                        "pose_path": pose_path,
                        "start": start,
                        "target": target_index,
                    }
                )

        if not sequences:
            raise ValueError(f"No valid pose sequences built for split={self.split}")
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        item = self.sequences[index]
        pose_path = Path(item["pose_path"])
        start = int(item["start"])
        target = int(item["target"])

        with np.load(pose_path) as npz:
            pose = npz["pose"]
        clip = pose[start : start + self.clip_len]
        if clip.shape[0] != self.clip_len:
            raise RuntimeError(f"Short pose clip read from {pose_path} at start={start}")

        pose_tensor = torch.from_numpy(clip.astype(np.float32))
        target_tensor = torch.tensor(target, dtype=torch.long)
        return pose_tensor, target_tensor

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for seq in self.sequences:
            counts[self.class_names[int(seq["target"])]] += 1
        return counts
