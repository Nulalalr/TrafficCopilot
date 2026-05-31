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

    @staticmethod
    def _find_label_runs(labels: list[int]) -> list[tuple[int, int, int]]:
        runs: list[tuple[int, int, int]] = []
        if not labels:
            return runs

        run_start = 0
        run_label = int(labels[0])
        for index in range(1, len(labels)):
            current_label = int(labels[index])
            if current_label == run_label:
                continue
            runs.append((run_start, index, run_label))
            run_start = index
            run_label = current_label
        runs.append((run_start, len(labels), run_label))
        return runs

    def _build_sequences(self) -> list[dict]:
        sequences: list[dict] = []
        for sample in self.samples:
            labels = self._load_labels(sample.csv_path)
            pose_path = self._pose_cache_path(sample.video_path)
            if not pose_path.exists():
                continue

            for start, end, label_id in self._find_label_runs(labels):
                if label_id == 0 and not self.include_background:
                    continue
                if label_id > 0:
                    target_index = label_id - 1
                else:
                    target_index = -1
                if target_index < 0 or target_index >= len(self.class_names):
                    continue

                segment_length = end - start
                if segment_length <= 0:
                    continue

                segment_starts: list[int]
                if segment_length <= self.clip_len:
                    segment_starts = [start]
                else:
                    segment_starts = list(range(start, end - self.clip_len + 1, self.stride))
                    last_start = end - self.clip_len
                    if not segment_starts or segment_starts[-1] != last_start:
                        segment_starts.append(last_start)

                for clip_start in segment_starts:
                    clip_end = min(end, clip_start + self.clip_len)
                    sequences.append(
                        {
                            "video_path": sample.video_path,
                            "pose_path": pose_path,
                            "start": clip_start,
                            "end": clip_end,
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
        end = int(item["end"])
        target = int(item["target"])

        with np.load(pose_path) as npz:
            pose = npz["pose"]
        clip = pose[start:end]
        if clip.shape[0] <= 0:
            raise RuntimeError(f"Empty pose clip read from {pose_path} at start={start}, end={end}")

        if clip.shape[0] >= self.clip_len:
            clip = clip[: self.clip_len]
        else:
            pad_count = self.clip_len - clip.shape[0]
            pad = np.repeat(clip[-1:, :], pad_count, axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        pose_tensor = torch.from_numpy(clip.astype(np.float32))
        target_tensor = torch.tensor(target, dtype=torch.long)
        return pose_tensor, target_tensor

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for seq in self.sequences:
            counts[self.class_names[int(seq["target"])]] += 1
        return counts
