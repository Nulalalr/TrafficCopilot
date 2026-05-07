from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VideoSequenceSample:
    video_path: Path
    csv_path: Path
    split: str


class TrafficGestureVideoDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        class_names: list[str],
        clip_len: int = 16,
        stride: int = 8,
        transform=None,
        include_background: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.class_names = list(class_names)
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.transform = transform
        self.include_background = bool(include_background)

        self.samples = self._discover_samples()
        self.sequences = self._build_sequences()

    def _discover_samples(self) -> list[VideoSequenceSample]:
        split_dir = self.dataset_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(str(split_dir))

        samples: list[VideoSequenceSample] = []
        for video_path in sorted(split_dir.glob("*.mp4")):
            csv_path = video_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            samples.append(VideoSequenceSample(video_path=video_path, csv_path=csv_path, split=self.split))

        if not samples:
            raise FileNotFoundError(f"No mp4/csv pairs found under {split_dir}")
        return samples

    def _load_labels(self, csv_path: Path) -> list[int]:
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            row = next(reader)
        return [int(x) for x in row]

    def _build_sequences(self) -> list[dict]:
        sequences: list[dict] = []
        for sample in self.samples:
            labels = self._load_labels(sample.csv_path)
            total_frames = len(labels)
            if total_frames < self.clip_len:
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
                        "start": start,
                        "target": target_index,
                        "frame_labels": clip_labels,
                    }
                )
        if not sequences:
            raise ValueError(f"No valid sequences built for split={self.split}")
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        item = self.sequences[index]
        video_path = item["video_path"]
        start = int(item["start"])
        target = int(item["target"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        try:
            for _ in range(self.clip_len):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                image = Image.fromarray(frame[:, :, ::-1]).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                frames.append(image)
        finally:
            cap.release()

        if len(frames) != self.clip_len:
            raise RuntimeError(f"Short clip read from {video_path} at start={start}")

        frame_tensor = torch.stack(frames, dim=0)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return frame_tensor, target_tensor

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for seq in self.sequences:
            counts[self.class_names[int(seq["target"])]] += 1
        return counts
