from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from core.plugins.mediapipe_pose_extractor import MediaPipePoseExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/raw/police_gesture_v1")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output-root", default="output/pose_sequences")
    parser.add_argument("--model-path", default="weights/pose_landmarker_lite.task")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_labels(csv_path: Path) -> list[int]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        row = next(csv.reader(handle))
    labels: list[int] = []
    for cell in row:
        cell = str(cell).strip()
        if not cell:
            continue
        labels.append(int(cell))
    return labels


def landmarks_to_array(landmarks: list, expected: int = 33) -> np.ndarray:
    arr = np.zeros((expected, 4), dtype=np.float32)
    n = min(expected, len(landmarks))
    for i in range(n):
        lm = landmarks[i]
        arr[i, 0] = float(getattr(lm, "x", 0.0))
        arr[i, 1] = float(getattr(lm, "y", 0.0))
        arr[i, 2] = float(getattr(lm, "z", 0.0))
        arr[i, 3] = float(getattr(lm, "visibility", 0.0))
    return arr


def process_video(extractor: MediaPipePoseExtractor, video_path: Path, csv_path: Path, out_path: Path, every: int, max_frames: int):
    labels = load_labels(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_idx = -1
    pose_list: list[np.ndarray] = []
    kept_index: list[int] = []

    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if frame_idx >= len(labels):
                break
            if max_frames and frame_idx >= int(max_frames):
                break
            if every > 1 and frame_idx % max(1, int(every)) != 0:
                continue

            ts_ms = int(round(frame_idx * 1000.0 / max(1e-6, fps)))
            poses = extractor.extract(frame, timestamp_ms=ts_ms)
            if poses:
                pose_arr = landmarks_to_array(poses[0])
            else:
                pose_arr = np.zeros((33, 4), dtype=np.float32)
            pose_list.append(pose_arr)
            kept_index.append(int(frame_idx))
    finally:
        cap.release()

    pose = np.stack(pose_list, axis=0) if pose_list else np.zeros((0, 33, 4), dtype=np.float32)
    kept = np.array(kept_index, dtype=np.int32)
    kept_labels = np.array([int(labels[i]) for i in kept_index], dtype=np.int32)
    elapsed = time.perf_counter() - start

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pose=pose.reshape((pose.shape[0], -1)),
        kept_index=kept,
        labels=kept_labels,
        fps=np.float32(fps),
        source=str(video_path),
        elapsed_sec=np.float32(elapsed),
    )
    return {
        "video": str(video_path),
        "frames_total": int(len(labels)),
        "frames_kept": int(len(kept_index)),
        "fps": float(fps),
        "elapsed_sec": round(float(elapsed), 3),
        "out": str(out_path),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    split_dir = dataset_root / str(args.split)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    out_split = output_root / str(args.split)
    out_split.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    extractor = MediaPipePoseExtractor(model_path=model_path, project_root=project_root)

    video_paths = sorted(split_dir.glob("*.mp4"))
    if args.max_videos and int(args.max_videos) > 0:
        video_paths = video_paths[: int(args.max_videos)]

    results = []
    for video_path in video_paths:
        csv_path = video_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        out_path = out_split / f"{video_path.stem}.npz"
        results.append(
            process_video(
                extractor=extractor,
                video_path=video_path,
                csv_path=csv_path,
                out_path=out_path,
                every=int(args.every),
                max_frames=int(args.max_frames or 0),
            )
        )

    if not results:
        raise SystemExit("No videos processed.")
    print({"videos": len(results), "details": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

