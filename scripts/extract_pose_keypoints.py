from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TASK_MODEL_PATH = PROJECT_ROOT / "weights" / "pose_landmarker_lite.task"
DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class PoseSample:
    image_path: Path
    label_name: str
    label_index: int
    split: str


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Pose keypoints for the traffic gesture dataset.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "mobilenetv3_baseline.yaml"),
        help="Training config used only for dataset_root resolution.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "pose_keypoints"),
        help="Directory to save extracted keypoints JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to extract.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_TASK_MODEL_PATH),
        help="Path to pose_landmarker.task model file when using MediaPipe Tasks API.",
    )
    return parser.parse_args()


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_landmark(landmark):
    return {
        "x": round(float(landmark.x), 6),
        "y": round(float(landmark.y), 6),
        "z": round(float(landmark.z), 6),
        "visibility": round(float(getattr(landmark, "visibility", 0.0)), 6),
        "presence": round(float(getattr(landmark, "presence", 0.0)), 6),
    }


def resolve_pose_api(model_path: str | Path):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("MediaPipe is not installed. Run: pip install mediapipe") from exc

    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise SystemExit(
                "MediaPipe Tasks API detected, but the pose model file is missing. "
                f"Expected at: {model_path}. Download pose_landmarker_lite.task first, for example from: "
                f"{DEFAULT_TASK_MODEL_URL}"
            )
        return "tasks", mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return "legacy", mp.solutions.pose

    mp_attrs = [name for name in dir(mp) if not name.startswith("_")]

    try:
        from mediapipe.python.solutions import pose as mp_pose

        return "legacy", mp_pose
    except Exception as exc:
        raise SystemExit(
            "Installed mediapipe package does not expose Pose Solution API. "
            f"Top-level attrs: {mp_attrs[:30]}. "
            f"Inner error: {type(exc).__name__}: {exc}"
        ) from exc


def load_split_samples(dataset_root: Path, split: str):
    csv_path = dataset_root / split / "_classes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    samples = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        class_names = reader.fieldnames[1:]
        class_to_index = {name: index for index, name in enumerate(class_names)}

        for row in reader:
            label_name = max(class_names, key=lambda name: int(row[name]))
            samples.append(
                PoseSample(
                    image_path=dataset_root / split / row["filename"],
                    label_name=label_name,
                    label_index=class_to_index[label_name],
                    split=split,
                )
            )
    return samples


def create_tasks_pose_landmarker(mp, model_path: str | Path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)


def extract_landmarks_with_tasks(mp, pose_landmarker, image_path: Path):
    rgb = np.array(Image.open(image_path).convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_landmarker.detect(mp_image)

    all_poses = []
    for pose_landmarks in result.pose_landmarks:
        all_poses.append([normalize_landmark(landmark) for landmark in pose_landmarks])
    return all_poses


def extract_landmarks_with_legacy(pose, image_path: Path):
    rgb = np.array(Image.open(image_path).convert("RGB"))
    result = pose.process(image=rgb)
    if result.pose_landmarks is None:
        return []
    return [[normalize_landmark(landmark) for landmark in result.pose_landmarks.landmark]]


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_root = PROJECT_ROOT / config["data"]["dataset_root"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_kind, pose_api = resolve_pose_api(args.model_path)

    if api_kind == "tasks":
        pose_runtime = create_tasks_pose_landmarker(pose_api, args.model_path)
    else:
        pose_runtime = pose_api.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    summary = {"dataset_root": str(dataset_root), "splits": {}, "output_dir": str(output_dir)}

    try:
        for split in args.splits:
            samples = load_split_samples(dataset_root=dataset_root, split=split)
            split_records = []
            found_count = 0

            for sample in samples:
                image_path = sample.image_path
                if api_kind == "tasks":
                    all_poses = extract_landmarks_with_tasks(pose_api, pose_runtime, image_path)
                else:
                    all_poses = extract_landmarks_with_legacy(pose_runtime, image_path)

                record = {
                    "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                    "label_index": sample.label_index,
                    "label_name": sample.label_name,
                    "pose_detected": len(all_poses) > 0,
                    "num_poses": len(all_poses),
                    "landmarks": all_poses[0] if all_poses else [],
                    "poses": all_poses,
                }

                if all_poses:
                    found_count += 1

                split_records.append(record)

            split_out = output_dir / f"{split}_pose_keypoints.json"
            with open(split_out, "w", encoding="utf-8") as handle:
                json.dump(split_records, handle, ensure_ascii=False, indent=2)

            summary["splits"][split] = {
                "num_samples": len(split_records),
                "pose_detected": found_count,
                "detection_rate": round(found_count / max(len(split_records), 1), 6),
                "file": str(split_out.relative_to(PROJECT_ROOT)),
            }
            print(f"[{split}] pose detected on {found_count}/{len(split_records)} images -> {split_out}")
    finally:
        if api_kind == "tasks":
            pose_runtime.close()
        else:
            pose_runtime.close()

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
