from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLASS_NAMES = [
    "change lanes",
    "go straight",
    "pull over",
    "slow down",
    "stop",
    "turn left",
    "turn right",
    "wait for  left turn",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert labeled videos into a frame classification dataset.")
    parser.add_argument("--input-root", default=str(PROJECT_ROOT / "police_gesture_v1"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "video_frames_ctp_v1"))
    parser.add_argument("--sample-every", type=int, default=5, help="Keep one frame every N frames.")
    parser.add_argument("--max-per-class-per-video", type=int, default=80)
    parser.add_argument("--skip-background", action="store_true", default=True)
    parser.add_argument("--max-videos-per-split", type=int, default=0, help="0 means no limit.")
    return parser.parse_args()


def ensure_split_dirs(root: Path):
    for split in ["train", "valid", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)


def write_classes_csv(split_dir: Path, rows: list[dict]):
    split_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", *CLASS_NAMES]
    with open(split_dir / "_classes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def convert_video(video_path: Path, csv_path: Path, output_split_dir: Path, sample_every: int, max_per_class_per_video: int):
    with open(csv_path, "r", encoding="utf-8") as handle:
        labels = [int(x) for x in next(csv.reader(handle))]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    rows = []
    kept_per_class = defaultdict(int)
    frame_index = -1
    saved = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_index += 1
            if frame_index >= len(labels):
                break

            label_id = int(labels[frame_index])
            if label_id <= 0:
                continue
            if frame_index % max(1, sample_every) != 0:
                continue

            class_index = label_id - 1
            if class_index < 0 or class_index >= len(CLASS_NAMES):
                continue
            label_name = CLASS_NAMES[class_index]
            if kept_per_class[label_name] >= max_per_class_per_video:
                continue

            filename = f"{video_path.stem}_f{frame_index:06d}.jpg"
            out_path = output_split_dir / filename
            ok_write = cv2.imwrite(str(out_path), frame)
            if not ok_write:
                continue

            row = {"filename": filename}
            for idx, class_name in enumerate(CLASS_NAMES):
                row[class_name] = 1 if idx == class_index else 0
            rows.append(row)
            kept_per_class[label_name] += 1
            saved += 1
    finally:
        cap.release()

    return rows, saved


def relocate_rows(rows: list[dict], source_dir: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = row["filename"]
        source_path = source_dir / filename
        target_path = target_dir / filename
        if source_path.exists():
            shutil.move(str(source_path), str(target_path))


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_split_dirs(output_root)

    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    test_rows: list[dict] = []

    summary = {}

    for split_name in ["train", "test"]:
        split_dir = input_root / split_name
        if not split_dir.exists():
            continue

        video_paths = sorted(split_dir.glob("*.mp4"))
        if args.max_videos_per_split and args.max_videos_per_split > 0:
            video_paths = video_paths[: args.max_videos_per_split]
        total_saved = 0
        split_rows: list[dict] = []

        for video_path in video_paths:
            csv_path = video_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            rows, saved = convert_video(
                video_path=video_path,
                csv_path=csv_path,
                output_split_dir=output_root / split_name,
                sample_every=args.sample_every,
                max_per_class_per_video=args.max_per_class_per_video,
            )
            split_rows.extend(rows)
            total_saved += saved

        summary[split_name] = {"videos": len(video_paths), "frames_saved": total_saved}
        if split_name == "train":
            split_rows.sort(key=lambda item: item["filename"])
            cut = max(1, int(len(split_rows) * 0.1))
            valid_rows = split_rows[-cut:]
            train_rows = split_rows[:-cut]
            relocate_rows(valid_rows, output_root / "train", output_root / "valid")
        else:
            test_rows = split_rows

    write_classes_csv(output_root / "train", train_rows)
    write_classes_csv(output_root / "valid", valid_rows)
    write_classes_csv(output_root / "test", test_rows)

    print(
        {
            "output_root": str(output_root),
            "train_samples": len(train_rows),
            "valid_samples": len(valid_rows),
            "test_samples": len(test_rows),
            "summary": summary,
        }
    )


if __name__ == "__main__":
    main()
