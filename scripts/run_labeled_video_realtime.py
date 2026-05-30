from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
from PIL import Image

from core.system.factory import build_system, load_yaml


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/web_video_frames.yaml")
    parser.add_argument("--dataset-root", default="data/raw/police_gesture_v1")
    parser.add_argument("--video", default="", help="Path to a .mp4 file. If empty, will evaluate all videos in split.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--no-display", action="store_true", default=False)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps-limit", type=float, default=0.0)
    parser.add_argument("--output-json", default="", help="Write evaluation summary as JSON to this path.")
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


def _vote(values: deque[str]) -> str:
    if not values:
        return ""
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def run_one_video(system, video_path: Path, csv_path: Path, sample_every: int, smooth_window: int, display: bool, max_frames: int, fps_limit: float):
    labels = load_labels(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_idx = -1
    total = 0
    correct = 0
    pred_buf: deque[str] = deque(maxlen=max(1, int(smooth_window)))
    last_pred = ""

    if display:
        cv2.namedWindow(video_path.name, cv2.WINDOW_NORMAL)

    last_tick = time.perf_counter()
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

            if fps_limit and fps_limit > 0:
                now = time.perf_counter()
                dt = now - last_tick
                min_dt = 1.0 / max(1e-6, float(fps_limit))
                if dt < min_dt:
                    time.sleep(max(0.0, min_dt - dt))
                last_tick = time.perf_counter()

            gt_id = int(labels[frame_idx])
            gt_label = CLASS_NAMES[gt_id - 1] if 1 <= gt_id <= len(CLASS_NAMES) else "BACKGROUND"

            do_predict = sample_every <= 1 or frame_idx % max(1, int(sample_every)) == 0
            if do_predict:
                pil = Image.fromarray(frame[:, :, ::-1]).convert("RGB")
                pred = system.predictor.predict(pil)
                last_pred = str(pred.label)
                pred_buf.append(last_pred)

            smooth_pred = _vote(pred_buf) if smooth_window and smooth_window > 1 else last_pred

            if gt_id > 0:
                total += 1
                if smooth_pred == gt_label:
                    correct += 1

            if display:
                acc = (correct / total) if total else 0.0
                line1 = f"GT: {gt_label}"
                line2 = f"PRED: {smooth_pred}"
                line3 = f"ACC: {acc:.4f} ({correct}/{total}) | fps_in={fps_in:.1f}"
                cv2.putText(frame, line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line3, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(video_path.name, frame)
                key = int(cv2.waitKey(1)) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyWindow(video_path.name)

    return {
        "video": str(video_path),
        "csv": str(csv_path),
        "frames_labeled": int(sum(1 for x in labels if int(x) > 0)),
        "frames_eval": int(total),
        "correct": int(correct),
        "accuracy": round(float(correct / max(1, total)), 6),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_yaml(config_path)
    system = build_system(config, project_root=project_root)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    display = bool(args.display) and not bool(args.no_display)
    sample_every = max(1, int(args.sample_every))
    smooth_window = max(1, int(args.smooth_window))

    video_arg = str(args.video or "").strip()
    results = []
    if video_arg:
        video_path = Path(video_arg)
        if not video_path.is_absolute():
            video_path = project_root / video_path
        csv_path = video_path.with_suffix(".csv")
        results.append(
            run_one_video(
                system=system,
                video_path=video_path,
                csv_path=csv_path,
                sample_every=sample_every,
                smooth_window=smooth_window,
                display=display,
                max_frames=int(args.max_frames or 0),
                fps_limit=float(args.fps_limit or 0.0),
            )
        )
    else:
        split_dir = dataset_root / str(args.split)
        for video_path in sorted(split_dir.glob("*.mp4")):
            csv_path = video_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            results.append(
                run_one_video(
                    system=system,
                    video_path=video_path,
                    csv_path=csv_path,
                    sample_every=sample_every,
                    smooth_window=smooth_window,
                    display=display,
                    max_frames=int(args.max_frames or 0),
                    fps_limit=float(args.fps_limit or 0.0),
                )
            )

    if not results:
        raise SystemExit("No videos found.")

    total = sum(int(r["frames_eval"]) for r in results)
    correct = sum(int(r["correct"]) for r in results)
    summary = {
        "videos": int(len(results)),
        "frames_eval": int(total),
        "correct": int(correct),
        "accuracy": round(float(correct / max(1, total)), 6),
        "config": str(Path(args.config)),
        "dataset_root": str(Path(args.dataset_root)),
        "sample_every": int(sample_every),
        "smooth_window": int(smooth_window),
        "max_frames": int(args.max_frames or 0),
        "fps_limit": float(args.fps_limit or 0.0),
        "details": results,
    }
    print(json.dumps(summary, ensure_ascii=False))

    output_json = str(args.output_json or "").strip()
    if output_json:
        out_path = Path(output_json)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
