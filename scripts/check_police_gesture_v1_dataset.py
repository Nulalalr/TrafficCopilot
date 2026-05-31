from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


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


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check police_gesture_v1 dataset (mp4 + per-frame csv labels).")
    parser.add_argument("--dataset-root", default="data/raw/police_gesture_v1")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--tolerance-frames", type=int, default=2, help="Allowed abs(video_frames - labels_len).")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--output-json", default="", help="Optional path to write JSON report.")
    return parser.parse_args()


def read_labels(csv_path: Path) -> list[int]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        row = next(csv.reader(handle))
    labels: list[int] = []
    for cell in row:
        cell = str(cell).strip()
        if cell == "":
            continue
        labels.append(int(cell))
    return labels


def get_video_meta(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": False, "error": "video_open_error"}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "ok": True,
        "frames": frame_count,
        "fps": round(fps, 4),
        "width": width,
        "height": height,
    }


def summarize_labels(labels: list[int]) -> dict:
    counts = {str(i): 0 for i in range(0, 9)}
    for x in labels:
        counts[str(int(x))] = counts.get(str(int(x)), 0) + 1
    background = int(counts.get("0", 0))
    non_bg = len(labels) - background
    return {
        "labels_len": len(labels),
        "background_frames": background,
        "non_background_frames": non_bg,
        "background_ratio": round(background / max(1, len(labels)), 6),
        "per_id_counts": counts,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    report: dict = {"dataset_root": str(dataset_root), "splits": {}}
    ok_all = True

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            report["splits"][split] = {"exists": False}
            ok_all = False
            continue

        mp4_all = sorted(split_dir.glob("*.mp4"))
        csv_all = sorted(split_dir.glob("*.csv"))
        mp4_paths = list(mp4_all)
        if args.max_videos and int(args.max_videos) > 0:
            mp4_paths = mp4_paths[: int(args.max_videos)]

        mp4_stems = {p.stem for p in mp4_all}
        csv_stems = {p.stem for p in csv_all}
        missing_csv = sorted(mp4_stems - csv_stems)
        missing_mp4 = sorted(csv_stems - mp4_stems)

        bad: list[dict] = []
        agg = {
            "videos": len(mp4_all),
            "csvs": len(csv_all),
            "videos_checked": len(mp4_paths),
            "missing_csv": len(missing_csv),
            "missing_mp4": len(missing_mp4),
            "bad": 0,
            "missing_csv_examples": missing_csv[:5],
            "missing_mp4_examples": missing_mp4[:5],
        }

        per_class_nonbg = {name: 0 for name in CLASS_NAMES}
        total_nonbg = 0

        for vp in mp4_paths:
            cp = vp.with_suffix(".csv")
            if not cp.exists():
                continue
            try:
                labels = read_labels(cp)
            except Exception as e:
                bad.append({"video": vp.name, "issue": "csv_read_error", "detail": str(e)})
                continue

            if not labels:
                bad.append({"video": vp.name, "issue": "empty_labels", "detail": ""})
                continue

            mn = min(labels)
            mx = max(labels)
            if mn < 0 or mx > 8:
                bad.append({"video": vp.name, "issue": "label_out_of_range", "detail": f"min={mn},max={mx}"})

            video_meta = get_video_meta(vp)
            if not video_meta["ok"]:
                bad.append({"video": vp.name, "issue": video_meta["error"], "detail": ""})
                continue

            frames = int(video_meta.get("frames", 0))
            tol = int(args.tolerance_frames)
            if frames and abs(frames - len(labels)) > tol:
                bad.append(
                    {
                        "video": vp.name,
                        "issue": "len_mismatch",
                        "detail": f"video_frames={frames}, labels_len={len(labels)}, tolerance={tol}",
                    }
                )

            for x in labels:
                if int(x) <= 0:
                    continue
                idx = int(x) - 1
                if 0 <= idx < len(CLASS_NAMES):
                    per_class_nonbg[CLASS_NAMES[idx]] += 1
                    total_nonbg += 1

        agg["bad"] = len(bad)
        agg["bad_examples"] = bad[:5]
        agg["non_background_frames_total"] = int(total_nonbg)
        agg["per_class_nonbg_frames"] = per_class_nonbg
        report["splits"][split] = agg

        if missing_csv or missing_mp4 or bad:
            ok_all = False

    report["ok"] = bool(ok_all)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = str(args.output_json or "").strip()
    if out:
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
