from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a markdown evaluation report from video eval JSON.")
    parser.add_argument("--eval-json", required=True, help="Path to eval summary JSON produced by run_labeled_video_realtime.py")
    parser.add_argument("--output-md", required=True, help="Path to write markdown report")
    parser.add_argument("--experiment-metrics", default="", help="Optional path to experiments/**/metrics.json")
    parser.add_argument("--title", default="视频识别评估报告（police_gesture_v1）")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_table(rows: list[list[str]]):
    if not rows:
        return ""
    header = rows[0]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    eval_json_path = Path(args.eval_json)
    if not eval_json_path.is_absolute():
        eval_json_path = project_root / eval_json_path
    eval_data = read_json(eval_json_path)

    exp_metrics = None
    exp_metrics_path = str(args.experiment_metrics or "").strip()
    if exp_metrics_path:
        p = Path(exp_metrics_path)
        if not p.is_absolute():
            p = project_root / p
        if p.exists():
            exp_metrics = read_json(p)

    title = str(args.title)
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## 1. 评测设置")
    lines.append("")
    lines.append(f"- 评测时间：自动生成")
    lines.append(f"- 使用配置：{eval_data.get('config', '')}")
    lines.append(f"- 数据集根目录：{eval_data.get('dataset_root', '')}")
    lines.append(f"- 抽样：每 {eval_data.get('sample_every', 1)} 帧做一次预测")
    lines.append(f"- 平滑：滑窗投票窗口 = {eval_data.get('smooth_window', 1)}")
    lines.append(f"- 最大帧数：{eval_data.get('max_frames', 0)}（0 表示不限制）")
    lines.append(f"- 限速：{eval_data.get('fps_limit', 0.0)}（0 表示不限制）")
    lines.append("")

    if exp_metrics is not None:
        lines.append("## 2. 训练侧指标（可选）")
        lines.append("")
        for key in ["best_valid_acc", "test_acc", "test_loss", "best_checkpoint", "last_checkpoint"]:
            if key in exp_metrics:
                lines.append(f"- {key}: {exp_metrics[key]}")
        lines.append("")

    lines.append("## 3. 视频级别评测结果（帧级准确率，忽略背景帧）")
    lines.append("")
    lines.append(f"- 总视频数：{eval_data.get('videos', 0)}")
    lines.append(f"- 评测帧数（非背景）：{eval_data.get('frames_eval', 0)}")
    lines.append(f"- 正确帧数：{eval_data.get('correct', 0)}")
    lines.append(f"- 总体准确率：{eval_data.get('accuracy', 0.0)}")
    lines.append("")

    details = list(eval_data.get("details") or [])
    details_sorted = sorted(details, key=lambda item: float(item.get("accuracy", 0.0)))
    table_rows: list[list[str]] = [["视频", "评测帧数", "正确帧数", "准确率"]]
    for item in details_sorted:
        table_rows.append(
            [
                str(item.get("video", "")),
                str(item.get("frames_eval", "")),
                str(item.get("correct", "")),
                str(item.get("accuracy", "")),
            ]
        )
    lines.append(format_table(table_rows))
    lines.append("")

    worst = details_sorted[: min(5, len(details_sorted))]
    if worst:
        lines.append("## 4. 最差视频 Top5（定位问题用）")
        lines.append("")
        for item in worst:
            lines.append(f"- {item.get('video', '')}: acc={item.get('accuracy', '')}, frames={item.get('frames_eval', '')}")
        lines.append("")

    out_path = Path(args.output_md)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

