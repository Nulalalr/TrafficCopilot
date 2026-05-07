from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from core.plugins.evaluator import ClassificationEvaluator
from core.plugins.pytorch_classifier import PyTorchMobileNetV3Classifier
from core.plugins.roboflow_dataset import RoboflowDataset
from core.system.factory import load_yaml


@dataclass(frozen=True)
class ModelVariant:
    name: str
    checkpoint_path: Path
    config_path: Path


def _mb(num_bytes: int) -> float:
    return float(num_bytes) / 1024.0 / 1024.0


def _safe_round(v: Any, ndigits: int = 4) -> Any:
    try:
        return round(float(v), ndigits)
    except Exception:
        return v


def _render_confusion_matrix_png(
    labels: list[str],
    cm_norm: list[list[float]],
    title: str,
    out_path: Path,
) -> None:
    n = len(labels)
    cell = 28 if n <= 10 else 22 if n <= 14 else 18
    margin_left = 220
    margin_top = 110
    margin_right = 40
    margin_bottom = 40
    width = margin_left + n * cell + margin_right
    height = margin_top + n * cell + margin_bottom
    img = Image.new("RGB", (width, height), (250, 250, 252))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((24, 20), title, fill=(20, 20, 20), font=font)
    draw.text((24, 44), "Rows: True label | Cols: Pred label (normalized)", fill=(60, 60, 60), font=font)

    for i, name in enumerate(labels):
        y = margin_top + i * cell + cell // 3
        draw.text((24, y), name[:28], fill=(20, 20, 20), font=font)

    for j, name in enumerate(labels):
        x = margin_left + j * cell
        draw.text((x, 76), name[:10], fill=(20, 20, 20), font=font)

    for i in range(n):
        for j in range(n):
            v = float(cm_norm[i][j]) if i < len(cm_norm) and j < len(cm_norm[i]) else 0.0
            v = max(0.0, min(1.0, v))
            if i == j:
                base = (60, 180, 120)
            else:
                base = (220, 100, 90)
            k = int(round(255 * (1.0 - v)))
            r = int(round(base[0] * v + k * (1.0 - v)))
            g = int(round(base[1] * v + k * (1.0 - v)))
            b = int(round(base[2] * v + k * (1.0 - v)))
            x1 = margin_left + j * cell
            y1 = margin_top + i * cell
            draw.rectangle((x1, y1, x1 + cell - 1, y1 + cell - 1), fill=(r, g, b), outline=(235, 235, 238))
            if n <= 12:
                draw.text((x1 + 2, y1 + 6), f"{v:.2f}", fill=(15, 15, 15), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _render_bar_png(items: list[tuple[str, float]], title: str, out_path: Path) -> None:
    width = 1100
    height = 520
    pad = 70
    img = Image.new("RGB", (width, height), (250, 250, 252))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((24, 20), title, fill=(20, 20, 20), font=font)

    if not items:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return

    max_v = max(v for _, v in items)
    max_v = max(1e-6, float(max_v))
    n = len(items)
    bar_h = int(math.floor((height - pad - 60) / max(1, n)))
    bar_h = max(16, min(26, bar_h))
    y = 70
    for name, v in items:
        v = float(v)
        w = int(round((width - 360) * (v / max_v)))
        draw.text((24, y + 4), name[:28], fill=(20, 20, 20), font=font)
        draw.rectangle((250, y, 250 + w, y + bar_h), fill=(60, 140, 210), outline=(200, 200, 210))
        draw.text((260 + w, y + 4), f"{v:.4f}", fill=(20, 20, 20), font=font)
        y += bar_h + 10

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _warmup(predictor: PyTorchMobileNetV3Classifier, dataset: RoboflowDataset, n: int = 6) -> None:
    samples = dataset.samples("train")
    for sample in samples[: max(0, int(n))]:
        with Image.open(sample.path) as img:
            _ = predictor.predict(img)


def _build_predictor(
    project_root: Path,
    dataset: RoboflowDataset,
    variant: ModelVariant,
    device: str,
    thresholds: dict[str, Any],
) -> tuple[PyTorchMobileNetV3Classifier, dict[str, Any]]:
    start = time.perf_counter()
    predictor = PyTorchMobileNetV3Classifier(
        checkpoint_path=variant.checkpoint_path,
        config_path=variant.config_path,
        class_names=dataset.class_names,
        project_root=project_root,
        device=device,
        model_name=variant.name,
        unknown_confidence_threshold=float(thresholds.get("confidence", 0.5)),
        unknown_margin_threshold=float(thresholds.get("margin", 0.08)),
    )
    load_ms = (time.perf_counter() - start) * 1000.0

    model = getattr(predictor, "_model", None)
    dtype = str(next(model.parameters()).dtype) if model is not None else "unknown"
    n_params = int(sum(p.numel() for p in model.parameters())) if model is not None else 0
    n_params_m = round(float(n_params) / 1e6, 3) if n_params else 0.0
    meta = {
        "variant": variant.name,
        "checkpoint_path": str(variant.checkpoint_path),
        "config_path": str(variant.config_path),
        "checkpoint_size_mb": round(_mb(variant.checkpoint_path.stat().st_size), 3) if variant.checkpoint_path.exists() else None,
        "device": str(getattr(predictor, "_device", "unknown")),
        "model_dtype": dtype,
        "params_m": n_params_m,
        "load_ms": round(float(load_ms), 2),
        "unknown_confidence_threshold": float(thresholds.get("confidence", 0.5)),
        "unknown_margin_threshold": float(thresholds.get("margin", 0.08)),
    }
    return predictor, meta


def _write_markdown_report(
    out_path: Path,
    meta: dict[str, Any],
    results: dict[str, Any],
    assets_rel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# 压缩模型性能报告\n")
    lines.append(f"- 生成时间：{meta['generated_at']}")
    lines.append(f"- 设备：{meta['device_summary']}")
    lines.append(f"- 数据集：{meta['dataset_root']}")
    lines.append(f"- 样本数：train={meta['num_train']} valid={meta['num_valid']} test={meta['num_test']}\n")

    lines.append("## 模型对比（概要）\n")
    lines.append("| 模型 | ckpt(MB) | dtype | params(M) | load(ms) | valid acc | valid f1(w) | valid p95(ms) | test acc | test f1(w) | test p95(ms) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, block in results["models"].items():
        valid = block["splits"].get("valid", {})
        test = block["splits"].get("test", {})
        v_lat = (valid.get("latency_ms") or {})
        t_lat = (test.get("latency_ms") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(block["meta"].get("checkpoint_size_mb", "")),
                    str(block["meta"].get("model_dtype", "")),
                    str(block["meta"].get("params_m", "")),
                    str(block["meta"].get("load_ms", "")),
                    str(valid.get("accuracy", "")),
                    str(valid.get("f1_weighted", "")),
                    str(v_lat.get("p95_ms", "")),
                    str(test.get("accuracy", "")),
                    str(test.get("f1_weighted", "")),
                    str(t_lat.get("p95_ms", "")),
                ]
            )
            + " |"
        )
    lines.append("")

    for name, block in results["models"].items():
        lines.append(f"## {name}\n")
        m = block["meta"]
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        for key in [
            "checkpoint_path",
            "checkpoint_size_mb",
            "config_path",
            "device",
            "model_dtype",
            "params_m",
            "load_ms",
            "unknown_confidence_threshold",
            "unknown_margin_threshold",
        ]:
            lines.append(f"| {key} | {m.get(key)} |")
        lines.append("")

        for split_name, split in block["splits"].items():
            lines.append(f"### {split_name} 指标\n")
            lines.append(
                "| acc | top3 | unknown | macro_f1 | weighted_f1 | mean(ms) | p50 | p90 | p95 | p99 | fps |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            lat = split.get("latency_ms") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(split.get("accuracy")),
                        str(split.get("top3_accuracy")),
                        str(split.get("unknown_rate")),
                        str(split.get("f1_macro")),
                        str(split.get("f1_weighted")),
                        str(lat.get("mean_ms")),
                        str(lat.get("p50_ms")),
                        str(lat.get("p90_ms")),
                        str(lat.get("p95_ms")),
                        str(lat.get("p99_ms")),
                        str(lat.get("throughput_fps")),
                    ]
                )
                + " |"
            )
            lines.append("")

            cm_png = split.get("assets", {}).get("cm_png")
            if cm_png:
                lines.append(f"![{name} {split_name} confusion matrix]({assets_rel}/{cm_png})\n")

            f1_png = split.get("assets", {}).get("f1_png")
            if f1_png:
                lines.append(f"![{name} {split_name} per-class f1]({assets_rel}/{f1_png})\n")

            lines.append("#### Top Confusions\n")
            lines.append("| true | pred | count | rate |")
            lines.append("|---|---|---:|---:|")
            for item in split.get("top_confusions", [])[:12]:
                lines.append(f"| {item.get('true')} | {item.get('pred')} | {item.get('count')} | {item.get('rate')} |")
            lines.append("")

            lines.append("#### Hard Classes\n")
            lines.append("| label | f1 | recall | support |")
            lines.append("|---|---:|---:|---:|")
            for item in split.get("hard_classes", [])[:12]:
                lines.append(f"| {item.get('label')} | {item.get('f1')} | {item.get('recall')} | {item.get('support')} |")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/web.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_yaml(config_path)
    dataset_root = project_root / config["data"]["dataset_root"]
    dataset = RoboflowDataset(dataset_root=dataset_root)
    thresholds = dict(config.get("thresholds", {}))

    variants = [
        ModelVariant(
            name="Baseline (FP32)",
            checkpoint_path=project_root / "experiments/mobilenetv3_albu_weather/best_model.pth",
            config_path=project_root / "experiments/mobilenetv3_albu_weather/config.yaml",
        ),
        ModelVariant(
            name="Compressed (FP16 ckpt)",
            checkpoint_path=project_root / "experiments/mobilenetv3_albu_weather/best_model_quantized.pth",
            config_path=project_root / "experiments/mobilenetv3_albu_weather/config.yaml",
        ),
    ]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / "docs" / "performance_reports" / run_id
    assets_dir = out_dir / "assets"

    measure_latency = not bool(args.no_latency)
    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else None

    results: dict[str, Any] = {"models": {}}
    for variant in variants:
        predictor, meta = _build_predictor(
            project_root=project_root,
            dataset=dataset,
            variant=variant,
            device=args.device,
            thresholds=thresholds,
        )
        _warmup(predictor, dataset, n=6)
        evaluator = ClassificationEvaluator(dataset=dataset, predictor=predictor)

        splits_out: dict[str, Any] = {}
        for split in ["valid", "test"]:
            split_result = evaluator.evaluate_split(split, max_samples=max_samples, measure_latency=measure_latency)
            labels = split_result.get("labels") or []
            cm_norm = split_result.get("confusion_matrix_normalized") or []

            assets: dict[str, str] = {}
            cm_png_name = f"cm_{variant.name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_{split}.png"
            _render_confusion_matrix_png(
                labels=list(labels),
                cm_norm=list(cm_norm),
                title=f"{variant.name} | {split} confusion matrix",
                out_path=assets_dir / cm_png_name,
            )
            assets["cm_png"] = cm_png_name

            per_class = split_result.get("per_class") or {}
            f1_items = [(k, float(v.get("f1", 0.0))) for k, v in per_class.items() if isinstance(v, dict)]
            f1_items.sort(key=lambda x: x[1])
            f1_png_name = f"f1_{variant.name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_{split}.png"
            _render_bar_png(
                items=f1_items[: min(12, len(f1_items))],
                title=f"{variant.name} | {split} lowest per-class F1",
                out_path=assets_dir / f1_png_name,
            )
            assets["f1_png"] = f1_png_name

            split_result["assets"] = assets
            splits_out[split] = split_result

        results["models"][variant.name] = {"meta": meta, "splits": splits_out}

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(config["data"]["dataset_root"]),
        "num_train": int(len(dataset.samples("train"))),
        "num_valid": int(len(dataset.samples("valid"))),
        "num_test": int(len(dataset.samples("test"))),
        "device_summary": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "measure_latency": bool(measure_latency),
        "max_samples": int(max_samples) if max_samples is not None else None,
    }

    raw_path = out_dir / "benchmark_raw.json"
    raw_path.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = out_dir / "性能报告.md"
    _write_markdown_report(report_path, meta=meta, results=results, assets_rel="assets")

    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

