from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from core.system.contracts import DatasetProvider, Predictor
import time


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t not in index or p not in index:
            continue
        cm[index[t], index[p]] += 1
    return cm


def _per_class_prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


@dataclass
class ClassificationEvaluator:
    dataset: DatasetProvider
    predictor: Predictor

    def evaluate_split(self, split: str, max_samples: int | None = None, measure_latency: bool = True) -> dict[str, Any]:
        samples = self.dataset.samples(split)
        if max_samples is not None and max_samples > 0:
            samples = samples[: int(max_samples)]
        labels = list(self.dataset.class_names)
        y_true: list[str] = []
        y_pred: list[str] = []
        y_top3: list[bool] = []
        unknown_flags: list[bool] = []
        latencies_ms: list[float] = []

        for sample in samples:
            with Image.open(sample.path) as img:
                if measure_latency:
                    start = time.perf_counter()
                    pred = self.predictor.predict(img)
                    latencies_ms.append((time.perf_counter() - start) * 1000)
                else:
                    pred = self.predictor.predict(img)
            y_true.append(sample.label)
            y_pred.append(pred.label)
            unknown_flags.append(bool(pred.is_unknown))
            topk_labels = [item.get("label") for item in pred.topk if isinstance(item, dict)]
            y_top3.append(sample.label in topk_labels)

        y_true_arr = np.array(y_true, dtype=object)
        y_pred_arr = np.array(y_pred, dtype=object)
        cm = _confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
        precision, recall, f1 = _per_class_prf(cm)

        accuracy = float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0
        macro_precision = float(np.mean(precision)) if len(labels) else 0.0
        macro_recall = float(np.mean(recall)) if len(labels) else 0.0
        macro_f1 = float(np.mean(f1)) if len(labels) else 0.0
        supports = cm.sum(axis=1).astype(np.float64)
        weights = supports / max(float(supports.sum()), 1.0)
        weighted_precision = float(np.sum(precision * weights)) if len(labels) else 0.0
        weighted_recall = float(np.sum(recall * weights)) if len(labels) else 0.0
        weighted_f1 = float(np.sum(f1 * weights)) if len(labels) else 0.0
        top3_accuracy = float(np.mean(np.array(y_top3, dtype=np.float32))) if y_top3 else 0.0
        unknown_rate = float(np.mean(np.array(unknown_flags, dtype=np.float32))) if unknown_flags else 0.0

        per_class = {}
        for i, name in enumerate(labels):
            per_class[name] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(cm[i, :].sum()),
            }

        cm_norm = (cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)).astype(np.float64)

        confusions: list[dict[str, Any]] = []
        for i, true_name in enumerate(labels):
            for j, pred_name in enumerate(labels):
                if i == j:
                    continue
                count = int(cm[i, j])
                if count <= 0:
                    continue
                confusions.append(
                    {
                        "true": true_name,
                        "pred": pred_name,
                        "count": count,
                        "rate": round(float(cm_norm[i, j]), 4),
                    }
                )
        confusions.sort(key=lambda x: (x["count"], x["rate"]), reverse=True)
        confusions = confusions[:12]

        hard_classes = sorted(
            [
                {
                    "label": name,
                    "f1": per_class[name]["f1"],
                    "recall": per_class[name]["recall"],
                    "support": per_class[name]["support"],
                }
                for name in labels
            ],
            key=lambda x: (x["f1"], x["support"]),
        )[:6]

        latency_summary = None
        if measure_latency and latencies_ms:
            lat = np.array(latencies_ms, dtype=np.float64)
            latency_summary = {
                "mean_ms": round(float(lat.mean()), 2),
                "p50_ms": round(float(np.percentile(lat, 50)), 2),
                "p90_ms": round(float(np.percentile(lat, 90)), 2),
                "p95_ms": round(float(np.percentile(lat, 95)), 2),
                "p99_ms": round(float(np.percentile(lat, 99)), 2),
                "max_ms": round(float(lat.max()), 2),
                "throughput_fps": round(float(1000.0 / max(lat.mean(), 1e-6)), 2),
            }

        return {
            "split": split,
            "num_samples": int(len(samples)),
            "accuracy": round(accuracy, 4),
            "precision_macro": round(macro_precision, 4),
            "recall_macro": round(macro_recall, 4),
            "f1_macro": round(macro_f1, 4),
            "precision_weighted": round(weighted_precision, 4),
            "recall_weighted": round(weighted_recall, 4),
            "f1_weighted": round(weighted_f1, 4),
            "top3_accuracy": round(top3_accuracy, 4),
            "unknown_rate": round(unknown_rate, 4),
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": [[round(float(v), 4) for v in row] for row in cm_norm.tolist()],
            "per_class": per_class,
            "top_confusions": confusions,
            "hard_classes": hard_classes,
            "latency_ms": latency_summary,
        }
