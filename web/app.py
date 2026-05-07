from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.system.factory import build_system, load_yaml

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "web.yaml"


DEMO_SCENARIOS = [
    {
        "id": "intersection_control",
        "name": "Intersection Control",
        "description": "STOP -> GO_STRAIGHT -> TURN_LEFT",
        "labels": ["stop", "stop", "stop", "go straight", "go straight", "go straight", "turn left", "turn left", "turn left"],
    },
    {
        "id": "lane_guidance",
        "name": "Lane Guidance",
        "description": "SLOW_DOWN -> CHANGE_LANES -> PULL_OVER",
        "labels": ["slow down", "slow down", "slow down", "change lanes", "change lanes", "change lanes", "pull over", "pull over", "pull over"],
    },
]


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _single_frame_intent(prediction: dict[str, Any], label_to_command: dict[str, str], command_descriptions: dict[str, str]):
    command = "UNKNOWN" if prediction.get("is_unknown") else label_to_command.get(prediction.get("label", ""), "UNKNOWN")
    return {
        "command": command,
        "state": "SINGLE_FRAME",
        "stability": prediction.get("confidence", 0.0),
        "reason": "single image inference",
        "window": [
            {
                "label": prediction.get("label", ""),
                "command": command,
                "confidence": prediction.get("confidence", 0.0),
            }
        ],
        "description": command_descriptions.get(command, "Undefined"),
    }


def create_app(config_path: str | Path | None = None) -> Flask:
    config_path = Path(config_path) if config_path else Path(os.getenv("TRAFFICCOPILOT_WEB_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = load_yaml(config_path)

    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "web" / "templates"),
        static_folder=str(PROJECT_ROOT / "web" / "static"),
        static_url_path="/static",
    )

    app.config["TRAFFICCOPILOT_CONFIG_PATH"] = str(config_path)
    app.config["TRAFFICCOPILOT_CONFIG"] = config
    app.config["TRAFFICCOPILOT_SYSTEM"] = build_system(config, PROJECT_ROOT)
    app.config["TRAFFICCOPILOT_ADMIN_TOKEN"] = os.getenv("TRAFFICCOPILOT_ADMIN_TOKEN", "")
    app.config["EVAL_CACHE"] = {}
    app.config["EVAL_CACHE_LOCK"] = Lock()

    intent_params = ((config.get("system") or {}).get("intent_engine") or {}).get("params") or {}
    app.config["LABEL_TO_COMMAND"] = dict(intent_params.get("label_to_command") or {})
    app.config["COMMAND_DESCRIPTIONS"] = dict(intent_params.get("command_descriptions") or {})

    @app.route("/")
    def index():
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        dataset = system.dataset
        dataset_root = Path(dataset.dataset_root) if hasattr(dataset, "dataset_root") else None
        samples_by_split = {k: dataset.samples(k) for k in ["train", "valid", "test"]}
        train_samples = samples_by_split["train"]
        class_names = dataset.class_names
        raw_metrics = {}
        metrics_path = (config.get("web") or {}).get("metrics_path")
        if metrics_path:
            p = PROJECT_ROOT / str(metrics_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as handle:
                    raw_metrics = json.load(handle)
        metrics = {
            "valid_accuracy": float(raw_metrics.get("best_valid_acc", 0.0)),
            "test_accuracy": float(raw_metrics.get("test_acc", 0.0)),
            "sample_counts": {split_name: len(items) for split_name, items in samples_by_split.items()},
            "class_distribution": {label: sum(1 for s in train_samples if s.label == label) for label in class_names},
        }
        demo_sequences = _build_demo_sequences(
            samples_by_split,
            class_names,
            dataset_root=dataset_root,
            label_to_command=app.config["LABEL_TO_COMMAND"],
        )
        return render_template(
            "index.html",
            metrics=metrics,
            classes=class_names,
            command_descriptions=app.config["COMMAND_DESCRIPTIONS"],
            demo_sequences=demo_sequences,
        )

    @app.route("/dataset-image/<path:relative_path>")
    def dataset_image(relative_path: str):
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        dataset_root = Path(system.dataset.dataset_root) if hasattr(system.dataset, "dataset_root") else None
        if dataset_root is None:
            return jsonify({"error": "dataset root not available"}), 500
        image_path = dataset_root / relative_path
        if not image_path.exists() or dataset_root not in image_path.resolve().parents:
            return jsonify({"error": "image not found"}), 404
        return send_file(image_path)

    @app.route("/api/session/reset", methods=["POST"])
    def reset_session():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        system.reset_session(session_id)
        return jsonify({"session_id": session_id, "status": "reset"})

    @app.route("/api/model-info")
    def model_info():
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        raw_metrics = {}
        metrics_path = (config.get("web") or {}).get("metrics_path")
        if metrics_path:
            p = PROJECT_ROOT / str(metrics_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as handle:
                    raw_metrics = json.load(handle)
        return jsonify(
            {
                "model_name": system.model_name,
                "num_classes": len(system.class_names),
                "valid_accuracy": float(raw_metrics.get("best_valid_acc", 0.0)),
                "test_accuracy": float(raw_metrics.get("test_acc", 0.0)),
            }
        )

    @app.route("/api/modules")
    def modules():
        cfg = app.config["TRAFFICCOPILOT_CONFIG"]
        return jsonify({"system": cfg.get("system", {}), "thresholds": cfg.get("thresholds", {})})

    @app.route("/api/admin/reload", methods=["POST"])
    def admin_reload():
        token = app.config.get("TRAFFICCOPILOT_ADMIN_TOKEN", "")
        sent = request.headers.get("X-Admin-Token", "")
        if not token or sent != token:
            return jsonify({"error": "forbidden"}), 403
        cfg_path = Path(app.config["TRAFFICCOPILOT_CONFIG_PATH"])
        cfg = load_yaml(cfg_path)
        app.config["TRAFFICCOPILOT_CONFIG"] = cfg
        app.config["TRAFFICCOPILOT_SYSTEM"] = build_system(cfg, PROJECT_ROOT)
        intent_params_local = ((cfg.get("system") or {}).get("intent_engine") or {}).get("params") or {}
        app.config["LABEL_TO_COMMAND"] = dict(intent_params_local.get("label_to_command") or {})
        app.config["COMMAND_DESCRIPTIONS"] = dict(intent_params_local.get("command_descriptions") or {})
        with app.config["EVAL_CACHE_LOCK"]:
            app.config["EVAL_CACHE"].clear()
        return jsonify({"status": "reloaded"})

    @app.route("/api/predict/sample", methods=["POST"])
    def predict_sample():
        payload = request.get_json(force=True)
        session_id = payload.get("session_id") or str(uuid.uuid4())
        image_relative_path = payload["image"]
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        dataset_root = Path(system.dataset.dataset_root) if hasattr(system.dataset, "dataset_root") else None
        if dataset_root is None:
            return jsonify({"error": "dataset root not available"}), 500
        image_path = dataset_root / image_relative_path
        if not image_path.exists():
            return jsonify({"error": "sample image not found"}), 404

        with Image.open(image_path) as img:
            start = time.perf_counter()
            result = system.predict_image(img, session_id=session_id)
            latency_ms = (time.perf_counter() - start) * 1000

        result["latency_ms"] = round(float(latency_ms), 2)
        result["image"] = image_relative_path
        return jsonify(_serialize_prediction(result))

    @app.route("/api/predict/upload", methods=["POST"])
    def predict_upload():
        session_id = request.form.get("session_id") or str(uuid.uuid4())

        if "file" in request.files:
            image = Image.open(request.files["file"].stream)
        else:
            image_data = request.form.get("image_data", "")
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            if not image_data:
                return jsonify({"error": "no image uploaded"}), 400
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        label_to_command = app.config["LABEL_TO_COMMAND"]
        command_descriptions = app.config["COMMAND_DESCRIPTIONS"]

        start = time.perf_counter()
        prediction = system.predictor.predict(image).as_dict()
        intent = _single_frame_intent(prediction, label_to_command, command_descriptions)
        pose_overlay_image = None
        pose_detected = False
        if system.pose_overlay is not None:
            overlay = system.pose_overlay.draw_overlay(image)
            pose_detected = overlay.pose_detected
            pose_overlay_image = overlay.image
        latency_ms = (time.perf_counter() - start) * 1000

        payload = {
            "session_id": session_id,
            "model_name": system.model_name,
            "prediction": prediction,
            "intent": intent,
            "status": "UNKNOWN" if prediction["is_unknown"] else "OK",
            "latency_ms": round(float(latency_ms), 2),
            "timestamp": int(time.time() * 1000),
            "pose_detected": pose_detected,
            "pose_overlay_image": pose_overlay_image,
        }
        return jsonify(_serialize_prediction(payload))

    @app.route("/api/demo-sequences")
    def demo_sequences():
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        dataset = system.dataset
        dataset_root = Path(dataset.dataset_root) if hasattr(dataset, "dataset_root") else None
        samples_by_split = {k: dataset.samples(k) for k in ["train", "valid", "test"]}
        return jsonify(
            {
                "scenarios": _build_demo_sequences(
                    samples_by_split,
                    dataset.class_names,
                    dataset_root=dataset_root,
                    label_to_command=app.config["LABEL_TO_COMMAND"],
                )
            }
        )

    @app.route("/api/evaluate/<split>")
    def evaluate_split(split: str):
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        if split not in ["train", "valid", "test"]:
            return jsonify({"error": "Invalid split"}), 400
        if system.evaluator is None:
            return jsonify({"error": "evaluator not enabled"}), 400
        max_samples = request.args.get("max_samples", type=int)
        measure_latency = request.args.get("measure_latency", default=1, type=int)
        refresh = request.args.get("refresh", default=0, type=int)
        ttl_sec = int((app.config["TRAFFICCOPILOT_CONFIG"].get("web") or {}).get("eval_cache_ttl_sec", 600))
        cache_key = _eval_cache_key(
            app.config["TRAFFICCOPILOT_CONFIG"],
            split=split,
            max_samples=max_samples,
            measure_latency=bool(measure_latency),
        )
        now = time.time()

        served_from_cache = False
        if not refresh:
            with app.config["EVAL_CACHE_LOCK"]:
                entry = app.config["EVAL_CACHE"].get(cache_key)
            if entry and (now - float(entry["ts"])) <= ttl_sec:
                result = entry["result"]
                served_from_cache = True
            else:
                result = None
        else:
            result = None

        if result is None:
            result = system.evaluator.evaluate_split(
                split,
                max_samples=max_samples,
                measure_latency=bool(measure_latency),
            )
            with app.config["EVAL_CACHE_LOCK"]:
                app.config["EVAL_CACHE"][cache_key] = {"ts": now, "result": result}
        return jsonify(
            {
                "accuracy": result.get("accuracy"),
                "precision": result.get("precision_macro"),
                "recall": result.get("recall_macro"),
                "f1": result.get("f1_macro"),
                "top3_accuracy": result.get("top3_accuracy"),
                "unknown_rate": result.get("unknown_rate"),
                "latency_ms": result.get("latency_ms"),
                "details": result,
                "cached": served_from_cache,
                "cache_ttl_sec": ttl_sec,
            }
        )

    @app.route("/api/evaluate/<split>/export")
    def export_evaluation(split: str):
        system = app.config["TRAFFICCOPILOT_SYSTEM"]
        if split not in ["train", "valid", "test"]:
            return jsonify({"error": "Invalid split"}), 400
        if system.evaluator is None:
            return jsonify({"error": "evaluator not enabled"}), 400

        fmt = (request.args.get("format") or "json").strip().lower()
        max_samples = request.args.get("max_samples", type=int)
        measure_latency = request.args.get("measure_latency", default=1, type=int)
        refresh = request.args.get("refresh", default=0, type=int)
        ttl_sec = int((app.config["TRAFFICCOPILOT_CONFIG"].get("web") or {}).get("eval_cache_ttl_sec", 600))
        cache_key = _eval_cache_key(
            app.config["TRAFFICCOPILOT_CONFIG"],
            split=split,
            max_samples=max_samples,
            measure_latency=bool(measure_latency),
        )
        now = time.time()

        result = None
        if not refresh:
            with app.config["EVAL_CACHE_LOCK"]:
                entry = app.config["EVAL_CACHE"].get(cache_key)
            if entry and (now - float(entry["ts"])) <= ttl_sec:
                result = entry["result"]

        if result is None:
            result = system.evaluator.evaluate_split(
                split,
                max_samples=max_samples,
                measure_latency=bool(measure_latency),
            )
            with app.config["EVAL_CACHE_LOCK"]:
                app.config["EVAL_CACHE"][cache_key] = {"ts": now, "result": result}

        meta = {
            "exported_at": int(now),
            "split": split,
            "max_samples": max_samples,
            "measure_latency": bool(measure_latency),
            "model_name": system.model_name,
            "predictor": (app.config["TRAFFICCOPILOT_CONFIG"].get("system") or {}).get("predictor"),
        }

        if fmt == "json":
            payload = {"meta": meta, "result": result}
            blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            buffer = io.BytesIO(blob)
            filename = f"evaluation_{split}.json"
            return send_file(
                buffer,
                mimetype="application/json",
                as_attachment=True,
                download_name=filename,
            )

        if fmt == "csv":
            csv_text = _evaluation_to_csv(meta, result)
            buffer = io.BytesIO(csv_text.encode("utf-8"))
            filename = f"evaluation_{split}.csv"
            return send_file(
                buffer,
                mimetype="text/csv",
                as_attachment=True,
                download_name=filename,
            )

        return jsonify({"error": "Invalid format. Use json or csv."}), 400

    return app


def _build_demo_sequences(
    samples_by_split: dict[str, list[Any]],
    class_names: list[str],
    dataset_root: Path | None,
    label_to_command: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[Any]] = {label: [] for label in class_names}
    for split_name, samples in samples_by_split.items():
        for sample in samples:
            by_label[sample.label].append(sample)

    def frame_number(filename: str) -> int:
        import re

        match = re.search(r"frame_(\\d+)", filename)
        return int(match.group(1)) if match else 0

    for label, items in by_label.items():
        items.sort(key=lambda s: frame_number(s.path.name))

    scenarios = []
    label_to_command = dict(label_to_command or {})
    for scenario in DEMO_SCENARIOS:
        frames = []
        cursor: dict[str, int] = {label: 0 for label in class_names}
        for label in scenario["labels"]:
            items = by_label.get(label) or []
            if not items:
                continue
            index = cursor.get(label, 0)
            cursor[label] = index + 1
            sample = items[index % len(items)]
            if dataset_root is None:
                relative_path = sample.path.name
            else:
                relative_path = sample.path.relative_to(dataset_root).as_posix()
            frames.append(
                {
                    "label": label,
                    "command": label_to_command.get(label, "UNKNOWN"),
                    "image": relative_path,
                    "filename": sample.path.name,
                }
            )
        scenarios.append({**scenario, "frames": frames})
    return scenarios


def _serialize_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    img = out.pop("pose_overlay_image", None)
    if isinstance(img, Image.Image):
        out["pose_overlay"] = image_to_data_url(img)
    else:
        out["pose_overlay"] = None
    return out


def _eval_cache_key(config: dict[str, Any], split: str, max_samples: int | None, measure_latency: bool) -> str:
    system_cfg = config.get("system") or {}
    data_cfg = config.get("data") or {}
    thresholds = config.get("thresholds") or {}
    web_cfg = config.get("web") or {}
    key_obj = {
        "system": {
            "dataset": system_cfg.get("dataset"),
            "predictor": system_cfg.get("predictor"),
            "pose_overlay": system_cfg.get("pose_overlay"),
            "intent_engine": system_cfg.get("intent_engine"),
            "evaluator": system_cfg.get("evaluator"),
        },
        "data": {"dataset_root": data_cfg.get("dataset_root")},
        "thresholds": thresholds,
        "metrics_path": web_cfg.get("metrics_path"),
        "split": split,
        "max_samples": max_samples,
        "measure_latency": bool(measure_latency),
    }
    return json.dumps(key_obj, ensure_ascii=False, sort_keys=True)


def _evaluation_to_csv(meta: dict[str, Any], result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("section,key,value")
    for k, v in meta.items():
        lines.append(f"meta,{k},{json.dumps(v, ensure_ascii=False)}")

    summary_keys = [
        "num_samples",
        "accuracy",
        "top3_accuracy",
        "unknown_rate",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    for k in summary_keys:
        if k in result:
            lines.append(f"summary,{k},{result.get(k)}")

    latency = result.get("latency_ms") or {}
    if isinstance(latency, dict) and latency:
        for k, v in latency.items():
            lines.append(f"latency_ms,{k},{v}")

    per_class = result.get("per_class") or {}
    if isinstance(per_class, dict) and per_class:
        lines.append("per_class,label,precision,recall,f1,support")
        for label, row in per_class.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"per_class,{label},{row.get('precision')},{row.get('recall')},{row.get('f1')},{row.get('support')}"
            )

    top_confusions = result.get("top_confusions") or []
    if isinstance(top_confusions, list) and top_confusions:
        lines.append("top_confusions,true,pred,count,rate")
        for item in top_confusions:
            if not isinstance(item, dict):
                continue
            lines.append(f"top_confusions,{item.get('true')},{item.get('pred')},{item.get('count')},{item.get('rate')}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    app = create_app()
    cfg = app.config["TRAFFICCOPILOT_CONFIG"]
    host = (cfg.get("web") or {}).get("host", "127.0.0.1")
    port = int((cfg.get("web") or {}).get("port", 5000))
    debug = bool((cfg.get("web") or {}).get("debug", False))
    app.run(host=host, port=port, debug=debug)
