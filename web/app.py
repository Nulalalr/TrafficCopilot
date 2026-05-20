from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.system.factory import build_system, load_yaml
from core.video.factory import build_video_pipeline

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
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    app.config["TRAFFICCOPILOT_CONFIG_PATH"] = str(config_path)
    app.config["TRAFFICCOPILOT_CONFIG"] = config
    try:
        app.config["TRAFFICCOPILOT_SYSTEM"] = build_system(config, PROJECT_ROOT)
        app.config["TRAFFICCOPILOT_SYSTEM_ERROR"] = ""
    except Exception as exc:
        app.config["TRAFFICCOPILOT_SYSTEM"] = None
        app.config["TRAFFICCOPILOT_SYSTEM_ERROR"] = f"{type(exc).__name__}: {exc}"
    app.config["TRAFFICCOPILOT_ADMIN_TOKEN"] = os.getenv("TRAFFICCOPILOT_ADMIN_TOKEN", "")
    app.config["EVAL_CACHE"] = {}
    app.config["EVAL_CACHE_LOCK"] = Lock()

    @dataclass
    class _CameraRuntime:
        running: bool
        stop_event: Event
        frame_cond: Condition
        latest_jpeg: bytes | None
        last_event: dict[str, Any] | None
        error: str
        fps_in: float | None
        width: int | None
        height: int | None
        processed_frames: int

    camera_lock = Lock()
    camera_runtime = _CameraRuntime(
        running=False,
        stop_event=Event(),
        frame_cond=Condition(),
        latest_jpeg=None,
        last_event=None,
        error="",
        fps_in=None,
        width=None,
        height=None,
        processed_frames=0,
    )
    app.config["CAMERA_RUNTIME"] = camera_runtime
    app.config["BROWSER_CAMERA_DETECTOR"] = None

    intent_params = ((config.get("system") or {}).get("intent_engine") or {}).get("params") or {}
    app.config["LABEL_TO_COMMAND"] = dict(intent_params.get("label_to_command") or {})
    app.config["COMMAND_DESCRIPTIONS"] = dict(intent_params.get("command_descriptions") or {})

    @app.after_request
    def _no_cache(resp):
        path = request.path or ""
        if path == "/" or path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    def _system_or_error():
        system = app.config.get("TRAFFICCOPILOT_SYSTEM")
        if system is None:
            err = app.config.get("TRAFFICCOPILOT_SYSTEM_ERROR") or "system unavailable"
            return None, err
        return system, ""

    def _load_camera_config() -> dict[str, Any]:
        cfg = app.config["TRAFFICCOPILOT_CONFIG"]
        web_cfg = cfg.get("web") or {}
        camera_path = web_cfg.get("camera_config_path") or "config/camera.yaml"
        p = Path(str(camera_path))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return load_yaml(p)

    def _get_browser_camera_detector():
        det = app.config.get("BROWSER_CAMERA_DETECTOR")
        if det is not None:
            return det
        cam_cfg = _load_camera_config()
        from core.system.registry import ComponentSpec, build_component

        detector_spec = ComponentSpec.from_obj(cam_cfg["system"]["police_detector"])
        det = build_component(detector_spec, project_root=PROJECT_ROOT)
        app.config["BROWSER_CAMERA_DETECTOR"] = det
        return det

    def _load_video_config() -> dict[str, Any]:
        cfg = app.config["TRAFFICCOPILOT_CONFIG"]
        web_cfg = cfg.get("web") or {}
        video_path = web_cfg.get("video_config_path") or "config/video.yaml"
        p = Path(str(video_path))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return load_yaml(p)

    def _start_camera_if_needed() -> None:
        with camera_lock:
            if camera_runtime.running:
                return
            camera_runtime.stop_event.clear()
            camera_runtime.error = ""
            camera_runtime.latest_jpeg = None
            camera_runtime.last_event = None
            camera_runtime.fps_in = None
            camera_runtime.width = None
            camera_runtime.height = None
            camera_runtime.processed_frames = 0

            def _loop():
                try:
                    system, system_error = _system_or_error()
                    if system is None:
                        raise RuntimeError(system_error)

                    cam_cfg = _load_camera_config()
                    video_cfg = dict(cam_cfg.get("video") or {})
                    thresholds = dict(cam_cfg.get("thresholds") or {})

                    from core.system.registry import ComponentSpec, build_component

                    detector_spec = ComponentSpec.from_obj(cam_cfg["system"]["police_detector"])
                    detector = build_component(detector_spec, project_root=PROJECT_ROOT)
                    tracker_spec = ComponentSpec.from_obj(cam_cfg["system"]["tracker"])
                    tracker = build_component(tracker_spec)
                    tracker.reset()
                    system.reset_session("server_camera")

                    import cv2

                    camera_index = int(video_cfg.get("camera_index", 0))
                    backend = video_cfg.get("camera_backend")
                    if backend not in (None, "", "null"):
                        cap = cv2.VideoCapture(camera_index, int(backend))
                    else:
                        cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open camera: index={camera_index}")

                    width = video_cfg.get("camera_width")
                    height = video_cfg.get("camera_height")
                    fps_target = video_cfg.get("camera_fps")
                    if width not in (None, "", "null"):
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                    if height not in (None, "", "null"):
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                    if fps_target not in (None, "", "null"):
                        cap.set(cv2.CAP_PROP_FPS, float(fps_target))

                    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

                    sample_every = int(video_cfg.get("sample_every", 1))
                    draw = bool(video_cfg.get("draw_overlay", True))
                    quality = int(video_cfg.get("jpeg_quality", 82))
                    max_fps = video_cfg.get("max_fps")
                    max_fps = float(max_fps) if max_fps not in (None, "", "null") else None

                    unknown_conf = float(thresholds.get("confidence", 0.5))
                    unknown_margin = float(thresholds.get("margin", 0.08))
                    system.predictor.unknown_confidence_threshold = unknown_conf
                    system.predictor.unknown_margin_threshold = unknown_margin

                    last_tick = time.perf_counter()
                    frame_idx = 0

                    with camera_lock:
                        camera_runtime.fps_in = fps_in
                        camera_runtime.width = w_in
                        camera_runtime.height = h_in

                    while not camera_runtime.stop_event.is_set():
                        if max_fps is not None:
                            now = time.perf_counter()
                            dt = now - last_tick
                            min_dt = 1.0 / max(1e-6, max_fps)
                            if dt < min_dt:
                                time.sleep(max(0.0, min_dt - dt))
                            last_tick = time.perf_counter()

                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break

                        frame_idx += 1
                        ts_ms = int(round(time.time() * 1000.0))

                        event: dict[str, Any] = {
                            "frame_index": frame_idx,
                            "timestamp_ms": ts_ms,
                            "detections": [],
                            "tracks": [],
                            "prediction": None,
                            "intent": None,
                        }

                        if sample_every <= 1 or (frame_idx - 1) % sample_every == 0:
                            detections = detector.detect(frame, timestamp_ms=ts_ms)
                            tracks = tracker.update(detections)
                            event["detections"] = [{"bbox": list(d.bbox.as_xyxy()), "score": float(d.score), "meta": d.meta or {}} for d in detections]
                            event["tracks"] = [
                                {"track_id": int(t.track_id), "bbox": list(t.bbox.as_xyxy()), "score": float(t.score), "age": int(t.age), "lost": int(t.lost)}
                                for t in tracks
                            ]

                            if tracks:
                                t0 = tracks[0]
                                x1, y1, x2, y2 = t0.bbox.clip(width=frame.shape[1], height=frame.shape[0]).as_xyxy()
                                roi = frame[y1:y2, x1:x2]
                                if roi.size > 0:
                                    pil = Image.fromarray(roi[:, :, ::-1]).convert("RGB")
                                    out = _predict_temporal_no_pose(system, pil, session_id="server_camera")
                                    event["prediction"] = out.get("prediction")
                                    event["intent"] = out.get("intent")

                                    if draw:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120), 2)
                                        cv2.putText(
                                            frame,
                                            f"Police#{int(t0.track_id)}",
                                            (x1, max(18, y1 - 8)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (80, 200, 120),
                                            2,
                                            cv2.LINE_AA,
                                        )
                                        pred_obj = event.get("prediction") or {}
                                        label = str(pred_obj.get("label", ""))
                                        conf = float(pred_obj.get("confidence", 0.0) or 0.0)
                                        intent_txt = ""
                                        if isinstance(event.get("intent"), dict):
                                            cmd = event["intent"].get("command")
                                            state = event["intent"].get("state")
                                            if cmd:
                                                intent_txt = f" | {cmd}"
                                            if state:
                                                intent_txt = f"{intent_txt} | {state}"
                                        txt = f"{label} ({conf:.2f}){intent_txt}"
                                        cv2.putText(frame, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                                        cv2.putText(frame, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 1, cv2.LINE_AA)

                        ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(95, quality))])
                        if ok_enc:
                            jpeg = bytes(buf)
                            with camera_runtime.frame_cond:
                                camera_runtime.latest_jpeg = jpeg
                                camera_runtime.last_event = event
                                camera_runtime.processed_frames += 1
                                camera_runtime.frame_cond.notify_all()

                    cap.release()

                except Exception as exc:
                    with camera_runtime.frame_cond:
                        camera_runtime.error = f"{type(exc).__name__}: {exc}"
                        camera_runtime.frame_cond.notify_all()
                finally:
                    with camera_lock:
                        camera_runtime.running = False

            camera_runtime.running = True
            thread = Thread(target=_loop, daemon=True)
            thread.start()

    def _stop_camera() -> None:
        with camera_lock:
            if not camera_runtime.running:
                return
            camera_runtime.stop_event.set()

    def _predict_temporal_no_pose(system, image: Image.Image, session_id: str | None) -> dict[str, Any]:
        start = time.perf_counter()
        prediction = system.predictor.predict(image)
        prediction_payload = prediction.as_dict()
        engine = system._get_engine(session_id) if session_id else None
        if engine is not None:
            intent = engine.update(prediction_payload)
        else:
            intent = {"command": "UNKNOWN", "state": "DISABLED", "stability": prediction_payload.get("confidence", 0.0)}
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "session_id": session_id,
            "prediction": prediction_payload,
            "intent": intent,
            "status": "UNKNOWN" if prediction_payload.get("is_unknown") else "OK",
            "latency_ms": round(float(latency_ms), 2),
            "timestamp": int(time.time() * 1000),
        }

    @app.route("/")
    def index():
        system, system_error = _system_or_error()
        if system is None:
            dataset_root = None
            samples_by_split = {k: [] for k in ["train", "valid", "test"]}
            train_samples = []
            class_names: list[str] = []
        else:
            dataset = system.dataset
            dataset_root = Path(dataset.dataset_root) if hasattr(dataset, "dataset_root") and getattr(dataset, "dataset_root") else None
            try:
                samples_by_split = {k: dataset.samples(k) for k in ["train", "valid", "test"]}
            except Exception:
                samples_by_split = {k: [] for k in ["train", "valid", "test"]}
            train_samples = samples_by_split["train"]
            try:
                class_names = dataset.class_names
            except Exception:
                class_names = list(getattr(system, "class_names", []))
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
            system_error=system_error,
        )

    @app.route("/dataset-image/<path:relative_path>")
    def dataset_image(relative_path: str):
        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
        dataset_root = Path(system.dataset.dataset_root) if hasattr(system.dataset, "dataset_root") else None
        if dataset_root is None:
            return jsonify({"error": "dataset root not available"}), 500
        dataset_root = dataset_root.resolve()
        image_path = (dataset_root / relative_path).resolve()
        if not image_path.exists() or (image_path != dataset_root and dataset_root not in image_path.parents):
            return jsonify({"error": "image not found"}), 404
        return send_file(image_path)

    @app.route("/artifacts/<path:relative_path>")
    def artifact_file(relative_path: str):
        outputs_root = (PROJECT_ROOT / "outputs").resolve()
        target_path = (outputs_root / relative_path).resolve()
        if not target_path.exists() or (target_path != outputs_root and outputs_root not in target_path.parents):
            return jsonify({"error": "artifact not found"}), 404
        return send_file(target_path)

    @app.route("/api/session/reset", methods=["POST"])
    def reset_session():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or str(uuid.uuid4())
        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
        system.reset_session(session_id)
        return jsonify({"session_id": session_id, "status": "reset"})

    @app.route("/api/model-info")
    def model_info():
        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
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

    @app.route("/api/debug/paths")
    def debug_paths():
        remote = request.remote_addr or ""
        if remote not in {"127.0.0.1", "::1"}:
            return jsonify({"error": "forbidden"}), 403
        token = app.config.get("TRAFFICCOPILOT_ADMIN_TOKEN", "")
        sent = request.headers.get("X-Admin-Token", "")
        if token and sent != token:
            return jsonify({"error": "forbidden"}), 403
        static_folder = Path(app.static_folder) if app.static_folder else None
        template_paths = []
        loader = getattr(app, "jinja_loader", None)
        searchpath = getattr(loader, "searchpath", None)
        if isinstance(searchpath, list):
            template_paths = [str(Path(p)) for p in searchpath]
        rules = []
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: str(r)):
            rules.append({"rule": str(rule), "endpoint": rule.endpoint, "methods": sorted(m for m in rule.methods if m not in {"HEAD", "OPTIONS"})})
        return jsonify(
            {
                "cwd": os.getcwd(),
                "project_root": str(PROJECT_ROOT),
                "static_url_path": app.static_url_path,
                "static_folder": str(static_folder) if static_folder else None,
                "static_style_exists": bool(static_folder and (static_folder / "style.css").exists()),
                "static_app_exists": bool(static_folder and (static_folder / "app.js").exists()),
                "template_searchpath": template_paths,
                "routes": rules,
            }
        )

    @app.route("/api/admin/reload", methods=["POST"])
    def admin_reload():
        token = app.config.get("TRAFFICCOPILOT_ADMIN_TOKEN", "")
        sent = request.headers.get("X-Admin-Token", "")
        if not token or sent != token:
            return jsonify({"error": "forbidden"}), 403
        cfg_path = Path(app.config["TRAFFICCOPILOT_CONFIG_PATH"])
        cfg = load_yaml(cfg_path)
        app.config["TRAFFICCOPILOT_CONFIG"] = cfg
        try:
            app.config["TRAFFICCOPILOT_SYSTEM"] = build_system(cfg, PROJECT_ROOT)
            app.config["TRAFFICCOPILOT_SYSTEM_ERROR"] = ""
        except Exception as exc:
            app.config["TRAFFICCOPILOT_SYSTEM"] = None
            app.config["TRAFFICCOPILOT_SYSTEM_ERROR"] = f"{type(exc).__name__}: {exc}"
        intent_params_local = ((cfg.get("system") or {}).get("intent_engine") or {}).get("params") or {}
        app.config["LABEL_TO_COMMAND"] = dict(intent_params_local.get("label_to_command") or {})
        app.config["COMMAND_DESCRIPTIONS"] = dict(intent_params_local.get("command_descriptions") or {})
        with app.config["EVAL_CACHE_LOCK"]:
            app.config["EVAL_CACHE"].clear()
        system, system_error = _system_or_error()
        return jsonify({"status": "reloaded", "system_ready": system is not None, "system_error": system_error})

    @app.route("/api/predict/sample", methods=["POST"])
    def predict_sample():
        payload = request.get_json(force=True)
        session_id = payload.get("session_id") or str(uuid.uuid4())
        image_relative_path = payload["image"]
        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
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

        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
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

    @app.route("/api/predict/camera-frame", methods=["POST"])
    def predict_camera_frame():
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

        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500

        image = image.convert("RGB")
        out = _predict_temporal_no_pose(system, image, session_id=session_id)

        detector_bbox = None
        detector_score = None
        try:
            detector = _get_browser_camera_detector()
            import numpy as np
            import cv2

            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            detections = detector.detect(frame, timestamp_ms=out.get("timestamp"))
            if detections:
                det0 = max(detections, key=lambda d: float(getattr(d, "score", 0.0)))
                x1, y1, x2, y2 = det0.bbox.clip(width=frame.shape[1], height=frame.shape[0]).as_xyxy()
                detector_bbox = [int(x1), int(y1), int(x2), int(y2)]
                detector_score = float(det0.score)
        except Exception:
            detector_bbox = None
            detector_score = None

        pose_detected = False
        pose_landmarks = None
        try:
            overlay_provider = system.pose_overlay
            if overlay_provider is not None and hasattr(overlay_provider, "_extract_landmarks"):
                landmarks = overlay_provider._extract_landmarks(image)  # type: ignore[attr-defined]
                if landmarks:
                    pose_detected = True
                    pose_landmarks = []
                    for lm in landmarks:
                        pose_landmarks.append(
                            {
                                "x": float(getattr(lm, "x", 0.0)),
                                "y": float(getattr(lm, "y", 0.0)),
                                "visibility": float(getattr(lm, "visibility", 1.0)),
                            }
                        )
        except Exception:
            pose_detected = False
            pose_landmarks = None

        out["police_bbox"] = detector_bbox
        out["police_score"] = detector_score
        out["pose_detected"] = pose_detected
        out["pose_landmarks"] = pose_landmarks
        return jsonify(_serialize_prediction(out))

    @app.route("/api/predict/video-upload", methods=["POST"])
    def predict_video_upload():
        if "file" not in request.files:
            return jsonify({"error": "no video uploaded"}), 400

        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "empty filename"}), 400

        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        work_dir = PROJECT_ROOT / "outputs" / "web_video_runs" / run_id
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(uploaded.filename).suffix or ".mp4"
        input_path = input_dir / f"upload{suffix}"
        uploaded.save(input_path)

        try:
            video_cfg = _load_video_config()
            video_cfg = dict(video_cfg)
            video_section = dict(video_cfg.get("video") or {})
            video_section["input_path"] = str(input_path)
            video_section["output_dir"] = str(output_dir)
            video_section["output_video_name"] = "result.mp4"
            video_section["output_jsonl_name"] = "events.jsonl"
            video_section["save_video"] = True
            video_section["save_jsonl"] = True
            video_cfg["video"] = video_section

            pipeline = build_video_pipeline(video_cfg, project_root=PROJECT_ROOT)
            result = pipeline.run()
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

        last_prediction = None
        last_intent = None
        event_count = 0
        event_path = Path(result.get("output_jsonl") or "")
        if event_path.exists():
            with open(event_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    event_count += 1
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("prediction"):
                        last_prediction = event.get("prediction")
                    if event.get("intent"):
                        last_intent = event.get("intent")

        output_video = Path(result["output_video"]) if result.get("output_video") else None
        output_jsonl = Path(result["output_jsonl"]) if result.get("output_jsonl") else None
        outputs_root = (PROJECT_ROOT / "outputs").resolve()
        output_video_url = None
        output_jsonl_url = None
        if output_video and output_video.exists():
            output_video_url = "/artifacts/" + output_video.resolve().relative_to(outputs_root).as_posix()
        if output_jsonl and output_jsonl.exists():
            output_jsonl_url = "/artifacts/" + output_jsonl.resolve().relative_to(outputs_root).as_posix()

        return jsonify(
            {
                "run_id": run_id,
                "model_name": ((app.config.get("TRAFFICCOPILOT_CONFIG") or {}).get("system") or {}).get("predictor", {}).get("params", {}).get("model_name", ""),
                "result": result,
                "event_count": event_count,
                "last_prediction": last_prediction,
                "last_intent": last_intent,
                "output_video_url": output_video_url,
                "output_jsonl_url": output_jsonl_url,
            }
        )

    @app.route("/api/demo-sequences")
    def demo_sequences():
        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
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
        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
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
        system, system_error = _system_or_error()
        if system is None:
            return jsonify({"error": system_error}), 500
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

    @app.route("/api/camera/start", methods=["POST"])
    def camera_start():
        _start_camera_if_needed()
        return jsonify({"running": bool(camera_runtime.running), "error": camera_runtime.error})

    @app.route("/api/camera/stop", methods=["POST"])
    def camera_stop():
        _stop_camera()
        return jsonify({"running": bool(camera_runtime.running)})

    @app.route("/api/camera/status")
    def camera_status():
        with camera_runtime.frame_cond:
            payload = {
                "running": bool(camera_runtime.running),
                "error": camera_runtime.error,
                "fps_in": camera_runtime.fps_in,
                "width": camera_runtime.width,
                "height": camera_runtime.height,
                "processed_frames": camera_runtime.processed_frames,
                "last_event": camera_runtime.last_event,
            }
        return jsonify(payload)

    @app.route("/camera/stream")
    def camera_stream():
        _start_camera_if_needed()

        def _gen():
            boundary = b"--frame\r\n"
            while True:
                with camera_runtime.frame_cond:
                    camera_runtime.frame_cond.wait(timeout=2.0)
                    jpeg = camera_runtime.latest_jpeg
                    err = camera_runtime.error
                    running = camera_runtime.running
                if err:
                    break
                if jpeg is None:
                    if not running:
                        break
                    continue
                header = b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                yield boundary + header + jpeg + b"\r\n"

        return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
