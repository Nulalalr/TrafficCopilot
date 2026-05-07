from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from core.system.system import TrafficCopilotSystem
from core.video.contracts import Detection, Tracker


@dataclass
class VideoPipeline:
    system: TrafficCopilotSystem
    detector: Any
    tracker: Tracker
    config: dict[str, Any]
    project_root: Path

    def run(self) -> dict[str, Any]:
        input_path = self._resolve_path(self.config.get("input_path"))
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(str(input_path) if input_path else "video.input_path not set")

        output_dir = self._resolve_dir(self.config.get("output_dir", "outputs/video_runs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        out_video_path = output_dir / (self.config.get("output_video_name") or f"{input_path.stem}_out.mp4")
        out_jsonl_path = output_dir / (self.config.get("output_jsonl_name") or f"{input_path.stem}_events.jsonl")

        sample_every = int(self.config.get("sample_every", 1))
        max_frames = self.config.get("max_frames")
        max_frames = int(max_frames) if max_frames not in (None, "", "null") else None

        draw = bool(self.config.get("draw_overlay", True))
        save_video = bool(self.config.get("save_video", True))
        save_jsonl = bool(self.config.get("save_jsonl", True))

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                writer = None
                save_video = False

        handle = open(out_jsonl_path, "w", encoding="utf-8") if save_jsonl else None

        session_id = f"video:{int(time.time() * 1000)}"
        self.system.reset_session(session_id)
        self.tracker.reset()

        frame_idx = 0
        processed = 0
        start = time.perf_counter()

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_idx += 1
                if sample_every > 1 and (frame_idx - 1) % sample_every != 0:
                    if save_video and writer is not None:
                        writer.write(frame)
                    continue

                ts_ms = int(round((frame_idx - 1) * 1000.0 / max(1e-6, fps)))
                detections: list[Detection] = self.detector.detect(frame, timestamp_ms=ts_ms)
                tracks = self.tracker.update(detections)

                event: dict[str, Any] = {
                    "frame_index": frame_idx,
                    "timestamp_ms": ts_ms,
                    "detections": [self._det_to_dict(d) for d in detections],
                    "tracks": [self._track_to_dict(t) for t in tracks],
                    "prediction": None,
                    "intent": None,
                }

                if tracks:
                    track = tracks[0]
                    x1, y1, x2, y2 = track.bbox.clip(width=frame.shape[1], height=frame.shape[0]).as_xyxy()
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        pil = Image.fromarray(roi[:, :, ::-1]).convert("RGB")
                        out = self.system.predict_image(pil, session_id=session_id)
                        event["prediction"] = out.get("prediction")
                        event["intent"] = out.get("intent")

                        if draw:
                            self._draw_track(frame, track_id=int(track.track_id), bbox=(x1, y1, x2, y2))
                            pred_obj = out.get("prediction") or {}
                            self._draw_text(
                                frame,
                                str(pred_obj.get("label", "")),
                                float(pred_obj.get("confidence", 0.0) or 0.0),
                                event["intent"],
                            )

                if handle is not None:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")

                if save_video and writer is not None:
                    writer.write(frame)

                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if handle is not None:
                handle.close()

        elapsed = time.perf_counter() - start
        return {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "output_video": str(out_video_path) if save_video else None,
            "output_jsonl": str(out_jsonl_path) if save_jsonl else None,
            "fps_in": fps,
            "frames_read": frame_idx,
            "frames_processed": processed,
            "elapsed_sec": round(float(elapsed), 3),
            "effective_fps": round(float(processed / max(1e-9, elapsed)), 2),
        }

    def run_camera(self) -> dict[str, Any]:
        camera_index = int(self.config.get("camera_index", 0))
        camera_backend = self.config.get("camera_backend")
        width = self.config.get("camera_width")
        height = self.config.get("camera_height")
        fps_target = self.config.get("camera_fps")
        sample_every = int(self.config.get("sample_every", 1))
        max_frames = self.config.get("max_frames")
        max_frames = int(max_frames) if max_frames not in (None, "", "null") else None

        draw = bool(self.config.get("draw_overlay", True))
        display = bool(self.config.get("display", True))
        window_name = str(self.config.get("window_name") or "TrafficCopilot Camera")
        save_video = bool(self.config.get("save_video", False))
        save_jsonl = bool(self.config.get("save_jsonl", False))

        cap = None
        if camera_backend not in (None, "", "null"):
            cap = cv2.VideoCapture(camera_index, int(camera_backend))
        else:
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera: index={camera_index}")

        if width not in (None, "", "null"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height not in (None, "", "null"):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps_target not in (None, "", "null"):
            cap.set(cv2.CAP_PROP_FPS, float(fps_target))

        fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        output_dir = self._resolve_dir(self.config.get("output_dir", "outputs/camera_runs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        out_video_path = output_dir / (self.config.get("output_video_name") or f"camera_{run_id}.mp4")
        out_jsonl_path = output_dir / (self.config.get("output_jsonl_name") or f"camera_{run_id}_events.jsonl")

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (int(width_in or 0), int(height_in or 0))
            if size[0] > 0 and size[1] > 0:
                writer = cv2.VideoWriter(str(out_video_path), fourcc, float(fps_in or 30.0), size)
                if not writer.isOpened():
                    writer = None
                    save_video = False
            else:
                save_video = False

        handle = open(out_jsonl_path, "w", encoding="utf-8") if save_jsonl else None

        session_id = f"camera:{int(time.time() * 1000)}"
        self.system.reset_session(session_id)
        self.tracker.reset()

        processed = 0
        frame_idx = 0
        start = time.perf_counter()

        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_idx += 1
                if sample_every > 1 and (frame_idx - 1) % sample_every != 0:
                    if save_video and writer is not None:
                        writer.write(frame)
                    if display:
                        cv2.imshow(window_name, frame)
                        key = int(cv2.waitKey(1)) & 0xFF
                        if key in (27, ord("q")):
                            break
                    continue

                ts_ms = int(round((time.perf_counter() - start) * 1000.0))
                detections: list[Detection] = self.detector.detect(frame, timestamp_ms=ts_ms)
                tracks = self.tracker.update(detections)

                event: dict[str, Any] = {
                    "frame_index": frame_idx,
                    "timestamp_ms": ts_ms,
                    "detections": [self._det_to_dict(d) for d in detections],
                    "tracks": [self._track_to_dict(t) for t in tracks],
                    "prediction": None,
                    "intent": None,
                }

                if tracks:
                    track = tracks[0]
                    x1, y1, x2, y2 = track.bbox.clip(width=frame.shape[1], height=frame.shape[0]).as_xyxy()
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        pil = Image.fromarray(roi[:, :, ::-1]).convert("RGB")
                        out = self.system.predict_image(pil, session_id=session_id)
                        event["prediction"] = out.get("prediction")
                        event["intent"] = out.get("intent")

                        if draw:
                            self._draw_track(frame, track_id=int(track.track_id), bbox=(x1, y1, x2, y2))
                            pred_obj = out.get("prediction") or {}
                            self._draw_text(
                                frame,
                                str(pred_obj.get("label", "")),
                                float(pred_obj.get("confidence", 0.0) or 0.0),
                                event["intent"],
                            )

                if handle is not None:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")

                if save_video and writer is not None:
                    writer.write(frame)

                if display:
                    cv2.imshow(window_name, frame)
                    key = int(cv2.waitKey(1)) & 0xFF
                    if key in (27, ord("q")):
                        break

                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if handle is not None:
                handle.close()
            if display:
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass

        elapsed = time.perf_counter() - start
        return {
            "camera_index": camera_index,
            "output_dir": str(output_dir),
            "output_video": str(out_video_path) if save_video else None,
            "output_jsonl": str(out_jsonl_path) if save_jsonl else None,
            "fps_in": fps_in,
            "width": width_in,
            "height": height_in,
            "frames_read": frame_idx,
            "frames_processed": processed,
            "elapsed_sec": round(float(elapsed), 3),
            "effective_fps": round(float(processed / max(1e-9, elapsed)), 2),
        }

    def _resolve_path(self, p: Any) -> Path | None:
        if not p:
            return None
        path = Path(str(p))
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _resolve_dir(self, p: Any) -> Path:
        path = Path(str(p))
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @staticmethod
    def _det_to_dict(det: Detection) -> dict[str, Any]:
        x1, y1, x2, y2 = det.bbox.as_xyxy()
        return {"bbox": [x1, y1, x2, y2], "score": round(float(det.score), 4), "meta": det.meta or {}}

    @staticmethod
    def _track_to_dict(track: Any) -> dict[str, Any]:
        x1, y1, x2, y2 = track.bbox.as_xyxy()
        return {
            "track_id": int(track.track_id),
            "bbox": [x1, y1, x2, y2],
            "score": round(float(track.score), 4),
            "age": int(track.age),
            "lost": int(track.lost),
        }

    @staticmethod
    def _draw_track(frame_bgr: Any, track_id: int, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (80, 200, 120), 2)
        cv2.putText(
            frame_bgr,
            f"Police#{track_id}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 200, 120),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_text(frame_bgr: Any, label: str, conf: float, intent: dict[str, Any] | None) -> None:
        h, w = frame_bgr.shape[:2]
        y = max(24, int(h * 0.06))
        txt = f"{label} ({conf:.2f})"
        if intent and isinstance(intent, dict):
            cmd = intent.get("command")
            state = intent.get("state")
            if cmd:
                txt = f"{txt} | {cmd}"
            if state:
                txt = f"{txt} | {state}"
        cv2.putText(frame_bgr, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 1, cv2.LINE_AA)
