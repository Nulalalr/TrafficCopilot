from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.video.contracts import BBox, Detection


@dataclass
class MediaPipePosePoliceDetector:
    model_path: str | Path | None = None
    project_root: str | Path | None = None
    num_poses: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    padding_ratio: float = 0.25
    min_score: float = 0.45

    def __post_init__(self):
        self.api_kind = None
        self.available = False
        self.runtime = None
        self.mp = None

        try:
            import mediapipe as mp
        except Exception:
            return

        self.mp = mp
        project_root = Path(self.project_root) if self.project_root else None

        model_path: Path | None
        if self.model_path is None:
            model_path = None
        else:
            model_path = Path(self.model_path)
            if project_root is not None and not model_path.is_absolute():
                model_path = project_root / model_path

        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and model_path is not None and model_path.exists():
            self.api_kind = "tasks"
            self.available = True
            self.runtime = self._create_tasks_runtime(model_path)
            return

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.api_kind = "legacy"
            self.available = True
            self.runtime = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    def detect(self, frame_bgr: Any, timestamp_ms: int | None = None) -> list[Detection]:
        if not self.available:
            return []

        landmarks_list = self._extract_landmarks(frame_bgr, timestamp_ms=timestamp_ms)
        if not landmarks_list:
            return []

        height, width = frame_bgr.shape[:2]
        best: Detection | None = None
        for landmarks in landmarks_list:
            det = self._landmarks_to_detection(landmarks, width=width, height=height)
            if det is None:
                continue
            if best is None:
                best = det
            else:
                if det.bbox.area() * float(det.score) > best.bbox.area() * float(best.score):
                    best = det

        if best is None or float(best.score) < float(self.min_score):
            return []
        return [best]

    def _landmarks_to_detection(self, landmarks: list[Any], width: int, height: int) -> Detection | None:
        idxs = [11, 12, 13, 14, 15, 16, 23, 24]
        xs: list[float] = []
        ys: list[float] = []
        vs: list[float] = []
        for idx in idxs:
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            x = float(getattr(lm, "x", 0.0))
            y = float(getattr(lm, "y", 0.0))
            v = float(getattr(lm, "visibility", 1.0))
            xs.append(x)
            ys.append(y)
            vs.append(v)

        x1 = min(xs) * width
        y1 = min(ys) * height
        x2 = max(xs) * width
        y2 = max(ys) * height

        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        pad = float(self.padding_ratio)
        x1 -= w * pad
        y1 -= h * pad
        x2 += w * pad
        y2 += h * pad

        bbox = BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)).clip(width=width, height=height)
        score = float(sum(vs) / max(1, len(vs)))
        return Detection(bbox=bbox, score=score, meta={"source": "mediapipe_pose"})

    def _create_tasks_runtime(self, model_path: Path):
        BaseOptions = self.mp.tasks.BaseOptions
        PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = self.mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        return PoseLandmarker.create_from_options(options)

    def _extract_landmarks(self, frame_bgr: Any, timestamp_ms: int | None):
        import numpy as np

        rgb = frame_bgr[:, :, ::-1].copy()
        if self.api_kind == "tasks":
            if timestamp_ms is None:
                timestamp_ms = 0
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            result = self.runtime.detect_for_video(mp_image, int(timestamp_ms))
            if not result.pose_landmarks:
                return []
            return [lm for lm in result.pose_landmarks]

        image = np.array(rgb)
        result = self.runtime.process(image=image)
        if result.pose_landmarks is None:
            return []
        return [result.pose_landmarks.landmark]

