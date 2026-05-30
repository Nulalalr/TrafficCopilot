from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MediaPipePoseExtractor:
    model_path: str | Path | None = None
    project_root: str | Path | None = None
    num_poses: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

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

    def extract(self, frame_bgr: Any, timestamp_ms: int | None = None) -> list[list[Any]]:
        if not self.available:
            return []
        return self._extract_landmarks(frame_bgr, timestamp_ms=timestamp_ms)

    def _create_tasks_runtime(self, model_path: Path):
        BaseOptions = self.mp.tasks.BaseOptions
        PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = self.mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=int(self.num_poses),
            min_pose_detection_confidence=float(self.min_detection_confidence),
            min_pose_presence_confidence=float(self.min_presence_confidence),
            min_tracking_confidence=float(self.min_tracking_confidence),
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

