from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from core.system.contracts import PoseOverlay


POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]


@dataclass
class MediaPipePoseOverlay:
    model_path: str | Path
    project_root: str | Path | None = None
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_poses: int = 1

    def __post_init__(self):
        project_root = Path(self.project_root) if self.project_root else None
        self.model_path = Path(self.model_path)
        if project_root is not None and not self.model_path.is_absolute():
            self.model_path = project_root / self.model_path
        self.api_kind = None
        self.available = False
        self.runtime = None
        self.mp = None
        try:
            import mediapipe as mp
        except Exception:
            return

        self.mp = mp
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and self.model_path.exists():
            self.api_kind = "tasks"
            self.available = True
            self.runtime = self._create_tasks_runtime(self.model_path)
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.api_kind = "legacy"
            self.available = True
            self.runtime = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
            )

    def draw_overlay(self, image: Image.Image) -> PoseOverlay:
        overlay = image.convert("RGB").copy()
        if not self.available:
            return PoseOverlay(image=overlay, pose_detected=False)

        landmarks = self._extract_landmarks(overlay)
        if not landmarks:
            return PoseOverlay(image=overlay, pose_detected=False)

        draw = ImageDraw.Draw(overlay)
        width, height = overlay.size

        def pt(idx: int):
            landmark = landmarks[idx]
            return (landmark.x * width, landmark.y * height)

        for start_idx, end_idx in POSE_CONNECTIONS:
            draw.line([pt(start_idx), pt(end_idx)], fill=(255, 140, 80), width=4)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
            x, y = pt(idx)
            visibility = float(getattr(landmarks[idx], "visibility", 1.0))
            if visibility >= 0.75:
                color = (48, 180, 90)
            elif visibility >= 0.45:
                color = (230, 170, 30)
            else:
                color = (220, 70, 70)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255), width=2)

        return PoseOverlay(image=overlay, pose_detected=True)

    def _create_tasks_runtime(self, model_path: Path):
        BaseOptions = self.mp.tasks.BaseOptions
        PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = self.mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        return PoseLandmarker.create_from_options(options)

    def _extract_landmarks(self, image: Image.Image):
        if self.api_kind == "tasks":
            import numpy as np

            rgb = np.array(image.convert("RGB"))
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            result = self.runtime.detect(mp_image)
            if not result.pose_landmarks:
                return []
            return result.pose_landmarks[0]

        import numpy as np

        result = self.runtime.process(image=np.array(image.convert("RGB")))
        if result.pose_landmarks is None:
            return []
        return result.pose_landmarks.landmark
