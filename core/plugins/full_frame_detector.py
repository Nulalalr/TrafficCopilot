from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.video.contracts import BBox, Detection


@dataclass
class FullFrameDetector:
    score: float = 1.0

    def detect(self, frame_bgr: Any, timestamp_ms: int | None = None) -> list[Detection]:
        if frame_bgr is None:
            return []
        height, width = frame_bgr.shape[:2]
        if width <= 1 or height <= 1:
            return []
        bbox = BBox(x1=0, y1=0, x2=int(width), y2=int(height))
        return [Detection(bbox=bbox, score=float(self.score), meta={"source": "full_frame"})]

