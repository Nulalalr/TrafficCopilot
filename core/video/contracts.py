from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> "BBox":
        x1 = max(0, min(int(self.x1), int(width - 1)))
        y1 = max(0, min(int(self.y1), int(height - 1)))
        x2 = max(0, min(int(self.x2), int(width - 1)))
        y2 = max(0, min(int(self.y2), int(height - 1)))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def area(self) -> int:
        return max(0, int(self.x2 - self.x1)) * max(0, int(self.y2 - self.y1))

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.area() + other.area() - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    score: float
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    score: float
    age: int
    lost: int


class PoliceDetector(Protocol):
    def detect(self, frame_bgr: Any, timestamp_ms: int | None = None) -> list[Detection]: ...


class Tracker(Protocol):
    def reset(self) -> None: ...

    def update(self, detections: list[Detection]) -> list[Track]: ...

