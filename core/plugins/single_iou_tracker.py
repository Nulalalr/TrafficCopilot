from __future__ import annotations

from dataclasses import dataclass

from core.video.contracts import BBox, Detection, Track


@dataclass
class SingleIouTracker:
    iou_threshold: float = 0.2
    max_lost: int = 15
    smooth: float = 0.65

    def __post_init__(self):
        self._track: Track | None = None
        self._next_id = 1

    def reset(self) -> None:
        self._track = None
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        det = self._pick_best(detections)
        if self._track is None:
            if det is None:
                return []
            self._track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                score=float(det.score),
                age=1,
                lost=0,
            )
            self._next_id += 1
            return [self._track]

        age = int(self._track.age) + 1
        if det is None:
            lost = int(self._track.lost) + 1
            if lost > int(self.max_lost):
                self._track = None
                return []
            self._track = Track(
                track_id=int(self._track.track_id),
                bbox=self._track.bbox,
                score=float(self._track.score) * 0.98,
                age=age,
                lost=lost,
            )
            return [self._track]

        iou = self._track.bbox.iou(det.bbox)
        if iou < float(self.iou_threshold):
            self._track = Track(
                track_id=int(self._track.track_id),
                bbox=self._track.bbox,
                score=float(self._track.score) * 0.98,
                age=age,
                lost=int(self._track.lost) + 1,
            )
            if int(self._track.lost) > int(self.max_lost):
                self._track = None
                return []
            return [self._track]

        bbox = self._smooth_bbox(self._track.bbox, det.bbox, alpha=float(self.smooth))
        self._track = Track(
            track_id=int(self._track.track_id),
            bbox=bbox,
            score=float(det.score),
            age=age,
            lost=0,
        )
        return [self._track]

    @staticmethod
    def _pick_best(detections: list[Detection]) -> Detection | None:
        if not detections:
            return None
        best = detections[0]
        for det in detections[1:]:
            if det.bbox.area() * float(det.score) > best.bbox.area() * float(best.score):
                best = det
        return best

    @staticmethod
    def _smooth_bbox(prev: BBox, cur: BBox, alpha: float) -> BBox:
        a = max(0.0, min(1.0, float(alpha)))
        x1 = int(round(prev.x1 * a + cur.x1 * (1 - a)))
        y1 = int(round(prev.y1 * a + cur.y1 * (1 - a)))
        x2 = int(round(prev.x2 * a + cur.x2 * (1 - a)))
        y2 = int(round(prev.y2 * a + cur.y2 * (1 - a)))
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

