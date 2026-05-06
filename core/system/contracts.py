from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float
    is_unknown: bool
    topk: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "margin": round(float(self.margin), 4),
            "is_unknown": bool(self.is_unknown),
            "topk": self.topk,
        }


@dataclass(frozen=True)
class PoseOverlay:
    image: Image.Image
    pose_detected: bool


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    split: str


class DatasetProvider(Protocol):
    @property
    def class_names(self) -> list[str]: ...

    def samples(self, split: str) -> list[Sample]: ...


class Predictor(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def class_names(self) -> list[str]: ...

    def predict(self, image: Image.Image) -> Prediction: ...


class PoseOverlayProvider(Protocol):
    def draw_overlay(self, image: Image.Image) -> PoseOverlay: ...


class IntentEngine(Protocol):
    def reset(self) -> None: ...

    def update(self, prediction: dict[str, Any]) -> dict[str, Any]: ...


class Evaluator(Protocol):
    def evaluate_split(
        self,
        split: str,
        max_samples: int | None = None,
        measure_latency: bool = True,
    ) -> dict[str, Any]: ...
