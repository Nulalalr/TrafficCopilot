from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any

from PIL import Image

from core.system.contracts import DatasetProvider, Evaluator, IntentEngine, PoseOverlayProvider, Prediction, Predictor


@dataclass
class SystemConfig:
    unknown_confidence_threshold: float = 0.5
    unknown_margin_threshold: float = 0.08


class TrafficCopilotSystem:
    def __init__(
        self,
        dataset: DatasetProvider,
        predictor: Predictor,
        evaluator: Evaluator | None = None,
        pose_overlay: PoseOverlayProvider | None = None,
        intent_engine_factory: callable | None = None,
        system_config: SystemConfig | None = None,
    ):
        self.dataset = dataset
        self.predictor = predictor
        self.evaluator = evaluator
        self.pose_overlay = pose_overlay
        self.intent_engine_factory = intent_engine_factory
        self.system_config = system_config or SystemConfig()
        self._sessions: dict[str, IntentEngine] = {}
        self._session_lock = Lock()

    @property
    def class_names(self) -> list[str]:
        return list(self.predictor.class_names)

    @property
    def model_name(self) -> str:
        return self.predictor.model_name

    def new_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.reset_session(session_id)
        return session_id

    def reset_session(self, session_id: str) -> None:
        with self._session_lock:
            if self.intent_engine_factory is None:
                self._sessions.pop(session_id, None)
                return
            self._sessions[session_id] = self.intent_engine_factory()

    def _get_engine(self, session_id: str) -> IntentEngine | None:
        if self.intent_engine_factory is None:
            return None
        with self._session_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = self.intent_engine_factory()
            return self._sessions[session_id]

    def new_intent_engine(self) -> IntentEngine | None:
        if self.intent_engine_factory is None:
            return None
        return self.intent_engine_factory()

    def predict_image(self, image: Image.Image, session_id: str | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        prediction = self.predictor.predict(image)
        prediction_payload = prediction.as_dict()

        engine = self._get_engine(session_id) if session_id else None
        if engine is not None:
            intent = engine.update(prediction_payload)
        else:
            intent = {"command": "UNKNOWN", "state": "DISABLED", "stability": prediction_payload["confidence"]}

        pose_overlay_payload = None
        pose_detected = False
        if self.pose_overlay is not None:
            overlay = self.pose_overlay.draw_overlay(image)
            pose_detected = overlay.pose_detected
            pose_overlay_payload = overlay.image

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "session_id": session_id,
            "prediction": prediction_payload,
            "intent": intent,
            "status": "UNKNOWN" if prediction_payload["is_unknown"] else "OK",
            "latency_ms": round(float(latency_ms), 2),
            "timestamp": int(time.time() * 1000),
            "pose_detected": pose_detected,
            "pose_overlay_image": pose_overlay_payload,
        }
