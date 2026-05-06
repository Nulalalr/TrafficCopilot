from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

from core.system.contracts import Prediction


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class OnnxGestureClassifier:
    model_path: str | Path
    config_path: str | Path
    class_names: list[str]
    project_root: str | Path | None = None
    providers: list[str] | None = None
    model_name: str = "ONNX"
    unknown_confidence_threshold: float = 0.5
    unknown_margin_threshold: float = 0.08

    def __post_init__(self):
        project_root = Path(self.project_root) if self.project_root else None
        self.model_path = Path(self.model_path)
        self.config_path = Path(self.config_path)
        if project_root is not None:
            if not self.model_path.is_absolute():
                self.model_path = project_root / self.model_path
            if not self.config_path.is_absolute():
                self.config_path = project_root / self.config_path
        self._sess = self._create_session()
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._image_size = self._load_image_size()

    def predict(self, image: Image.Image) -> Prediction:
        if not isinstance(image, Image.Image):
            raise TypeError("predict expects PIL.Image.Image")
        image = image.convert("RGB")

        x = self._preprocess(image, self._image_size)
        logits = self._sess.run([self._output_name], {self._input_name: x})[0]
        probs = self._softmax(logits[0].astype(np.float32))

        topk = min(3, len(self.class_names))
        top_indices = np.argsort(-probs)[:topk]
        top_probs = probs[top_indices]

        top_label = self.class_names[int(top_indices[0])]
        top_probability = float(top_probs[0])
        runner_up_probability = float(top_probs[1]) if len(top_probs) > 1 else 0.0
        margin = top_probability - runner_up_probability
        is_unknown = top_probability < self.unknown_confidence_threshold or margin < self.unknown_margin_threshold

        topk_payload: list[dict[str, Any]] = []
        for idx, p in zip(top_indices.tolist(), top_probs.tolist()):
            topk_payload.append({"label": self.class_names[int(idx)], "probability": round(float(p), 4)})

        return Prediction(
            label=top_label,
            confidence=top_probability,
            margin=margin,
            is_unknown=is_unknown,
            topk=topk_payload,
        )

    def _create_session(self) -> ort.InferenceSession:
        providers = self.providers
        if not providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ort.InferenceSession(str(self.model_path), providers=providers)

    def _load_image_size(self) -> int:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return int(config["data"]["image_size"])

    @staticmethod
    def _preprocess(image: Image.Image, image_size: int) -> np.ndarray:
        target = image.resize((image_size + 32, image_size + 32), resample=Image.BILINEAR)
        left = (target.width - image_size) // 2
        top = (target.height - image_size) // 2
        target = target.crop((left, top, left + image_size, top + image_size))

        arr = np.asarray(target, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))
        return arr[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)
