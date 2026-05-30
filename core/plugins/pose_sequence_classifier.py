from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from core.model.pose_sequence_classifier import PoseGRUClassifier
from core.plugins.mediapipe_pose_extractor import MediaPipePoseExtractor
from core.system.contracts import Prediction


@dataclass
class PoseSequenceClassifier:
    checkpoint_path: str | Path
    config_path: str | Path
    class_names: list[str]
    project_root: str | Path | None = None
    device: str = "auto"
    model_name: str = "Pose GRU"
    model_path: str | Path = "weights/pose_landmarker_lite.task"
    clip_len: int = 16
    unknown_confidence_threshold: float = 0.5
    unknown_margin_threshold: float = 0.08

    def __post_init__(self):
        project_root = Path(self.project_root) if self.project_root else None
        self.checkpoint_path = Path(self.checkpoint_path)
        self.config_path = Path(self.config_path)
        self.model_path = Path(self.model_path)
        if project_root is not None:
            if not self.checkpoint_path.is_absolute():
                self.checkpoint_path = project_root / self.checkpoint_path
            if not self.config_path.is_absolute():
                self.config_path = project_root / self.config_path
            if not self.model_path.is_absolute():
                self.model_path = project_root / self.model_path

        self._device = self._resolve_device(self.device)
        self._extractor = MediaPipePoseExtractor(model_path=self.model_path, project_root=project_root)
        self._buffer: list[torch.Tensor] = []
        self._pose_dim = 132
        self._model = self._load_model()

    def reset(self) -> None:
        self._buffer = []

    def predict(self, image: Image.Image) -> Prediction:
        import numpy as np

        if not isinstance(image, Image.Image):
            raise TypeError("predict expects PIL.Image.Image")
        rgb = np.asarray(image.convert("RGB"))
        frame_bgr = rgb[:, :, ::-1].copy()
        poses = self._extractor.extract(frame_bgr, timestamp_ms=None)
        if poses:
            lm = poses[0]
            vec = torch.zeros((self._pose_dim,), dtype=torch.float32)
            n = min(33, len(lm))
            for i in range(n):
                p = lm[i]
                base = i * 4
                vec[base + 0] = float(getattr(p, "x", 0.0))
                vec[base + 1] = float(getattr(p, "y", 0.0))
                vec[base + 2] = float(getattr(p, "z", 0.0))
                vec[base + 3] = float(getattr(p, "visibility", 0.0))
        else:
            vec = torch.zeros((self._pose_dim,), dtype=torch.float32)

        self._buffer.append(vec)
        if len(self._buffer) > int(self.clip_len):
            self._buffer = self._buffer[-int(self.clip_len) :]

        if len(self._buffer) < int(self.clip_len):
            return Prediction(
                label="UNKNOWN",
                confidence=0.0,
                margin=0.0,
                is_unknown=True,
                topk=[],
            )

        seq = torch.stack(self._buffer, dim=0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(seq)[0].detach().cpu()
            probs = torch.softmax(logits, dim=0)

        top_probs, top_indices = torch.topk(probs, k=min(3, len(self.class_names)))
        top_label = self.class_names[int(top_indices[0])]
        top_probability = float(top_probs[0])
        runner_up_probability = float(top_probs[1]) if len(top_probs) > 1 else 0.0
        margin = top_probability - runner_up_probability
        is_unknown = top_probability < self.unknown_confidence_threshold or margin < self.unknown_margin_threshold

        topk: list[dict[str, Any]] = []
        for probability, index in zip(top_probs.tolist(), top_indices.tolist()):
            topk.append({"label": self.class_names[int(index)], "probability": round(float(probability), 4)})

        return Prediction(
            label=top_label,
            confidence=top_probability,
            margin=margin,
            is_unknown=is_unknown,
            topk=topk,
        )

    def _load_model(self):
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        pose_dim = int((config.get("model") or {}).get("pose_dim", 132))
        hidden_dim = int((config.get("model") or {}).get("hidden_dim", 128))
        num_layers = int((config.get("model") or {}).get("num_layers", 1))
        dropout = float((config.get("model") or {}).get("dropout", 0.2))
        self._pose_dim = pose_dim
        self.clip_len = int((config.get("data") or {}).get("clip_len", self.clip_len))

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        ckpt_class_names = ckpt.get("class_names")
        if ckpt_class_names is not None and list(ckpt_class_names) != list(self.class_names):
            raise ValueError("Checkpoint class names do not match configured dataset labels.")

        model = PoseGRUClassifier(
            num_classes=len(self.class_names),
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()
        return model

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        device = (device or "auto").strip().lower()
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

