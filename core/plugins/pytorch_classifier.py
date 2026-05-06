from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms

from core.model.mobilenetv3_classifier import build_mobilenetv3_small
from core.system.contracts import Prediction


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PyTorchMobileNetV3Classifier:
    checkpoint_path: str | Path
    config_path: str | Path
    class_names: list[str]
    project_root: str | Path | None = None
    device: str = "auto"
    model_name: str = "MobileNetV3"
    unknown_confidence_threshold: float = 0.5
    unknown_margin_threshold: float = 0.08

    def __post_init__(self):
        project_root = Path(self.project_root) if self.project_root else None
        self.checkpoint_path = Path(self.checkpoint_path)
        self.config_path = Path(self.config_path)
        if project_root is not None:
            if not self.checkpoint_path.is_absolute():
                self.checkpoint_path = project_root / self.checkpoint_path
            if not self.config_path.is_absolute():
                self.config_path = project_root / self.config_path
        self._device = self._resolve_device(self.device)
        self._model, self._transform = self._load_model()

    def predict(self, image: Image.Image) -> Prediction:
        if not isinstance(image, Image.Image):
            raise TypeError("predict expects PIL.Image.Image")

        image = image.convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        if next(self._model.parameters()).dtype == torch.float16:
            tensor = tensor.half()

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()

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

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        ckpt_class_names = checkpoint.get("class_names")
        if ckpt_class_names is not None and list(ckpt_class_names) != list(self.class_names):
            raise ValueError("Checkpoint class names do not match configured dataset labels.")

        model = build_mobilenetv3_small(
            num_classes=len(self.class_names),
            pretrained=False,
            dropout=config["model"]["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("compressed", False) and checkpoint.get("compression_type") == "fp16":
            model = model.half()

        model.to(self._device)
        model.eval()

        image_size = int(config["data"]["image_size"])
        transform = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        return model, transform

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        device = (device or "auto").strip().lower()
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
