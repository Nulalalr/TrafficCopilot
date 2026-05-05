from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model.mobilenetv3_classifier import build_mobilenetv3_small


CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "mobilenetv3_albu_weather" / "best_model.pth"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "mobilenetv3_albu_weather" / "config.yaml"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "mobilenetv3_albu_weather" / "best_model_quantized.pth"


def compress_model(checkpoint_path: Path, config_path: Path, output_path: Path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    print(f"Building model with {num_classes} classes...")
    model = build_mobilenetv3_small(
        num_classes=num_classes,
        pretrained=False,
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Converting model to FP16...")
    model = model.half()

    fp16_state_dict = {k: v.half() for k, v in model.state_dict().items()}

    compressed_checkpoint = {
        "model_state_dict": fp16_state_dict,
        "class_names": class_names,
        "compressed": True,
        "compression_type": "fp16",
    }

    print(f"Saving compressed model to: {output_path}")
    torch.save(compressed_checkpoint, output_path)

    original_size = checkpoint_path.stat().st_size / (1024 * 1024)
    compressed_size = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = original_size / compressed_size

    print("\n" + "=" * 50)
    print("Compression Complete!")
    print("=" * 50)
    print(f"Original size:   {original_size:.2f} MB")
    print(f"Compressed size: {compressed_size:.2f} MB")
    print(f"Compression:      {compression_ratio:.2f}x smaller")
    print(f"Output file:     {output_path}")
    print("=" * 50)

    return model


if __name__ == "__main__":
    compress_model(CHECKPOINT_PATH, CONFIG_PATH, OUTPUT_PATH)