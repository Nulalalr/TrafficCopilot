from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model.mobilenetv3_classifier import build_mobilenetv3_small


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained MobileNetV3 checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth or last_model.pth")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "mobilenetv3_baseline.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "weights" / "mobilenetv3_baseline.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    return parser.parse_args()


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    args = parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = build_mobilenetv3_small(
        num_classes=len(class_names),
        pretrained=False,
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = config["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=args.opset,
    )
    print(f"ONNX exported to {output_path}")


if __name__ == "__main__":
    main()
