from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize an ONNX classification model.")
    parser.add_argument("--input", required=True, help="Input ONNX model path.")
    parser.add_argument(
        "--output",
        default="weights/mobilenetv3_baseline_int8.onnx",
        help="Output quantized ONNX model path.",
    )
    parser.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode. Dynamic is framework-only baseline; static needs calibration data later.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit("onnxruntime is not installed. Run: pip install onnxruntime onnxruntime-tools") from exc

    if args.mode == "static":
        raise SystemExit(
            "Static quantization calibration is not wired yet. Start with --mode dynamic, "
            "then add a calibration reader once the export pipeline is stable."
        )

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {output_path}")


if __name__ == "__main__":
    main()
