from __future__ import annotations

import argparse
from pathlib import Path

from core.system.factory import load_yaml
from core.video.factory import build_video_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/video.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_yaml(config_path)
    pipeline = build_video_pipeline(config, project_root=project_root)
    result = pipeline.run()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

