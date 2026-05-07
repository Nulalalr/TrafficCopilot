from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.app import create_app

CONFIG_PATH = PROJECT_ROOT / "config" / "web_video_frames.yaml"

app = create_app(CONFIG_PATH)


if __name__ == "__main__":
    web_cfg = (app.config.get("TRAFFICCOPILOT_CONFIG") or {}).get("web") or {}
    app.run(
        host=str(web_cfg.get("host", "127.0.0.1")),
        port=int(web_cfg.get("port", 5000)),
        debug=bool(web_cfg.get("debug", False)),
    )
