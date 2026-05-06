from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.app import create_app

app = create_app()

if __name__ == "__main__":
    cfg = app.config["TRAFFICCOPILOT_CONFIG"]
    host = (cfg.get("web") or {}).get("host", "127.0.0.1")
    port = int((cfg.get("web") or {}).get("port", 5000))
    debug = bool((cfg.get("web") or {}).get("debug", False))
    app.run(host=host, port=port, debug=debug)
