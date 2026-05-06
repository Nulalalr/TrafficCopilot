from __future__ import annotations

from pathlib import Path
from typing import Any

from core.system.factory import build_system
from core.system.registry import ComponentSpec, build_component
from core.video.pipeline import VideoPipeline


def build_video_pipeline(config: dict[str, Any], project_root: Path) -> VideoPipeline:
    system = build_system(config, project_root)

    spec_detector = ComponentSpec.from_obj(config["system"]["police_detector"])
    detector = build_component(spec_detector, project_root=project_root)

    spec_tracker = ComponentSpec.from_obj(config["system"]["tracker"])
    tracker = build_component(spec_tracker)

    return VideoPipeline(
        system=system,
        detector=detector,
        tracker=tracker,
        config=dict(config.get("video", {})),
        project_root=project_root,
    )

