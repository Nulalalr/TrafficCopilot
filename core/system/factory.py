from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from core.plugins.evaluator import ClassificationEvaluator
from core.plugins.static_dataset import StaticDataset
from core.system.contracts import DatasetProvider, Evaluator, Predictor
from core.system.registry import ComponentSpec, build_component
from core.system.system import SystemConfig, TrafficCopilotSystem


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def build_system(config: dict[str, Any], project_root: Path) -> TrafficCopilotSystem:
    spec_dataset = ComponentSpec.from_obj(config["system"]["dataset"])
    dataset: DatasetProvider = build_component(
        spec_dataset,
        dataset_root=project_root / config["data"]["dataset_root"],
    )

    def _load_class_names_fallback() -> list[str]:
        predictor_params = ((config.get("system") or {}).get("predictor") or {}).get("params") or {}
        ckpt = predictor_params.get("checkpoint_path")
        if ckpt:
            ckpt_path = Path(str(ckpt))
            if not ckpt_path.is_absolute():
                ckpt_path = project_root / ckpt_path
            candidate = ckpt_path.parent / "class_names.json"
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return list(data)

        metrics_path = ((config.get("web") or {}).get("metrics_path")) or ""
        if metrics_path:
            mp = Path(str(metrics_path))
            if not mp.is_absolute():
                mp = project_root / mp
            if mp.exists():
                raw = json.loads(mp.read_text(encoding="utf-8"))
                names = raw.get("class_names")
                if isinstance(names, list) and all(isinstance(x, str) for x in names):
                    return list(names)

        raise FileNotFoundError("Dataset not available and fallback class names not found (class_names.json / metrics.json).")

    try:
        class_names = dataset.class_names
    except Exception:
        class_names = _load_class_names_fallback()
        dataset = StaticDataset(class_names=class_names, dataset_root=None)

    spec_predictor = ComponentSpec.from_obj(config["system"]["predictor"])
    predictor: Predictor = build_component(
        spec_predictor,
        class_names=class_names,
        project_root=project_root,
        unknown_confidence_threshold=float(config.get("thresholds", {}).get("confidence", 0.5)),
        unknown_margin_threshold=float(config.get("thresholds", {}).get("margin", 0.08)),
    )

    evaluator: Evaluator | None
    if config["system"].get("evaluator"):
        spec_eval = ComponentSpec.from_obj(config["system"]["evaluator"])
        evaluator = build_component(spec_eval, dataset=dataset, predictor=predictor)
    else:
        evaluator = ClassificationEvaluator(dataset=dataset, predictor=predictor)

    pose_overlay = None
    if config["system"].get("pose_overlay"):
        spec_pose = ComponentSpec.from_obj(config["system"]["pose_overlay"])
        pose_overlay = build_component(spec_pose, project_root=project_root)

    intent_engine_factory = None
    if config["system"].get("intent_engine"):
        spec_intent = ComponentSpec.from_obj(config["system"]["intent_engine"])

        def _factory():
            return build_component(spec_intent)

        intent_engine_factory = _factory

    sys_cfg = SystemConfig(
        unknown_confidence_threshold=float(config.get("thresholds", {}).get("confidence", 0.5)),
        unknown_margin_threshold=float(config.get("thresholds", {}).get("margin", 0.08)),
    )

    return TrafficCopilotSystem(
        dataset=dataset,
        predictor=predictor,
        evaluator=evaluator,
        pose_overlay=pose_overlay,
        intent_engine_factory=intent_engine_factory,
        system_config=sys_cfg,
    )

