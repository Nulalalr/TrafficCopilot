from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComponentSpec:
    target: str
    params: dict[str, Any] | None = None

    @staticmethod
    def from_obj(obj: Any) -> "ComponentSpec":
        if isinstance(obj, ComponentSpec):
            return obj
        if isinstance(obj, str):
            return ComponentSpec(target=obj, params={})
        if isinstance(obj, dict):
            target = obj.get("target")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("Component spec missing non-empty 'target'")
            params = obj.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("Component spec 'params' must be a dict")
            return ComponentSpec(target=target, params=dict(params))
        raise TypeError("Unsupported component spec type")


def import_object(target: str) -> Any:
    if ":" in target:
        module_path, attr = target.split(":", 1)
    else:
        module_path, attr = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def build_component(spec: ComponentSpec, **extra_kwargs: Any) -> Any:
    obj = import_object(spec.target)
    params = dict(spec.params or {})
    params.update(extra_kwargs)
    if callable(obj):
        return obj(**params)
    return obj

