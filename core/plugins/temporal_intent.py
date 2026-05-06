from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class TemporalIntentEngine:
    label_to_command: dict[str, str]
    command_descriptions: dict[str, str] | None = None
    window_size: int = 5
    activation_count: int = 3
    hold_frames: int = 2

    def __post_init__(self):
        self.command_descriptions = dict(self.command_descriptions or {})
        self.reset()

    def reset(self) -> None:
        self.history = deque(maxlen=self.window_size)
        self.active_command = "UNKNOWN"
        self.cooldown = 0
        self.state = "IDLE"

    def update(self, prediction: dict[str, Any]) -> dict[str, Any]:
        label = prediction.get("label", "")
        command = self.label_to_command.get(label, "UNKNOWN")
        confidence = float(prediction.get("confidence", 0.0))
        if bool(prediction.get("is_unknown", False)):
            command = "UNKNOWN"

        self.history.append({"label": label, "command": command, "confidence": confidence})
        weights: dict[str, float] = {}
        for index, item in enumerate(self.history):
            recency = (index + 1) / len(self.history)
            weights[item["command"]] = weights.get(item["command"], 0.0) + float(item["confidence"]) * recency

        ranked = sorted(weights.items(), key=lambda it: it[1], reverse=True)
        candidate_command, candidate_score = ranked[0] if ranked else ("UNKNOWN", 0.0)
        candidate_count = sum(1 for item in self.history if item["command"] == candidate_command)

        reason = "waiting for enough stable evidence"
        if candidate_command == "UNKNOWN":
            self.state = "DETECTING"
            self.cooldown = max(self.cooldown - 1, 0)
            if self.cooldown == 0:
                self.active_command = "UNKNOWN"
            reason = "raw predictions are uncertain"
        elif candidate_count >= self.activation_count:
            self.active_command = candidate_command
            self.state = "COMMAND_ACTIVE"
            self.cooldown = self.hold_frames
            reason = f"{candidate_count} / {len(self.history)} recent frames support the same command"
        elif self.cooldown > 0 and self.active_command != "UNKNOWN":
            self.cooldown -= 1
            self.state = "HOLDING"
            reason = "holding last stable command to suppress jitter"
        else:
            self.active_command = "UNKNOWN"
            self.state = "DETECTING"

        description = self.command_descriptions.get(self.active_command, "Undefined")
        return {
            "command": self.active_command,
            "state": self.state,
            "stability": round(float(candidate_score / max(len(self.history), 1)), 4),
            "reason": reason,
            "window": list(self.history),
            "description": description,
        }

