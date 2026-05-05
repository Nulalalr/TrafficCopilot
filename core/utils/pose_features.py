from __future__ import annotations

import json
from pathlib import Path

import numpy as np


UPPER_BODY_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24]
KEYPOINT_NAMES = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}


def load_pose_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_pose_lookup(records: list[dict]):
    lookup = {}
    for record in records:
        lookup[record["image_path"].replace("\\", "/")] = record
    return lookup


def _safe_visibility(landmark: dict):
    return float(landmark.get("visibility", 0.0))


def _coords(landmark: dict):
    return np.array([float(landmark["x"]), float(landmark["y"]), float(landmark["z"])], dtype=np.float32)


def _normalize_point(point: np.ndarray, center: np.ndarray, scale: float):
    if scale < 1e-6:
        scale = 1.0
    return (point - center) / scale


def _vector_feature(a: np.ndarray, b: np.ndarray):
    vec = b - a
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.concatenate([vec, np.array([0.0], dtype=np.float32)])
    return np.concatenate([vec, np.array([norm], dtype=np.float32)])


def extract_pose_feature_vector(record: dict):
    landmarks = record.get("landmarks", [])
    if len(landmarks) < 25:
        return np.zeros(52, dtype=np.float32)

    left_shoulder = _coords(landmarks[11])
    right_shoulder = _coords(landmarks[12])
    left_hip = _coords(landmarks[23])
    right_hip = _coords(landmarks[24])

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    torso_center = (shoulder_center + hip_center) / 2.0

    shoulder_width = float(np.linalg.norm(right_shoulder[:2] - left_shoulder[:2]))
    torso_height = float(np.linalg.norm(hip_center[:2] - shoulder_center[:2]))
    body_scale = max(shoulder_width, torso_height, 1e-3)

    feature_parts = []

    for idx in UPPER_BODY_KEYPOINTS:
        landmark = landmarks[idx]
        point = _coords(landmark)
        normalized = _normalize_point(point, torso_center, body_scale)
        visibility = _safe_visibility(landmark)
        feature_parts.append(np.concatenate([normalized, np.array([visibility], dtype=np.float32)]))

    vectors = [
        _vector_feature(_coords(landmarks[11]), _coords(landmarks[13])),
        _vector_feature(_coords(landmarks[13]), _coords(landmarks[15])),
        _vector_feature(_coords(landmarks[12]), _coords(landmarks[14])),
        _vector_feature(_coords(landmarks[14]), _coords(landmarks[16])),
        _vector_feature(_coords(landmarks[11]), _coords(landmarks[12])),
    ]
    feature_parts.extend(vectors)

    global_stats = np.array(
        [
            shoulder_width,
            torso_height,
            float(_coords(landmarks[15])[1] - _coords(landmarks[11])[1]),
            float(_coords(landmarks[16])[1] - _coords(landmarks[12])[1]),
        ],
        dtype=np.float32,
    )
    feature_parts.append(global_stats)

    return np.concatenate(feature_parts).astype(np.float32)


def estimate_pose_feature_dim():
    dummy_record = {
        "landmarks": [
            {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0, "presence": 1.0} for _ in range(33)
        ]
    }
    return int(extract_pose_feature_vector(dummy_record).shape[0])
