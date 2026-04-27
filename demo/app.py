import base64
import csv
import io
import re
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image


DATASET_ROOT = Path(__file__).parent / "traffic gestures.v1i.multiclass"
CLASS_CSV_NAME = "_classes.csv"
IMAGE_SIZE = (32, 32)
HOG_SIZE = (64, 64)
TOP_K = 3

LABEL_TO_COMMAND = {
    "change lanes": "CHANGE_LANES",
    "go straight": "GO_STRAIGHT",
    "pull over": "PULL_OVER",
    "slow down": "SLOW_DOWN",
    "stop": "STOP",
    "turn left": "TURN_LEFT",
    "turn right": "TURN_RIGHT",
    "wait for  left turn": "WAIT_LEFT_TURN",
}

COMMAND_DESCRIPTIONS = {
    "CHANGE_LANES": "建议车辆变道通过",
    "GO_STRAIGHT": "允许直行",
    "PULL_OVER": "靠边停车",
    "SLOW_DOWN": "减速观察",
    "STOP": "立即停车等待",
    "TURN_LEFT": "允许左转",
    "TURN_RIGHT": "允许右转",
    "WAIT_LEFT_TURN": "左转待转",
    "UNKNOWN": "置信度不足，进入安全降级",
}

DEMO_SCENARIOS = [
    {
        "id": "intersection_control",
        "name": "路口放行演示",
        "description": "模拟 STOP -> GO_STRAIGHT -> TURN_LEFT 的指挥过程。",
        "labels": ["stop", "stop", "stop", "go straight", "go straight", "go straight", "turn left", "turn left", "turn left"],
    },
    {
        "id": "lane_guidance",
        "name": "车辆引导演示",
        "description": "模拟 SLOW_DOWN -> CHANGE_LANES -> PULL_OVER 的连续指令。",
        "labels": ["slow down", "slow down", "slow down", "change lanes", "change lanes", "change lanes", "pull over", "pull over", "pull over"],
    },
]


@dataclass
class Sample:
    path: Path
    label: str
    split: str


class FeatureExtractor:
    @staticmethod
    def open_image(image_source) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        return Image.open(image_source).convert("RGB")

    @staticmethod
    def resize_gray(image_source, size):
        image = FeatureExtractor.open_image(image_source).convert("L").resize(size)
        return np.asarray(image, dtype=np.float32) / 255.0

    @staticmethod
    def hog(gray, cell=8, bins=9):
        grad_y = np.zeros_like(gray)
        grad_x = np.zeros_like(gray)
        grad_y[1:-1, :] = gray[2:, :] - gray[:-2, :]
        grad_x[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        angle = (np.arctan2(grad_y, grad_x) % np.pi) / np.pi * bins

        height, width = gray.shape
        feats = []
        for y in range(0, height, cell):
            for x in range(0, width, cell):
                hist = np.zeros(bins, dtype=np.float32)
                mag_block = magnitude[y : y + cell, x : x + cell].ravel()
                ang_block = angle[y : y + cell, x : x + cell].ravel()
                for mag_value, ang_value in zip(mag_block, ang_block):
                    hist[int(ang_value) % bins] += mag_value
                feats.append(hist / (np.linalg.norm(hist) + 1e-6))
        return np.concatenate(feats)

    @staticmethod
    def extract(image_source):
        small_gray = FeatureExtractor.resize_gray(image_source, IMAGE_SIZE).ravel()
        hog_gray = FeatureExtractor.resize_gray(image_source, HOG_SIZE)
        hog_feature = FeatureExtractor.hog(hog_gray)
        return np.concatenate([small_gray, hog_feature]).astype(np.float32)


class GestureDataset:
    def __init__(self, root: Path):
        self.root = root
        self.class_names = []

    def load_split(self, split: str):
        csv_path = self.root / split / CLASS_CSV_NAME
        samples = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.class_names = reader.fieldnames[1:]
            for row in reader:
                label = max(self.class_names, key=lambda name: int(row[name]))
                samples.append(Sample(path=self.root / split / row["filename"], label=label, split=split))
        return samples

    def all_samples(self):
        train = self.load_split("train")
        valid = self.load_split("valid")
        test = self.load_split("test")
        return {"train": train, "valid": valid, "test": test}


class KNNGestureClassifier:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.dataset = GestureDataset(dataset_root)
        self.samples_by_split = self.dataset.all_samples()
        self.class_names = list(self.dataset.class_names)
        self.feature_mean = None
        self.feature_std = None
        self.support_features = None
        self.support_labels = []
        self.metrics = {}
        self.demo_sequences = []
        self._build()

    def _vectorize(self, samples):
        return np.stack([FeatureExtractor.extract(sample.path) for sample in samples])

    def _standardize(self, features):
        return (features - self.feature_mean) / self.feature_std

    def _predict_from_matrix(self, feature_vector, features, labels):
        distances = np.sum((features - feature_vector) ** 2, axis=1)
        nearest_ids = np.argsort(distances)[:TOP_K]
        nearest_distances = distances[nearest_ids]
        nearest_labels = [labels[idx] for idx in nearest_ids]

        weights = 1.0 / (nearest_distances + 1e-6)
        class_scores = {label: 0.0 for label in self.class_names}
        for label, weight in zip(nearest_labels, weights):
            class_scores[label] += float(weight)

        total_score = sum(class_scores.values()) + 1e-6
        probabilities = {label: score / total_score for label, score in class_scores.items()}
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

        top_label = nearest_labels[0]
        top_probability = probabilities[top_label]
        runner_up_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_probability - runner_up_probability
        support_count = Counter(nearest_labels)[top_label]

        is_unknown = top_probability < 0.44 or margin < 0.08
        return {
            "label": top_label,
            "confidence": round(float(top_probability), 4),
            "margin": round(float(margin), 4),
            "is_unknown": is_unknown,
            "neighbors": [
                {
                    "label": labels[idx],
                    "distance": round(float(distances[idx]), 4),
                    "split": getattr(self, "support_split_names", {}).get(idx, "support"),
                }
                for idx in nearest_ids
            ],
            "topk": [{"label": label, "probability": round(float(probability), 4)} for label, probability in ranked[:3]],
        }

    def _evaluate(self, train_samples, eval_samples):
        train_features = self._vectorize(train_samples)
        self.feature_mean = train_features.mean(axis=0)
        self.feature_std = train_features.std(axis=0) + 1e-6
        train_features = self._standardize(train_features)
        eval_features = self._standardize(self._vectorize(eval_samples))

        predictions = []
        for feature_vector in eval_features:
            predictions.append(self._predict_from_matrix(feature_vector, train_features, [sample.label for sample in train_samples])["label"])
        accuracy = sum(pred == sample.label for pred, sample in zip(predictions, eval_samples)) / len(eval_samples)
        return round(float(accuracy), 4)

    def _build_demo_sequences(self):
        by_label = {label: [] for label in self.class_names}
        for split_name, samples in self.samples_by_split.items():
            for sample in samples:
                by_label[sample.label].append(sample)

        for label, items in by_label.items():
            items.sort(key=lambda sample: self._frame_number(sample.path.name))

        scenarios = []
        for scenario in DEMO_SCENARIOS:
            frames = []
            local_cursor = Counter()
            for label in scenario["labels"]:
                index = local_cursor[label]
                sample = by_label[label][index % len(by_label[label])]
                local_cursor[label] += 1
                relative_path = sample.path.relative_to(self.dataset_root).as_posix()
                frames.append(
                    {
                        "label": label,
                        "command": LABEL_TO_COMMAND[label],
                        "image": relative_path,
                        "filename": sample.path.name,
                    }
                )
            scenarios.append({**scenario, "frames": frames})
        self.demo_sequences = scenarios

    @staticmethod
    def _frame_number(filename):
        match = re.search(r"frame_(\d+)", filename)
        return int(match.group(1)) if match else 0

    def _build(self):
        train_samples = self.samples_by_split["train"]
        valid_samples = self.samples_by_split["valid"]
        test_samples = self.samples_by_split["test"]

        self.metrics = {
            "valid_accuracy": self._evaluate(train_samples, valid_samples),
            "test_accuracy": self._evaluate(train_samples, test_samples),
            "sample_counts": {
                split_name: len(samples) for split_name, samples in self.samples_by_split.items()
            },
            "class_distribution": {
                label: sum(1 for sample in train_samples if sample.label == label) for label in self.class_names
            },
        }

        support_samples = train_samples + valid_samples
        support_features = self._vectorize(support_samples)
        self.feature_mean = support_features.mean(axis=0)
        self.feature_std = support_features.std(axis=0) + 1e-6
        self.support_features = self._standardize(support_features)
        self.support_labels = [sample.label for sample in support_samples]
        self.support_split_names = {index: sample.split for index, sample in enumerate(support_samples)}
        self._build_demo_sequences()

    def predict(self, image_source):
        feature_vector = self._standardize(FeatureExtractor.extract(image_source))
        return self._predict_from_matrix(feature_vector, self.support_features, self.support_labels)


class TemporalIntentEngine:
    def __init__(self, window_size=5, activation_count=3, hold_frames=2):
        self.window_size = window_size
        self.activation_count = activation_count
        self.hold_frames = hold_frames
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window_size)
        self.active_command = "UNKNOWN"
        self.cooldown = 0
        self.state = "IDLE"

    def update(self, prediction):
        label = prediction["label"]
        command = LABEL_TO_COMMAND.get(label, "UNKNOWN")
        confidence = prediction["confidence"]
        if prediction["is_unknown"]:
            command = "UNKNOWN"

        self.history.append({"label": label, "command": command, "confidence": confidence})
        weights = {}
        for index, item in enumerate(self.history):
            recency = (index + 1) / len(self.history)
            weights[item["command"]] = weights.get(item["command"], 0.0) + item["confidence"] * recency

        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
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

        return {
            "command": self.active_command,
            "state": self.state,
            "stability": round(float(candidate_score / max(len(self.history), 1)), 4),
            "reason": reason,
            "window": list(self.history),
            "description": COMMAND_DESCRIPTIONS.get(self.active_command, "未定义"),
        }


class SessionStore:
    def __init__(self):
        self._store = {}
        self._lock = Lock()

    def get_engine(self, session_id: str):
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = TemporalIntentEngine()
            return self._store[session_id]

    def reset(self, session_id: str):
        with self._lock:
            self._store[session_id] = TemporalIntentEngine()


classifier = KNNGestureClassifier(DATASET_ROOT)
session_store = SessionStore()
app = Flask(__name__)


def make_prediction_payload(prediction, temporal_result, latency_ms):
    return {
        "prediction": prediction,
        "intent": temporal_result,
        "status": "UNKNOWN" if prediction["is_unknown"] else "OK",
        "latency_ms": round(latency_ms, 2),
        "timestamp": int(time.time() * 1000),
    }


def make_single_frame_intent(prediction):
    command = "UNKNOWN" if prediction["is_unknown"] else LABEL_TO_COMMAND.get(prediction["label"], "UNKNOWN")
    return {
        "command": command,
        "state": "SINGLE_FRAME",
        "stability": prediction["confidence"],
        "reason": "single image inference without temporal smoothing",
        "window": [
            {
                "label": prediction["label"],
                "command": command,
                "confidence": prediction["confidence"],
            }
        ],
        "description": COMMAND_DESCRIPTIONS.get(command, "未定义"),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=classifier.metrics,
        classes=classifier.class_names,
        command_descriptions=COMMAND_DESCRIPTIONS,
        demo_sequences=classifier.demo_sequences,
    )


@app.route("/dataset-image/<path:relative_path>")
def dataset_image(relative_path):
    image_path = DATASET_ROOT / relative_path
    if not image_path.exists() or DATASET_ROOT not in image_path.resolve().parents:
        return jsonify({"error": "image not found"}), 404
    return send_file(image_path)


@app.route("/api/session/reset", methods=["POST"])
def reset_session():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id") or str(uuid.uuid4())
    session_store.reset(session_id)
    return jsonify({"session_id": session_id, "status": "reset"})


@app.route("/api/predict/sample", methods=["POST"])
def predict_sample():
    payload = request.get_json(force=True)
    session_id = payload.get("session_id") or str(uuid.uuid4())
    image_relative_path = payload["image"]
    image_path = DATASET_ROOT / image_relative_path
    if not image_path.exists():
        return jsonify({"error": "sample image not found"}), 404

    engine = session_store.get_engine(session_id)
    start = time.perf_counter()
    prediction = classifier.predict(image_path)
    temporal_result = engine.update(prediction)
    latency_ms = (time.perf_counter() - start) * 1000

    return jsonify(
        {
            "session_id": session_id,
            "image": image_relative_path,
            **make_prediction_payload(prediction, temporal_result, latency_ms),
        }
    )


@app.route("/api/predict/upload", methods=["POST"])
def predict_upload():
    session_id = request.form.get("session_id") or str(uuid.uuid4())

    if "file" in request.files:
        image = Image.open(request.files["file"].stream)
    else:
        image_data = request.form.get("image_data", "")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        if not image_data:
            return jsonify({"error": "no image uploaded"}), 400
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    start = time.perf_counter()
    prediction = classifier.predict(image)
    temporal_result = make_single_frame_intent(prediction)
    latency_ms = (time.perf_counter() - start) * 1000

    return jsonify(
        {
            "session_id": session_id,
            **make_prediction_payload(prediction, temporal_result, latency_ms),
        }
    )


@app.route("/api/demo-sequences")
def demo_sequences():
    return jsonify({"scenarios": classifier.demo_sequences})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
