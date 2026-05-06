from __future__ import annotations

import base64
import csv
import io
import re
import sys
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
import yaml
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model.mobilenetv3_classifier import build_mobilenetv3_small


DATASET_ROOT = Path(__file__).parent / "traffic gestures.v1i.multiclass"
CLASS_CSV_NAME = "_classes.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "mobilenetv3_albu_weather" / "best_model_quantized.pth"
CHECKPOINT_METRICS_PATH = PROJECT_ROOT / "experiments" / "mobilenetv3_albu_weather" / "metrics.json"
CHECKPOINT_CONFIG_PATH = PROJECT_ROOT / "experiments" / "mobilenetv3_albu_weather" / "config.yaml"
POSE_TASK_MODEL_PATH = PROJECT_ROOT / "weights" / "pose_landmarker_lite.task"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
    "CHANGE_LANES": "Suggest lane change",
    "GO_STRAIGHT": "Allow going straight",
    "PULL_OVER": "Pull over to roadside",
    "SLOW_DOWN": "Slow down and observe",
    "STOP": "Stop immediately",
    "TURN_LEFT": "Allow left turn",
    "TURN_RIGHT": "Allow right turn",
    "WAIT_LEFT_TURN": "Wait for left turn",
    "UNKNOWN": "Confidence too low, degrade to safe state",
}

DEMO_SCENARIOS = [
    {
        "id": "intersection_control",
        "name": "Intersection Control",
        "description": "STOP -> GO_STRAIGHT -> TURN_LEFT",
        "labels": ["stop", "stop", "stop", "go straight", "go straight", "go straight", "turn left", "turn left", "turn left"],
    },
    {
        "id": "lane_guidance",
        "name": "Lane Guidance",
        "description": "SLOW_DOWN -> CHANGE_LANES -> PULL_OVER",
        "labels": ["slow down", "slow down", "slow down", "change lanes", "change lanes", "change lanes", "pull over", "pull over", "pull over"],
    },
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]


@dataclass
class Sample:
    path: Path
    label: str
    split: str


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


class ModelGestureClassifier:
    def __init__(self, dataset_root: Path, checkpoint_path: Path, metrics_path: Path, config_path: Path):
        self.dataset_root = dataset_root
        self.dataset = GestureDataset(dataset_root)
        self.samples_by_split = self.dataset.all_samples()
        self.class_names = list(self.dataset.class_names)
        self.metrics = self._build_metrics(metrics_path)
        self.demo_sequences = self._build_demo_sequences()
        self.model_name = "MobileNetV3 + Weather Aug"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = self._load_model(checkpoint_path, config_path)

    def _build_metrics(self, metrics_path: Path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            raw_metrics = yaml.safe_load(handle)
        train_samples = self.samples_by_split["train"]
        return {
            "valid_accuracy": raw_metrics["best_valid_acc"],
            "test_accuracy": raw_metrics["test_acc"],
            "sample_counts": {
                split_name: len(samples) for split_name, samples in self.samples_by_split.items()
            },
            "class_distribution": {
                label: sum(1 for sample in train_samples if sample.label == label) for label in self.class_names
            },
        }

    def _load_model(self, checkpoint_path: Path, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        class_names = checkpoint["class_names"]
        if class_names != self.class_names:
            raise ValueError("Checkpoint class names do not match dataset labels.")

        model = build_mobilenetv3_small(
            num_classes=len(class_names),
            pretrained=False,
            dropout=config["model"]["dropout"],
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if checkpoint.get("compressed", False) and checkpoint.get("compression_type") == "fp16":
            model = model.half()
        
        model.to(self.device)
        model.eval()

        image_size = config["data"]["image_size"]
        transform = transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return model, transform

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
        return scenarios

    @staticmethod
    def _frame_number(filename):
        match = re.search(r"frame_(\d+)", filename)
        return int(match.group(1)) if match else 0

    def predict(self, image_source):
        if isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        else:
            image = Image.open(image_source).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        if next(self.model.parameters()).dtype == torch.float16:
            tensor = tensor.half()

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        top_probs, top_indices = torch.topk(probs, k=min(3, len(self.class_names)))
        top_label = self.class_names[int(top_indices[0])]
        top_probability = float(top_probs[0])
        runner_up_probability = float(top_probs[1]) if len(top_probs) > 1 else 0.0
        margin = top_probability - runner_up_probability
        is_unknown = top_probability < 0.5 or margin < 0.08

        return {
            "label": top_label,
            "confidence": round(top_probability, 4),
            "margin": round(margin, 4),
            "is_unknown": is_unknown,
            "topk": [
                {
                    "label": self.class_names[int(index)],
                    "probability": round(float(probability), 4),
                }
                for probability, index in zip(top_probs.tolist(), top_indices.tolist())
            ],
        }


class PoseVisualizer:
    def __init__(self, model_path: Path):
        self.api_kind = None
        self.runtime = None
        self.mp = None
        try:
            import mediapipe as mp
        except Exception:
            self.available = False
            return

        self.mp = mp
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and model_path.exists():
            self.api_kind = "tasks"
            self.available = True
            self.runtime = self._create_tasks_runtime(model_path)
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.api_kind = "legacy"
            self.available = True
            self.runtime = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
        else:
            self.available = False

    def _create_tasks_runtime(self, model_path: Path):
        BaseOptions = self.mp.tasks.BaseOptions
        PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = self.mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        return PoseLandmarker.create_from_options(options)

    def _extract_landmarks(self, image: Image.Image):
        if not self.available:
            return []

        if self.api_kind == "tasks":
            rgb = torch.tensor(0)  # placeholder to satisfy linter path separation
            del rgb
            import numpy as np

            rgb = np.array(image.convert("RGB"))
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            result = self.runtime.detect(mp_image)
            if not result.pose_landmarks:
                return []
            return result.pose_landmarks[0]

        import numpy as np

        result = self.runtime.process(image=np.array(image.convert("RGB")))
        if result.pose_landmarks is None:
            return []
        return result.pose_landmarks.landmark

    def draw_overlay(self, image: Image.Image):
        overlay = image.convert("RGB").copy()
        draw = ImageDraw.Draw(overlay)
        landmarks = self._extract_landmarks(overlay)
        if not landmarks:
            return overlay, False

        width, height = overlay.size

        def pt(idx):
            landmark = landmarks[idx]
            return (landmark.x * width, landmark.y * height)

        for start_idx, end_idx in POSE_CONNECTIONS:
            start_point = pt(start_idx)
            end_point = pt(end_idx)
            draw.line([start_point, end_point], fill=(255, 140, 80), width=4)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
            x, y = pt(idx)
            visibility = float(getattr(landmarks[idx], "visibility", 1.0))
            color = (48, 180, 90) if visibility >= 0.75 else (230, 170, 30) if visibility >= 0.45 else (220, 70, 70)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255), width=2)

        return overlay, True


def image_to_data_url(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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
            "description": COMMAND_DESCRIPTIONS.get(self.active_command, "Undefined"),
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


classifier = ModelGestureClassifier(
    dataset_root=DATASET_ROOT,
    checkpoint_path=CHECKPOINT_PATH,
    metrics_path=CHECKPOINT_METRICS_PATH,
    config_path=CHECKPOINT_CONFIG_PATH,
)
pose_visualizer = PoseVisualizer(POSE_TASK_MODEL_PATH)
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
        "reason": f"single image inference with {classifier.model_name}",
        "window": [
            {
                "label": prediction["label"],
                "command": command,
                "confidence": prediction["confidence"],
            }
        ],
        "description": COMMAND_DESCRIPTIONS.get(command, "Undefined"),
    }

from typing import Sequence, Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def flatten_labels(sequences: Sequence[Dict]) -> np.ndarray:
    """
    展平演示场景序列中的真实标签
    Args:
        sequences: 由_build_demo_sequences生成的演示场景序列  
    Returns:
        展平后的真实标签数组
    """
    labels = []
    for scenario in sequences:
        for frame in scenario["frames"]:
            labels.append(frame["label"])
    return np.array(labels)

def flatten_preds(preds: Sequence[np.ndarray]) -> np.ndarray:
    """
    展平模型预测结果
    Args:
        preds: 由predict函数生成的预测结果序列  
    Returns:
        展平后的预测标签数组
    """
    return np.concatenate(preds)

def evaluate_model(classifier, split="valid"):
    """
    评估模型在指定数据集划分上的性能
    Args:
        classifier: ModelGestureClassifier实例
        split: 数据集划分（"train"/"valid"/"test"）
    Returns:
        包含评估指标的字典
    """
    samples = classifier.samples_by_split[split]
    
    preds = []
    true_labels = []
    for sample in samples:
        prediction = classifier.predict(sample.path)
        preds.append(np.array([prediction["label"]]))
        true_labels.append(sample.label)
    
    pred_labels = flatten_preds(preds)
    
    true_labels = np.array(true_labels)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="macro")
    recall = recall_score(true_labels, pred_labels, average="macro")
    f1 = f1_score(true_labels, pred_labels, average="macro")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 评估验证集
valid_result = evaluate_model(classifier, split="valid")
print(f"Valid Accuracy: {valid_result['accuracy']:.4f}")
print(f"Valid Precision: {valid_result['precision']:.4f}")
print(f"Valid Recall: {valid_result['recall']:.4f}")
print(f"Valid F1: {valid_result['f1']:.4f}")

# 评估测试集
test_result = evaluate_model(classifier, split="test")
print(f"Test Accuracy: {test_result['accuracy']:.4f}")
print(f"Test Precision: {test_result['precision']:.4f}")
print(f"Test Recall: {test_result['recall']:.4f}")
print(f"Test F1: {test_result['f1']:.4f}")


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


@app.route("/api/model-info")
def model_info():
    return jsonify(
        {
            "model_name": classifier.model_name,
            "valid_accuracy": classifier.metrics["valid_accuracy"],
            "test_accuracy": classifier.metrics["test_accuracy"],
        }
    )


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
    pose_overlay, pose_detected = pose_visualizer.draw_overlay(image)

    return jsonify(
        {
            "session_id": session_id,
            "pose_overlay": image_to_data_url(pose_overlay),
            "pose_detected": pose_detected,
            "model_name": classifier.model_name,
            **make_prediction_payload(prediction, temporal_result, latency_ms),
        }
    )


@app.route("/api/demo-sequences")
def demo_sequences():
    return jsonify({"scenarios": classifier.demo_sequences})


@app.route("/api/evaluate/<split>")
def evaluate_split(split):
    if split not in ["train", "valid", "test"]:
        return jsonify({"error": "Invalid split"}), 400
    result = evaluate_model(classifier, split=split)
    return jsonify({
        "accuracy": round(result["accuracy"], 4),
        "precision": round(result["precision"], 4),
        "recall": round(result["recall"], 4),
        "f1": round(result["f1"], 4)
    })

@app.route("/evaluate")
def evaluate_page():
    valid_result = evaluate_model(classifier, split="valid")
    test_result = evaluate_model(classifier, split="test")
    return render_template(
        "evaluate.html",
        valid_accuracy=round(valid_result["accuracy"], 4),
        valid_precision=round(valid_result["precision"], 4),
        valid_recall=round(valid_result["recall"], 4),
        valid_f1=round(valid_result["f1"], 4),
        test_accuracy=round(test_result["accuracy"], 4),
        test_precision=round(test_result["precision"], 4),
        test_recall=round(test_result["recall"], 4),
        test_f1=round(test_result["f1"], 4)
    )

if __name__ == "__main__":
    app.run(debug=False, port=5000)
