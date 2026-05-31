from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model.pose_sequence_classifier import PoseGRUClassifier
from core.utils.pose_sequence_dataset import PoliceGesturePoseSequenceDataset
from core.utils.training import AverageMeter, build_class_weights, save_json, seed_everything, top1_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pose sequence classifier (GRU) on police_gesture_v1 videos.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "pose_sequence_classifier.yaml"))
    return parser.parse_args()


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_run_dir(config: dict) -> Path:
    output_root = PROJECT_ROOT / config["train"]["output_root"]
    run_name = config["train"]["run_name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_training_log(rows: list[dict], path: Path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "valid_loss",
        "valid_acc",
        "lr",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_dataloaders(config: dict):
    dataset_root = PROJECT_ROOT / config["data"]["dataset_root"]
    pose_root = PROJECT_ROOT / config["data"]["pose_root"]
    clip_len = int(config["data"]["clip_len"])
    stride = int(config["data"]["stride"])

    train_ds = PoliceGesturePoseSequenceDataset(
        dataset_root=dataset_root,
        pose_root=pose_root,
        split="train",
        clip_len=clip_len,
        stride=stride,
        include_background=bool(config["data"].get("include_background", False)),
    )
    valid_ds = PoliceGesturePoseSequenceDataset(
        dataset_root=dataset_root,
        pose_root=pose_root,
        split="train",
        clip_len=clip_len,
        stride=max(clip_len, stride),
        include_background=bool(config["data"].get("include_background", False)),
    )
    test_ds = PoliceGesturePoseSequenceDataset(
        dataset_root=dataset_root,
        pose_root=pose_root,
        split="test",
        clip_len=clip_len,
        stride=max(clip_len, stride),
        include_background=bool(config["data"].get("include_background", False)),
    )

    loader_kwargs = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": int(config["train"]["num_workers"]),
        "pin_memory": bool(config["train"]["pin_memory"]),
    }
    if int(config["train"]["num_workers"]) > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for pose, targets in loader:
        pose = pose.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pose)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits, targets), batch_size)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    predictions_all = []
    targets_all = []
    for pose, targets in loader:
        pose = pose.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(pose)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits, targets), batch_size)
        predictions_all.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    return loss_meter.avg, acc_meter.avg, predictions_all, targets_all


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["train"]["seed"]))

    if config["train"]["device"] == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = build_run_dir(config)
    shutil.copy2(args.config, run_dir / "config.yaml")

    train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader = create_dataloaders(config)
    class_names = train_ds.class_names

    pose_dim = int(config["model"]["pose_dim"])
    model = PoseGRUClassifier(
        num_classes=len(class_names),
        pose_dim=pose_dim,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if bool(config["train"]["use_class_weights"]):
        class_weights = build_class_weights(train_ds.class_counts(), class_names).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=float(config["train"]["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["train"]["epochs"]), eta_min=float(config["train"]["min_lr"]))

    best_valid_acc = -1.0
    log_rows = []
    best_ckpt_path = run_dir / "best_model.pth"
    last_ckpt_path = run_dir / "last_model.pth"

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "train_acc": round(float(train_acc), 6),
                "valid_loss": round(float(valid_loss), 6),
                "valid_acc": round(float(valid_acc), 6),
                "lr": round(float(current_lr), 10),
            }
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} lr={current_lr:.6f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_names": class_names,
            "config": config,
            "valid_acc": float(valid_acc),
        }
        torch.save(state, last_ckpt_path)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(state, best_ckpt_path)
        scheduler.step()

    best_state = torch.load(best_ckpt_path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_state["model_state_dict"])
    else:
        model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_acc, test_predictions, test_targets = evaluate(model, test_loader, criterion, device)

    metrics = {
        "device": str(device),
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_sequences": len(train_ds),
        "valid_sequences": len(valid_ds),
        "test_sequences": len(test_ds),
        "best_valid_acc": round(float(best_valid_acc), 6),
        "test_loss": round(float(test_loss), 6),
        "test_acc": round(float(test_acc), 6),
        "best_checkpoint": str(best_ckpt_path.relative_to(PROJECT_ROOT)),
        "last_checkpoint": str(last_ckpt_path.relative_to(PROJECT_ROOT)),
        "clip_len": int(config["data"]["clip_len"]),
        "stride": int(config["data"]["stride"]),
    }

    write_training_log(log_rows, run_dir / "training_log.csv")
    save_json(metrics, run_dir / "metrics.json")
    save_json({"class_names": class_names}, run_dir / "class_names.json")

    print(f"Training finished. Best validation accuracy: {best_valid_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Artifacts saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
