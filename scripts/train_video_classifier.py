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
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model.video_classifier import MobileNetV3GRUClassifier
from core.utils.training import AverageMeter, build_class_weights, save_json, seed_everything, top1_accuracy
from core.utils.video_dataset import TrafficGestureVideoDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = [
    "Stop",
    "Forward",
    "Left Turn",
    "Left Turn Waiting",
    "Right Turn",
    "Lane Changing",
    "Slow Down",
    "Pull Over",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight video gesture classifier.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "video_classifier.yaml"))
    return parser.parse_args()


def load_config(config_path: str | Path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_transforms(config: dict):
    image_size = int(config["data"]["image_size"])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def create_dataloaders(config: dict):
    dataset_root = PROJECT_ROOT / config["data"]["dataset_root"]
    train_tf, eval_tf = build_transforms(config)

    train_dataset = TrafficGestureVideoDataset(
        dataset_root=dataset_root,
        split="train",
        class_names=CLASS_NAMES,
        clip_len=config["data"]["clip_len"],
        stride=config["data"]["train_stride"],
        transform=train_tf,
        include_background=config["data"].get("include_background", False),
    )
    test_dataset = TrafficGestureVideoDataset(
        dataset_root=dataset_root,
        split="test",
        class_names=CLASS_NAMES,
        clip_len=config["data"]["clip_len"],
        stride=config["data"]["eval_stride"],
        transform=eval_tf,
        include_background=config["data"].get("include_background", False),
    )

    loader_kwargs = {
        "batch_size": config["train"]["batch_size"],
        "num_workers": config["train"]["num_workers"],
        "pin_memory": config["train"]["pin_memory"],
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, test_dataset, train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for frames, targets in loader:
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
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

    for frames, targets in loader:
        frames = frames.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(frames)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits, targets), batch_size)
        predictions_all.extend(predictions.cpu().tolist())
        targets_all.extend(targets.cpu().tolist())

    return loss_meter.avg, acc_meter.avg, predictions_all, targets_all


def build_run_dir(config: dict):
    output_root = PROJECT_ROOT / config["train"]["output_root"]
    run_name = config["train"].get("run_name") or datetime.now().strftime("video_classifier_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_training_log(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["train"]["seed"])

    if config["train"]["device"] == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = build_run_dir(config)
    shutil.copy2(args.config, run_dir / "config.yaml")

    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(config)

    model = MobileNetV3GRUClassifier(
        num_classes=len(CLASS_NAMES),
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
    ).to(device)

    if config["train"]["use_class_weights"]:
        class_weights = build_class_weights(train_dataset.class_counts(), CLASS_NAMES).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"], eta_min=config["train"]["min_lr"])

    best_test_acc = -1.0
    log_rows = []
    best_ckpt_path = run_dir / "best_model.pth"
    last_ckpt_path = run_dir / "last_model.pth"

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(test_acc, 6),
                "lr": round(current_lr, 10),
            }
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} lr={current_lr:.6f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_names": CLASS_NAMES,
            "config": config,
            "test_acc": test_acc,
        }
        torch.save(state, last_ckpt_path)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(state, best_ckpt_path)

        scheduler.step()

    best_state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_acc, test_predictions, test_targets = evaluate(model, test_loader, criterion, device)

    metrics = {
        "device": str(device),
        "class_names": CLASS_NAMES,
        "train_sequences": len(train_dataset),
        "test_sequences": len(test_dataset),
        "best_test_acc": round(float(best_test_acc), 6),
        "final_test_loss": round(float(test_loss), 6),
        "final_test_acc": round(float(test_acc), 6),
        "test_predictions": test_predictions,
        "test_targets": test_targets,
        "best_checkpoint": str(best_ckpt_path.relative_to(PROJECT_ROOT)),
    }

    write_training_log(log_rows, run_dir / "training_log.csv")
    save_json(metrics, run_dir / "metrics.json")
    save_json({"class_names": CLASS_NAMES}, run_dir / "class_names.json")

    print(f"Training finished. Best test accuracy: {best_test_acc:.4f}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
