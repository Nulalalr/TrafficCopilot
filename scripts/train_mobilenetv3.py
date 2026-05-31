from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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

from core.model.mobilenetv3_classifier import build_mobilenetv3_small
from core.utils.dataset import TrafficGestureDataset
from core.utils.training import AverageMeter, build_class_weights, save_json, seed_everything, top1_accuracy


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        return augmented["image"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MobileNetV3 baseline on the traffic gesture dataset.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "mobilenetv3_baseline.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_torchvision_transforms(config: dict):
    image_size = config["data"]["image_size"]
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
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


def build_albumentations_transforms(config: dict):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    image_size = config["data"]["image_size"]
    aug = config["augment"]

    train_tf = A.Compose(
        [
            A.Resize(height=image_size + 32, width=image_size + 32),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.75, 1.0), p=1.0),
            A.HorizontalFlip(p=aug["horizontal_flip"]),
            A.Affine(
                translate_percent={
                    "x": (-aug["shift_limit"], aug["shift_limit"]),
                    "y": (-aug["shift_limit"], aug["shift_limit"]),
                },
                scale=(1.0 - aug["scale_limit"], 1.0 + aug["scale_limit"]),
                rotate=(-aug["rotate_limit"], aug["rotate_limit"]),
                border_mode=0,
                p=aug["shift_scale_rotate_p"],
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug["brightness_limit"],
                contrast_limit=aug["contrast_limit"],
                p=aug["brightness_contrast_p"],
            ),
            A.HueSaturationValue(
                hue_shift_limit=aug["hue_shift_limit"],
                sat_shift_limit=aug["sat_shift_limit"],
                val_shift_limit=aug["val_shift_limit"],
                p=aug["hsv_p"],
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_range=(50, 90), p=1.0),
                ],
                p=aug["degrade_p"],
            ),
            A.OneOf(
                [
                    A.RandomRain(
                        slant_range=(-10, 10),
                        drop_length=16,
                        drop_width=1,
                        blur_value=3,
                        brightness_coefficient=0.9,
                        p=1.0,
                    ),
                    A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0),
                    A.RandomSnow(brightness_coeff=1.5, snow_point_range=(0.1, 0.3), p=1.0),
                ],
                p=aug["weather_p"],
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.18),
                hole_width_range=(0.05, 0.18),
                fill=0,
                p=aug["coarse_dropout_p"],
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    eval_tf = A.Compose(
        [
            A.Resize(height=image_size + 32, width=image_size + 32),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return AlbumentationsWrapper(train_tf), AlbumentationsWrapper(eval_tf)


def build_transforms(config: dict):
    backend = config["augment"]["backend"]
    if backend == "albumentations":
        return build_albumentations_transforms(config)
    return build_torchvision_transforms(config)


def create_dataloaders(config: dict):
    dataset_root = PROJECT_ROOT / config["data"]["dataset_root"]
    train_tf, eval_tf = build_transforms(config)

    train_dataset = TrafficGestureDataset(dataset_root=dataset_root, split="train", transform=train_tf)
    valid_dataset = TrafficGestureDataset(dataset_root=dataset_root, split="valid", transform=eval_tf)
    test_dataset = TrafficGestureDataset(dataset_root=dataset_root, split="test", transform=eval_tf)

    loader_kwargs = {
        "batch_size": config["train"]["batch_size"],
        "num_workers": config["train"]["num_workers"],
        "pin_memory": config["train"]["pin_memory"],
    }
    if int(config["train"]["num_workers"]) > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
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
    all_predictions = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits, targets), batch_size)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    return loss_meter.avg, acc_meter.avg, all_predictions, all_targets


def build_run_dir(config: dict):
    output_root = PROJECT_ROOT / config["train"]["output_root"]
    run_name = config["train"].get("run_name") or datetime.now().strftime("mobilenetv3_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
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


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["train"]["seed"])

    if config["train"]["device"] == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = build_run_dir(config)
    shutil.copy2(args.config, run_dir / "config.yaml")

    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = create_dataloaders(config)
    class_names = train_dataset.class_names

    model = build_mobilenetv3_small(
        num_classes=len(class_names),
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if config["train"]["use_class_weights"]:
        class_weights = build_class_weights(train_dataset.class_counts(), class_names).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"], eta_min=config["train"]["min_lr"])

    best_valid_acc = -1.0
    log_rows = []
    best_ckpt_path = run_dir / "best_model.pth"
    last_ckpt_path = run_dir / "last_model.pth"

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "valid_loss": round(valid_loss, 6),
            "valid_acc": round(valid_acc, 6),
            "lr": round(current_lr, 10),
        }
        log_rows.append(log_row)

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
            "valid_acc": valid_acc,
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
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "test_samples": len(test_dataset),
        "best_valid_acc": round(float(best_valid_acc), 6),
        "test_loss": round(float(test_loss), 6),
        "test_acc": round(float(test_acc), 6),
        "test_predictions": test_predictions,
        "test_targets": test_targets,
        "best_checkpoint": str(best_ckpt_path.relative_to(PROJECT_ROOT)),
        "last_checkpoint": str(last_ckpt_path.relative_to(PROJECT_ROOT)),
    }

    write_training_log(log_rows, run_dir / "training_log.csv")
    save_json(metrics, run_dir / "metrics.json")
    save_json({"class_names": class_names}, run_dir / "class_names.json")

    print(f"Training finished. Best validation accuracy: {best_valid_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
