"""Command line entry point for training a semantic segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from autonomous_seg import DataConfig, Trainer, TrainingConfig, build_model, create_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an autonomous driving segmentation model")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to save the training history as JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device specified in the configuration (e.g. 'cuda' or 'cpu')",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_data_config(cfg: Dict[str, Any]) -> DataConfig:
    class_map = cfg.get("class_map")
    if class_map is not None:
        class_map = {int(k): int(v) for k, v in class_map.items()}

    image_size = tuple(int(dim) for dim in cfg.get("image_size", [512, 512]))

    return DataConfig(
        train_dir=Path(cfg["train_dir"]),
        val_dir=Path(cfg["val_dir"]) if cfg.get("val_dir") else None,
        image_size=image_size,
        batch_size=int(cfg.get("batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 4)),
        class_map=class_map,
        augmentations=bool(cfg.get("augmentations", True)),
    )


def build_training_config(cfg: Dict[str, Any], num_classes: int, device_override: str | None) -> TrainingConfig:
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    resume_from = Path(cfg["resume_from"]) if cfg.get("resume_from") else None

    grad_clip = cfg.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None

    training_cfg = TrainingConfig(
        epochs=int(cfg.get("epochs", 50)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        device=device_override or cfg.get("device", ("cuda" if torch.cuda.is_available() else "cpu")),
        mixed_precision=bool(cfg.get("mixed_precision", True)),
        checkpoint_dir=checkpoint_dir,
        val_interval=int(cfg.get("val_interval", 1)),
        grad_clip=grad_clip,
        num_classes=num_classes,
        ignore_index=int(cfg.get("ignore_index", 255)),
        resume_from=resume_from,
    )
    return training_cfg


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    data_cfg = build_data_config(config["data"])
    model_cfg = config["model"]
    training_cfg = build_training_config(config.get("training", {}), model_cfg["num_classes"], args.device)

    train_loader, val_loader = create_dataloaders(data_cfg)
    model = build_model(model_cfg)
    trainer = Trainer(model, train_loader, val_loader, training_cfg)

    history = trainer.train()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Training completed. Best mIoU:", trainer.best_miou)


if __name__ == "__main__":
    main()
