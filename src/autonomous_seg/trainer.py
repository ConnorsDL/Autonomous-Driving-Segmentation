"""Training utilities for semantic segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Configuration that controls the training loop."""

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    val_interval: int = 1
    grad_clip: Optional[float] = None
    num_classes: int = 2
    ignore_index: int = 255
    resume_from: Optional[Path] = None

    def to_dict(self) -> Dict:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "checkpoint_dir": str(self.checkpoint_dir),
            "val_interval": self.val_interval,
            "grad_clip": self.grad_clip,
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "resume_from": str(self.resume_from) if self.resume_from else None,
        }


class Trainer:
    """High level helper that orchestrates model training."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and self.device.type == "cuda")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_miou = 0.0

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def train(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "train_miou": [], "val_loss": [], "val_miou": []}

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_miou = self._train_one_epoch(epoch)
            history["train_loss"].append(train_loss)
            history["train_miou"].append(train_miou)

            if self.val_loader is not None and epoch % self.config.val_interval == 0:
                val_loss, val_miou = self.evaluate()
                history["val_loss"].append(val_loss)
                history["val_miou"].append(val_miou)
                if val_miou > self.best_miou:
                    self.best_miou = val_miou
                    self._save_checkpoint(epoch, best=True)
            else:
                val_loss = None
                val_miou = None

            log_message = (
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train_loss: {train_loss:.4f} | train_mIoU: {train_miou:.4f}"
            )
            if val_loss is not None and val_miou is not None:
                log_message += f" | val_loss: {val_loss:.4f} | val_mIoU: {val_miou:.4f}"
            print(log_message)

            self._save_checkpoint(epoch, best=False, metrics={
                "train_loss": train_loss,
                "train_miou": train_miou,
                "val_loss": val_loss,
                "val_miou": val_miou,
            })

        return history

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        epoch_loss = 0.0
        intersections = torch.zeros(self.config.num_classes, device=self.device)
        unions = torch.zeros(self.config.num_classes, device=self.device)
        total_batches = len(self.train_loader)

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            if self.config.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            batch_inter, batch_union = self._intersection_and_union(logits, targets)
            intersections += batch_inter
            unions += batch_union

        miou = self._mean_iou(intersections, unions)
        return epoch_loss / total_batches, miou

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        epoch_loss = 0.0
        intersections = torch.zeros(self.config.num_classes, device=self.device)
        unions = torch.zeros(self.config.num_classes, device=self.device)
        total_batches = len(self.val_loader) if self.val_loader is not None else 1

        if self.val_loader is None:
            return 0.0, 0.0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            epoch_loss += loss.item()
            batch_inter, batch_union = self._intersection_and_union(logits, targets)
            intersections += batch_inter
            unions += batch_union

        miou = self._mean_iou(intersections, unions)
        return epoch_loss / total_batches, miou

    def _intersection_and_union(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.argmax(logits, dim=1)
        valid_mask = targets != self.config.ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        if preds.numel() == 0:
            return (
                torch.zeros(self.config.num_classes, device=self.device),
                torch.zeros(self.config.num_classes, device=self.device),
            )

        conf_matrix = torch.zeros((self.config.num_classes, self.config.num_classes), device=self.device)
        indices = targets * self.config.num_classes + preds
        conf_matrix.view(-1).index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
        intersection = torch.diag(conf_matrix)
        union = conf_matrix.sum(dim=0) + conf_matrix.sum(dim=1) - intersection
        return intersection, union

    @staticmethod
    def _mean_iou(intersections: torch.Tensor, unions: torch.Tensor) -> float:
        valid = unions > 0
        if valid.sum() == 0:
            return 0.0
        iou = intersections[valid] / unions[valid]
        return iou.mean().item()

    def _save_checkpoint(self, epoch: int, best: bool = False, metrics: Optional[Dict[str, Optional[float]]] = None) -> None:
        checkpoint_path = self.checkpoint_dir / ("best.pt" if best else f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "config": self.config.to_dict(),
                "metrics": metrics or {},
            },
            checkpoint_path,
        )

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Resumed training from {path} at epoch {checkpoint['epoch']}")
