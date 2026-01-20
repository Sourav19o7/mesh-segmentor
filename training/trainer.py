"""
Training loop for Point Transformer segmentation model.

Features:
- Mixed precision training (AMP)
- Gradient clipping
- Learning rate scheduling
- Early stopping
- Checkpoint management
- S3 integration for checkpoints
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Any
from pathlib import Path
from utils.logging import get_logger
from utils.s3 import S3Client
from training.metrics import MetricsAccumulator

logger = get_logger(__name__)


class Trainer:
    """
    Trainer for Point Transformer segmentation model.

    Handles:
    - Training loop with validation
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint saving/loading
    - Early stopping
    - S3 checkpoint upload

    Example:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cuda",
        )
        trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_classes: int = 3,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "models/checkpoints/",
        early_stopping_patience: int = 15,
        log_interval: int = 10,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            num_classes: Number of segmentation classes
            use_amp: Use automatic mixed precision
            grad_clip: Gradient clipping value (0 to disable)
            checkpoint_dir: Local directory for checkpoints
            s3_bucket: S3 bucket for checkpoint upload
            s3_prefix: S3 key prefix for checkpoints
            early_stopping_patience: Epochs without improvement before stopping
            log_interval: Log every N batches
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.use_amp = use_amp and device == "cuda"
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.early_stopping_patience = early_stopping_patience
        self.log_interval = log_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # S3 client
        if s3_bucket:
            self.s3_client = S3Client(bucket=s3_bucket)
        else:
            self.s3_client = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.epochs_without_improvement = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            resume_from: Path to checkpoint to resume from

        Returns:
            Dictionary with training history
        """
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(
            f"Starting training for {epochs} epochs, "
            f"device={self.device}, amp={self.use_amp}"
        )

        history = {
            "train_loss": [],
            "train_miou": [],
            "val_loss": [],
            "val_miou": [],
            "lr": [],
        }

        start_epoch = self.epoch
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_miou"].append(train_metrics["miou"])

            # Validation epoch
            val_metrics = self._validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_miou"].append(val_metrics["miou"])

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["lr"].append(current_lr)

            # Log epoch summary
            logger.info("=" * 70)
            logger.info(
                f"Epoch {epoch + 1}/{epochs} Complete | "
                f"Time: {train_metrics.get('epoch_time', 0):.1f}s | "
                f"LR: {current_lr:.6f}"
            )
            logger.info(
                f"  Train -> Loss: {train_metrics['loss']:.4f} | "
                f"mIoU: {train_metrics['miou']:.4f} | "
                f"Accuracy: {train_metrics.get('accuracy', 0):.4f}"
            )
            logger.info(
                f"  Val   -> Loss: {val_metrics['loss']:.4f} | "
                f"mIoU: {val_metrics['miou']:.4f} | "
                f"Accuracy: {val_metrics.get('accuracy', 0):.4f}"
            )
            # Log per-class IoU if available
            if 'class_iou' in val_metrics:
                class_names = {0: 'Background', 1: 'Metal', 2: 'Gem'}
                iou_str = " | ".join(
                    f"{class_names.get(i, f'Class{i}')}: {iou:.3f}"
                    for i, iou in enumerate(val_metrics['class_iou'])
                )
                logger.info(f"  Val IoU per class -> {iou_str}")
            logger.info("=" * 70)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Checkpointing
            is_best = val_metrics["miou"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["miou"]
                self.epochs_without_improvement = 0
                logger.info(f"New best mIoU: {self.best_metric:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping after {epoch + 1} epochs "
                    f"({self.early_stopping_patience} epochs without improvement)"
                )
                break

        logger.info(f"Training complete. Best mIoU: {self.best_metric:.4f}")
        return history

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        accumulator = MetricsAccumulator(self.num_classes)

        num_batches = len(train_loader)
        epoch_start = time.time()
        running_loss = 0.0

        for batch_idx, (points, labels) in enumerate(train_loader):
            # Move to device
            points = points.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with AMP
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(points)
                    loss = self.criterion(logits, labels)

                # Backward pass with scaler
                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(points)
                loss = self.criterion(logits, labels)

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

            # Update metrics
            predictions = logits.argmax(dim=-1)
            accumulator.update(predictions, labels, loss.item())
            running_loss += loss.item()

            self.global_step += 1

            # Log batch progress with progress percentage
            if (batch_idx + 1) % self.log_interval == 0:
                progress = (batch_idx + 1) / num_batches * 100
                avg_loss = running_loss / (batch_idx + 1)
                elapsed = time.time() - epoch_start
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (num_batches - batch_idx - 1) / batches_per_sec if batches_per_sec > 0 else 0

                logger.info(
                    f"  [{progress:5.1f}%] Batch {batch_idx + 1}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                    f"Speed: {batches_per_sec:.1f} batch/s | ETA: {eta:.0f}s"
                )

        epoch_time = time.time() - epoch_start
        metrics = accumulator.compute()
        metrics["epoch_time"] = epoch_time

        return metrics

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        accumulator = MetricsAccumulator(self.num_classes)

        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        logits = self.model(points)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(points)
                    loss = self.criterion(logits, labels)

                predictions = logits.argmax(dim=-1)
                accumulator.update(predictions, labels, loss.item())

        return accumulator.compute()

    def save_checkpoint(
        self,
        is_best: bool = False,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save training checkpoint.

        Args:
            is_best: Whether this is the best model so far
            filename: Custom filename

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch + 1}.pt"

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler"] = self.scaler.state_dict()

        # Save locally
        local_path = self.checkpoint_dir / filename
        torch.save(checkpoint, local_path)
        logger.info(f"Saved checkpoint: {local_path}")

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

            # Upload to S3
            if self.s3_client:
                s3_key = f"{self.s3_prefix}best_model.pt"
                self.s3_client.upload_file(best_path, s3_key)
                logger.info(f"Uploaded best model to S3: {s3_key}")

        return str(local_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (local or S3 URI)
        """
        # Handle S3 path
        if checkpoint_path.startswith("s3://"):
            from utils.s3 import parse_s3_uri
            bucket, key = parse_s3_uri(checkpoint_path)
            local_path = self.checkpoint_dir / "resume_checkpoint.pt"

            client = S3Client(bucket=bucket)
            client.download_file(key, local_path)
            checkpoint_path = str(local_path)

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint["epoch"] + 1  # Start from next epoch
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        self.epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )

        if self.scheduler and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        logger.info(
            f"Resumed from epoch {checkpoint['epoch'] + 1}, "
            f"best mIoU: {self.best_metric:.4f}"
        )


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "AdamW",
    lr: float = 0.001,
    weight_decay: float = 0.01,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer.

    Args:
        model: Model to optimize
        optimizer_name: Optimizer type
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    params = model.parameters()

    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            params, lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "CosineAnnealingLR",
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_name: Scheduler type
        **kwargs: Scheduler arguments

    Returns:
        Scheduler instance
    """
    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
