#!/usr/bin/env python3
"""
Main training script for Point Transformer jewelry segmentation.

Usage:
    python -m training.train --config configs/training_config.yaml

AWS EC2 Usage:
    # First, sync data from S3
    aws s3 sync s3://mesh-segmentor/datasets/train ./data/processed/train
    aws s3 sync s3://mesh-segmentor/datasets/labels/train ./data/labels/train

    # Run training
    python -m training.train \
        --train-data ./data/processed/train \
        --train-labels ./data/labels/train \
        --val-data ./data/processed/val \
        --val-labels ./data/labels/val \
        --s3-bucket mesh-segmentor \
        --epochs 100 \
        --batch-size 8
"""

import argparse
import sys
import os
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.point_transformer import create_point_transformer
from models.losses import SegmentationLoss
from preprocessing.dataset import JewelrySegmentationDataset
from training.trainer import Trainer, create_optimizer, create_scheduler
from training.augmentations import create_training_augmentation
from utils.logging import setup_logging, get_logger
from utils.config import load_config, get_config_path

from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Point Transformer for jewelry segmentation"
    )

    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        default="./data/processed/train",
        help="Path to training data (local or S3 URI)",
    )
    parser.add_argument(
        "--train-labels",
        type=str,
        default="./data/labels/train",
        help="Path to training labels (local or S3 URI)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="./data/processed/val",
        help="Path to validation data",
    )
    parser.add_argument(
        "--val-labels",
        type=str,
        default="./data/labels/val",
        help="Path to validation labels",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--num-points", type=int, default=20000, help="Points per sample"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Data loader workers"
    )

    # Model parameters
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["small", "base", "large"],
        help="Model size",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Pretrained weights path"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket for checkpoint upload",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=True, help="Use mixed precision"
    )
    parser.add_argument(
        "--no-amp", action="store_false", dest="amp", help="Disable mixed precision"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level, format_type="text")
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Point Transformer Training for Jewelry Segmentation")
    logger.info("=" * 60)

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logger.info(f"Device: {args.device}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create datasets
    logger.info("Creating datasets...")

    train_augment = create_training_augmentation()

    train_dataset = JewelrySegmentationDataset(
        data_dir=args.train_data,
        label_dir=args.train_labels,
        num_points=args.num_points,
        transform=train_augment,
        normalize=True,
    )

    val_dataset = JewelrySegmentationDataset(
        data_dir=args.val_data,
        label_dir=args.val_labels,
        num_points=args.num_points,
        transform=None,
        normalize=True,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    logger.info(f"Creating {args.model_size} Point Transformer model...")

    model = create_point_transformer(
        num_classes=3,
        model_size=args.model_size,
        pretrained=args.pretrained,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_name="AdamW",
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = create_scheduler(
        optimizer,
        scheduler_name="CosineAnnealingLR",
        T_max=args.epochs,
        eta_min=1e-5,
    )

    # Create loss function
    # Class weights: background=0.1, metal=1.0, gem=2.0
    criterion = SegmentationLoss(
        num_classes=3,
        class_weights=[0.1, 1.0, 2.0],
        ce_weight=1.0,
        dice_weight=0.5,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        scheduler=scheduler,
        num_classes=3,
        use_amp=args.amp,
        grad_clip=1.0,
        checkpoint_dir=args.checkpoint_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix="models/checkpoints/",
        early_stopping_patience=15,
        log_interval=10,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        resume_from=args.resume,
    )

    logger.info("Training complete!")
    logger.info(f"Best validation mIoU: {trainer.best_metric:.4f}")

    # Final checkpoint upload
    if args.s3_bucket:
        logger.info("Uploading final model to S3...")
        from utils.s3 import S3Client

        s3 = S3Client(bucket=args.s3_bucket)
        s3.upload_file(
            f"{args.checkpoint_dir}/best_model.pt",
            "models/best_model.pt",
        )
        logger.info("Done!")


if __name__ == "__main__":
    main()
