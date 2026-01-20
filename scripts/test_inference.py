#!/usr/bin/env python3
"""
Test inference on synthetic data or sample files.

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --model checkpoints/best_model.pt
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


def create_synthetic_mesh():
    """Create a synthetic ring mesh for testing."""
    import trimesh

    # Create a torus (ring body)
    ring = trimesh.creation.torus(major_radius=1.0, minor_radius=0.15, major_sections=64, minor_sections=32)

    # Create a sphere (gem)
    gem = trimesh.creation.icosphere(radius=0.2, subdivisions=3)
    gem.apply_translation([1.0, 0, 0.15])

    # Combine
    combined = trimesh.util.concatenate([ring, gem])

    return combined, len(ring.faces), len(gem.faces)


def test_predictor(model_path: str, device: str = "auto"):
    """Test the predictor on synthetic data."""
    from inference.predictor import Predictor

    logger.info("=" * 60)
    logger.info("Testing Predictor")
    logger.info("=" * 60)

    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_path}")

    # Check model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return False

    # Load predictor
    predictor = Predictor(
        model_path=model_path,
        device=device,
        model_size="small",
        use_amp=False,
    )

    # Create synthetic point cloud
    logger.info("Creating synthetic test data...")
    num_points = 20000

    # Generate ring + gem points (similar to training data)
    ring_points = num_points * 7 // 10  # 70% ring
    gem_points = num_points - ring_points  # 30% gem

    # Ring (torus)
    u = np.random.uniform(0, 2 * np.pi, ring_points)
    v = np.random.uniform(0, 2 * np.pi, ring_points)
    R, r = 1.0, 0.15
    ring_x = (R + r * np.cos(v)) * np.cos(u)
    ring_y = (R + r * np.cos(v)) * np.sin(u)
    ring_z = r * np.sin(v)
    ring = np.stack([ring_x, ring_y, ring_z], axis=1)

    # Gem (sphere)
    phi = np.random.uniform(0, np.pi, gem_points)
    theta = np.random.uniform(0, 2 * np.pi, gem_points)
    gem_r = 0.2
    gem_x = gem_r * np.sin(phi) * np.cos(theta) + R
    gem_y = gem_r * np.sin(phi) * np.sin(theta)
    gem_z = gem_r * np.cos(phi) + r + 0.1
    gem = np.stack([gem_x, gem_y, gem_z], axis=1)

    # Combine
    points = np.vstack([ring, gem]).astype(np.float32)

    # Normalize
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    points = points / scale

    # Ground truth labels
    gt_labels = np.array([1] * ring_points + [2] * gem_points)

    logger.info(f"Test points shape: {points.shape}")
    logger.info(f"Ground truth: {ring_points} metal, {gem_points} gem")

    # Run inference
    logger.info("Running inference...")
    pred_labels = predictor.predict(points)

    # Calculate metrics
    correct = (pred_labels == gt_labels).sum()
    accuracy = correct / len(gt_labels)

    metal_correct = ((pred_labels == 1) & (gt_labels == 1)).sum()
    metal_total = (gt_labels == 1).sum()
    metal_acc = metal_correct / metal_total if metal_total > 0 else 0

    gem_correct = ((pred_labels == 2) & (gt_labels == 2)).sum()
    gem_total = (gt_labels == 2).sum()
    gem_acc = gem_correct / gem_total if gem_total > 0 else 0

    logger.info("=" * 60)
    logger.info("Results:")
    logger.info(f"  Overall Accuracy: {accuracy:.4f} ({correct}/{len(gt_labels)})")
    logger.info(f"  Metal Accuracy:   {metal_acc:.4f} ({metal_correct}/{metal_total})")
    logger.info(f"  Gem Accuracy:     {gem_acc:.4f} ({gem_correct}/{gem_total})")
    logger.info("=" * 60)

    # Prediction distribution
    unique, counts = np.unique(pred_labels, return_counts=True)
    logger.info("Prediction distribution:")
    class_names = {0: "Background", 1: "Metal", 2: "Gem"}
    for cls, count in zip(unique, counts):
        pct = count / len(pred_labels) * 100
        logger.info(f"  {class_names.get(cls, f'Class {cls}')}: {count} ({pct:.1f}%)")

    return accuracy > 0.5


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test inference")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Model path")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    args = parser.parse_args()

    success = test_predictor(args.model, args.device)

    if success:
        logger.info("\n✓ Inference test PASSED")
    else:
        logger.error("\n✗ Inference test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
