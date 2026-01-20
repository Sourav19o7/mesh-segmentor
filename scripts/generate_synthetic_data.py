#!/usr/bin/env python3
"""
Generate Synthetic Training Data

Creates synthetic point clouds that mimic jewelry structure:
- Ring body (torus) = Metal (class 1)
- Gemstones (spheres) = Gem (class 2)

This allows testing the full pipeline without downloading external datasets.

USAGE:
    python scripts/generate_synthetic_data.py

    # Custom number of samples
    python scripts/generate_synthetic_data.py --num-train 500 --num-val 100

    # Custom points per sample
    python scripts/generate_synthetic_data.py --num-points 20000
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


def sample_torus(
    n_points: int,
    R: float = 1.0,  # Major radius (ring radius)
    r: float = 0.15,  # Minor radius (band thickness)
) -> np.ndarray:
    """
    Sample points uniformly from a torus surface.

    Args:
        n_points: Number of points to sample
        R: Major radius
        r: Minor radius

    Returns:
        (n_points, 3) point cloud
    """
    # Parametric torus: sample u, v uniformly
    u = np.random.uniform(0, 2 * np.pi, n_points)
    v = np.random.uniform(0, 2 * np.pi, n_points)

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    return np.stack([x, y, z], axis=1)


def sample_sphere(
    n_points: int,
    center: np.ndarray,
    radius: float = 0.2,
) -> np.ndarray:
    """
    Sample points uniformly from a sphere surface.

    Args:
        n_points: Number of points to sample
        center: (3,) center position
        radius: Sphere radius

    Returns:
        (n_points, 3) point cloud
    """
    # Use rejection sampling for uniform distribution
    points = []
    while len(points) < n_points:
        # Generate random points in cube
        p = np.random.uniform(-1, 1, (n_points * 2, 3))
        # Normalize to sphere surface
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        p = p / norms
        points.extend(p[:n_points - len(points)])

    points = np.array(points[:n_points])
    return points * radius + center


def sample_ellipsoid(
    n_points: int,
    center: np.ndarray,
    radii: np.ndarray,  # (3,) radii for x, y, z
) -> np.ndarray:
    """Sample points from ellipsoid surface."""
    # Sample unit sphere then scale
    points = []
    while len(points) < n_points:
        p = np.random.randn(n_points * 2, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        p = p / norms
        points.extend(p[:n_points - len(points)])

    points = np.array(points[:n_points])
    return points * radii + center


def generate_ring_with_gems(
    num_points: int = 20000,
    num_gems: int = None,
    gem_size_range: tuple = (0.08, 0.2),
    ring_radius: float = 1.0,
    band_thickness: float = 0.15,
) -> tuple:
    """
    Generate a synthetic ring with gemstones.

    Args:
        num_points: Total number of points
        num_gems: Number of gems (random 1-5 if None)
        gem_size_range: (min, max) gem radius
        ring_radius: Ring major radius
        band_thickness: Ring band thickness

    Returns:
        Tuple of (points, labels)
        - points: (num_points, 3)
        - labels: (num_points,) with 1=metal, 2=gem
    """
    if num_gems is None:
        num_gems = np.random.randint(1, 6)

    # Decide point distribution
    # More points for ring (typically larger surface area)
    gem_point_ratio = 0.3  # 30% of points for gems
    total_gem_points = int(num_points * gem_point_ratio)
    ring_points = num_points - total_gem_points

    # Sample ring (torus)
    ring = sample_torus(ring_points, R=ring_radius, r=band_thickness)

    # Sample gems at random positions on top of ring
    gems = []
    gem_labels = []
    points_per_gem = total_gem_points // num_gems

    for i in range(num_gems):
        # Position gem on top of ring
        angle = np.random.uniform(0, 2 * np.pi)
        if i == 0:
            # Main center stone at top
            angle = 0

        # Gem center slightly above ring surface
        gem_x = ring_radius * np.cos(angle)
        gem_y = ring_radius * np.sin(angle)
        gem_z = band_thickness + np.random.uniform(0.05, 0.15)

        center = np.array([gem_x, gem_y, gem_z])

        # Random gem size
        gem_radius = np.random.uniform(*gem_size_range)

        # First gem (center stone) is larger
        if i == 0:
            gem_radius = gem_size_range[1]

        # Sample gem points
        n_pts = points_per_gem if i < num_gems - 1 else total_gem_points - len(gem_labels)

        # Use ellipsoid for more realistic gem shape
        radii = np.array([gem_radius, gem_radius, gem_radius * 1.2])
        gem_points = sample_ellipsoid(n_pts, center, radii)

        gems.append(gem_points)
        gem_labels.extend([2] * n_pts)

    # Combine
    if gems:
        all_gems = np.vstack(gems)
        points = np.vstack([ring, all_gems])
        labels = np.array([1] * ring_points + gem_labels)
    else:
        points = ring
        labels = np.array([1] * ring_points)

    # Normalize to unit sphere
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    if scale > 0:
        points = points / scale

    return points.astype(np.float32), labels.astype(np.int64)


def generate_earring(num_points: int = 20000) -> tuple:
    """Generate synthetic earring (hoop + gems)."""
    num_gems = np.random.randint(0, 4)

    # Smaller torus for earring hoop
    ring_points = num_points - (num_points // 4) * num_gems
    ring = sample_torus(ring_points, R=0.8, r=0.08)

    gems = []
    gem_labels = []

    for i in range(num_gems):
        angle = (i / num_gems) * np.pi  # Bottom half of hoop
        gem_x = 0.8 * np.cos(angle)
        gem_y = 0.8 * np.sin(angle)
        gem_z = 0

        center = np.array([gem_x, gem_y, gem_z])
        gem_radius = np.random.uniform(0.1, 0.15)

        n_pts = num_points // 4
        gem_points = sample_sphere(n_pts, center, gem_radius)
        gems.append(gem_points)
        gem_labels.extend([2] * n_pts)

    if gems:
        all_gems = np.vstack(gems)
        points = np.vstack([ring, all_gems])
        labels = np.array([1] * ring_points + gem_labels)
    else:
        points = ring
        labels = np.array([1] * ring_points)

    # Normalize
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    points = points / scale

    return points.astype(np.float32), labels.astype(np.int64)


def generate_pendant(num_points: int = 20000) -> tuple:
    """Generate synthetic pendant (chain + gem)."""
    # Chain points (cylinder-like)
    chain_points = num_points // 3

    # Simple chain as elongated points
    t = np.random.uniform(0, 1, chain_points)
    chain_x = np.random.normal(0, 0.02, chain_points)
    chain_y = np.random.normal(0, 0.02, chain_points)
    chain_z = t * 2 - 1  # -1 to 1

    chain = np.stack([chain_x, chain_y, chain_z], axis=1)

    # Large center gem
    gem_points = num_points - chain_points
    gem_center = np.array([0, 0, -0.5])
    gem = sample_ellipsoid(gem_points, gem_center, np.array([0.3, 0.3, 0.4]))

    points = np.vstack([chain, gem])
    labels = np.array([1] * chain_points + [2] * gem_points)

    # Normalize
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    points = points / scale

    return points.astype(np.float32), labels.astype(np.int64)


def generate_dataset(
    output_points_dir: Path,
    output_labels_dir: Path,
    num_train: int = 500,
    num_val: int = 100,
    num_test: int = 100,
    num_points: int = 20000,
) -> dict:
    """
    Generate complete synthetic dataset.

    Args:
        output_points_dir: Output directory for points
        output_labels_dir: Output directory for labels
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
        num_points: Points per sample

    Returns:
        Statistics dictionary
    """
    generators = [
        ("ring", generate_ring_with_gems),
        ("earring", generate_earring),
        ("pendant", generate_pendant),
    ]

    stats = {
        "total_samples": 0,
        "by_split": {},
        "label_distribution": {1: 0, 2: 0},
    }

    for split_name, num_samples in [("train", num_train), ("val", num_val), ("test", num_test)]:
        if num_samples == 0:
            continue

        split_points_dir = output_points_dir / split_name
        split_labels_dir = output_labels_dir / split_name
        split_points_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {num_samples} {split_name} samples...")

        for i in range(num_samples):
            # Randomly choose generator
            gen_name, generator = generators[np.random.randint(len(generators))]

            # Generate sample
            points, labels = generator(num_points)

            # Save
            sample_id = f"{gen_name}_{i:05d}"
            np.save(split_points_dir / f"{sample_id}.npy", points)
            np.save(split_labels_dir / f"{sample_id}.npy", labels)

            # Update stats
            for label in [1, 2]:
                stats["label_distribution"][label] += (labels == label).sum()

            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples}")

        stats["by_split"][split_name] = num_samples
        stats["total_samples"] += num_samples

    return stats


def print_stats(stats: dict):
    """Print generation statistics."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)

    print(f"\nTotal samples: {stats['total_samples']}")

    print("\nBy split:")
    for split, count in stats["by_split"].items():
        print(f"  {split}: {count}")

    print("\nLabel distribution (points):")
    total = sum(stats["label_distribution"].values())
    for label, count in stats["label_distribution"].items():
        pct = count / total * 100 if total > 0 else 0
        name = {1: "Metal", 2: "Gem"}[label]
        print(f"  {name} ({label}): {count:,} ({pct:.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic jewelry point cloud data"
    )

    parser.add_argument(
        "--output-points",
        type=str,
        default="data/processed",
        help="Output directory for point clouds",
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        default="data/labels",
        help="Output directory for labels",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=500,
        help="Number of training samples (default: 500)",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=100,
        help="Number of validation samples (default: 100)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=100,
        help="Number of test samples (default: 100)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=20000,
        help="Points per sample (default: 20000)",
    )

    args = parser.parse_args()

    stats = generate_dataset(
        output_points_dir=Path(args.output_points),
        output_labels_dir=Path(args.output_labels),
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        num_points=args.num_points,
    )

    print_stats(stats)

    print("\nNext steps:")
    print("  1. Validate: python scripts/validate_dataset.py --splits")
    print("  2. Visualize: python scripts/visualize_sample.py --random")
    print("  3. Train: python -m training.train --epochs 50")


if __name__ == "__main__":
    main()
