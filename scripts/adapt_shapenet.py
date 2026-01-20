#!/usr/bin/env python3
"""
ShapeNetPart Dataset Adapter

Converts ShapeNetPart HDF5 data to mesh-segmentor training format.

ShapeNetPart has 16 categories with 50 total part labels.
We'll map these to our 3-class format:
  - Class 0 (background): Not used (ShapeNet has no background)
  - Class 1 (metal): Structural/body parts
  - Class 2 (gem): Decorative/accent parts

This validates the training pipeline before collecting real jewelry data.

USAGE:
    # First download the dataset
    ./scripts/download_shapenet.sh

    # Then run this adapter
    python scripts/adapt_shapenet.py

    # Or specify custom paths
    python scripts/adapt_shapenet.py \
        --input data/shapenet/hdf5_data \
        --output-points data/processed \
        --output-labels data/labels

    # Use specific categories only
    python scripts/adapt_shapenet.py --categories Earphone,Ring
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


# ShapeNetPart category information
# Format: category_name -> (num_parts, part_start_idx)
SHAPENET_CATEGORIES = {
    "Airplane": (4, 0),      # Parts 0-3
    "Bag": (2, 4),           # Parts 4-5
    "Cap": (2, 6),           # Parts 6-7
    "Car": (4, 8),           # Parts 8-11
    "Chair": (4, 12),        # Parts 12-15
    "Earphone": (3, 16),     # Parts 16-18
    "Guitar": (3, 19),       # Parts 19-21
    "Knife": (2, 22),        # Parts 22-23
    "Lamp": (4, 24),         # Parts 24-27
    "Laptop": (2, 28),       # Parts 28-29
    "Motorbike": (6, 30),    # Parts 30-35
    "Mug": (2, 36),          # Parts 36-37
    "Pistol": (3, 38),       # Parts 38-40
    "Rocket": (3, 41),       # Parts 41-43
    "Skateboard": (3, 44),   # Parts 44-46
    "Table": (3, 47),        # Parts 47-49
}

# Map ShapeNet parts to our classes (1=metal/body, 2=gem/accent)
# This is a heuristic mapping - structural parts -> metal, decorative -> gem
PART_TO_CLASS = {
    # Airplane: body=metal, wings/tail/engine=accent
    0: 1, 1: 2, 2: 2, 3: 2,
    # Bag: body=metal, handle=accent
    4: 1, 5: 2,
    # Cap: body=metal, visor=accent
    6: 1, 7: 2,
    # Car: body=metal, wheels/roof/hood=mixed
    8: 1, 9: 2, 10: 1, 11: 2,
    # Chair: seat=metal, legs/back/arms=accent
    12: 1, 13: 2, 14: 2, 15: 2,
    # Earphone: body=metal, ear_pad=accent, wire=accent
    16: 1, 17: 2, 18: 2,
    # Guitar: body=metal, neck=accent, head=accent
    19: 1, 20: 2, 21: 2,
    # Knife: blade=metal, handle=accent
    22: 1, 23: 2,
    # Lamp: base=metal, shade=accent, bulb=accent, pole=metal
    24: 1, 25: 2, 26: 2, 27: 1,
    # Laptop: base=metal, screen=accent
    28: 1, 29: 2,
    # Motorbike: various parts
    30: 1, 31: 2, 32: 1, 33: 2, 34: 1, 35: 2,
    # Mug: body=metal, handle=accent
    36: 1, 37: 2,
    # Pistol: body=metal, barrel=accent, trigger=accent
    38: 1, 39: 2, 40: 2,
    # Rocket: body=metal, fins=accent, nose=accent
    41: 1, 42: 2, 43: 2,
    # Skateboard: deck=metal, wheels=accent, trucks=accent
    44: 1, 45: 2, 46: 2,
    # Table: top=metal, legs=accent, shelf=accent
    47: 1, 48: 2, 49: 2,
}


def load_hdf5_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ShapeNetPart HDF5 file.

    Args:
        filepath: Path to .h5 file

    Returns:
        Tuple of (points, labels, categories)
        - points: (N, 2048, 3) point clouds
        - labels: (N, 2048) part labels (0-49)
        - categories: (N,) category indices (0-15)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install h5py")

    with h5py.File(filepath, "r") as f:
        points = f["data"][:]  # (N, 2048, 3)
        labels = f["label"][:]  # (N, 2048)

        # Some files have 'pid' (part id), others have 'seg'
        if "pid" in f:
            part_labels = f["pid"][:]
        else:
            part_labels = f["seg"][:]  # (N, 2048)

    return points, labels.flatten(), part_labels


def map_labels_to_classes(
    part_labels: np.ndarray,
    category_idx: int,
) -> np.ndarray:
    """
    Map ShapeNet part labels to our 3-class format.

    Args:
        part_labels: (N,) ShapeNet part labels
        category_idx: Category index (0-15)

    Returns:
        (N,) class labels (1=metal, 2=gem)
    """
    classes = np.zeros_like(part_labels)

    for part_id in np.unique(part_labels):
        mask = part_labels == part_id
        # Map to our class (default to 1 if not in mapping)
        our_class = PART_TO_CLASS.get(int(part_id), 1)
        classes[mask] = our_class

    return classes


def resample_points(
    points: np.ndarray,
    labels: np.ndarray,
    target_points: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample point cloud to target number of points.

    Args:
        points: (N, 3) points
        labels: (N,) labels
        target_points: Target number of points

    Returns:
        Resampled (points, labels)
    """
    n = len(points)

    if n == target_points:
        return points, labels

    if n > target_points:
        # Subsample without replacement
        indices = np.random.choice(n, target_points, replace=False)
    else:
        # Oversample with replacement
        indices = np.random.choice(n, target_points, replace=True)

    return points[indices], labels[indices]


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere centered at origin."""
    center = points.mean(axis=0)
    points = points - center

    scale = np.abs(points).max()
    if scale > 0:
        points = points / scale

    return points


def process_shapenet(
    input_dir: Path,
    output_points_dir: Path,
    output_labels_dir: Path,
    categories: List[str] = None,
    num_points: int = 20000,
    max_samples_per_category: int = None,
) -> Dict:
    """
    Process ShapeNetPart dataset.

    Args:
        input_dir: Directory containing hdf5_data/
        output_points_dir: Output directory for points
        output_labels_dir: Output directory for labels
        categories: List of categories to include (None = all)
        num_points: Number of points per sample
        max_samples_per_category: Limit samples per category

    Returns:
        Statistics dictionary
    """
    hdf5_dir = input_dir / "hdf5_data" if (input_dir / "hdf5_data").exists() else input_dir

    if not hdf5_dir.exists():
        raise FileNotFoundError(f"HDF5 data not found at {hdf5_dir}")

    # Find all train/val/test files
    splits = {
        "train": list(hdf5_dir.glob("*train*.h5")),
        "val": list(hdf5_dir.glob("*val*.h5")),
        "test": list(hdf5_dir.glob("*test*.h5")),
    }

    # If no split files found, try generic pattern
    if not any(splits.values()):
        all_h5 = list(hdf5_dir.glob("*.h5"))
        if all_h5:
            # Split 70/15/15
            np.random.shuffle(all_h5)
            n = len(all_h5)
            splits = {
                "train": all_h5[:int(0.7*n)],
                "val": all_h5[int(0.7*n):int(0.85*n)],
                "test": all_h5[int(0.85*n):],
            }

    stats = {
        "total_samples": 0,
        "by_split": {},
        "by_category": {},
        "label_distribution": {1: 0, 2: 0},
    }

    # Category name to index mapping
    cat_names = list(SHAPENET_CATEGORIES.keys())
    cat_filter = None
    if categories:
        cat_filter = set(categories)
        logger.info(f"Filtering to categories: {cat_filter}")

    for split_name, h5_files in splits.items():
        if not h5_files:
            continue

        split_points_dir = output_points_dir / split_name
        split_labels_dir = output_labels_dir / split_name
        split_points_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        split_count = 0
        category_counts = {}

        for h5_file in sorted(h5_files):
            logger.info(f"Processing {h5_file.name}...")

            try:
                points_batch, cat_labels, part_labels = load_hdf5_file(h5_file)
            except Exception as e:
                logger.error(f"Failed to load {h5_file}: {e}")
                continue

            for i in range(len(points_batch)):
                cat_idx = int(cat_labels[i]) if len(cat_labels.shape) > 0 else int(cat_labels)
                cat_name = cat_names[cat_idx] if cat_idx < len(cat_names) else f"cat_{cat_idx}"

                # Filter categories
                if cat_filter and cat_name not in cat_filter:
                    continue

                # Limit samples per category
                if max_samples_per_category:
                    if category_counts.get(cat_name, 0) >= max_samples_per_category:
                        continue

                # Get points and part labels for this sample
                pts = points_batch[i]  # (2048, 3)
                plabels = part_labels[i]  # (2048,)

                # Map to our classes
                class_labels = map_labels_to_classes(plabels, cat_idx)

                # Resample to target points
                pts, class_labels = resample_points(pts, class_labels, num_points)

                # Normalize
                pts = normalize_points(pts)

                # Generate filename
                sample_id = f"{cat_name.lower()}_{split_count:05d}"

                # Save
                np.save(split_points_dir / f"{sample_id}.npy", pts.astype(np.float32))
                np.save(split_labels_dir / f"{sample_id}.npy", class_labels.astype(np.int64))

                # Update stats
                split_count += 1
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

                for label in [1, 2]:
                    stats["label_distribution"][label] += (class_labels == label).sum()

        stats["by_split"][split_name] = split_count
        stats["total_samples"] += split_count

        for cat, count in category_counts.items():
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + count

        logger.info(f"  {split_name}: {split_count} samples")

    return stats


def print_stats(stats: Dict):
    """Print processing statistics."""
    print("\n" + "=" * 60)
    print("SHAPENET ADAPTATION COMPLETE")
    print("=" * 60)

    print(f"\nTotal samples: {stats['total_samples']}")

    print("\nBy split:")
    for split, count in stats["by_split"].items():
        print(f"  {split}: {count}")

    print("\nBy category:")
    for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nLabel distribution (points):")
    total = sum(stats["label_distribution"].values())
    for label, count in stats["label_distribution"].items():
        pct = count / total * 100 if total > 0 else 0
        name = {1: "Metal/Body", 2: "Gem/Accent"}[label]
        print(f"  {name} ({label}): {count:,} ({pct:.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Adapt ShapeNetPart dataset for mesh-segmentor"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/shapenet",
        help="Input directory containing hdf5_data/",
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
        "--num-points",
        type=int,
        default=20000,
        help="Number of points per sample (default: 20000)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories (default: all)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Maximum samples per category",
    )

    args = parser.parse_args()

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Process dataset
    stats = process_shapenet(
        input_dir=Path(args.input),
        output_points_dir=Path(args.output_points),
        output_labels_dir=Path(args.output_labels),
        categories=categories,
        num_points=args.num_points,
        max_samples_per_category=args.max_per_category,
    )

    print_stats(stats)

    print("\nNext steps:")
    print("  1. Validate: python scripts/validate_dataset.py --splits")
    print("  2. Train:    python -m training.train --epochs 50")


if __name__ == "__main__":
    main()
