#!/usr/bin/env python3
"""
Dataset Validation Script

Validates preprocessed dataset for training readiness.

Checks:
- File integrity (points and labels match)
- Point cloud shape and normalization
- Label distribution
- Missing files
- Corrupted files

USAGE:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data-dir data/processed/train --label-dir data/labels/train
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


def validate_file_pair(
    points_path: Path,
    labels_path: Path,
    expected_points: int = 20000,
) -> dict:
    """Validate a single point cloud / label pair."""
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check files exist
    if not points_path.exists():
        result["valid"] = False
        result["errors"].append(f"Points file missing: {points_path}")
        return result

    if not labels_path.exists():
        result["valid"] = False
        result["errors"].append(f"Labels file missing: {labels_path}")
        return result

    try:
        points = np.load(points_path)
        labels = np.load(labels_path)
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to load files: {e}")
        return result

    # Check shapes
    if points.ndim != 2 or points.shape[1] != 3:
        result["valid"] = False
        result["errors"].append(f"Invalid points shape: {points.shape}, expected (N, 3)")

    if labels.ndim != 1:
        result["valid"] = False
        result["errors"].append(f"Invalid labels shape: {labels.shape}, expected (N,)")

    if points.shape[0] != labels.shape[0]:
        result["valid"] = False
        result["errors"].append(
            f"Shape mismatch: points={points.shape[0]}, labels={labels.shape[0]}"
        )

    if points.shape[0] != expected_points:
        result["warnings"].append(
            f"Unexpected point count: {points.shape[0]}, expected {expected_points}"
        )

    # Check data types
    if points.dtype != np.float32:
        result["warnings"].append(f"Points dtype: {points.dtype}, expected float32")

    if labels.dtype not in [np.int32, np.int64]:
        result["warnings"].append(f"Labels dtype: {labels.dtype}, expected int32/int64")

    # Check normalization
    point_max = np.abs(points).max()
    if point_max > 1.5:
        result["warnings"].append(f"Points may not be normalized: max={point_max:.2f}")

    # Check labels are valid
    unique_labels = np.unique(labels)
    invalid_labels = [l for l in unique_labels if l not in [0, 1, 2]]
    if invalid_labels:
        result["valid"] = False
        result["errors"].append(f"Invalid labels found: {invalid_labels}")

    # Compute statistics
    result["stats"] = {
        "num_points": len(points),
        "point_range": (float(points.min()), float(points.max())),
        "label_distribution": {
            int(l): int((labels == l).sum()) for l in [0, 1, 2]
        },
    }

    return result


def validate_dataset(
    data_dir: str,
    label_dir: str,
    expected_points: int = 20000,
) -> dict:
    """Validate entire dataset."""
    data_path = Path(data_dir)
    label_path = Path(label_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label directory not found: {label_path}")

    # Find all files
    data_files = {f.stem: f for f in data_path.glob("*.npy")}
    label_files = {f.stem: f for f in label_path.glob("*.npy")}

    all_names = set(data_files.keys()) | set(label_files.keys())

    results = {
        "total_files": len(all_names),
        "valid_files": 0,
        "invalid_files": 0,
        "missing_points": [],
        "missing_labels": [],
        "errors": [],
        "warnings": [],
        "label_totals": {0: 0, 1: 0, 2: 0},
    }

    logger.info(f"Validating {len(all_names)} files...")

    for name in sorted(all_names):
        points_path = data_files.get(name)
        labels_path = label_files.get(name)

        if points_path is None:
            results["missing_points"].append(name)
            results["invalid_files"] += 1
            continue

        if labels_path is None:
            results["missing_labels"].append(name)
            results["invalid_files"] += 1
            continue

        file_result = validate_file_pair(
            points_path, labels_path, expected_points
        )

        if file_result["valid"]:
            results["valid_files"] += 1

            # Accumulate label statistics
            for label, count in file_result["stats"]["label_distribution"].items():
                results["label_totals"][label] += count
        else:
            results["invalid_files"] += 1
            for error in file_result["errors"]:
                results["errors"].append(f"{name}: {error}")

        for warning in file_result["warnings"]:
            results["warnings"].append(f"{name}: {warning}")

    return results


def print_validation_report(results: dict):
    """Print validation report."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal files:    {results['total_files']}")
    print(f"Valid files:    {results['valid_files']}")
    print(f"Invalid files:  {results['invalid_files']}")

    if results["missing_points"]:
        print(f"\nMissing point files ({len(results['missing_points'])}):")
        for name in results["missing_points"][:5]:
            print(f"  - {name}")
        if len(results["missing_points"]) > 5:
            print(f"  ... and {len(results['missing_points']) - 5} more")

    if results["missing_labels"]:
        print(f"\nMissing label files ({len(results['missing_labels'])}):")
        for name in results["missing_labels"][:5]:
            print(f"  - {name}")
        if len(results["missing_labels"]) > 5:
            print(f"  ... and {len(results['missing_labels']) - 5} more")

    print("\n" + "-" * 70)
    print("LABEL DISTRIBUTION (across all valid files)")
    print("-" * 70)

    total_points = sum(results["label_totals"].values())
    if total_points > 0:
        for label, count in results["label_totals"].items():
            pct = count / total_points * 100
            label_name = {0: "Background", 1: "Metal", 2: "Gem"}[label]
            bar = "█" * int(pct / 2)
            print(f"  {label_name:12} ({label}): {count:>10,} ({pct:5.1f}%) {bar}")

    if results["errors"]:
        print("\n" + "-" * 70)
        print(f"ERRORS ({len(results['errors'])})")
        print("-" * 70)
        for error in results["errors"][:10]:
            print(f"  ✗ {error}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")

    if results["warnings"]:
        print("\n" + "-" * 70)
        print(f"WARNINGS ({len(results['warnings'])})")
        print("-" * 70)
        for warning in results["warnings"][:10]:
            print(f"  ⚠ {warning}")
        if len(results["warnings"]) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more warnings")

    print("\n" + "=" * 70)

    if results["invalid_files"] == 0 and len(results["errors"]) == 0:
        print("✓ DATASET VALIDATION PASSED")
    else:
        print("✗ DATASET VALIDATION FAILED")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed dataset")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing point cloud .npy files",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="data/labels",
        help="Directory containing label .npy files",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=20000,
        help="Expected number of points per file",
    )
    parser.add_argument(
        "--splits",
        action="store_true",
        help="Validate train/val/test splits separately",
    )

    args = parser.parse_args()

    if args.splits:
        for split in ["train", "val", "test"]:
            data_path = Path(args.data_dir) / split
            label_path = Path(args.label_dir) / split

            if not data_path.exists():
                continue

            print(f"\n{'#' * 70}")
            print(f"# SPLIT: {split.upper()}")
            print(f"{'#' * 70}")

            results = validate_dataset(
                str(data_path),
                str(label_path),
                args.num_points,
            )
            print_validation_report(results)
    else:
        results = validate_dataset(
            args.data_dir,
            args.label_dir,
            args.num_points,
        )
        print_validation_report(results)


if __name__ == "__main__":
    main()
