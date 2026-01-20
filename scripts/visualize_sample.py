#!/usr/bin/env python3
"""
Visualization Script for Preprocessed Data

Visualizes point clouds with labels to verify preprocessing quality.

USAGE:
    # Visualize a single sample
    python scripts/visualize_sample.py data/processed/train/ring_001.npy

    # Visualize with labels
    python scripts/visualize_sample.py data/processed/train/ring_001.npy --labels data/labels/train/ring_001.npy

    # Save to image instead of interactive display
    python scripts/visualize_sample.py sample.npy --output visualization.png

    # Visualize random samples from dataset
    python scripts/visualize_sample.py --random --data-dir data/processed/train --num 4
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_point_cloud(
    points: np.ndarray,
    labels: np.ndarray = None,
    title: str = "Point Cloud",
    output_path: str = None,
    point_size: float = 1.0,
):
    """
    Visualize point cloud using matplotlib.

    Args:
        points: (N, 3) point positions
        labels: (N,) optional labels for coloring
        title: Plot title
        output_path: Save to file instead of displaying
        point_size: Size of points
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Color by labels if provided
    if labels is not None:
        # Color map: background=gray, metal=gold, gem=blue
        colors = np.zeros((len(points), 4))
        colors[labels == 0] = [0.5, 0.5, 0.5, 0.3]  # Background - gray, transparent
        colors[labels == 1] = [0.83, 0.69, 0.22, 1.0]  # Metal - gold
        colors[labels == 2] = [0.15, 0.65, 0.85, 1.0]  # Gem - blue

        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=point_size,
        )

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.5, 0.5, 0.5], label=f"Background ({(labels == 0).sum():,})"),
            Patch(facecolor=[0.83, 0.69, 0.22], label=f"Metal ({(labels == 1).sum():,})"),
            Patch(facecolor=[0.15, 0.65, 0.85], label=f"Gem ({(labels == 2).sum():,})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],  # Color by Z coordinate
            cmap="viridis",
            s=point_size,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}\n{len(points):,} points")

    # Equal aspect ratio
    max_range = np.abs(points).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_multiple(
    data_dir: str,
    label_dir: str = None,
    num_samples: int = 4,
    output_path: str = None,
):
    """Visualize multiple random samples in a grid."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    data_path = Path(data_dir)
    files = list(data_path.glob("*.npy"))

    if len(files) == 0:
        print(f"No .npy files found in {data_dir}")
        return

    # Random selection
    np.random.seed(42)
    selected = np.random.choice(len(files), min(num_samples, len(files)), replace=False)
    selected_files = [files[i] for i in selected]

    # Create grid
    cols = 2
    rows = (num_samples + 1) // 2
    fig = plt.figure(figsize=(12, 6 * rows))

    for idx, points_path in enumerate(selected_files):
        points = np.load(points_path)

        # Try to load labels
        labels = None
        if label_dir:
            label_path = Path(label_dir) / points_path.name
            if label_path.exists():
                labels = np.load(label_path)

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")

        if labels is not None:
            colors = np.zeros((len(points), 4))
            colors[labels == 0] = [0.5, 0.5, 0.5, 0.3]
            colors[labels == 1] = [0.83, 0.69, 0.22, 1.0]
            colors[labels == 2] = [0.15, 0.65, 0.85, 1.0]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5)
        else:
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=points[:, 2], cmap="viridis", s=0.5
            )

        ax.set_title(points_path.stem)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        max_range = np.abs(points).max()
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_point_cloud_stats(points: np.ndarray, labels: np.ndarray = None):
    """Print statistics about a point cloud."""
    print("\n" + "=" * 50)
    print("POINT CLOUD STATISTICS")
    print("=" * 50)

    print(f"Number of points:  {len(points):,}")
    print(f"Shape:             {points.shape}")
    print(f"Data type:         {points.dtype}")

    print(f"\nCoordinate ranges:")
    print(f"  X: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
    print(f"  Y: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    print(f"  Z: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")

    print(f"\nBounding box size:")
    bbox = points.max(axis=0) - points.min(axis=0)
    print(f"  {bbox[0]:.4f} x {bbox[1]:.4f} x {bbox[2]:.4f}")

    if labels is not None:
        print(f"\nLabel distribution:")
        for label in [0, 1, 2]:
            count = (labels == label).sum()
            pct = count / len(labels) * 100
            name = {0: "Background", 1: "Metal", 2: "Gem"}[label]
            print(f"  {name:12} ({label}): {count:>7,} ({pct:5.1f}%)")

    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed point clouds"
    )

    parser.add_argument(
        "points_file",
        type=str,
        nargs="?",
        help="Path to points .npy file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels .npy file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save visualization to file instead of displaying",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Visualize random samples from dataset",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/train",
        help="Data directory for random sampling",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=None,
        help="Label directory for random sampling",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help="Number of random samples to visualize",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, no visualization",
    )

    args = parser.parse_args()

    if args.random:
        # Visualize random samples
        label_dir = args.label_dir
        if label_dir is None:
            # Try to infer label directory
            label_dir = args.data_dir.replace("processed", "labels")

        visualize_multiple(
            args.data_dir,
            label_dir,
            args.num,
            args.output,
        )
    elif args.points_file:
        # Visualize single file
        points = np.load(args.points_file)

        labels = None
        if args.labels:
            labels = np.load(args.labels)
        else:
            # Try to find labels automatically
            points_path = Path(args.points_file)
            label_path = points_path.parent.parent.parent / "labels" / points_path.parent.name / points_path.name
            if label_path.exists():
                labels = np.load(label_path)
                print(f"Auto-loaded labels from: {label_path}")

        print_point_cloud_stats(points, labels)

        if not args.stats_only:
            visualize_point_cloud(
                points,
                labels,
                title=Path(args.points_file).stem,
                output_path=args.output,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
