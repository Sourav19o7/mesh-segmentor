#!/usr/bin/env python3
"""
Generate More Realistic Synthetic Jewelry Data

Creates jewelry-like meshes with:
- Realistic ring bands with varying profiles
- Multiple gem settings (prong, bezel, pave)
- Various gem cuts (round, princess, oval)
- Earrings, pendants, bracelets

Usage:
    python scripts/generate_realistic_data.py --num-train 1000 --num-val 200
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import trimesh
from typing import Tuple, List
import argparse
from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


def create_ring_band(
    major_radius: float = 8.0,  # Ring size (mm)
    minor_radius: float = 1.5,  # Band thickness
    width: float = 3.0,         # Band width
    profile: str = "round",     # round, flat, comfort
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Create a ring band with realistic dimensions."""

    n_major = 64
    n_minor = 32

    u = np.linspace(0, 2 * np.pi, n_major)
    v = np.linspace(0, 2 * np.pi, n_minor)
    u, v = np.meshgrid(u, v)

    if profile == "flat":
        # Flat band profile
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        z = width * (v / (2 * np.pi) - 0.5)
    elif profile == "comfort":
        # Comfort fit (rounded inside)
        inner_r = minor_radius * 0.8
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        z = minor_radius * np.sin(v)
    else:  # round
        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        z = minor_radius * np.sin(v)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Create faces
    faces = []
    for i in range(n_minor - 1):
        for j in range(n_major - 1):
            p1 = i * n_major + j
            p2 = i * n_major + (j + 1)
            p3 = (i + 1) * n_major + (j + 1)
            p4 = (i + 1) * n_major + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    labels = np.ones(len(mesh.faces), dtype=np.int64)  # Metal

    return mesh, labels


def create_round_gem(
    radius: float = 2.0,
    height: float = 1.5,
    facets: int = 16,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Create a round brilliant cut gem."""

    # Crown (top)
    crown_vertices = [[0, 0, height]]  # Table center

    # Table facets
    for i in range(facets):
        angle = 2 * np.pi * i / facets
        r = radius * 0.6
        crown_vertices.append([r * np.cos(angle), r * np.sin(angle), height * 0.9])

    # Crown facets
    for i in range(facets):
        angle = 2 * np.pi * i / facets + np.pi / facets
        r = radius * 0.85
        crown_vertices.append([r * np.cos(angle), r * np.sin(angle), height * 0.6])

    # Girdle
    for i in range(facets):
        angle = 2 * np.pi * i / facets
        crown_vertices.append([radius * np.cos(angle), radius * np.sin(angle), 0])

    # Pavilion (bottom)
    for i in range(facets):
        angle = 2 * np.pi * i / facets + np.pi / facets
        r = radius * 0.5
        crown_vertices.append([r * np.cos(angle), r * np.sin(angle), -height * 0.5])

    # Culet (bottom point)
    crown_vertices.append([0, 0, -height * 0.8])

    vertices = np.array(crown_vertices)

    # Create faces (simplified)
    faces = []

    # Table
    for i in range(facets):
        faces.append([0, i + 1, ((i + 1) % facets) + 1])

    # Crown
    for i in range(facets):
        i1 = i + 1
        i2 = ((i + 1) % facets) + 1
        i3 = facets + 1 + i
        i4 = facets + 1 + ((i + 1) % facets)
        faces.append([i1, i2, i3])
        faces.append([i2, i4, i3])

    # Upper girdle
    for i in range(facets):
        i1 = facets + 1 + i
        i2 = facets + 1 + ((i + 1) % facets)
        i3 = 2 * facets + 1 + i
        i4 = 2 * facets + 1 + ((i + 1) % facets)
        faces.append([i1, i2, i3])
        faces.append([i2, i4, i3])

    # Lower girdle to pavilion
    for i in range(facets):
        i1 = 2 * facets + 1 + i
        i2 = 2 * facets + 1 + ((i + 1) % facets)
        i3 = 3 * facets + 1 + i
        faces.append([i1, i2, i3])

    # Pavilion to culet
    culet_idx = len(vertices) - 1
    for i in range(facets):
        i1 = 3 * facets + 1 + i
        i2 = 3 * facets + 1 + ((i + 1) % facets)
        faces.append([i1, i2, culet_idx])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    labels = np.full(len(mesh.faces), 2, dtype=np.int64)  # Gem

    return mesh, labels


def create_prong_setting(
    gem_radius: float,
    num_prongs: int = 4,
    prong_height: float = 2.0,
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Create prong setting for a gem."""

    meshes = []

    for i in range(num_prongs):
        angle = 2 * np.pi * i / num_prongs + np.pi / num_prongs

        # Prong as small cylinder
        prong = trimesh.creation.cylinder(
            radius=0.3,
            height=prong_height,
            sections=8,
        )

        # Position prong
        x = (gem_radius + 0.2) * np.cos(angle)
        y = (gem_radius + 0.2) * np.sin(angle)

        prong.apply_translation([x, y, prong_height / 2])

        meshes.append(prong)

    combined = trimesh.util.concatenate(meshes)
    labels = np.ones(len(combined.faces), dtype=np.int64)  # Metal

    return combined, labels


def create_solitaire_ring(
    ring_radius: float = 8.0,
    gem_radius: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a solitaire ring with center stone."""

    all_points = []
    all_labels = []

    # Ring band
    band, band_labels = create_ring_band(ring_radius, 1.5, 3.0, "comfort")
    band_points = band.sample(10000)
    all_points.append(band_points)
    all_labels.append(np.ones(len(band_points), dtype=np.int64))

    # Center gem
    gem, gem_labels = create_round_gem(gem_radius, gem_radius * 0.7)
    gem.apply_translation([0, 0, 2.5])  # Position on top of band
    gem_points = gem.sample(5000)
    all_points.append(gem_points)
    all_labels.append(np.full(len(gem_points), 2, dtype=np.int64))

    # Prong setting
    prongs, prong_labels = create_prong_setting(gem_radius, 6, 3.0)
    prong_points = prongs.sample(3000)
    all_points.append(prong_points)
    all_labels.append(np.ones(len(prong_points), dtype=np.int64))

    # Optional: Side stones
    if np.random.random() > 0.5:
        for side in [-1, 1]:
            side_gem, _ = create_round_gem(gem_radius * 0.4, gem_radius * 0.3)
            angle = side * np.pi / 4
            x = (ring_radius + 1) * np.cos(angle)
            y = (ring_radius + 1) * np.sin(angle)
            side_gem.apply_translation([x, y, 1.5])
            side_points = side_gem.sample(1000)
            all_points.append(side_points)
            all_labels.append(np.full(len(side_points), 2, dtype=np.int64))

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    return points, labels


def create_three_stone_ring(
    ring_radius: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a three-stone ring."""

    all_points = []
    all_labels = []

    # Ring band
    band, _ = create_ring_band(ring_radius, 1.2, 4.0, "flat")
    band_points = band.sample(8000)
    all_points.append(band_points)
    all_labels.append(np.ones(len(band_points), dtype=np.int64))

    # Center stone (larger)
    center_gem, _ = create_round_gem(2.5, 1.8)
    center_gem.apply_translation([0, 0, 2.0])
    center_points = center_gem.sample(4000)
    all_points.append(center_points)
    all_labels.append(np.full(len(center_points), 2, dtype=np.int64))

    # Side stones
    for angle in [-0.4, 0.4]:
        side_gem, _ = create_round_gem(1.8, 1.3)
        x = (ring_radius + 0.5) * np.cos(angle)
        y = (ring_radius + 0.5) * np.sin(angle)
        side_gem.apply_translation([x, y, 1.8])
        side_points = side_gem.sample(2000)
        all_points.append(side_points)
        all_labels.append(np.full(len(side_points), 2, dtype=np.int64))

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    return points, labels


def create_pave_ring(
    ring_radius: float = 8.0,
    num_stones: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a pave-set ring with many small stones."""

    all_points = []
    all_labels = []

    # Ring band
    band, _ = create_ring_band(ring_radius, 1.5, 4.0, "flat")
    band_points = band.sample(8000)
    all_points.append(band_points)
    all_labels.append(np.ones(len(band_points), dtype=np.int64))

    # Small pave stones around the band
    for i in range(num_stones):
        angle = 2 * np.pi * i / num_stones
        gem, _ = create_round_gem(0.5, 0.4)

        x = ring_radius * np.cos(angle)
        y = ring_radius * np.sin(angle)
        z = 1.8 + np.random.uniform(-0.2, 0.2)

        gem.apply_translation([x, y, z])
        gem_points = gem.sample(500)
        all_points.append(gem_points)
        all_labels.append(np.full(len(gem_points), 2, dtype=np.int64))

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    return points, labels


def create_hoop_earring() -> Tuple[np.ndarray, np.ndarray]:
    """Create a hoop earring with optional stones."""

    all_points = []
    all_labels = []

    # Hoop
    hoop = trimesh.creation.torus(major_radius=10, minor_radius=1.0)
    hoop_points = hoop.sample(8000)
    all_points.append(hoop_points)
    all_labels.append(np.ones(len(hoop_points), dtype=np.int64))

    # Optional stones on hoop
    if np.random.random() > 0.3:
        num_stones = np.random.randint(3, 8)
        for i in range(num_stones):
            angle = np.pi * i / num_stones  # Half circle
            gem, _ = create_round_gem(0.8, 0.6)
            x = 10 * np.cos(angle)
            y = 10 * np.sin(angle)
            gem.apply_translation([x, y, 0])
            gem_points = gem.sample(1000)
            all_points.append(gem_points)
            all_labels.append(np.full(len(gem_points), 2, dtype=np.int64))

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    return points, labels


def create_pendant() -> Tuple[np.ndarray, np.ndarray]:
    """Create a pendant with bail and stone."""

    all_points = []
    all_labels = []

    # Bail (loop at top)
    bail = trimesh.creation.torus(major_radius=2, minor_radius=0.5)
    bail.apply_translation([0, 0, 5])
    bail_points = bail.sample(2000)
    all_points.append(bail_points)
    all_labels.append(np.ones(len(bail_points), dtype=np.int64))

    # Setting/frame
    frame = trimesh.creation.cylinder(radius=3.5, height=1.5)
    frame.apply_translation([0, 0, 0])
    frame_points = frame.sample(3000)
    all_points.append(frame_points)
    all_labels.append(np.ones(len(frame_points), dtype=np.int64))

    # Main stone
    gem, _ = create_round_gem(3.0, 2.0)
    gem.apply_translation([0, 0, 1])
    gem_points = gem.sample(6000)
    all_points.append(gem_points)
    all_labels.append(np.full(len(gem_points), 2, dtype=np.int64))

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    return points, labels


def generate_sample(num_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random jewelry sample."""

    generators = [
        (create_solitaire_ring, 0.3),
        (create_three_stone_ring, 0.2),
        (create_pave_ring, 0.2),
        (create_hoop_earring, 0.15),
        (create_pendant, 0.15),
    ]

    # Weighted random choice
    weights = [g[1] for g in generators]
    weights = np.array(weights) / sum(weights)
    idx = np.random.choice(len(generators), p=weights)

    points, labels = generators[idx][0]()

    # Resample to target points
    if len(points) != num_points:
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
        else:
            indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]
        labels = labels[indices]

    # Normalize
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    if scale > 0:
        points = points / scale

    # Random rotation
    angle = np.random.uniform(0, 2 * np.pi)
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    points = points @ rot.T

    # Safety check: replace NaN/inf values
    if not np.isfinite(points).all():
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

    return points.astype(np.float32), labels.astype(np.int64)


def generate_dataset(
    output_points_dir: Path,
    output_labels_dir: Path,
    num_train: int,
    num_val: int,
    num_test: int,
    num_points: int,
):
    """Generate the complete dataset."""

    stats = {"total": 0, "by_split": {}}

    for split_name, num_samples in [("train", num_train), ("val", num_val), ("test", num_test)]:
        if num_samples == 0:
            continue

        split_points_dir = output_points_dir / split_name
        split_labels_dir = output_labels_dir / split_name
        split_points_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {num_samples} {split_name} samples...")

        for i in range(num_samples):
            points, labels = generate_sample(num_points)

            sample_id = f"jewelry_{i:05d}"
            np.save(split_points_dir / f"{sample_id}.npy", points)
            np.save(split_labels_dir / f"{sample_id}.npy", labels)

            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples}")

        stats["by_split"][split_name] = num_samples
        stats["total"] += num_samples

    logger.info(f"\nGeneration complete! Total: {stats['total']} samples")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate realistic synthetic jewelry data")
    parser.add_argument("--output-points", type=str, default="data/processed", help="Output points directory")
    parser.add_argument("--output-labels", type=str, default="data/labels", help="Output labels directory")
    parser.add_argument("--num-train", type=int, default=1000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=200, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=200, help="Test samples")
    parser.add_argument("--num-points", type=int, default=20000, help="Points per sample")

    args = parser.parse_args()

    generate_dataset(
        output_points_dir=Path(args.output_points),
        output_labels_dir=Path(args.output_labels),
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        num_points=args.num_points,
    )


if __name__ == "__main__":
    main()
