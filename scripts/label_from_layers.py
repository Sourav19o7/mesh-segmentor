#!/usr/bin/env python3
"""
Label meshes using Rhino layer names.

If your .3dm files have components organized by layers like:
- "Metal", "Band", "Ring", "Setting" -> Metal (class 1)
- "Gem", "Stone", "Diamond", "Ruby" -> Gem (class 2)

This script will automatically label based on layer names.

Usage:
    python scripts/label_from_layers.py --input jewelry.3dm --output-dir data/labeled

    # Specify custom layer mappings
    python scripts/label_from_layers.py --input jewelry.3dm \
        --metal-layers "Metal,Band,Prong" \
        --gem-layers "Diamond,Ruby,Sapphire"
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import trimesh
import argparse
from typing import List, Dict, Set
from preprocessing.rhino_loader import RhinoLoader
from preprocessing.mesh_converter import MeshConverter
from preprocessing.point_sampler import PointSampler
from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


# Default layer name patterns
DEFAULT_METAL_PATTERNS = [
    "metal", "band", "ring", "setting", "prong", "shank", "bezel",
    "gold", "silver", "platinum", "chain", "clasp", "hoop"
]

DEFAULT_GEM_PATTERNS = [
    "gem", "stone", "diamond", "ruby", "sapphire", "emerald",
    "pearl", "crystal", "jewel", "cubic", "cz", "moissanite"
]


def match_layer_to_class(
    layer_name: str,
    metal_patterns: List[str],
    gem_patterns: List[str],
) -> int:
    """
    Match a layer name to a class based on patterns.

    Returns:
        1 for metal, 2 for gem, 1 as default
    """
    layer_lower = layer_name.lower()

    # Check gem patterns first (usually more specific)
    for pattern in gem_patterns:
        if pattern in layer_lower:
            return 2

    # Check metal patterns
    for pattern in metal_patterns:
        if pattern in layer_lower:
            return 1

    # Default to metal
    return 1


def process_file(
    input_path: str,
    output_dir: str,
    metal_patterns: List[str],
    gem_patterns: List[str],
    num_points: int = 20000,
):
    """Process a single .3dm file using layer-based labeling."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    logger.info(f"Loading: {input_path}")
    loader = RhinoLoader()
    converter = MeshConverter()

    geometries = loader.load(input_path)
    logger.info(f"Found {len(geometries)} geometries")

    # Analyze layers
    layer_stats: Dict[str, Dict] = {}
    for geom in geometries:
        layer = geom.layer_name or "Default"
        if layer not in layer_stats:
            layer_stats[layer] = {"count": 0, "class": match_layer_to_class(layer, metal_patterns, gem_patterns)}
        layer_stats[layer]["count"] += 1

    logger.info("\nLayer analysis:")
    for layer, stats in sorted(layer_stats.items()):
        class_name = "Metal" if stats["class"] == 1 else "Gem"
        logger.info(f"  '{layer}': {stats['count']} objects -> {class_name}")

    # Convert and label meshes
    all_meshes = []
    all_labels = []

    for geom in geometries:
        mesh = converter.convert(geom)
        if mesh is None:
            continue

        layer = geom.layer_name or "Default"
        label = layer_stats[layer]["class"]

        all_meshes.append(mesh)
        all_labels.append(np.full(len(mesh.faces), label, dtype=np.int64))

    if not all_meshes:
        logger.error("No valid meshes found")
        return

    # Combine meshes
    combined = trimesh.util.concatenate(all_meshes)
    face_labels = np.concatenate(all_labels)

    logger.info(f"\nCombined mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    # Sample points
    sampler = PointSampler(num_points=num_points)
    points, face_indices = sampler.sample_with_face_indices(combined)

    # Get point labels from face labels
    point_labels = face_labels[face_indices]

    # Normalize points
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max()
    points = points / scale

    # Save
    basename = Path(input_path).stem
    points_path = output_dir / f"{basename}_points.npy"
    labels_path = output_dir / f"{basename}_labels.npy"

    np.save(points_path, points.astype(np.float32))
    np.save(labels_path, point_labels.astype(np.int64))

    # Print statistics
    unique, counts = np.unique(point_labels, return_counts=True)
    logger.info(f"\nSaved to: {output_dir}")
    logger.info(f"  Points: {points_path}")
    logger.info(f"  Labels: {labels_path}")
    logger.info(f"\nLabel distribution:")
    class_names = {1: "Metal", 2: "Gem"}
    for cls, count in zip(unique, counts):
        pct = count / len(point_labels) * 100
        logger.info(f"  {class_names.get(cls, f'Class {cls}')}: {count} ({pct:.1f}%)")

    return points, point_labels


def main():
    parser = argparse.ArgumentParser(description="Label jewelry using Rhino layers")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .3dm file or directory")
    parser.add_argument("--output-dir", "-o", type=str, default="data/labeled", help="Output directory")
    parser.add_argument("--num-points", type=int, default=20000, help="Points per sample")
    parser.add_argument(
        "--metal-layers",
        type=str,
        default=",".join(DEFAULT_METAL_PATTERNS),
        help="Comma-separated metal layer patterns"
    )
    parser.add_argument(
        "--gem-layers",
        type=str,
        default=",".join(DEFAULT_GEM_PATTERNS),
        help="Comma-separated gem layer patterns"
    )

    args = parser.parse_args()

    metal_patterns = [p.strip().lower() for p in args.metal_layers.split(",")]
    gem_patterns = [p.strip().lower() for p in args.gem_layers.split(",")]

    input_path = Path(args.input)

    if input_path.is_dir():
        files = list(input_path.glob("*.3dm"))
        logger.info(f"Found {len(files)} .3dm files")

        for f in files:
            try:
                process_file(str(f), args.output_dir, metal_patterns, gem_patterns, args.num_points)
            except Exception as e:
                logger.error(f"Failed to process {f}: {e}")
    else:
        process_file(str(input_path), args.output_dir, metal_patterns, gem_patterns, args.num_points)


if __name__ == "__main__":
    main()
