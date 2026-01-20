#!/usr/bin/env python3
"""
Interactive Labeling Tool for Jewelry Meshes

Helps you label real jewelry .3dm files by:
1. Loading the mesh and displaying it
2. Using geometric heuristics to pre-label (gems are usually small, convex, separate)
3. Letting you correct labels interactively
4. Saving labeled data for training

Usage:
    python scripts/label_tool.py --input jewelry.3dm --output-dir data/labeled
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import trimesh
import argparse
from typing import List, Tuple, Dict
from preprocessing.rhino_loader import RhinoLoader
from preprocessing.mesh_converter import MeshConverter
from preprocessing.point_sampler import PointSampler
from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


class GeometricLabeler:
    """
    Use geometric heuristics to pre-label jewelry components.

    Heuristics for gems:
    - Small relative volume (< 10% of total)
    - High convexity (gems are usually convex shapes)
    - Separated from main body (not the largest connected component)
    - Often positioned on top (higher Z coordinate)

    Heuristics for metal:
    - Largest connected component(s)
    - Ring/band shapes (torus-like)
    - Lower convexity (more complex shapes)
    """

    def __init__(
        self,
        gem_volume_threshold: float = 0.15,  # Max 15% of total volume
        gem_convexity_threshold: float = 0.7,  # Min convexity for gems
        min_component_faces: int = 10,  # Minimum faces to consider
    ):
        self.gem_volume_threshold = gem_volume_threshold
        self.gem_convexity_threshold = gem_convexity_threshold
        self.min_component_faces = min_component_faces

    def label_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Auto-label mesh faces using geometric heuristics.

        Returns:
            (num_faces,) array with labels: 1=metal, 2=gem
        """
        labels = np.ones(len(mesh.faces), dtype=np.int64)  # Default to metal

        # Split into connected components
        components = mesh.split(only_watertight=False)

        if len(components) <= 1:
            logger.warning("Mesh has only one component - using convexity heuristic")
            return self._label_single_mesh(mesh)

        # Calculate properties for each component
        total_volume = sum(abs(c.volume) if c.is_watertight else c.area for c in components)

        component_info = []
        for i, comp in enumerate(components):
            if len(comp.faces) < self.min_component_faces:
                continue

            volume = abs(comp.volume) if comp.is_watertight else comp.area
            volume_ratio = volume / total_volume if total_volume > 0 else 0

            # Calculate convexity (volume / convex hull volume)
            try:
                convex_hull = comp.convex_hull
                convexity = volume / abs(convex_hull.volume) if convex_hull.volume > 0 else 0
            except:
                convexity = 0.5

            # Get centroid Z position (relative to mesh bounds)
            z_position = (comp.centroid[2] - mesh.bounds[0, 2]) / (mesh.bounds[1, 2] - mesh.bounds[0, 2] + 1e-8)

            component_info.append({
                'index': i,
                'component': comp,
                'volume_ratio': volume_ratio,
                'convexity': convexity,
                'z_position': z_position,
                'face_count': len(comp.faces),
            })

        # Sort by volume (largest first)
        component_info.sort(key=lambda x: x['volume_ratio'], reverse=True)

        # Label components
        # Largest component(s) = metal, small convex components = gem
        for info in component_info:
            is_gem = (
                info['volume_ratio'] < self.gem_volume_threshold and
                info['convexity'] > self.gem_convexity_threshold
            )

            # Also consider position - gems often on top
            if info['z_position'] > 0.6 and info['volume_ratio'] < 0.2:
                is_gem = True

            label = 2 if is_gem else 1

            # Find which faces in original mesh belong to this component
            # This is approximate - we match by vertex positions
            comp_vertices = set(map(tuple, np.round(info['component'].vertices, 6)))

            for face_idx, face in enumerate(mesh.faces):
                face_verts = set(map(tuple, np.round(mesh.vertices[face], 6)))
                if face_verts.issubset(comp_vertices) or len(face_verts & comp_vertices) >= 2:
                    labels[face_idx] = label

            logger.info(
                f"Component {info['index']}: "
                f"vol={info['volume_ratio']:.2%}, "
                f"convex={info['convexity']:.2f}, "
                f"z={info['z_position']:.2f} -> "
                f"{'GEM' if is_gem else 'METAL'}"
            )

        return labels

    def _label_single_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Label a single mesh using face-level heuristics."""
        labels = np.ones(len(mesh.faces), dtype=np.int64)

        # Use face normals and positions
        face_centers = mesh.triangles_center
        face_normals = mesh.face_normals

        # Normalize positions
        center = mesh.centroid
        positions = face_centers - center

        # Faces pointing upward and in upper region might be gems
        z_threshold = np.percentile(positions[:, 2], 75)
        upward_facing = face_normals[:, 2] > 0.5
        upper_region = positions[:, 2] > z_threshold

        # Simple heuristic: upper, upward-facing regions might be gem settings
        potential_gem = upward_facing & upper_region
        labels[potential_gem] = 2

        return labels


def process_file(
    input_path: str,
    output_dir: str,
    num_points: int = 20000,
    auto_label: bool = True,
):
    """Process a single .3dm file and save labeled data."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    logger.info(f"Loading: {input_path}")
    loader = RhinoLoader()
    converter = MeshConverter()

    geometries = loader.load(input_path)
    logger.info(f"Found {len(geometries)} geometries")

    # Convert to trimesh
    meshes = []
    for geom in geometries:
        mesh = converter.convert(geom)
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        logger.error("No valid meshes found")
        return

    # Combine meshes
    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    logger.info(f"Combined mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    # Auto-label using heuristics
    if auto_label:
        labeler = GeometricLabeler()
        face_labels = labeler.label_mesh(combined)
    else:
        face_labels = np.ones(len(combined.faces), dtype=np.int64)

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

    # Save visualization
    try:
        colors = np.zeros((len(points), 4), dtype=np.uint8)
        colors[point_labels == 1] = [192, 192, 192, 255]  # Silver for metal
        colors[point_labels == 2] = [0, 191, 255, 255]    # Blue for gem

        pc = trimesh.PointCloud(points, colors=colors)
        viz_path = output_dir / f"{basename}_preview.ply"
        pc.export(viz_path)
        logger.info(f"  Preview: {viz_path}")
    except Exception as e:
        logger.warning(f"Could not save preview: {e}")

    return points, point_labels


def main():
    parser = argparse.ArgumentParser(description="Label jewelry meshes for training")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .3dm file or directory")
    parser.add_argument("--output-dir", "-o", type=str, default="data/labeled", help="Output directory")
    parser.add_argument("--num-points", type=int, default=20000, help="Points per sample")
    parser.add_argument("--no-auto-label", action="store_true", help="Disable auto-labeling")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        # Process all .3dm files in directory
        files = list(input_path.glob("*.3dm"))
        logger.info(f"Found {len(files)} .3dm files")

        for f in files:
            try:
                process_file(str(f), args.output_dir, args.num_points, not args.no_auto_label)
            except Exception as e:
                logger.error(f"Failed to process {f}: {e}")
    else:
        process_file(str(input_path), args.output_dir, args.num_points, not args.no_auto_label)


if __name__ == "__main__":
    main()
