#!/usr/bin/env python3
"""
Segment a GLB file into Metal and Gem components.

Usage:
    python scripts/segment_glb.py test.glb -o segmented.glb
    python scripts/segment_glb.py test.glb --model checkpoints/best_model.pt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import trimesh
import numpy as np
from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load mesh from GLB/OBJ/STL file."""
    logger.info(f"Loading mesh from {path}...")
    scene = trimesh.load(path)

    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError("No meshes found in file")
        elif len(meshes) == 1:
            mesh = meshes[0]
        else:
            logger.info(f"Combining {len(meshes)} meshes...")
            mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def main():
    parser = argparse.ArgumentParser(description="Segment GLB file into Metal/Gem components")
    parser.add_argument("input", type=str, help="Input GLB/OBJ/STL file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output GLB file")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "base", "large"])
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--num-points", type=int, default=20000, help="Points to sample")

    args = parser.parse_args()

    # Default output name
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_segmented.glb")

    # Load mesh
    mesh = load_mesh(args.input)

    # Import here to avoid slow startup
    import torch
    from inference.predictor import Predictor
    from inference.mesh_segmenter import MeshSegmenter
    from inference.component_splitter import ComponentSplitter
    from inference.glb_exporter import GLBExporter

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    logger.info(f"Using device: {args.device}")

    # Load model
    logger.info(f"Loading model from {args.model}...")
    predictor = Predictor(
        model_path=args.model,
        device=args.device,
        model_size=args.model_size,
    )

    # Setup pipeline
    segmenter = MeshSegmenter(predictor=predictor, num_points=args.num_points)
    splitter = ComponentSplitter()
    exporter = GLBExporter()

    # Run segmentation
    logger.info("Running segmentation...")
    face_labels = segmenter.segment(mesh)

    # Show label distribution
    unique, counts = np.unique(face_labels, return_counts=True)
    logger.info("Face label distribution:")
    class_names = {0: "Background", 1: "Metal", 2: "Gem"}
    for label, count in zip(unique, counts):
        pct = count / len(face_labels) * 100
        logger.info(f"  {class_names.get(label, f'Class {label}')}: {count} faces ({pct:.1f}%)")

    # Split into components
    logger.info("Splitting into components...")
    components = splitter.split(mesh, face_labels)

    # Export
    logger.info(f"Exporting to {args.output}...")
    exporter.export(components, args.output)

    # Summary
    print()
    print("=" * 60)
    print("Segmentation Complete!")
    print("=" * 60)
    print()
    print("Components found:")
    for c in components:
        print(f"  {c.name}: {c.face_count} faces")
    print()
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
