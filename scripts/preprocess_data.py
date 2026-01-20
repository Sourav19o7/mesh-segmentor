#!/usr/bin/env python3
"""
Data Preprocessing Script for Mesh Segmentor

Converts .3dm files into training-ready point clouds with labels.

LABELING STRATEGIES:
1. BY_NAME: Infer labels from object/layer names (default)
2. BY_LAYER: Use layer names exclusively
3. BY_MATERIAL: Use material assignments
4. INTERACTIVE: Prompt user for each object

EXPECTED INPUT STRUCTURE:
    data/raw/
        ├── train/
        │   ├── ring_001.3dm
        │   ├── ring_002.3dm
        │   └── ...
        └── val/
            ├── ring_101.3dm
            └── ...

OUTPUT STRUCTURE:
    data/processed/
        ├── train/
        │   ├── ring_001.npy  # (20000, 3) points
        │   └── ...
        └── val/
            └── ...
    data/labels/
        ├── train/
        │   ├── ring_001.npy  # (20000,) labels
        │   └── ...
        └── val/
            └── ...

USAGE:
    # Process all files in data/raw/
    python scripts/preprocess_data.py

    # Process specific directory
    python scripts/preprocess_data.py --input data/raw/train --output-dir data/processed/train

    # Interactive labeling mode
    python scripts/preprocess_data.py --labeling interactive

    # Specify custom label mapping file
    python scripts/preprocess_data.py --label-map label_mapping.yaml
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.rhino_loader import RhinoLoader, ExtractedMesh, merge_meshes
from preprocessing.mesh_converter import MeshConverter
from preprocessing.point_sampler import PointSampler
from utils.logging import setup_logging, get_logger

setup_logging(level="INFO", format_type="text")
logger = get_logger(__name__)


# ============================================================================
# LABEL MAPPING CONFIGURATION
# ============================================================================

# Keywords that indicate METAL (class 1)
METAL_KEYWORDS = [
    # Metals
    "metal", "gold", "silver", "platinum", "palladium", "rhodium",
    "brass", "bronze", "copper", "titanium", "steel", "alloy",
    # Jewelry parts (typically metal)
    "ring", "band", "shank", "setting", "prong", "bezel", "basket",
    "bail", "clasp", "chain", "hoop", "post", "back", "mount",
    "gallery", "head", "crown", "collet", "frame", "body",
]

# Keywords that indicate GEM (class 2)
GEM_KEYWORDS = [
    # Gemstones
    "gem", "stone", "diamond", "ruby", "sapphire", "emerald",
    "topaz", "amethyst", "opal", "pearl", "garnet", "peridot",
    "aquamarine", "tourmaline", "tanzanite", "citrine", "onyx",
    "turquoise", "jade", "alexandrite", "spinel", "zircon",
    # Generic
    "crystal", "cubic", "cz", "moissanite", "brilliant",
    # Parts
    "center", "accent", "side", "halo", "pave",
]

# Default class for unrecognized objects
DEFAULT_CLASS = 1  # Default to metal (most common in jewelry)


@dataclass
class LabeledMesh:
    """Mesh with assigned label."""
    mesh: ExtractedMesh
    label: int
    label_source: str  # How the label was determined


class LabelMapper:
    """
    Map mesh objects to class labels.

    Supports multiple labeling strategies:
    - by_name: Use object and layer names
    - by_layer: Use layer names only
    - by_material: Use material names
    - from_file: Load explicit mapping from YAML/JSON
    """

    def __init__(
        self,
        strategy: str = "by_name",
        label_map_file: Optional[str] = None,
        default_class: int = DEFAULT_CLASS,
    ):
        self.strategy = strategy
        self.default_class = default_class
        self.explicit_map: Dict[str, int] = {}

        if label_map_file and Path(label_map_file).exists():
            self._load_label_map(label_map_file)

    def _load_label_map(self, filepath: str):
        """Load explicit label mapping from file."""
        path = Path(filepath)

        if path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported label map format: {path.suffix}")

        # Expected format:
        # metal: ["ring_body", "prong_*", "Layer::Metal"]
        # gem: ["diamond_*", "Layer::Gems"]

        for class_name, patterns in data.items():
            class_id = {"background": 0, "metal": 1, "gem": 2}.get(class_name.lower())
            if class_id is None:
                continue

            for pattern in patterns:
                self.explicit_map[pattern.lower()] = class_id

        logger.info(f"Loaded {len(self.explicit_map)} label mappings from {filepath}")

    def get_label(
        self,
        mesh: ExtractedMesh,
        interactive: bool = False,
    ) -> Tuple[int, str]:
        """
        Determine class label for a mesh.

        Args:
            mesh: Extracted mesh object
            interactive: Prompt user if label is uncertain

        Returns:
            Tuple of (label, source_description)
        """
        name = (mesh.name or "").lower()
        layer = (mesh.layer_name or "").lower()
        combined = f"{name} {layer}"

        # Check explicit mapping first
        for pattern, label in self.explicit_map.items():
            if self._matches_pattern(pattern, name, layer):
                return label, f"explicit:{pattern}"

        # Strategy-based labeling
        if self.strategy == "by_name":
            return self._label_by_keywords(combined, name, layer)
        elif self.strategy == "by_layer":
            return self._label_by_keywords(layer, "", layer)
        elif self.strategy == "interactive" or interactive:
            return self._interactive_label(mesh)
        else:
            return self._label_by_keywords(combined, name, layer)

    def _matches_pattern(self, pattern: str, name: str, layer: str) -> bool:
        """Check if name/layer matches a pattern (supports wildcards)."""
        import fnmatch

        if pattern.startswith("layer::"):
            return fnmatch.fnmatch(layer, pattern[7:])

        return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(layer, pattern)

    def _label_by_keywords(
        self,
        text: str,
        name: str,
        layer: str,
    ) -> Tuple[int, str]:
        """Determine label by keyword matching."""
        # Check for gem keywords first (gems are usually more specifically named)
        for keyword in GEM_KEYWORDS:
            if keyword in text:
                return 2, f"keyword:{keyword}"

        # Check for metal keywords
        for keyword in METAL_KEYWORDS:
            if keyword in text:
                return 1, f"keyword:{keyword}"

        # Default
        return self.default_class, "default"

    def _interactive_label(self, mesh: ExtractedMesh) -> Tuple[int, str]:
        """Prompt user for label."""
        print(f"\n{'='*60}")
        print(f"Object: {mesh.name or '(unnamed)'}")
        print(f"Layer:  {mesh.layer_name or '(no layer)'}")
        print(f"Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}")
        print(f"{'='*60}")
        print("Labels: [0] Background  [1] Metal  [2] Gem  [s] Skip")

        while True:
            choice = input("Enter label: ").strip().lower()

            if choice == "s":
                return -1, "skipped"
            elif choice in ["0", "1", "2"]:
                return int(choice), "interactive"
            else:
                print("Invalid choice. Enter 0, 1, 2, or s to skip.")


class DataPreprocessor:
    """
    Main preprocessing pipeline.

    Converts .3dm files to point clouds with labels.
    """

    def __init__(
        self,
        num_points: int = 20000,
        labeling_strategy: str = "by_name",
        label_map_file: Optional[str] = None,
        min_faces: int = 10,
        normalize: bool = True,
    ):
        """
        Initialize preprocessor.

        Args:
            num_points: Points to sample per model
            labeling_strategy: How to assign labels
            label_map_file: Optional explicit label mapping
            min_faces: Minimum faces to include a mesh
            normalize: Normalize points to unit sphere
        """
        self.num_points = num_points
        self.min_faces = min_faces
        self.normalize = normalize

        self.loader = RhinoLoader(triangulate=True)
        self.converter = MeshConverter()
        self.sampler = PointSampler(
            num_points=num_points,
            return_normals=False,
            normalize=False,  # We'll normalize after labeling
        )
        self.label_mapper = LabelMapper(
            strategy=labeling_strategy,
            label_map_file=label_map_file,
        )

        # Statistics
        self.stats = defaultdict(int)

    def process_file(
        self,
        input_path: Path,
        output_points_path: Path,
        output_labels_path: Path,
        interactive: bool = False,
    ) -> bool:
        """
        Process a single .3dm file.

        Args:
            input_path: Path to .3dm file
            output_points_path: Path for output points .npy
            output_labels_path: Path for output labels .npy
            interactive: Use interactive labeling

        Returns:
            True if successful
        """
        logger.info(f"Processing: {input_path.name}")

        try:
            # Step 1: Load .3dm file
            extracted_meshes = self.loader.load(input_path)

            if len(extracted_meshes) == 0:
                logger.warning(f"No meshes found in {input_path.name}")
                self.stats["empty_files"] += 1
                return False

            # Step 2: Label each mesh
            labeled_meshes = []
            for mesh in extracted_meshes:
                if mesh.num_faces < self.min_faces:
                    self.stats["skipped_small"] += 1
                    continue

                label, source = self.label_mapper.get_label(mesh, interactive)

                if label == -1:  # Skipped
                    self.stats["skipped_interactive"] += 1
                    continue

                labeled_meshes.append(LabeledMesh(mesh, label, source))
                self.stats[f"class_{label}"] += 1

                logger.debug(
                    f"  {mesh.name or '(unnamed)'}: class={label} ({source})"
                )

            if len(labeled_meshes) == 0:
                logger.warning(f"No valid meshes after labeling in {input_path.name}")
                self.stats["no_valid_meshes"] += 1
                return False

            # Step 3: Convert to trimesh and merge
            trimeshes = []
            mesh_labels = []  # Track label for each mesh's faces

            for lm in labeled_meshes:
                tm = self.converter.convert(lm.mesh)
                if tm is not None:
                    trimeshes.append(tm)
                    mesh_labels.append((len(tm.faces), lm.label))

            if len(trimeshes) == 0:
                logger.warning(f"Failed to convert meshes in {input_path.name}")
                self.stats["conversion_failed"] += 1
                return False

            merged = self.converter.merge(trimeshes)

            # Step 4: Create face-level labels
            face_labels = np.zeros(len(merged.faces), dtype=np.int32)
            face_idx = 0
            for num_faces, label in mesh_labels:
                face_labels[face_idx:face_idx + num_faces] = label
                face_idx += num_faces

            # Step 5: Sample points with face tracking
            points, point_face_indices = self._sample_with_faces(merged)

            # Step 6: Transfer face labels to points
            point_labels = face_labels[point_face_indices]

            # Step 7: Normalize if requested
            if self.normalize:
                points = self._normalize_points(points)

            # Step 8: Save outputs
            output_points_path.parent.mkdir(parents=True, exist_ok=True)
            output_labels_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(output_points_path, points.astype(np.float32))
            np.save(output_labels_path, point_labels.astype(np.int64))

            # Log statistics
            unique, counts = np.unique(point_labels, return_counts=True)
            label_dist = {int(u): int(c) for u, c in zip(unique, counts)}
            logger.info(
                f"  Saved: {len(points)} points, labels={label_dist}"
            )

            self.stats["processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing {input_path.name}: {e}")
            self.stats["errors"] += 1
            return False

    def _sample_with_faces(
        self,
        mesh,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points and return face indices."""
        import trimesh

        points, face_indices = trimesh.sample.sample_surface(
            mesh, self.num_points
        )

        return points.astype(np.float32), face_indices

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize to unit sphere centered at origin."""
        center = points.mean(axis=0)
        points = points - center

        scale = np.abs(points).max()
        if scale > 0:
            points = points / scale

        return points

    def process_directory(
        self,
        input_dir: Path,
        output_points_dir: Path,
        output_labels_dir: Path,
        interactive: bool = False,
    ) -> Dict:
        """
        Process all .3dm files in a directory.

        Args:
            input_dir: Directory containing .3dm files
            output_points_dir: Directory for point cloud outputs
            output_labels_dir: Directory for label outputs
            interactive: Use interactive labeling

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all .3dm files
        files = list(input_dir.glob("*.3dm"))

        if len(files) == 0:
            logger.warning(f"No .3dm files found in {input_dir}")
            return self.stats

        logger.info(f"Found {len(files)} .3dm files in {input_dir}")

        for filepath in sorted(files):
            stem = filepath.stem

            output_points = Path(output_points_dir) / f"{stem}.npy"
            output_labels = Path(output_labels_dir) / f"{stem}.npy"

            self.process_file(
                filepath,
                output_points,
                output_labels,
                interactive,
            )

        return dict(self.stats)

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "=" * 60)
        print("PREPROCESSING STATISTICS")
        print("=" * 60)

        print(f"Files processed:     {self.stats.get('processed', 0)}")
        print(f"Empty files:         {self.stats.get('empty_files', 0)}")
        print(f"Conversion failed:   {self.stats.get('conversion_failed', 0)}")
        print(f"Errors:              {self.stats.get('errors', 0)}")
        print()
        print("Label distribution (meshes):")
        print(f"  Background (0):    {self.stats.get('class_0', 0)}")
        print(f"  Metal (1):         {self.stats.get('class_1', 0)}")
        print(f"  Gem (2):           {self.stats.get('class_2', 0)}")
        print("=" * 60)


def create_sample_label_map(output_path: str):
    """Create a sample label mapping file."""
    sample_map = {
        "metal": [
            "ring_body*",
            "band*",
            "shank*",
            "prong*",
            "setting*",
            "Layer::Metal",
            "Layer::Gold",
            "Layer::Silver",
        ],
        "gem": [
            "diamond*",
            "stone*",
            "gem*",
            "center_stone",
            "accent*",
            "Layer::Gems",
            "Layer::Stones",
        ],
        "background": [
            "helper*",
            "construction*",
            "reference*",
        ],
    }

    with open(output_path, "w") as f:
        yaml.dump(sample_map, f, default_flow_style=False)

    print(f"Created sample label map: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess .3dm files for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory with .3dm files (default: data/raw)",
    )
    parser.add_argument(
        "--output-points",
        type=str,
        default="data/processed",
        help="Output directory for point clouds (default: data/processed)",
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        default="data/labels",
        help="Output directory for labels (default: data/labels)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=20000,
        help="Points per model (default: 20000)",
    )
    parser.add_argument(
        "--labeling",
        type=str,
        choices=["by_name", "by_layer", "interactive"],
        default="by_name",
        help="Labeling strategy (default: by_name)",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Path to label mapping YAML/JSON file",
    )
    parser.add_argument(
        "--create-sample-map",
        type=str,
        metavar="PATH",
        help="Create a sample label mapping file and exit",
    )
    parser.add_argument(
        "--splits",
        action="store_true",
        help="Process train/val subdirectories separately",
    )

    args = parser.parse_args()

    # Create sample label map if requested
    if args.create_sample_map:
        create_sample_label_map(args.create_sample_map)
        return

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        num_points=args.num_points,
        labeling_strategy=args.labeling,
        label_map_file=args.label_map,
    )

    input_path = Path(args.input)

    if args.splits:
        # Process train/ and val/ subdirectories
        for split in ["train", "val", "test"]:
            split_input = input_path / split

            if not split_input.exists():
                continue

            logger.info(f"\nProcessing {split} split...")

            preprocessor.process_directory(
                split_input,
                Path(args.output_points) / split,
                Path(args.output_labels) / split,
                interactive=(args.labeling == "interactive"),
            )
    else:
        # Process single directory
        preprocessor.process_directory(
            input_path,
            Path(args.output_points),
            Path(args.output_labels),
            interactive=(args.labeling == "interactive"),
        )

    preprocessor.print_statistics()


if __name__ == "__main__":
    main()
