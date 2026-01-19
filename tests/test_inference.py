"""
Unit tests for inference pipeline.
"""

import pytest
import numpy as np
import trimesh

from inference.component_splitter import ComponentSplitter, SegmentedComponent
from inference.glb_exporter import GLBExporter


class TestComponentSplitter:
    """Tests for component splitting."""

    def test_split_simple_mesh(self):
        """Test splitting a simple labeled mesh."""
        # Create a mesh with two separate boxes
        box1 = trimesh.creation.box()
        box2 = trimesh.creation.box()
        box2.apply_translation([3, 0, 0])

        # Merge them
        mesh = trimesh.util.concatenate([box1, box2])

        # Create labels: first box = metal (1), second = gem (2)
        face_labels = np.zeros(len(mesh.faces), dtype=np.int32)
        face_labels[:len(box1.faces)] = 1  # metal
        face_labels[len(box1.faces):] = 2  # gem

        splitter = ComponentSplitter(min_volume_ratio=0.0)
        components = splitter.split(mesh, face_labels)

        assert len(components) == 2

        # Check names
        names = {c.name for c in components}
        assert "Metal_01" in names
        assert "Gem_01" in names

    def test_volume_ordering(self):
        """Test that components are ordered by volume."""
        # Create two boxes of different sizes
        small_box = trimesh.creation.box(extents=[1, 1, 1])
        large_box = trimesh.creation.box(extents=[2, 2, 2])
        large_box.apply_translation([5, 0, 0])

        mesh = trimesh.util.concatenate([small_box, large_box])

        # Both are metal
        face_labels = np.ones(len(mesh.faces), dtype=np.int32)

        splitter = ComponentSplitter(min_volume_ratio=0.0)
        components = splitter.split(mesh, face_labels)

        # Should have 2 components, largest first
        assert len(components) == 2
        assert components[0].name == "Metal_01"
        assert components[1].name == "Metal_02"
        assert components[0].volume > components[1].volume

    def test_min_volume_filter(self):
        """Test that small components are filtered out."""
        # Create one large and one tiny box
        large_box = trimesh.creation.box(extents=[10, 10, 10])
        tiny_box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        tiny_box.apply_translation([20, 0, 0])

        mesh = trimesh.util.concatenate([large_box, tiny_box])

        # Both are metal
        face_labels = np.ones(len(mesh.faces), dtype=np.int32)

        # Filter components < 1% of total volume
        splitter = ComponentSplitter(min_volume_ratio=0.01)
        components = splitter.split(mesh, face_labels)

        # Tiny box should be filtered
        assert len(components) == 1
        assert components[0].name == "Metal_01"


class TestGLBExporter:
    """Tests for GLB export."""

    def test_export_single_component(self):
        """Test exporting a single component."""
        box = trimesh.creation.box()

        component = SegmentedComponent(
            name="Metal_01",
            class_id=1,
            class_name="metal",
            mesh=box,
            volume=1.0,
            face_count=12,
            original_face_indices=np.arange(12),
        )

        exporter = GLBExporter(include_materials=True)
        glb_bytes = exporter.export([component])

        assert len(glb_bytes) > 0
        # GLB files start with "glTF"
        assert glb_bytes[:4] == b"glTF"

    def test_export_multiple_components(self):
        """Test exporting multiple components."""
        box1 = trimesh.creation.box()
        box2 = trimesh.creation.icosphere()

        components = [
            SegmentedComponent(
                name="Metal_01",
                class_id=1,
                class_name="metal",
                mesh=box1,
                volume=1.0,
                face_count=12,
                original_face_indices=np.arange(12),
            ),
            SegmentedComponent(
                name="Gem_01",
                class_id=2,
                class_name="gem",
                mesh=box2,
                volume=0.5,
                face_count=80,
                original_face_indices=np.arange(80),
            ),
        ]

        exporter = GLBExporter(include_materials=True)
        glb_bytes = exporter.export(components)

        assert len(glb_bytes) > 0

    def test_export_to_file(self, tmp_path):
        """Test exporting to file."""
        box = trimesh.creation.box()

        component = SegmentedComponent(
            name="Metal_01",
            class_id=1,
            class_name="metal",
            mesh=box,
            volume=1.0,
            face_count=12,
            original_face_indices=np.arange(12),
        )

        output_path = tmp_path / "test.glb"

        exporter = GLBExporter()
        glb_bytes = exporter.export([component], output_path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size == len(glb_bytes)


class TestNamingConvention:
    """Tests for naming convention."""

    def test_naming_format(self):
        """Test component naming format."""
        splitter = ComponentSplitter(zero_pad=2)

        # Verify prefix mapping
        assert splitter.CLASS_PREFIXES[1] == "Metal"
        assert splitter.CLASS_PREFIXES[2] == "Gem"

    def test_multiple_components_per_class(self):
        """Test naming with multiple components of same class."""
        # Create 3 separate metal boxes
        boxes = []
        for i in range(3):
            box = trimesh.creation.box(extents=[1 + i * 0.5, 1, 1])
            box.apply_translation([i * 5, 0, 0])
            boxes.append(box)

        mesh = trimesh.util.concatenate(boxes)
        face_labels = np.ones(len(mesh.faces), dtype=np.int32)

        splitter = ComponentSplitter(min_volume_ratio=0.0)
        components = splitter.split(mesh, face_labels)

        # Should have 3 metal components
        assert len(components) == 3

        names = [c.name for c in components]
        assert names == ["Metal_01", "Metal_02", "Metal_03"]

        # Verify ordering by volume (largest first)
        volumes = [c.volume for c in components]
        assert volumes == sorted(volumes, reverse=True)
