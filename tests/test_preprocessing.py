"""
Unit tests for preprocessing pipeline.
"""

import pytest
import numpy as np
import trimesh

from preprocessing.mesh_converter import MeshConverter, from_arrays
from preprocessing.point_sampler import PointSampler
from preprocessing.dataset import JewelrySegmentationDataset, create_label_from_mesh_name


class TestMeshConverter:
    """Tests for mesh conversion."""

    def test_from_arrays(self):
        """Test creating mesh from arrays."""
        # Create simple cube vertices
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)

        # Cube faces (triangulated)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5],  # right
        ], dtype=np.int64)

        mesh = from_arrays(vertices, faces, process=True)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_merge_meshes(self):
        """Test merging multiple meshes."""
        converter = MeshConverter()

        # Create two simple meshes
        mesh1 = trimesh.creation.box()
        mesh2 = trimesh.creation.box()
        mesh2.apply_translation([2, 0, 0])

        merged = converter.merge([mesh1, mesh2])

        # Merged should have more vertices
        assert len(merged.vertices) > len(mesh1.vertices)


class TestPointSampler:
    """Tests for point sampling."""

    def test_sample_from_mesh(self):
        """Test sampling points from mesh."""
        # Create a simple mesh
        mesh = trimesh.creation.icosphere(subdivisions=2)

        sampler = PointSampler(
            num_points=1000,
            return_normals=True,
            normalize=True,
        )

        points, normals = sampler.sample(mesh)

        assert points.shape == (1000, 3)
        assert normals.shape == (1000, 3)

        # Points should be normalized to unit sphere
        assert np.abs(points).max() <= 1.0 + 1e-6

    def test_sample_without_normals(self):
        """Test sampling without normals."""
        mesh = trimesh.creation.box()

        sampler = PointSampler(
            num_points=500,
            return_normals=False,
            normalize=False,
        )

        points = sampler.sample(mesh)

        assert points.shape == (500, 3)

    def test_sample_with_face_indices(self):
        """Test sampling with face index tracking."""
        mesh = trimesh.creation.box()

        sampler = PointSampler(num_points=500)
        points, normals, face_indices = sampler.sample_with_face_indices(mesh)

        assert points.shape == (500, 3)
        assert face_indices.shape == (500,)
        assert face_indices.max() < len(mesh.faces)


class TestLabelInference:
    """Tests for label inference from names."""

    @pytest.mark.parametrize("name,layer,expected", [
        ("Gold_Ring", None, 1),  # metal
        ("diamond_stone", None, 2),  # gem
        ("background", None, 0),  # background
        (None, "Metal Parts", 1),  # metal layer
        (None, "Gems", 2),  # gem layer
        ("prong_01", None, 1),  # metal (prong is metal)
        ("ruby_center", None, 2),  # gem
        ("unknown_part", None, 0),  # default to background
    ])
    def test_label_inference(self, name, layer, expected):
        """Test label inference from mesh/layer names."""
        label = create_label_from_mesh_name(name, layer)
        assert label == expected
