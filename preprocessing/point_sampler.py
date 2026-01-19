"""
Sample point clouds from meshes using pytorch3d.

Provides:
- Uniform surface sampling
- Area-weighted sampling
- Normal estimation at sampled points
- Point cloud normalization
"""

import torch
import numpy as np
import trimesh
from typing import Tuple, Optional, Union
from utils.logging import get_logger

logger = get_logger(__name__)

# Import pytorch3d sampling functions
try:
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    logger.warning("pytorch3d not available, falling back to trimesh sampling")


class PointSampler:
    """
    Sample point clouds from triangle meshes using pytorch3d.

    Uses pytorch3d's efficient GPU-accelerated sampling when available,
    with fallback to trimesh's CPU sampling.

    Example:
        sampler = PointSampler(num_points=20000)
        points, normals = sampler.sample(mesh)
        points_normalized = sampler.normalize(points)
    """

    def __init__(
        self,
        num_points: int = 20000,
        device: str = "cuda",
        return_normals: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize point sampler.

        Args:
            num_points: Number of points to sample (default: 20000)
            device: torch device ('cuda' or 'cpu')
            return_normals: Whether to compute normals at sampled points
            normalize: Whether to normalize point cloud to unit sphere
        """
        self.num_points = num_points
        self.device = device if torch.cuda.is_available() else "cpu"
        self.return_normals = return_normals
        self.normalize = normalize

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"

    def sample(
        self,
        mesh: trimesh.Trimesh,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Sample points from a mesh surface.

        Args:
            mesh: trimesh.Trimesh to sample from

        Returns:
            If return_normals=False: (N, 3) points
            If return_normals=True: ((N, 3) points, (N, 3) normals)
        """
        if PYTORCH3D_AVAILABLE:
            return self._sample_pytorch3d(mesh)
        else:
            return self._sample_trimesh(mesh)

    def _sample_pytorch3d(
        self,
        mesh: trimesh.Trimesh,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Sample using pytorch3d (GPU-accelerated)."""
        # Convert trimesh to pytorch3d Meshes
        vertices = torch.tensor(
            mesh.vertices, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, V, 3)

        faces = torch.tensor(
            mesh.faces, dtype=torch.int64, device=self.device
        ).unsqueeze(0)  # (1, F, 3)

        meshes = Meshes(verts=vertices, faces=faces)

        # Sample points
        if self.return_normals:
            points, normals = sample_points_from_meshes(
                meshes,
                num_samples=self.num_points,
                return_normals=True,
            )
            points = points.squeeze(0).cpu().numpy()  # (N, 3)
            normals = normals.squeeze(0).cpu().numpy()  # (N, 3)
        else:
            points = sample_points_from_meshes(
                meshes,
                num_samples=self.num_points,
                return_normals=False,
            )
            points = points.squeeze(0).cpu().numpy()
            normals = None

        # Normalize if requested
        if self.normalize:
            points, center, scale = self._normalize_points(points)
            # Store normalization params in case needed for inverse transform
            self._last_center = center
            self._last_scale = scale

        logger.debug(
            f"Sampled {len(points)} points using pytorch3d"
        )

        if self.return_normals:
            return points, normals
        return points

    def _sample_trimesh(
        self,
        mesh: trimesh.Trimesh,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Sample using trimesh (CPU fallback)."""
        # Sample points from mesh surface
        points, face_indices = trimesh.sample.sample_surface(
            mesh, self.num_points
        )

        # Get normals from face indices
        if self.return_normals:
            normals = mesh.face_normals[face_indices]
        else:
            normals = None

        # Normalize if requested
        if self.normalize:
            points, center, scale = self._normalize_points(points)
            self._last_center = center
            self._last_scale = scale

        logger.debug(
            f"Sampled {len(points)} points using trimesh"
        )

        if self.return_normals:
            return points, normals
        return points

    def _normalize_points(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Normalize point cloud to unit sphere centered at origin.

        Args:
            points: (N, 3) point cloud

        Returns:
            Tuple of (normalized_points, center, scale)
        """
        # Center at origin
        center = points.mean(axis=0)
        points_centered = points - center

        # Scale to unit sphere
        scale = np.abs(points_centered).max()
        if scale > 0:
            points_normalized = points_centered / scale
        else:
            points_normalized = points_centered
            scale = 1.0

        return points_normalized, center, scale

    def denormalize_points(
        self,
        points: np.ndarray,
        center: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Reverse normalization to get original coordinates.

        Args:
            points: Normalized points
            center: Original center (uses last sample if not provided)
            scale: Original scale (uses last sample if not provided)

        Returns:
            Points in original coordinates
        """
        if center is None:
            center = getattr(self, "_last_center", np.zeros(3))
        if scale is None:
            scale = getattr(self, "_last_scale", 1.0)

        return points * scale + center

    def sample_with_face_indices(
        self,
        mesh: trimesh.Trimesh,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample points and return corresponding face indices.

        Useful for transferring face labels to points.

        Args:
            mesh: trimesh.Trimesh to sample from

        Returns:
            Tuple of (points, normals, face_indices)
        """
        # Use trimesh for this since pytorch3d doesn't return face indices
        points, face_indices = trimesh.sample.sample_surface(
            mesh, self.num_points
        )

        normals = mesh.face_normals[face_indices]

        # Normalize if requested
        if self.normalize:
            points, center, scale = self._normalize_points(points)
            self._last_center = center
            self._last_scale = scale

        return points, normals, face_indices


def sample_points_uniform(
    mesh: trimesh.Trimesh,
    num_points: int = 20000,
    return_normals: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function for uniform point sampling.

    Args:
        mesh: Mesh to sample
        num_points: Number of points
        return_normals: Whether to return normals

    Returns:
        Points (and optionally normals)
    """
    sampler = PointSampler(
        num_points=num_points,
        return_normals=return_normals,
        normalize=True,
    )
    return sampler.sample(mesh)


def create_pytorch3d_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str = "cuda",
) -> "Meshes":
    """
    Create a pytorch3d Meshes object from numpy arrays.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        device: torch device

    Returns:
        pytorch3d Meshes object
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("pytorch3d is required for this function")

    device = device if torch.cuda.is_available() else "cpu"

    verts = torch.tensor(
        vertices, dtype=torch.float32, device=device
    ).unsqueeze(0)

    faces_tensor = torch.tensor(
        faces, dtype=torch.int64, device=device
    ).unsqueeze(0)

    return Meshes(verts=verts, faces=faces_tensor)
