"""
Transfer point cloud predictions to mesh faces.

Maps per-point labels to mesh faces using:
1. Sample points from mesh surface
2. Run model inference on sampled points
3. Map point labels back to mesh faces via nearest neighbor
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict
from preprocessing.point_sampler import PointSampler
from inference.predictor import Predictor
from utils.logging import get_logger

logger = get_logger(__name__)


class MeshSegmenter:
    """
    Segment mesh into labeled regions using point cloud classification.

    Pipeline:
    1. Sample points from mesh surface
    2. Track which face each point came from
    3. Predict labels for sampled points
    4. Vote on face labels based on point predictions

    Example:
        segmenter = MeshSegmenter(
            predictor=predictor,
            num_points=20000,
        )
        face_labels = segmenter.segment(mesh)
    """

    def __init__(
        self,
        predictor: Predictor,
        num_points: int = 20000,
        voting_method: str = "majority",
        min_votes: int = 1,
    ):
        """
        Initialize mesh segmenter.

        Args:
            predictor: Model predictor instance
            num_points: Number of points to sample
            voting_method: 'majority' or 'weighted' (by confidence)
            min_votes: Minimum votes needed for face label
        """
        self.predictor = predictor
        self.num_points = num_points
        self.voting_method = voting_method
        self.min_votes = min_votes

        self.sampler = PointSampler(
            num_points=num_points,
            return_normals=False,
            normalize=False,  # Predictor handles normalization
        )

    def segment(
        self,
        mesh: trimesh.Trimesh,
    ) -> np.ndarray:
        """
        Segment mesh faces into labeled regions.

        Args:
            mesh: Input mesh to segment

        Returns:
            (num_faces,) array of face labels
        """
        logger.info(
            f"Segmenting mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        # Step 1: Sample points with face indices
        points, face_indices = self._sample_with_face_indices(mesh)
        logger.debug(f"Sampled {len(points)} points")

        # Step 2: Normalize points for model
        points_normalized, center, scale = self._normalize_points(points)

        # Step 3: Predict labels
        if self.voting_method == "weighted":
            point_labels, point_probs = self.predictor.predict(
                points_normalized, return_probs=True
            )
            point_weights = point_probs.max(axis=-1)
        else:
            point_labels = self.predictor.predict(points_normalized)
            point_weights = None

        logger.debug(f"Predicted labels for {len(point_labels)} points")

        # Step 4: Transfer to face labels
        face_labels = self._vote_face_labels(
            mesh, face_indices, point_labels, point_weights
        )

        # Log label distribution
        unique, counts = np.unique(face_labels, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = self.predictor.CLASS_NAMES.get(label, f"class_{label}")
            pct = count / len(face_labels) * 100
            logger.info(f"  {class_name}: {count} faces ({pct:.1f}%)")

        return face_labels

    def _sample_with_face_indices(
        self,
        mesh: trimesh.Trimesh,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points from mesh surface with face tracking."""
        points, face_indices = trimesh.sample.sample_surface(
            mesh, self.num_points
        )
        return points.astype(np.float32), face_indices

    def _normalize_points(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Normalize point cloud to unit sphere."""
        center = points.mean(axis=0)
        points_centered = points - center

        scale = np.abs(points_centered).max()
        if scale > 0:
            points_normalized = points_centered / scale
        else:
            points_normalized = points_centered
            scale = 1.0

        return points_normalized, center, scale

    def _vote_face_labels(
        self,
        mesh: trimesh.Trimesh,
        face_indices: np.ndarray,
        point_labels: np.ndarray,
        point_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vote on face labels based on point predictions.

        Each face's label is determined by majority vote from
        all points sampled from that face.
        """
        num_faces = len(mesh.faces)
        num_classes = self.predictor.num_classes

        # Accumulate votes per face
        if self.voting_method == "weighted" and point_weights is not None:
            # Weighted voting
            vote_matrix = np.zeros((num_faces, num_classes), dtype=np.float64)
            for point_idx, face_idx in enumerate(face_indices):
                label = point_labels[point_idx]
                weight = point_weights[point_idx]
                vote_matrix[face_idx, label] += weight
        else:
            # Majority voting
            vote_matrix = np.zeros((num_faces, num_classes), dtype=np.int32)
            for point_idx, face_idx in enumerate(face_indices):
                label = point_labels[point_idx]
                vote_matrix[face_idx, label] += 1

        # Determine face labels
        face_labels = np.zeros(num_faces, dtype=np.int32)

        for face_idx in range(num_faces):
            votes = vote_matrix[face_idx]
            total_votes = votes.sum()

            if total_votes >= self.min_votes:
                face_labels[face_idx] = votes.argmax()
            else:
                # Face has no votes - use nearest neighbor from voted faces
                face_labels[face_idx] = self._nearest_neighbor_label(
                    mesh, face_idx, face_labels, vote_matrix
                )

        return face_labels

    def _nearest_neighbor_label(
        self,
        mesh: trimesh.Trimesh,
        face_idx: int,
        current_labels: np.ndarray,
        vote_matrix: np.ndarray,
    ) -> int:
        """Get label from nearest face that has votes."""
        # Find faces with votes
        voted_faces = vote_matrix.sum(axis=1) > 0

        if not voted_faces.any():
            return 0  # Default to background

        # Get face centers
        face_centers = mesh.triangles_center

        # Find nearest voted face
        query = face_centers[face_idx].reshape(1, -1)
        tree = cKDTree(face_centers[voted_faces])
        _, idx = tree.query(query, k=1)

        # Get label of nearest voted face
        voted_indices = np.where(voted_faces)[0]
        nearest_face = voted_indices[idx[0]]

        return vote_matrix[nearest_face].argmax()

    def segment_with_smoothing(
        self,
        mesh: trimesh.Trimesh,
        smooth_iterations: int = 3,
    ) -> np.ndarray:
        """
        Segment mesh with spatial smoothing of labels.

        Applies iterative neighbor-based smoothing to reduce noise.

        Args:
            mesh: Input mesh
            smooth_iterations: Number of smoothing passes

        Returns:
            Smoothed face labels
        """
        face_labels = self.segment(mesh)

        if smooth_iterations <= 0:
            return face_labels

        # Build face adjacency
        adjacency = mesh.face_adjacency

        for _ in range(smooth_iterations):
            new_labels = face_labels.copy()

            for face_idx in range(len(mesh.faces)):
                # Get adjacent faces
                adj_mask = (adjacency[:, 0] == face_idx) | (
                    adjacency[:, 1] == face_idx
                )
                adj_faces = adjacency[adj_mask].flatten()
                adj_faces = adj_faces[adj_faces != face_idx]

                if len(adj_faces) == 0:
                    continue

                # Vote from neighbors
                neighbor_labels = face_labels[adj_faces]
                unique, counts = np.unique(neighbor_labels, return_counts=True)

                # Only change if clear majority
                if counts.max() > len(adj_faces) / 2:
                    new_labels[face_idx] = unique[counts.argmax()]

            face_labels = new_labels

        return face_labels
