"""
Point cloud augmentations for training.

Implements:
- Random rotation
- Random scaling
- Random translation
- Random jitter
- Random point dropout
"""

import numpy as np
from typing import Tuple, Optional
import torch


class PointCloudAugmentation:
    """
    Augmentation pipeline for point clouds.

    All transformations preserve labels since they operate
    on point positions only.

    Example:
        augment = PointCloudAugmentation(
            rotate=True,
            scale=True,
            translate=True,
        )
        points, labels = augment(points, labels)
    """

    def __init__(
        self,
        rotate: bool = True,
        rotate_range: Tuple[float, float] = (-180, 180),
        scale: bool = True,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translate: bool = True,
        translate_range: float = 0.1,
        jitter: bool = True,
        jitter_sigma: float = 0.01,
        jitter_clip: float = 0.05,
        dropout: bool = True,
        dropout_ratio: float = 0.1,
        flip: bool = False,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            rotate: Apply random rotation around Z axis
            rotate_range: (min, max) rotation in degrees
            scale: Apply random uniform scaling
            scale_range: (min, max) scale factors
            translate: Apply random translation
            translate_range: Max translation as fraction of bounding box
            jitter: Apply random position jitter
            jitter_sigma: Standard deviation of jitter noise
            jitter_clip: Maximum jitter displacement
            dropout: Randomly drop points
            dropout_ratio: Fraction of points to drop
            flip: Random flip along X or Y axis
        """
        self.rotate = rotate
        self.rotate_range = rotate_range
        self.scale = scale
        self.scale_range = scale_range
        self.translate = translate
        self.translate_range = translate_range
        self.jitter = jitter
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.dropout = dropout
        self.dropout_ratio = dropout_ratio
        self.flip = flip

    def __call__(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations.

        Args:
            points: (N, 3) point positions
            labels: (N,) point labels

        Returns:
            Tuple of (augmented_points, labels)
        """
        points = points.copy()

        # Random rotation around Z axis
        if self.rotate:
            points = self._rotate_z(points)

        # Random scaling
        if self.scale:
            points = self._random_scale(points)

        # Random translation
        if self.translate:
            points = self._random_translate(points)

        # Random jitter
        if self.jitter:
            points = self._jitter(points)

        # Random flip
        if self.flip:
            points = self._random_flip(points)

        # Random dropout (affects both points and labels)
        if self.dropout:
            points, labels = self._random_dropout(points, labels)

        return points.astype(np.float32), labels

    def _rotate_z(self, points: np.ndarray) -> np.ndarray:
        """Rotate around Z axis."""
        angle = np.random.uniform(*self.rotate_range)
        angle_rad = np.deg2rad(angle)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

        return points @ rotation_matrix.T

    def _random_scale(self, points: np.ndarray) -> np.ndarray:
        """Apply random uniform scaling."""
        scale = np.random.uniform(*self.scale_range)
        return points * scale

    def _random_translate(self, points: np.ndarray) -> np.ndarray:
        """Apply random translation."""
        # Compute bounding box size
        bbox_size = points.max(axis=0) - points.min(axis=0)
        max_offset = bbox_size * self.translate_range

        translation = np.random.uniform(-max_offset, max_offset)
        return points + translation

    def _jitter(self, points: np.ndarray) -> np.ndarray:
        """Apply random position jitter."""
        noise = np.random.normal(0, self.jitter_sigma, points.shape)
        noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
        return points + noise

    def _random_flip(self, points: np.ndarray) -> np.ndarray:
        """Random flip along X or Y axis."""
        if np.random.random() > 0.5:
            points[:, 0] = -points[:, 0]
        if np.random.random() > 0.5:
            points[:, 1] = -points[:, 1]
        return points

    def _random_dropout(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly drop points."""
        n = len(points)
        keep_ratio = 1.0 - self.dropout_ratio

        # Randomly select points to keep
        keep_indices = np.random.choice(
            n,
            size=int(n * keep_ratio),
            replace=False,
        )

        return points[keep_indices], labels[keep_indices]


class ComposeAugmentation:
    """Compose multiple augmentation transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            points, labels = t(points, labels)
        return points, labels


def create_training_augmentation() -> PointCloudAugmentation:
    """Create default training augmentation."""
    return PointCloudAugmentation(
        rotate=True,
        rotate_range=(-180, 180),
        scale=True,
        scale_range=(0.8, 1.2),
        translate=True,
        translate_range=0.1,
        jitter=True,
        jitter_sigma=0.01,
        jitter_clip=0.05,
        dropout=True,
        dropout_ratio=0.1,
        flip=False,
    )
