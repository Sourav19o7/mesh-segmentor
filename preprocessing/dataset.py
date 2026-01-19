"""
PyTorch Dataset for jewelry segmentation training.

Expected data format:
- Point clouds: .npy files with shape (N, 3)
- Labels: .npy files with shape (N,) containing class indices
  - 0 = background
  - 1 = metal
  - 2 = gem

Directory structure:
    data/processed/
        train/
            model_001.npy
            model_002.npy
        val/
            model_101.npy
        test/
            model_201.npy
    data/labels/
        train/
            model_001.npy
            model_002.npy
        ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Callable
from utils.logging import get_logger
from utils.s3 import S3Client, parse_s3_uri

logger = get_logger(__name__)


class JewelrySegmentationDataset(Dataset):
    """
    Dataset for point cloud semantic segmentation of jewelry.

    Loads preprocessed point clouds and their labels from disk or S3.

    Example:
        dataset = JewelrySegmentationDataset(
            data_dir="data/processed/train",
            label_dir="data/labels/train",
            num_points=20000,
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    """

    # Class names for reference
    CLASS_NAMES = ["background", "metal", "gem"]
    NUM_CLASSES = 3

    def __init__(
        self,
        data_dir: str,
        label_dir: str,
        num_points: int = 20000,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        cache_in_memory: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to point cloud files (.npy) or S3 URI
            label_dir: Path to label files (.npy) or S3 URI
            num_points: Number of points per sample (resample if different)
            transform: Optional augmentation transform
            normalize: Whether to normalize point clouds
            cache_in_memory: Cache all data in memory (faster but more RAM)
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.num_points = num_points
        self.transform = transform
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory

        # Detect S3 paths
        self.use_s3 = data_dir.startswith("s3://")

        if self.use_s3:
            self._init_s3()
        else:
            self._init_local()

        # In-memory cache
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        logger.info(
            f"Initialized dataset with {len(self)} samples, "
            f"num_points={num_points}, normalize={normalize}"
        )

    def _init_local(self):
        """Initialize from local filesystem."""
        data_path = Path(self.data_dir)
        label_path = Path(self.label_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label directory not found: {label_path}")

        # Find all .npy files in data directory
        self.data_files = sorted(data_path.glob("*.npy"))
        self.label_files = sorted(label_path.glob("*.npy"))

        # Verify matching files
        data_names = {f.stem for f in self.data_files}
        label_names = {f.stem for f in self.label_files}

        common = data_names & label_names
        if len(common) < len(data_names):
            missing = data_names - label_names
            logger.warning(f"Missing labels for: {missing}")

        # Keep only matching files
        self.data_files = [
            f for f in self.data_files if f.stem in common
        ]
        self.label_files = [
            label_path / f"{f.stem}.npy" for f in self.data_files
        ]

    def _init_s3(self):
        """Initialize from S3."""
        data_bucket, data_prefix = parse_s3_uri(self.data_dir)
        label_bucket, label_prefix = parse_s3_uri(self.label_dir)

        self.s3_client = S3Client(bucket=data_bucket)

        # List data files
        data_keys = self.s3_client.list_objects(data_prefix)
        data_keys = [k for k in data_keys if k.endswith(".npy")]

        # List label files
        label_keys = self.s3_client.list_objects(label_prefix)
        label_keys = [k for k in label_keys if k.endswith(".npy")]

        # Match by filename
        data_names = {Path(k).stem: k for k in data_keys}
        label_names = {Path(k).stem: k for k in label_keys}

        common = set(data_names.keys()) & set(label_names.keys())

        self.data_files = [data_names[n] for n in sorted(common)]
        self.label_files = [label_names[n] for n in sorted(common)]

        self.data_bucket = data_bucket
        self.label_bucket = label_bucket

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (points, labels)
            - points: (N, 3) float tensor
            - labels: (N,) long tensor
        """
        # Check cache
        if idx in self._cache:
            points, labels = self._cache[idx]
        else:
            points, labels = self._load_sample(idx)
            if self.cache_in_memory:
                self._cache[idx] = (points.copy(), labels.copy())

        # Resample if needed
        if len(points) != self.num_points:
            points, labels = self._resample(points, labels)

        # Normalize
        if self.normalize:
            points = self._normalize(points)

        # Apply transform (augmentation)
        if self.transform is not None:
            points, labels = self.transform(points, labels)

        # Convert to tensors
        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()

        return points_tensor, labels_tensor

    def _load_sample(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single sample from disk or S3."""
        if self.use_s3:
            # Download from S3
            data_bytes = self.s3_client.download_bytes(
                self.data_files[idx], bucket=self.data_bucket
            )
            label_bytes = self.s3_client.download_bytes(
                self.label_files[idx], bucket=self.label_bucket
            )

            points = np.load(
                __import__("io").BytesIO(data_bytes)
            )
            labels = np.load(
                __import__("io").BytesIO(label_bytes)
            )
        else:
            points = np.load(self.data_files[idx])
            labels = np.load(self.label_files[idx])

        return points.astype(np.float32), labels.astype(np.int64)

    def _resample(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample to target number of points."""
        n = len(points)

        if n >= self.num_points:
            # Subsample
            indices = np.random.choice(n, self.num_points, replace=False)
        else:
            # Oversample with replacement
            indices = np.random.choice(n, self.num_points, replace=True)

        return points[indices], labels[indices]

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere centered at origin."""
        center = points.mean(axis=0)
        points = points - center

        scale = np.abs(points).max()
        if scale > 0:
            points = points / scale

        return points


class JewelryDataModule:
    """
    Data module for managing train/val/test splits.

    Example:
        dm = JewelryDataModule(
            train_data="s3://bucket/train",
            val_data="s3://bucket/val",
            batch_size=8,
        )
        train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        train_data: str,
        train_labels: str,
        val_data: str,
        val_labels: str,
        test_data: Optional[str] = None,
        test_labels: Optional[str] = None,
        batch_size: int = 8,
        num_points: int = 20000,
        num_workers: int = 4,
        train_transform: Optional[Callable] = None,
    ):
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_workers = num_workers
        self.train_transform = train_transform

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        dataset = JewelrySegmentationDataset(
            data_dir=self.train_data,
            label_dir=self.train_labels,
            num_points=self.num_points,
            transform=self.train_transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        dataset = JewelrySegmentationDataset(
            data_dir=self.val_data,
            label_dir=self.val_labels,
            num_points=self.num_points,
            transform=None,  # No augmentation for validation
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader if test data provided."""
        if self.test_data is None:
            return None

        dataset = JewelrySegmentationDataset(
            data_dir=self.test_data,
            label_dir=self.test_labels,
            num_points=self.num_points,
            transform=None,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def create_label_from_mesh_name(
    mesh_name: Optional[str],
    layer_name: Optional[str],
) -> int:
    """
    Infer label from mesh name or layer name.

    Common naming conventions in jewelry CAD:
    - "Metal", "Gold", "Silver", "Platinum" → metal (1)
    - "Gem", "Stone", "Diamond", "Ruby", etc. → gem (2)
    - Others → background (0)

    Args:
        mesh_name: Name of the mesh object
        layer_name: Name of the layer

    Returns:
        Class label (0, 1, or 2)
    """
    metal_keywords = [
        "metal", "gold", "silver", "platinum", "palladium",
        "brass", "bronze", "copper", "ring", "band", "shank",
        "setting", "prong", "bezel"
    ]

    gem_keywords = [
        "gem", "stone", "diamond", "ruby", "sapphire", "emerald",
        "topaz", "amethyst", "opal", "pearl", "crystal", "cubic"
    ]

    # Check both name and layer
    text = f"{mesh_name or ''} {layer_name or ''}".lower()

    for keyword in gem_keywords:
        if keyword in text:
            return 2  # gem

    for keyword in metal_keywords:
        if keyword in text:
            return 1  # metal

    return 0  # background
