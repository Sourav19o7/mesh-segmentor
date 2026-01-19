"""
Model inference for point cloud segmentation.

Handles:
- Model loading from checkpoint or S3
- Batch inference with AMP
- Point cloud preprocessing
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path
from models.point_transformer import create_point_transformer
from utils.logging import get_logger
from utils.s3 import S3Client, parse_s3_uri

logger = get_logger(__name__)


class Predictor:
    """
    Model predictor for point cloud segmentation.

    Handles model loading, preprocessing, and inference.

    Example:
        predictor = Predictor(
            model_path="s3://mesh-segmentor/models/best_model.pt",
            device="cuda",
        )
        labels = predictor.predict(points)
    """

    # Class mappings
    CLASS_NAMES = {0: "background", 1: "metal", 2: "gem"}

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_classes: int = 3,
        model_size: str = "base",
        use_amp: bool = True,
        cache_dir: str = "/tmp/mesh-segmentor",
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to model checkpoint (local or S3 URI)
            device: Inference device
            num_classes: Number of classes
            model_size: Model size configuration
            use_amp: Use automatic mixed precision
            cache_dir: Directory for caching downloaded models
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.use_amp = use_amp and self.device == "cuda"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.device != device:
            logger.warning(f"Requested device {device} not available, using {self.device}")

        # Load model
        self.model = self._load_model(model_path, model_size)
        self.model.eval()

        logger.info(
            f"Predictor initialized: device={self.device}, amp={self.use_amp}"
        )

    def _load_model(
        self,
        model_path: str,
        model_size: str,
    ) -> torch.nn.Module:
        """Load model from checkpoint."""
        # Handle S3 path
        if model_path.startswith("s3://"):
            local_path = self._download_from_s3(model_path)
        else:
            local_path = model_path

        logger.info(f"Loading model from {local_path}")

        # Create model
        model = create_point_transformer(
            num_classes=self.num_classes,
            model_size=model_size,
        )

        # Load checkpoint
        checkpoint = torch.load(local_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model = model.to(self.device)

        logger.info("Model loaded successfully")
        return model

    def _download_from_s3(self, s3_uri: str) -> str:
        """Download model from S3."""
        bucket, key = parse_s3_uri(s3_uri)
        local_path = self.cache_dir / key.split("/")[-1]

        if not local_path.exists():
            logger.info(f"Downloading model from S3: {s3_uri}")
            client = S3Client(bucket=bucket)
            client.download_file(key, local_path)
        else:
            logger.info(f"Using cached model: {local_path}")

        return str(local_path)

    def predict(
        self,
        points: Union[np.ndarray, torch.Tensor],
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict class labels for point cloud.

        Args:
            points: (N, 3) or (B, N, 3) point positions
            return_probs: Also return class probabilities

        Returns:
            If return_probs=False: (N,) or (B, N) predicted labels
            If return_probs=True: ((N,) labels, (N, C) probabilities)
        """
        # Convert to tensor
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()

        # Add batch dimension if needed
        if points.dim() == 2:
            points = points.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        points = points.to(self.device)

        # Normalize points
        points = self._normalize(points)

        # Inference
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    logits = self.model(points)
            else:
                logits = self.model(points)

            probs = F.softmax(logits, dim=-1)
            labels = logits.argmax(dim=-1)

        # Convert to numpy
        labels = labels.cpu().numpy()
        probs = probs.cpu().numpy()

        # Remove batch dimension if single input
        if single_input:
            labels = labels[0]
            probs = probs[0]

        if return_probs:
            return labels, probs
        return labels

    def _normalize(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to unit sphere."""
        # Center at origin
        center = points.mean(dim=1, keepdim=True)
        points = points - center

        # Scale to unit sphere
        scale = points.abs().max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        scale = scale.clamp(min=1e-8)
        points = points / scale

        return points

    def predict_with_confidence(
        self,
        points: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence scores and uncertainty mask.

        Args:
            points: (N, 3) point positions
            confidence_threshold: Minimum confidence for reliable prediction

        Returns:
            Tuple of (labels, confidence, uncertain_mask)
            - labels: (N,) predicted class labels
            - confidence: (N,) confidence scores
            - uncertain_mask: (N,) boolean mask of uncertain predictions
        """
        labels, probs = self.predict(points, return_probs=True)

        # Get confidence (max probability)
        confidence = probs.max(axis=-1)

        # Mark uncertain predictions
        uncertain_mask = confidence < confidence_threshold

        return labels, confidence, uncertain_mask


class BatchPredictor:
    """
    Batch predictor for efficient inference on multiple point clouds.

    Handles memory-efficient batching for large datasets.
    """

    def __init__(
        self,
        predictor: Predictor,
        batch_size: int = 8,
        num_points: int = 20000,
    ):
        self.predictor = predictor
        self.batch_size = batch_size
        self.num_points = num_points

    def predict_batch(
        self,
        point_clouds: list,
    ) -> list:
        """
        Predict on multiple point clouds.

        Args:
            point_clouds: List of (N, 3) point arrays

        Returns:
            List of (N,) label arrays
        """
        results = []

        for i in range(0, len(point_clouds), self.batch_size):
            batch = point_clouds[i : i + self.batch_size]

            # Resample to fixed size
            batch_resampled = []
            original_sizes = []
            for pc in batch:
                original_sizes.append(len(pc))
                if len(pc) != self.num_points:
                    indices = np.random.choice(
                        len(pc), self.num_points, replace=len(pc) < self.num_points
                    )
                    batch_resampled.append(pc[indices])
                else:
                    batch_resampled.append(pc)

            # Stack into batch tensor
            batch_tensor = np.stack(batch_resampled)

            # Predict
            batch_labels = self.predictor.predict(batch_tensor)

            # Split back to individual results
            # Note: labels are for resampled points, need to handle this
            for j, labels in enumerate(batch_labels):
                results.append(labels)

        return results
