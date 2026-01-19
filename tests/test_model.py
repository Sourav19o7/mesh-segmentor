"""
Unit tests for Point Transformer model.
"""

import pytest
import torch
import numpy as np

from models.point_transformer import PointTransformer, create_point_transformer
from models.losses import SegmentationLoss, FocalLoss


class TestPointTransformer:
    """Tests for Point Transformer model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = create_point_transformer(num_classes=3, model_size="small")
        assert model is not None
        assert model.num_classes == 3

    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = create_point_transformer(num_classes=3, model_size="small")
        model.eval()

        batch_size = 2
        num_points = 1024
        x = torch.randn(batch_size, num_points, 3)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_points, 3)

    def test_predict(self):
        """Test prediction method."""
        model = create_point_transformer(num_classes=3, model_size="small")
        model.eval()

        x = torch.randn(1, 1024, 3)

        with torch.no_grad():
            labels = model.predict(x)

        assert labels.shape == (1, 1024)
        assert labels.min() >= 0
        assert labels.max() <= 2

    def test_predict_proba(self):
        """Test probability prediction."""
        model = create_point_transformer(num_classes=3, model_size="small")
        model.eval()

        x = torch.randn(1, 1024, 3)

        with torch.no_grad():
            probs = model.predict_proba(x)

        assert probs.shape == (1, 1024, 3)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 1024), atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        """Test forward pass on GPU."""
        model = create_point_transformer(num_classes=3, model_size="small")
        model = model.cuda()
        model.eval()

        x = torch.randn(2, 1024, 3).cuda()

        with torch.no_grad():
            output = model(x)

        assert output.device.type == "cuda"
        assert output.shape == (2, 1024, 3)


class TestSegmentationLoss:
    """Tests for loss functions."""

    def test_segmentation_loss(self):
        """Test combined segmentation loss."""
        loss_fn = SegmentationLoss(
            num_classes=3,
            class_weights=[0.1, 1.0, 2.0],
            ce_weight=1.0,
            dice_weight=0.5,
        )

        logits = torch.randn(2, 1024, 3)
        targets = torch.randint(0, 3, (2, 1024))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_focal_loss(self):
        """Test focal loss."""
        loss_fn = FocalLoss(num_classes=3, gamma=2.0)

        logits = torch.randn(2, 1024, 3)
        targets = torch.randint(0, 3, (2, 1024))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_with_ignore_index(self):
        """Test loss ignores specified index."""
        loss_fn = SegmentationLoss(num_classes=3, ignore_index=-100)

        logits = torch.randn(2, 100, 3)
        targets = torch.randint(0, 3, (2, 100))
        targets[0, :50] = -100  # Mark half as ignored

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestModelSizes:
    """Test different model sizes."""

    @pytest.mark.parametrize("size", ["small", "base", "large"])
    def test_model_sizes(self, size):
        """Test all model sizes can be created and run."""
        model = create_point_transformer(num_classes=3, model_size=size)
        model.eval()

        x = torch.randn(1, 512, 3)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 512, 3)
