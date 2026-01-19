"""
Loss functions for point cloud segmentation.

Implements:
- Weighted CrossEntropy for class imbalance
- Dice loss for better boundary handling
- Combined loss with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SegmentationLoss(nn.Module):
    """
    Combined loss for point cloud semantic segmentation.

    Combines:
    - Weighted Cross Entropy: handles class imbalance
    - Dice Loss: improves boundary segmentation

    Args:
        num_classes: Number of classes
        class_weights: Optional class weights for CE loss
        ce_weight: Weight for cross entropy loss
        dice_weight: Weight for dice loss
        ignore_index: Label index to ignore
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_weights: Optional[List[float]] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        # Class weights for cross entropy
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            logits: (B, N, C) predicted logits
            targets: (B, N) ground truth labels

        Returns:
            Combined loss scalar
        """
        B, N, C = logits.shape

        # Reshape for loss computation
        logits_flat = logits.view(-1, C)  # (B*N, C)
        targets_flat = targets.view(-1)  # (B*N,)

        # Cross entropy loss
        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        # Dice loss
        if self.dice_weight > 0:
            dice_loss = self._dice_loss(logits, targets)
            total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        else:
            total_loss = ce_loss

        return total_loss

    def _dice_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute multi-class dice loss.

        Args:
            logits: (B, N, C) predicted logits
            targets: (B, N) ground truth labels
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice loss scalar
        """
        probs = F.softmax(logits, dim=-1)  # (B, N, C)

        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets.clamp(0, self.num_classes - 1),
            num_classes=self.num_classes,
        ).float()  # (B, N, C)

        # Compute dice per class
        dims = (0, 1)  # Reduce over batch and points

        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

        # Average over classes (optionally weight)
        if self.weight is not None:
            dice_loss = 1.0 - (dice_per_class * self.weight).sum() / self.weight.sum()
        else:
            dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling extreme class imbalance.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        num_classes: Number of classes
        alpha: Class balancing weights
        gamma: Focusing parameter (higher = more focus on hard examples)
        ignore_index: Label index to ignore
    """

    def __init__(
        self,
        num_classes: int = 3,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index

        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (B, N, C) predicted logits
            targets: (B, N) ground truth labels

        Returns:
            Focal loss scalar
        """
        B, N, C = logits.shape

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # (B, N, C)

        # Get probability of true class
        targets_clamped = targets.clamp(0, C - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes=C).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (B, N)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, C),
            targets.view(-1),
            reduction="none",
            ignore_index=self.ignore_index,
        ).view(B, N)

        # Apply focal weight and alpha
        if self.alpha is not None:
            alpha_t = self.alpha[targets_clamped]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Handle ignore_index
        mask = targets != self.ignore_index
        focal_loss = focal_loss * mask.float()

        return focal_loss.sum() / mask.float().sum()


class LovaszSoftmax(nn.Module):
    """
    Lovasz-Softmax loss for multi-class segmentation.

    Better IoU optimization than cross entropy.
    """

    def __init__(
        self,
        num_classes: int = 3,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Lovasz-Softmax loss.

        Args:
            logits: (B, N, C) predicted logits
            targets: (B, N) ground truth labels

        Returns:
            Loss scalar
        """
        probs = F.softmax(logits, dim=-1)
        return self._lovasz_softmax_flat(probs, targets)

    def _lovasz_softmax_flat(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten version of Lovasz-Softmax."""
        B, N, C = probs.shape

        losses = []
        for c in range(C):
            fg = (targets == c).float()  # (B, N)
            errors = (fg - probs[:, :, c]).abs()

            # Sort errors
            errors_sorted, indices = torch.sort(errors.view(-1), descending=True)
            fg_sorted = fg.view(-1)[indices]

            # Compute Lovasz extension
            grad = self._lovasz_grad(fg_sorted)
            loss = (errors_sorted * grad).sum()
            losses.append(loss)

        return sum(losses) / C

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of Lovasz extension."""
        n = len(gt_sorted)
        gts = gt_sorted.sum()

        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union

        if n > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]

        return jaccard
