"""
Evaluation metrics for point cloud segmentation.

Implements:
- Mean Intersection over Union (mIoU)
- Per-class IoU
- Overall accuracy
- Per-class accuracy
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        predictions: (N,) predicted class indices
        targets: (N,) ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        (num_classes, num_classes) confusion matrix
        conf[i, j] = count of targets==i predicted as j
    """
    # Filter ignored indices
    mask = targets != ignore_index
    predictions = predictions[mask]
    targets = targets[mask]

    # Compute confusion matrix
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=predictions.device)

    for t, p in zip(targets, predictions):
        conf[t, p] += 1

    return conf


def compute_iou_from_confusion(
    confusion_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """
    Compute IoU from confusion matrix.

    Args:
        confusion_matrix: (C, C) confusion matrix

    Returns:
        Tuple of (per_class_iou, mean_iou)
    """
    # True positives: diagonal
    tp = torch.diag(confusion_matrix).float()

    # False positives: column sum - diagonal
    fp = confusion_matrix.sum(dim=0).float() - tp

    # False negatives: row sum - diagonal
    fn = confusion_matrix.sum(dim=1).float() - tp

    # IoU = TP / (TP + FP + FN)
    denominator = tp + fp + fn
    iou = torch.where(
        denominator > 0,
        tp / denominator,
        torch.zeros_like(tp),
    )

    # Mean IoU (only over classes with samples)
    valid_classes = denominator > 0
    if valid_classes.sum() > 0:
        mean_iou = iou[valid_classes].mean().item()
    else:
        mean_iou = 0.0

    return iou, mean_iou


def compute_miou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute mean Intersection over Union.

    Args:
        predictions: (B, N) or (N,) predicted class indices
        targets: (B, N) or (N,) ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        Dictionary with 'miou' and per-class 'iou_class_X' values
    """
    # Flatten if batched
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute confusion matrix
    conf = compute_confusion_matrix(predictions, targets, num_classes, ignore_index)

    # Compute IoU
    per_class_iou, mean_iou = compute_iou_from_confusion(conf)

    # Build result dict
    result = {"miou": mean_iou}
    class_names = ["background", "metal", "gem"]

    for i in range(num_classes):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        result[f"iou_{name}"] = per_class_iou[i].item()

    return result


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute overall and per-class accuracy.

    Args:
        predictions: (B, N) or (N,) predicted class indices
        targets: (B, N) or (N,) ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        Dictionary with 'accuracy' and per-class accuracies
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Filter ignored
    mask = targets != ignore_index
    predictions = predictions[mask]
    targets = targets[mask]

    # Overall accuracy
    correct = (predictions == targets).sum().item()
    total = len(targets)
    overall_acc = correct / total if total > 0 else 0.0

    # Per-class accuracy
    result = {"accuracy": overall_acc}
    class_names = ["background", "metal", "gem"]

    for i in range(num_classes):
        class_mask = targets == i
        class_total = class_mask.sum().item()
        if class_total > 0:
            class_correct = ((predictions == i) & class_mask).sum().item()
            class_acc = class_correct / class_total
        else:
            class_acc = 0.0

        name = class_names[i] if i < len(class_names) else f"class_{i}"
        result[f"acc_{name}"] = class_acc

    return result


class MetricsAccumulator:
    """
    Accumulate metrics over multiple batches.

    Example:
        accumulator = MetricsAccumulator(num_classes=3)
        for batch in dataloader:
            predictions = model(batch)
            accumulator.update(predictions, targets)
        metrics = accumulator.compute()
    """

    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset accumulated state."""
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )
        self.total_loss = 0.0
        self.num_samples = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ):
        """
        Update with a batch of predictions.

        Args:
            predictions: (B, N) predicted class indices
            targets: (B, N) ground truth class indices
            loss: Optional loss value
        """
        # Move to CPU for accumulation
        predictions = predictions.detach().cpu().flatten()
        targets = targets.detach().cpu().flatten()

        # Update confusion matrix
        batch_conf = compute_confusion_matrix(
            predictions, targets, self.num_classes, self.ignore_index
        )
        self.confusion_matrix += batch_conf

        # Update loss
        if loss is not None:
            self.total_loss += loss
            self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary with all metrics
        """
        # IoU metrics
        per_class_iou, mean_iou = compute_iou_from_confusion(self.confusion_matrix)

        # Accuracy from confusion matrix
        correct = torch.diag(self.confusion_matrix).sum().item()
        total = self.confusion_matrix.sum().item()
        accuracy = correct / total if total > 0 else 0.0

        # Build result
        result = {
            "miou": mean_iou,
            "accuracy": accuracy,
            "class_iou": [per_class_iou[i].item() for i in range(self.num_classes)],
        }

        # Per-class metrics
        class_names = ["background", "metal", "gem"]
        class_accs = []
        for i in range(self.num_classes):
            name = class_names[i] if i < len(class_names) else f"class_{i}"
            result[f"iou_{name}"] = per_class_iou[i].item()

            # Per-class accuracy
            class_total = self.confusion_matrix[i, :].sum().item()
            if class_total > 0:
                class_correct = self.confusion_matrix[i, i].item()
                class_acc = class_correct / class_total
            else:
                class_acc = 0.0

            result[f"acc_{name}"] = class_acc
            class_accs.append(class_acc)

        result["class_accuracy"] = class_accs

        # Average loss
        if self.num_samples > 0:
            result["loss"] = self.total_loss / self.num_samples
        else:
            result["loss"] = 0.0

        return result
