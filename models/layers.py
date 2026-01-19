"""
Custom layers for Point Transformer architecture.

Implements:
- Point Transformer attention layer with position encoding
- Transition Down (downsampling)
- Transition Up (upsampling)
- k-NN grouping utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k-nearest neighbors.

    Args:
        x: (B, N, C) point features
        k: number of neighbors

    Returns:
        (B, N, k) indices of k nearest neighbors
    """
    # Compute pairwise distances
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # (B, N, N)
    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
    distances = xx + inner + xx.transpose(2, 1)  # (B, N, N)

    # Get k smallest distances
    _, indices = distances.topk(k, dim=-1, largest=False)  # (B, N, k)

    return indices


def index_points(
    points: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Index points using given indices.

    Args:
        points: (B, N, C) input points
        idx: (B, M) or (B, M, K) indices

    Returns:
        Indexed points with shape (B, M, C) or (B, M, K, C)
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    batch_indices = batch_indices.view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]

    return new_points


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
) -> torch.Tensor:
    """
    Farthest point sampling.

    Args:
        xyz: (B, N, 3) point positions
        npoint: number of points to sample

    Returns:
        (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Random starting point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)

        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, -1)[1]

    return centroids


class PositionEncoding(nn.Module):
    """
    Position encoding for point transformer.

    Encodes relative positions between query and key points.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pos_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_diff: (B, N, k, 3) relative positions

        Returns:
            (B, N, k, C) position encodings
        """
        return self.mlp(pos_diff)


class PointTransformerLayer(nn.Module):
    """
    Point Transformer attention layer.

    Implements vector attention with position encoding:
    y_i = sum_j softmax(phi(x_i) - psi(x_j) + delta) * (alpha(x_j) + delta)

    where delta is the position encoding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        k_neighbors: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.k = k_neighbors

        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels

        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)

        # Position encoding
        self.pos_enc = PositionEncoding(out_channels)

        # Attention MLP (for computing attention weights)
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C_in) point features
            pos: (B, N, 3) point positions

        Returns:
            (B, N, C_out) output features
        """
        B, N, _ = x.shape

        # Get k-nearest neighbors
        idx = knn(pos, self.k)  # (B, N, k)

        # Project Q, K, V
        q = self.q_proj(x)  # (B, N, C)
        k = self.k_proj(x)  # (B, N, C)
        v = self.v_proj(x)  # (B, N, C)

        # Gather neighbors
        k_grouped = index_points(k, idx)  # (B, N, k, C)
        v_grouped = index_points(v, idx)  # (B, N, k, C)
        pos_grouped = index_points(pos, idx)  # (B, N, k, 3)

        # Relative positions
        pos_diff = pos_grouped - pos.unsqueeze(2)  # (B, N, k, 3)

        # Position encoding
        pos_enc = self.pos_enc(pos_diff)  # (B, N, k, C)

        # Attention: q_i - k_j + pos_enc
        q_expanded = q.unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, N, k, C)
        attn_input = q_expanded - k_grouped + pos_enc  # (B, N, k, C)

        # Compute attention weights
        attn_weights = self.attn_mlp(attn_input)  # (B, N, k, C)
        attn_weights = F.softmax(attn_weights, dim=2)  # (B, N, k, C)

        # Apply attention to values
        v_with_pos = v_grouped + pos_enc  # (B, N, k, C)
        out = (attn_weights * v_with_pos).sum(dim=2)  # (B, N, C)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class PointTransformerBlock(nn.Module):
    """
    Point Transformer block with residual connection.

    Structure:
    x → Linear → PT Layer → Linear → + → out
    |______________________________|
                (residual)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        k_neighbors: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pre-transformer linear
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Point transformer layer
        self.transformer = PointTransformerLayer(
            out_channels, out_channels, num_heads, k_neighbors, dropout
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Post-transformer linear
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # Residual projection if needed
        self.residual = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C_in) input features
            pos: (B, N, 3) point positions

        Returns:
            (B, N, C_out) output features
        """
        B, N, _ = x.shape

        # Residual
        identity = self.residual(x)

        # Forward pass
        out = self.linear1(x)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)

        out = self.transformer(out, pos)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)

        out = self.linear2(out)
        out = self.bn3(out.transpose(1, 2)).transpose(1, 2)

        # Add residual
        out = F.relu(out + identity)

        return out


class TransitionDown(nn.Module):
    """
    Downsampling module using farthest point sampling.

    Reduces the number of points while increasing feature dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        k_neighbors: int = 16,
    ):
        super().__init__()
        self.stride = stride
        self.k = k_neighbors

        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, C) input features
            pos: (B, N, 3) input positions

        Returns:
            Tuple of:
            - new_x: (B, N//stride, C_out) downsampled features
            - new_pos: (B, N//stride, 3) downsampled positions
            - idx: (B, N//stride) indices of sampled points
        """
        B, N, C = x.shape
        n_new = N // self.stride

        # Farthest point sampling
        idx = farthest_point_sample(pos, n_new)  # (B, n_new)

        # Get new positions
        new_pos = index_points(pos, idx)  # (B, n_new, 3)

        # Get k-nearest neighbors in original points
        knn_idx = knn(new_pos, self.k)  # (B, n_new, k)

        # Gather neighbor features and positions
        grouped_x = index_points(x, knn_idx)  # (B, n_new, k, C)
        grouped_pos = index_points(pos, knn_idx)  # (B, n_new, k, 3)

        # Relative positions
        rel_pos = grouped_pos - new_pos.unsqueeze(2)  # (B, n_new, k, 3)

        # Concatenate features and relative positions
        grouped_features = torch.cat([grouped_x, rel_pos], dim=-1)  # (B, n_new, k, C+3)

        # MLP and max pooling
        B, M, K, D = grouped_features.shape
        grouped_features = grouped_features.view(B * M * K, D)
        grouped_features = self.mlp[0](grouped_features)  # Linear
        grouped_features = grouped_features.view(B * M, K, -1).transpose(1, 2)
        grouped_features = self.mlp[1](grouped_features).transpose(1, 2)  # BN
        grouped_features = self.mlp[2](grouped_features.contiguous().view(B * M * K, -1))  # ReLU
        grouped_features = self.mlp[3](grouped_features)  # Linear
        grouped_features = grouped_features.view(B * M, K, -1).transpose(1, 2)
        grouped_features = self.mlp[4](grouped_features).transpose(1, 2)  # BN
        grouped_features = self.mlp[5](grouped_features.contiguous().view(B * M * K, -1))  # ReLU
        grouped_features = grouped_features.view(B, M, K, -1)

        # Max pool over neighbors
        new_x = grouped_features.max(dim=2)[0]  # (B, n_new, C_out)

        return new_x, new_pos, idx


class TransitionUp(nn.Module):
    """
    Upsampling module using interpolation.

    Increases number of points while decreasing feature dimension.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels + skip_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        skip_x: torch.Tensor,
        skip_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) coarse features
            pos: (B, N, 3) coarse positions
            skip_x: (B, M, C_skip) skip connection features (M > N)
            skip_pos: (B, M, 3) skip connection positions

        Returns:
            (B, M, C_out) upsampled features
        """
        B, N, C = x.shape
        _, M, _ = skip_pos.shape

        # Interpolate features to skip positions
        # Using 3-nearest neighbor interpolation
        dist = torch.cdist(skip_pos, pos)  # (B, M, N)
        dist, idx = dist.topk(3, dim=-1, largest=False)  # (B, M, 3)

        # Inverse distance weights
        dist = torch.clamp(dist, min=1e-10)
        weights = 1.0 / dist
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (B, M, 3)

        # Gather and interpolate
        interpolated = index_points(x, idx)  # (B, M, 3, C)
        interpolated = (interpolated * weights.unsqueeze(-1)).sum(dim=2)  # (B, M, C)

        # Concatenate with skip features
        concat = torch.cat([interpolated, skip_x], dim=-1)  # (B, M, C + C_skip)

        # MLP
        B, M, D = concat.shape
        out = self.mlp[0](concat.view(B * M, D))
        out = self.mlp[1](out.view(B, M, -1).transpose(1, 2)).transpose(1, 2)
        out = self.mlp[2](out.contiguous().view(B * M, -1))
        out = out.view(B, M, -1)

        return out
