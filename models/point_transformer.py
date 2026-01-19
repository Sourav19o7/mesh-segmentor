"""
Point Transformer for semantic segmentation of 3D point clouds.

Architecture follows the Point Transformer paper with modifications
for segmentation tasks.

Input: (B, N, 3) point positions
Output: (B, N, num_classes) per-point class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from models.layers import (
    PointTransformerBlock,
    TransitionDown,
    TransitionUp,
    index_points,
)
from utils.logging import get_logger

logger = get_logger(__name__)


class PointTransformer(nn.Module):
    """
    Point Transformer for semantic segmentation.

    U-Net style architecture with:
    - Encoder: Progressive downsampling with transformer blocks
    - Decoder: Progressive upsampling with skip connections
    - Segmentation head: MLP for per-point classification

    Args:
        in_channels: Input feature dimension (3 for xyz)
        num_classes: Number of segmentation classes
        embed_dim: Initial embedding dimension
        depths: Number of transformer blocks at each stage
        channels: Feature dimensions at each stage
        num_heads: Number of attention heads at each stage
        k_neighbors: k for k-NN in attention
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        embed_dim: int = 32,
        depths: List[int] = [2, 2, 2, 2],
        channels: List[int] = [32, 64, 128, 256],
        num_heads: List[int] = [2, 4, 8, 8],
        k_neighbors: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(depths)

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.down_transitions = nn.ModuleList()

        for i in range(self.num_stages):
            # Transformer blocks at this stage
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                blocks.append(
                    PointTransformerBlock(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        num_heads=num_heads[i],
                        k_neighbors=k_neighbors,
                        dropout=dropout,
                    )
                )
            self.encoder_blocks.append(blocks)

            # Transition down (except last stage)
            if i < self.num_stages - 1:
                self.down_transitions.append(
                    TransitionDown(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        stride=4,
                        k_neighbors=k_neighbors,
                    )
                )

        # Decoder
        self.up_transitions = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        decoder_channels = list(reversed(channels))
        decoder_num_heads = list(reversed(num_heads))

        for i in range(self.num_stages - 1):
            # Transition up
            self.up_transitions.append(
                TransitionUp(
                    in_channels=decoder_channels[i],
                    skip_channels=decoder_channels[i + 1],
                    out_channels=decoder_channels[i + 1],
                )
            )

            # Transformer blocks
            blocks = nn.ModuleList()
            for j in range(depths[self.num_stages - 2 - i]):
                blocks.append(
                    PointTransformerBlock(
                        in_channels=decoder_channels[i + 1],
                        out_channels=decoder_channels[i + 1],
                        num_heads=decoder_num_heads[i + 1],
                        k_neighbors=k_neighbors,
                        dropout=dropout,
                    )
                )
            self.decoder_blocks.append(blocks)

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels[0], channels[0] // 2),
            nn.BatchNorm1d(channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels[0] // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

        # Log model info
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"PointTransformer initialized: "
            f"{num_params:,} parameters, "
            f"{num_classes} classes"
        )

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, N, 3) input point positions

        Returns:
            (B, N, num_classes) per-point class logits
        """
        B, N, _ = x.shape
        pos = x  # Use positions directly

        # Input embedding
        x_flat = x.view(B * N, -1)
        x = self.input_embed[0](x_flat)
        x = self.input_embed[1](x.view(B, N, -1).transpose(1, 2)).transpose(1, 2)
        x = self.input_embed[2](x.contiguous().view(B * N, -1))
        x = self.input_embed[3](x)
        x = self.input_embed[4](x.view(B, N, -1).transpose(1, 2)).transpose(1, 2)
        x = self.input_embed[5](x.contiguous().view(B * N, -1))
        x = x.view(B, N, -1)

        # Encoder with skip connections
        encoder_features = []
        encoder_positions = []

        for i in range(self.num_stages):
            # Transformer blocks
            for block in self.encoder_blocks[i]:
                x = block(x, pos)

            # Store for skip connections
            encoder_features.append(x)
            encoder_positions.append(pos)

            # Transition down
            if i < self.num_stages - 1:
                x, pos, _ = self.down_transitions[i](x, pos)

        # Decoder
        for i in range(self.num_stages - 1):
            skip_idx = self.num_stages - 2 - i

            # Transition up
            x = self.up_transitions[i](
                x,
                pos,
                encoder_features[skip_idx],
                encoder_positions[skip_idx],
            )
            pos = encoder_positions[skip_idx]

            # Transformer blocks
            for block in self.decoder_blocks[i]:
                x = block(x, pos)

        # Segmentation head
        B, N, C = x.shape
        x_flat = x.view(B * N, C)

        # Apply seg head with proper BN
        x = self.seg_head[0](x_flat)  # Linear
        x = self.seg_head[1](x.view(B, N, -1).transpose(1, 2)).transpose(1, 2)  # BN
        x = self.seg_head[2](x.contiguous().view(B * N, -1))  # ReLU
        x = self.seg_head[3](x)  # Dropout
        x = self.seg_head[4](x)  # Linear
        x = self.seg_head[5](x.view(B, N, -1).transpose(1, 2)).transpose(1, 2)  # BN
        x = self.seg_head[6](x.contiguous().view(B * N, -1))  # ReLU
        x = self.seg_head[7](x)  # Dropout
        x = self.seg_head[8](x)  # Final linear

        x = x.view(B, N, self.num_classes)

        return x

    def predict(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get class predictions.

        Args:
            x: (B, N, 3) input point positions

        Returns:
            (B, N) predicted class indices
        """
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            x: (B, N, 3) input point positions

        Returns:
            (B, N, num_classes) class probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def create_point_transformer(
    num_classes: int = 3,
    model_size: str = "base",
    pretrained: Optional[str] = None,
) -> PointTransformer:
    """
    Factory function to create Point Transformer models.

    Args:
        num_classes: Number of segmentation classes
        model_size: 'small', 'base', or 'large'
        pretrained: Path to pretrained weights

    Returns:
        PointTransformer model
    """
    configs = {
        "small": {
            "embed_dim": 32,
            "depths": [1, 1, 1, 1],
            "channels": [32, 64, 128, 256],
            "num_heads": [2, 4, 4, 8],
            "k_neighbors": 16,
            "dropout": 0.1,
        },
        "base": {
            "embed_dim": 32,
            "depths": [2, 2, 2, 2],
            "channels": [32, 64, 128, 256],
            "num_heads": [2, 4, 8, 8],
            "k_neighbors": 16,
            "dropout": 0.1,
        },
        "large": {
            "embed_dim": 64,
            "depths": [3, 3, 3, 3],
            "channels": [64, 128, 256, 512],
            "num_heads": [4, 8, 8, 16],
            "k_neighbors": 24,
            "dropout": 0.1,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")

    config = configs[model_size]

    model = PointTransformer(
        in_channels=3,
        num_classes=num_classes,
        **config,
    )

    if pretrained is not None:
        logger.info(f"Loading pretrained weights from {pretrained}")
        state_dict = torch.load(pretrained, map_location="cpu")

        # Handle different state dict formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model.load_state_dict(state_dict, strict=False)

    return model
