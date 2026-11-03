"""
SolarCNN1D: 1D Convolutional Neural Network for Solar Radiation (GHI) Prediction.

Methodology Note:
Each row of 21 meteorological features is processed across multi-scale 1D convolutional
kernels with residual skip connections, batch normalization, GELU activations, and
adaptive pooling leading to an MLP regression head.
This architecture supports real-time single-row inference for web deployment.
"""

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Residual 1D Convolutional block with BatchNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.2):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual shortcut if channel dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class SolarCNN1D(nn.Module):
    """
    1D CNN regression model for Solar Radiation (GHI) Prediction.

    Args:
        input_dim (int): Number of input meteorological features (default: 21).
        base_channels (int): Base channel depth (default: 32).
        dropout (float): Dropout probability (default: 0.2).
    """

    def __init__(
        self,
        input_dim: int = 21,
        base_channels: int = 32,
        dropout: float = 0.2
    ):
        super(SolarCNN1D, self).__init__()
        self.input_dim = input_dim

        # Initial stem convolution: 1 input channel -> base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.GELU()
        )

        # Multi-scale feature extraction stages
        self.block1 = ConvBlock1D(base_channels, base_channels * 2, kernel_size=3, dropout=dropout)
        self.block2 = ConvBlock1D(base_channels * 2, base_channels * 4, kernel_size=3, dropout=dropout)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(base_channels * 4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 21).
        Returns:
            torch.Tensor: Predicted GHI of shape (batch_size, 1).
        """
        # Reshape (B, 21) -> (B, 1, 21)
        x_channels = x.unsqueeze(1)

        out = self.stem(x_channels)
        out = self.block1(out)
        out = self.block2(out)

        pooled = self.flatten(self.global_pool(out))
        return self.head(pooled)
