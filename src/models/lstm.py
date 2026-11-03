"""
SolarLSTM: Long Short-Term Memory Network for Solar Radiation (GHI) Prediction.

Methodology Note:
Each row of 21 meteorological features is modeled as an ordered feature representation
(row-level pseudo-sequence) with feature embedding projection, stacked LSTM layers,
recurrent dropout, LayerNorm, and a multi-layer perceptron regression head.
This architecture supports real-time single-row inference for web deployment.
"""

import torch
import torch.nn as nn


class SolarLSTM(nn.Module):
    """
    LSTM regression model for Solar Radiation (GHI) Prediction.
    
    Args:
        input_dim (int): Number of input meteorological features (default: 21).
        embed_dim (int): Projection dimension for each feature token (default: 32).
        hidden_dim (int): LSTM hidden state dimension (default: 64).
        num_layers (int): Number of stacked LSTM layers (default: 2).
        dropout (float): Dropout probability (default: 0.2).
        bidirectional (bool): Whether to use bidirectional LSTM (default: True).
    """

    def __init__(
        self,
        input_dim: int = 21,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(SolarLSTM, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Project each scalar feature into an embedding dimension
        self.feature_projection = nn.Linear(1, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.post_lstm_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)

        # Non-linear regression head
        lstm_out_dim = hidden_dim * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
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
        # Reshape (B, 21) -> (B, 21, 1)
        x_seq = x.unsqueeze(-1)

        # Project to (B, 21, embed_dim)
        embeddings = self.input_norm(self.feature_projection(x_seq))

        # LSTM output: (B, 21, hidden_dim * num_directions)
        lstm_out, (h_n, _) = self.lstm(embeddings)

        # Global average pooling across feature sequence + last hidden state
        avg_pooled = torch.mean(lstm_out, dim=1)
        pooled = self.dropout(self.post_lstm_norm(avg_pooled))

        # Output prediction: (B, 1)
        return self.head(pooled)
