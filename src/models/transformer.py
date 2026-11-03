"""
SolarTransformer: Feature Tokenizer Transformer (FT-Transformer) for Solar Radiation (GHI) Prediction.

Methodology Note:
Each continuous numerical meteorological feature is independently tokenized via a learned
linear projection into a vector embedding space (d_model). A learned [CLS] token is
prepended, and the token sequence undergoes Multi-Head Self-Attention across all
meteorological variables. The output [CLS] representation is passed through LayerNorm
and a linear head to predict continuous GHI.
This architecture directly models all inter-feature atmospheric interactions and supports
real-time single-row inference for web deployment.
"""

import torch
import torch.nn as nn


class NumericalFeatureTokenizer(nn.Module):
    """
    Learned Feature Tokenizer for numerical features (Gorishniy et al., NeurIPS 2021).
    Transforms each scalar feature x_i into a d_model embedding vector.
    """

    def __init__(self, num_features: int, d_model: int):
        super(NumericalFeatureTokenizer, self).__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Weight and bias per numerical feature
        self.weights = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.biases = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        Returns:
            torch.Tensor: Feature embeddings of shape (batch_size, num_features, d_model).
        """
        # x.unsqueeze(-1) -> (B, num_features, 1)
        # weights.unsqueeze(0) -> (1, num_features, d_model)
        # Result: (B, num_features, d_model)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class SolarTransformer(nn.Module):
    """
    FT-Transformer regression model for Solar Radiation (GHI) Prediction.

    Args:
        num_features (int): Number of input meteorological features (default: 21).
        d_model (int): Token embedding dimension (default: 48).
        nhead (int): Number of self-attention heads (default: 4).
        num_layers (int): Number of Transformer Encoder blocks (default: 2).
        dim_feedforward (int): Feedforward MLP dimension (default: 96).
        dropout (float): Dropout probability (default: 0.15).
    """

    def __init__(
        self,
        num_features: int = 21,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 96,
        dropout: float = 0.15
    ):
        super(SolarTransformer, self).__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Feature tokenizer
        self.tokenizer = NumericalFeatureTokenizer(num_features, d_model)

        # Learned [CLS] token prepended to the feature tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer Encoder Stack with Pre-LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # Final regression head operating on the [CLS] representation
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 21).
        Returns:
            torch.Tensor: Predicted GHI of shape (batch_size, 1).
        """
        batch_size = x.size(0)

        # Tokenize features: (B, 21, d_model)
        feature_tokens = self.tokenizer(x)

        # Expand [CLS] token to batch size: (B, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate [CLS] + feature tokens -> (B, 22, d_model)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)

        # Transformer self-attention pass
        encoded = self.transformer_encoder(tokens)

        # Extract [CLS] token representation at index 0
        cls_out = self.final_norm(encoded[:, 0, :])

        # Project to scalar GHI
        return self.head(cls_out)
