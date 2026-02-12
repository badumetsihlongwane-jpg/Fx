"""
Regime Detector - Latent market state classification.

Detects whether the market is in trending, ranging, volatile, or quiet regimes.
Uses the slow CMS blocks as input, since regime = slow macro knowledge.

Regime detection feeds back into the model to modulate predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RegimeDetector(nn.Module):
    """
    Detects latent market regimes from slow memory blocks.
    
    Regimes:
    0: Trending (directional movement, follow momentum)
    1: Ranging (mean reversion, fade extremes)
    2: Volatile (high uncertainty, reduce position size)
    3: Quiet (low volume, widen stops, reduce frequency)
    
    The regime is detected from the slowest CMS level, which contains
    the most persistent macro knowledge.
    
    Args:
        input_dim: Input dimension (from slow CMS level)
        hidden_dim: Hidden dimension for regime classifier
        num_regimes: Number of regime classes
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_regimes: int = 4,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        # Regime feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Regime classifier head
        self.classifier = nn.Linear(hidden_dim, num_regimes)
        
        # Regime embedding (for feeding back to the model)
        self.regime_embeddings = nn.Embedding(num_regimes, hidden_dim)
        
    def forward(
        self,
        slow_memory_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect market regime.
        
        Args:
            slow_memory_state: Output from slowest CMS level (batch, input_dim)
        
        Returns:
            regime_logits: Logits for each regime class (batch, num_regimes)
            regime_features: Regime feature embedding (batch, hidden_dim)
        """
        # Extract features
        regime_features = self.feature_extractor(slow_memory_state)
        
        # Classify regime
        regime_logits = self.classifier(regime_features)
        
        return regime_logits, regime_features
    
    def get_regime_embedding(
        self,
        regime_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get soft regime embedding based on regime probabilities.
        
        Args:
            regime_probs: Probabilities for each regime (batch, num_regimes)
        
        Returns:
            regime_emb: Soft regime embedding (batch, hidden_dim)
        """
        # Get all regime embeddings
        all_embeddings = self.regime_embeddings.weight  # (num_regimes, hidden_dim)
        
        # Soft combination based on probabilities
        regime_emb = torch.matmul(regime_probs, all_embeddings)  # (batch, hidden_dim)
        
        return regime_emb
