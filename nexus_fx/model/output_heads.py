"""
Output Heads - Multi-task prediction outputs.

NEXUS-FX predicts multiple targets simultaneously:
1. Direction (up/neutral/down classification)
2. Volatility (regression on future realized volatility)
3. Regime (current regime classification)
4. Confidence (uncertainty quantification)

Multi-task learning improves generalization and provides richer signals
for trading decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class OutputHeads(nn.Module):
    """
    Multi-task prediction heads for NEXUS-FX.
    
    Args:
        input_dim: Dimension of input features from backbone
        num_direction_classes: Number of direction classes (default: 3 for up/neutral/down)
        predict_volatility: Whether to predict volatility
        predict_regime: Whether to predict regime
        output_confidence: Whether to output confidence scores
        num_regimes: Number of regime classes
    """
    
    def __init__(
        self,
        input_dim: int,
        num_direction_classes: int = 3,
        predict_volatility: bool = True,
        predict_regime: bool = True,
        output_confidence: bool = True,
        num_regimes: int = 4,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_direction_classes = num_direction_classes
        self.predict_volatility = predict_volatility
        self.predict_regime = predict_regime
        self.output_confidence = output_confidence
        self.num_regimes = num_regimes
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )
        
        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_direction_classes),
        )
        
        # Volatility prediction head (regression)
        if predict_volatility:
            self.volatility_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
                nn.Softplus(),  # Ensure positive volatility predictions
            )
        
        # Regime prediction head
        if predict_regime:
            self.regime_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, num_regimes),
            )
        
        # Confidence head (predicts model uncertainty)
        if output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid(),  # Confidence in [0, 1]
            )
    
    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for all tasks.
        
        Args:
            features: Input features (batch, input_dim) or (batch, seq_len, input_dim)
        
        Returns:
            outputs: Dictionary containing:
                - direction_logits: (batch, num_direction_classes)
                - volatility: (batch, 1) if predict_volatility
                - regime_logits: (batch, num_regimes) if predict_regime
                - confidence: (batch, 1) if output_confidence
        """
        # Shared processing
        shared_features = self.shared(features)
        
        outputs = {}
        
        # Direction prediction
        outputs['direction_logits'] = self.direction_head(shared_features)
        
        # Volatility prediction
        if self.predict_volatility:
            outputs['volatility'] = self.volatility_head(shared_features)
        
        # Regime prediction
        if self.predict_regime:
            outputs['regime_logits'] = self.regime_head(shared_features)
        
        # Confidence prediction
        if self.output_confidence:
            outputs['confidence'] = self.confidence_head(shared_features)
        
        return outputs
    
    def get_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw outputs to predictions.
        
        Args:
            outputs: Raw outputs from forward pass
        
        Returns:
            predictions: Dictionary with:
                - direction_probs: Softmax probabilities
                - direction_class: Predicted class
                - volatility: Volatility prediction
                - regime_probs: Regime probabilities
                - regime_class: Predicted regime
                - confidence: Confidence score
        """
        predictions = {}
        
        # Direction
        direction_probs = F.softmax(outputs['direction_logits'], dim=-1)
        predictions['direction_probs'] = direction_probs
        predictions['direction_class'] = torch.argmax(direction_probs, dim=-1)
        
        # Volatility
        if 'volatility' in outputs:
            predictions['volatility'] = outputs['volatility']
        
        # Regime
        if 'regime_logits' in outputs:
            regime_probs = F.softmax(outputs['regime_logits'], dim=-1)
            predictions['regime_probs'] = regime_probs
            predictions['regime_class'] = torch.argmax(regime_probs, dim=-1)
        
        # Confidence
        if 'confidence' in outputs:
            predictions['confidence'] = outputs['confidence']
        
        return predictions
