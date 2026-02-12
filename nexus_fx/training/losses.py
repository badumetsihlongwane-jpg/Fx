"""
Multi-objective Loss Functions for NEXUS-FX.

Combines multiple prediction objectives:
1. Direction prediction (classification)
2. Volatility prediction (regression)
3. Regime prediction (classification)
4. Confidence calibration (alignment of predicted confidence with accuracy)

Each objective has an adaptive weight that adjusts during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class NexusFXLoss(nn.Module):
    """
    Multi-objective loss for NEXUS-FX model.
    
    Combines:
    - Direction loss (cross-entropy)
    - Volatility loss (MSE)
    - Regime loss (cross-entropy)
    - Calibration loss (confidence vs accuracy alignment)
    
    Args:
        direction_weight: Weight for direction loss
        volatility_weight: Weight for volatility loss
        regime_weight: Weight for regime loss
        calibration_weight: Weight for calibration loss
        adaptive_weights: Whether to adapt weights during training
    """
    
    def __init__(
        self,
        direction_weight: float = 1.0,
        volatility_weight: float = 0.5,
        regime_weight: float = 0.3,
        calibration_weight: float = 0.2,
        adaptive_weights: bool = True,
    ):
        super().__init__()
        
        self.register_buffer('direction_weight', torch.tensor(direction_weight))
        self.register_buffer('volatility_weight', torch.tensor(volatility_weight))
        self.register_buffer('regime_weight', torch.tensor(regime_weight))
        self.register_buffer('calibration_weight', torch.tensor(calibration_weight))
        
        self.adaptive_weights = adaptive_weights
        
        # Track loss magnitudes for adaptive weighting
        self.register_buffer('direction_ema', torch.tensor(1.0))
        self.register_buffer('volatility_ema', torch.tensor(1.0))
        self.register_buffer('regime_ema', torch.tensor(1.0))
        self.register_buffer('calibration_ema', torch.tensor(1.0))
        
        self.ema_momentum = 0.9
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-objective loss.
        
        Args:
            outputs: Model outputs dict with keys:
                - direction_logits: (batch, 3)
                - volatility: (batch, 1)
                - regime_logits: (batch, num_regimes)
                - confidence: (batch, 1)
            targets: Target dict with keys:
                - direction: (batch,) with values in {0, 1, 2}
                - volatility: (batch,) target volatility
                - regime: (batch,) regime labels (optional)
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual losses for logging
        """
        losses = {}
        
        # 1. Direction loss (cross-entropy)
        if 'direction_logits' in outputs and 'direction' in targets:
            direction_loss = F.cross_entropy(
                outputs['direction_logits'],
                targets['direction'].long()
            )
            losses['direction'] = direction_loss
        
        # 2. Volatility loss (MSE)
        if 'volatility' in outputs and 'volatility' in targets:
            volatility_loss = F.mse_loss(
                outputs['volatility'].squeeze(-1),
                targets['volatility']
            )
            losses['volatility'] = volatility_loss
        
        # 3. Regime loss (cross-entropy)
        if 'regime_logits' in outputs and 'regime' in targets:
            regime_loss = F.cross_entropy(
                outputs['regime_logits'],
                targets['regime'].long()
            )
            losses['regime'] = regime_loss
        
        # 4. Calibration loss
        if 'confidence' in outputs and 'direction_logits' in outputs and 'direction' in targets:
            calibration_loss = self._compute_calibration_loss(
                outputs['direction_logits'],
                outputs['confidence'],
                targets['direction']
            )
            losses['calibration'] = calibration_loss
        
        # Update EMAs
        if self.training and self.adaptive_weights:
            self._update_loss_emas(losses)
        
        # Compute weighted sum
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        if 'direction' in losses:
            weight = self._get_adaptive_weight('direction')
            total_loss = total_loss + weight * losses['direction']
        
        if 'volatility' in losses:
            weight = self._get_adaptive_weight('volatility')
            total_loss = total_loss + weight * losses['volatility']
        
        if 'regime' in losses:
            weight = self._get_adaptive_weight('regime')
            total_loss = total_loss + weight * losses['regime']
        
        if 'calibration' in losses:
            weight = self._get_adaptive_weight('calibration')
            total_loss = total_loss + weight * losses['calibration']
        
        # Add total to dict
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _compute_calibration_loss(
        self,
        direction_logits: torch.Tensor,
        confidence: torch.Tensor,
        direction_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calibration loss: predicted confidence should match actual accuracy.
        
        High confidence predictions should be more accurate than low confidence ones.
        """
        # Get predictions
        pred_classes = torch.argmax(direction_logits, dim=-1)
        correct = (pred_classes == direction_targets.long()).float()
        
        # Confidence should match correctness
        # If model is 90% confident, it should be right 90% of the time
        calibration_loss = F.mse_loss(confidence.squeeze(-1), correct)
        
        return calibration_loss
    
    def _update_loss_emas(self, losses: Dict[str, torch.Tensor]) -> None:
        """Update exponential moving averages of loss magnitudes"""
        if 'direction' in losses:
            self.direction_ema = (
                self.ema_momentum * self.direction_ema +
                (1 - self.ema_momentum) * losses['direction'].detach()
            )
        
        if 'volatility' in losses:
            self.volatility_ema = (
                self.ema_momentum * self.volatility_ema +
                (1 - self.ema_momentum) * losses['volatility'].detach()
            )
        
        if 'regime' in losses:
            self.regime_ema = (
                self.ema_momentum * self.regime_ema +
                (1 - self.ema_momentum) * losses['regime'].detach()
            )
        
        if 'calibration' in losses:
            self.calibration_ema = (
                self.ema_momentum * self.calibration_ema +
                (1 - self.ema_momentum) * losses['calibration'].detach()
            )
    
    def _get_adaptive_weight(self, loss_name: str) -> torch.Tensor:
        """
        Get adaptive weight for a loss component.
        
        Balances losses by normalizing by their typical magnitude.
        """
        if not self.adaptive_weights:
            if loss_name == 'direction':
                return self.direction_weight
            elif loss_name == 'volatility':
                return self.volatility_weight
            elif loss_name == 'regime':
                return self.regime_weight
            elif loss_name == 'calibration':
                return self.calibration_weight
        
        # Adaptive weighting based on EMA magnitudes
        # Normalize so all losses contribute roughly equally
        total_ema = (
            self.direction_ema + self.volatility_ema +
            self.regime_ema + self.calibration_ema
        )
        
        if loss_name == 'direction':
            base_weight = self.direction_weight
            adaptive_factor = total_ema / (self.direction_ema + 1e-8)
        elif loss_name == 'volatility':
            base_weight = self.volatility_weight
            adaptive_factor = total_ema / (self.volatility_ema + 1e-8)
        elif loss_name == 'regime':
            base_weight = self.regime_weight
            adaptive_factor = total_ema / (self.regime_ema + 1e-8)
        elif loss_name == 'calibration':
            base_weight = self.calibration_weight
            adaptive_factor = total_ema / (self.calibration_ema + 1e-8)
        else:
            return torch.tensor(1.0)
        
        return base_weight * adaptive_factor / 4  # Normalize by number of losses
