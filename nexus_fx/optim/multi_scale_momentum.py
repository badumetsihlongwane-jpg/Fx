"""
Multi-Scale Momentum Muon (M3) - Multi-timescale momentum optimizer.

Maintains momentum at multiple timescales, inspired by the nested memory
concept in NSAM. Combines short-term and long-term momentum adaptively.

Theoretical Background:
    Just as the model has memories at different timescales, the optimizer
    should have momentum at different timescales:
    
    - Short-term momentum: Captures recent gradient direction (fast adaptation)
    - Long-term momentum: Captures persistent optimization landscape (stability)
    
    The combination is learned during training, allowing the optimizer to
    balance rapid adaptation with stable convergence.
"""

import torch
from torch.optim import Optimizer
import math


class MultiScaleMomentumMuon(Optimizer):
    """
    Multi-Scale Momentum Muon optimizer.
    
    Maintains momentum buffers at multiple timescales and combines them
    via learned mixing weights.
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        betas: Tuple of (short_momentum, long_momentum) (default: (0.9, 0.999))
        weight_decay: Weight decay factor (default: 0.01)
        eps: Small constant for numerical stability (default: 1e-8)
        adaptive_mixing: Whether to adapt momentum mixing (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        adaptive_mixing: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            adaptive_mixing=adaptive_mixing,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        
        Returns:
            loss (optional)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta_short, beta_long = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            adaptive_mixing = group['adaptive_mixing']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_short'] = torch.zeros_like(p.data)
                    state['momentum_long'] = torch.zeros_like(p.data)
                    state['mixing_weight'] = 0.5  # Start with equal mixing
                    state['grad_variance_short'] = torch.zeros_like(p.data)
                    state['grad_variance_long'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                t = state['step']
                
                # Update short-term momentum
                state['momentum_short'] = (
                    beta_short * state['momentum_short'] +
                    (1 - beta_short) * grad
                )
                
                # Update long-term momentum
                state['momentum_long'] = (
                    beta_long * state['momentum_long'] +
                    (1 - beta_long) * grad
                )
                
                # Bias correction
                momentum_short_hat = state['momentum_short'] / (1 - beta_short ** t)
                momentum_long_hat = state['momentum_long'] / (1 - beta_long ** t)
                
                # Adaptive mixing
                if adaptive_mixing:
                    # Update gradient variance estimates
                    grad_sq = grad ** 2
                    state['grad_variance_short'] = (
                        beta_short * state['grad_variance_short'] +
                        (1 - beta_short) * grad_sq
                    )
                    state['grad_variance_long'] = (
                        beta_long * state['grad_variance_long'] +
                        (1 - beta_long) * grad_sq
                    )
                    
                    # Adapt mixing based on gradient variance
                    # If short-term variance is high, rely more on long-term
                    var_short = state['grad_variance_short'] / (1 - beta_short ** t)
                    var_long = state['grad_variance_long'] / (1 - beta_long ** t)
                    
                    # Mixing weight: 0 = all short, 1 = all long
                    var_ratio = var_short / (var_long + eps)
                    mixing_raw = torch.sigmoid(torch.log(var_ratio + eps))
                    
                    # Smooth mixing weight updates
                    state['mixing_weight'] = (
                        0.9 * state['mixing_weight'] +
                        0.1 * mixing_raw.mean().item()
                    )
                
                # Combine momenta
                alpha = state['mixing_weight']
                combined_momentum = (
                    (1 - alpha) * momentum_short_hat +
                    alpha * momentum_long_hat
                )
                
                # Update parameters
                p.data.add_(combined_momentum, alpha=-lr)
        
        return loss
    
    def get_momentum_stats(self) -> dict:
        """Get statistics about momentum mixing"""
        stats = {
            'mean_mixing_weight': [],
            'momentum_short_norm': [],
            'momentum_long_norm': [],
        }
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'mixing_weight' in state:
                    stats['mean_mixing_weight'].append(state['mixing_weight'])
                if 'momentum_short' in state:
                    stats['momentum_short_norm'].append(
                        state['momentum_short'].norm().item()
                    )
                if 'momentum_long' in state:
                    stats['momentum_long_norm'].append(
                        state['momentum_long'].norm().item()
                    )
        
        # Average across all parameters
        for key in stats:
            if stats[key]:
                stats[key] = sum(stats[key]) / len(stats[key])
            else:
                stats[key] = 0.0
        
        return stats
