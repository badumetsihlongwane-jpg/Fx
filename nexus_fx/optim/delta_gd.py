"""
Delta Gradient Descent (DGD) - L2-regression based optimizer.

Replaces standard dot-product weight update with L2-regression objective,
resulting in adaptive decay based on current data distribution.

Theoretical Background:
    In NSAM, parameter updates are framed as L2-regression problems rather
    than gradient descent. This creates adaptive learning rates and decay
    factors that respond to the current data distribution.
    
    Standard SGD: w ← w - lr * grad
    
    DGD: w ← w - lr * (grad + λ * (w - w_ref))
    
    where λ (decay) adapts based on mini-batch statistics, creating
    automatic regularization that strengthens when data is sparse/noisy
    and weakens when data is informative.
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional


class DeltaGradientDescent(Optimizer):
    """
    Delta Gradient Descent optimizer.
    
    Implements adaptive decay based on gradient statistics.
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        base_decay: Base decay factor (default: 0.01)
        adaptive_decay: Whether to adapt decay based on gradients (default: True)
        decay_momentum: Momentum for decay adaptation (default: 0.9)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        base_decay: float = 0.01,
        adaptive_decay: bool = True,
        decay_momentum: float = 0.9,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if base_decay < 0.0:
            raise ValueError(f"Invalid base_decay: {base_decay}")
        
        defaults = dict(
            lr=lr,
            base_decay=base_decay,
            adaptive_decay=adaptive_decay,
            decay_momentum=decay_momentum,
            eps=eps,
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
            base_decay = group['base_decay']
            adaptive_decay = group['adaptive_decay']
            decay_momentum = group['decay_momentum']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['reference_param'] = p.data.clone()
                    state['grad_variance'] = torch.zeros_like(p.data)
                    state['adaptive_lambda'] = torch.ones_like(p.data) * base_decay
                
                state['step'] += 1
                
                # Compute adaptive decay (λ)
                if adaptive_decay:
                    # Update gradient variance estimate (exponential moving average)
                    grad_sq = grad ** 2
                    state['grad_variance'] = (
                        decay_momentum * state['grad_variance'] +
                        (1 - decay_momentum) * grad_sq
                    )
                    
                    # Adapt lambda based on gradient variance
                    # High variance → higher decay (more regularization)
                    # Low variance → lower decay (trust the gradient)
                    grad_std = torch.sqrt(state['grad_variance'] + eps)
                    lambda_adaptive = base_decay * (1 + grad_std)
                    
                    # Smooth lambda updates
                    state['adaptive_lambda'] = (
                        decay_momentum * state['adaptive_lambda'] +
                        (1 - decay_momentum) * lambda_adaptive
                    )
                    
                    lambda_t = state['adaptive_lambda']
                else:
                    lambda_t = base_decay
                
                # DGD update: w ← w - lr * (grad + λ * (w - w_ref))
                # This is equivalent to L2 regression with reference point
                decay_term = lambda_t * (p.data - state['reference_param'])
                p.data.add_(grad + decay_term, alpha=-lr)
                
                # Periodically update reference (acts as slow-moving target)
                if state['step'] % 1000 == 0:
                    state['reference_param'] = p.data.clone()
        
        return loss
    
    def get_decay_stats(self) -> dict:
        """Get statistics about adaptive decay factors"""
        stats = {
            'mean_lambda': [],
            'std_lambda': [],
            'max_lambda': [],
        }
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'adaptive_lambda' in state:
                    lambda_t = state['adaptive_lambda']
                    stats['mean_lambda'].append(lambda_t.mean().item())
                    stats['std_lambda'].append(lambda_t.std().item())
                    stats['max_lambda'].append(lambda_t.max().item())
        
        # Average across all parameters
        for key in stats:
            if stats[key]:
                stats[key] = sum(stats[key]) / len(stats[key])
            else:
                stats[key] = 0.0
        
        return stats
