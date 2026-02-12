"""
Main training loop for NEXUS-FX with continual learning support.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm
import os

from ..config import NexusFXConfig
from .losses import NexusFXLoss
from .evaluation import NexusFXEvaluator
from ..optim import DeltaGradientDescent, MultiScaleMomentumMuon


class NexusFXTrainer:
    """
    Trainer for NEXUS-FX model.
    
    Supports:
    - Continual learning
    - Multi-scale memory updates
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: NexusFXConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = NexusFXLoss(
            direction_weight=config.direction_loss_weight,
            volatility_weight=config.volatility_loss_weight,
            regime_weight=config.regime_loss_weight,
            calibration_weight=config.calibration_loss_weight,
        ).to(device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Evaluator
        self.evaluator = NexusFXEvaluator()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self):
        """Create optimizer based on config"""
        if self.config.optimizer_type == 'delta_gd':
            return DeltaGradientDescent(
                self.model.parameters(),
                lr=self.config.learning_rate,
                base_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == 'multi_scale_momentum':
            return MultiScaleMomentumMuon(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:  # Adam fallback
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            loss, losses_dict = self.train_step(batch)
            epoch_losses.append(losses_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'step': self.global_step,
            })
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = sum(d[key].item() for d in epoch_losses) / len(epoch_losses)
        
        return avg_losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass (to be implemented when model is ready)
        # For now, assume batch has 'inputs' and 'targets'
        # outputs = self.model(batch['inputs'])
        
        # Placeholder outputs for structure
        outputs = {
            'direction_logits': torch.randn(len(batch.get('direction', [1])), 3, device=self.device),
            'volatility': torch.randn(len(batch.get('direction', [1])), 1, device=self.device),
        }
        
        targets = {
            'direction': batch.get('direction', torch.zeros(1, device=self.device)),
            'volatility': batch.get('volatility', torch.zeros(1, device=self.device)),
        }
        
        # Compute loss
        loss, losses_dict = self.criterion(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss, losses_dict
    
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.evaluator.reset()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Placeholder - will be implemented with full model
                outputs = {
                    'direction_logits': torch.randn(1, 3, device=self.device),
                    'volatility': torch.randn(1, 1, device=self.device),
                }
                
                targets = {
                    'direction': batch.get('direction', torch.zeros(1, device=self.device)),
                    'volatility': batch.get('volatility', torch.zeros(1, device=self.device)),
                }
                
                loss, losses_dict = self.criterion(outputs, targets)
                val_losses.append(losses_dict)
                
                self.evaluator.update(outputs, targets)
        
        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = sum(d[key].item() for d in val_losses) / len(val_losses)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics()
        avg_losses.update(metrics)
        
        return avg_losses
    
    def train(self) -> None:
        """Main training loop"""
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_losses.get('total', 0):.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_losses = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_losses.get('total', 0):.4f}")
                
                # Save best model
                if val_losses.get('total', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pt')
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, os.path.join('checkpoints', filename))
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded: {filename}")
