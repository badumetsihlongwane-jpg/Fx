"""
Associative Memory Module - The fundamental building block of NEXUS-FX.

This module implements associative memory based on L2-regression rather than
dot-product attention, following the NSAM (Nested Sequential Associative Memory)
framework. Each memory stores key-value pairs and retrieves via minimizing
an internal L2 objective.

Theoretical Background:
    In NSAM, memories are nested optimization problems. Traditional attention
    uses dot-product similarity, but associative memory uses L2-regression:
    
    M(q) = argmin_v || v - Σ_i α_i * V_i ||^2
    
    where α_i are attention weights based on L2 distance to keys:
    α_i = softmin(|| q - K_i ||^2, temperature=τ)
    
    This formulation naturally leads to surprise-gated writing: the memory
    writes more when prediction error is high (surprise = ||v_actual - v_predicted||^2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AssociativeMemory(nn.Module):
    """
    Core associative memory: stores key-value pairs and retrieves
    via L2-minimization rather than dot-product attention.
    
    Each memory has an update_frequency parameter controlling how often
    its internal state updates. This is the core mechanism for creating
    a hierarchy of memories at different timescales.
    
    Args:
        key_dim: Dimension of memory keys
        value_dim: Dimension of memory values
        num_slots: Number of key-value slots in the memory
        update_frequency: How often this memory updates (1 = every step)
        temperature: Temperature for attention weight computation
        use_surprise_gating: If True, gate writes by prediction error
        use_dgd: If True, use Delta GD principles for adaptive decay
    """
    
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_slots: int,
        update_frequency: int = 1,
        temperature: float = 1.0,
        use_surprise_gating: bool = True,
        use_dgd: bool = True,
        memory_decay: float = 0.99,
    ):
        super().__init__()
        
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_slots = num_slots
        self.update_frequency = update_frequency
        self.temperature = temperature
        self.use_surprise_gating = use_surprise_gating
        self.use_dgd = use_dgd
        self.memory_decay = memory_decay
        
        # Initialize memory slots
        # Keys and values are learnable parameters that get updated during forward pass
        self.register_buffer('keys', torch.randn(num_slots, key_dim))
        self.register_buffer('values', torch.randn(num_slots, value_dim))
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))
        self.register_buffer('slot_age', torch.zeros(num_slots))
        
        # Normalization layers
        self.key_norm = nn.LayerNorm(key_dim)
        self.value_norm = nn.LayerNorm(value_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        value_target: Optional[torch.Tensor] = None,
        write_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: read from memory and optionally write.
        
        Args:
            query: Query tensor of shape (batch, key_dim)
            value_target: Target values to write (batch, value_dim), if in write mode
            write_mode: If True and update frequency allows, write to memory
        
        Returns:
            retrieved_values: Retrieved values (batch, value_dim)
            surprise: Prediction error for each query (batch,)
        """
        batch_size = query.shape[0]
        
        # Normalize query
        query = self.key_norm(query)
        
        # Compute L2 distances to all keys
        # distances shape: (batch, num_slots)
        distances = torch.cdist(query, self.keys, p=2)
        
        # Compute attention weights using softmin (negative softmax of distances)
        # α_i = exp(-d_i^2 / τ) / Σ_j exp(-d_j^2 / τ)
        attention_logits = -distances.pow(2) / self.temperature
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Retrieve values via weighted sum
        # retrieved = Σ_i α_i * V_i
        retrieved_values = torch.matmul(attention_weights, self.values)  # (batch, value_dim)
        
        # Compute surprise (prediction error)
        if value_target is not None:
            surprise = F.mse_loss(retrieved_values, value_target, reduction='none').mean(dim=-1)
        else:
            surprise = torch.zeros(batch_size, device=query.device)
        
        # Write to memory if conditions are met
        if write_mode and value_target is not None:
            self._write_to_memory(query, value_target, surprise)
        
        return retrieved_values, surprise
    
    def _write_to_memory(
        self,
        keys_new: torch.Tensor,
        values_new: torch.Tensor,
        surprise: torch.Tensor,
    ) -> None:
        """
        Write new key-value pairs to memory slots.
        
        Writing strategy:
        1. Check if current step is a multiple of update_frequency
        2. If using surprise gating, weight writes by prediction error
        3. Replace oldest slots or use DGD-style adaptive update
        
        Args:
            keys_new: New keys to write (batch, key_dim)
            values_new: New values to write (batch, value_dim)
            surprise: Surprise scores for gating (batch,)
        """
        # Check if we should update at this step
        should_update = (self.step_counter % self.update_frequency) == 0
        
        if not should_update:
            self.step_counter += 1
            return
        
        batch_size = keys_new.shape[0]
        
        # Apply surprise gating if enabled
        if self.use_surprise_gating:
            # Higher surprise → higher write weight
            write_weights = torch.sigmoid(surprise - surprise.mean())
        else:
            write_weights = torch.ones(batch_size, device=keys_new.device)
        
        # Find slots to replace (oldest slots)
        num_to_replace = min(batch_size, self.num_slots)
        _, oldest_indices = torch.topk(self.slot_age, num_to_replace, largest=True)
        
        # Write new keys and values
        for i in range(num_to_replace):
            idx = oldest_indices[i]
            weight = write_weights[i]
            
            if self.use_dgd:
                # DGD-style update: blend old and new with adaptive decay
                decay = self.memory_decay * (1 - weight)
                self.keys[idx] = decay * self.keys[idx] + (1 - decay) * keys_new[i]
                self.values[idx] = decay * self.values[idx] + (1 - decay) * values_new[i]
            else:
                # Direct replacement
                self.keys[idx] = keys_new[i]
                self.values[idx] = values_new[i]
            
            # Reset age for this slot
            self.slot_age[idx] = 0
        
        # Age all other slots
        self.slot_age += 1
        self.step_counter += 1
    
    def reset(self) -> None:
        """Reset memory to initial state"""
        self.keys.normal_()
        self.values.normal_()
        self.step_counter.zero_()
        self.slot_age.zero_()
    
    def get_memory_state(self) -> dict:
        """Get current memory state for inspection/saving"""
        return {
            'keys': self.keys.clone(),
            'values': self.values.clone(),
            'step_counter': self.step_counter.item(),
            'slot_age': self.slot_age.clone(),
        }
    
    def load_memory_state(self, state: dict) -> None:
        """Load memory state from checkpoint"""
        self.keys.copy_(state['keys'])
        self.values.copy_(state['values'])
        self.step_counter.copy_(torch.tensor(state['step_counter']))
        self.slot_age.copy_(state['slot_age'])
