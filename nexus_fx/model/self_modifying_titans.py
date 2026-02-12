"""
Self-Modifying Titans - In-context adaptive sequence processing.

Unlike standard Transformers, this module generates its OWN keys, values,
and learning rates during the forward pass. This enables deep self-modification:
the model adapts its memory writing strategy based on the current market regime.

Theoretical Background:
    In NSAM, each optimization problem can generate its own learning rate schedule.
    Applied to forex: during high-volatility events (NFP, rate decisions), the model
    should "pay more attention" by increasing its in-context learning rate.
    
    The Titans architecture processes sequences while maintaining a persistent
    associative memory. At each timestep:
    1. Generate query from current input
    2. Read from memory using the query
    3. Generate new (key, value, lr) triple based on current state
    4. Write to memory with self-generated learning rate
    5. Output prediction
    
    This allows the model to adapt without weight updates, crucial for live trading
    where we can't retrain on each tick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .associative_memory import AssociativeMemory


class SelfModifyingTitansLayer(nn.Module):
    """
    A single layer of Self-Modifying Titans.
    
    Processes input through:
    1. Query generation
    2. Memory read
    3. Key-value-lr generation
    4. Memory write
    5. Output generation
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_memory_slots: Number of slots in the associative memory
        memory_update_frequency: How often the memory updates
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_memory_slots: int = 128,
        memory_update_frequency: int = 1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Query generation
        self.query_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Associative memory
        self.memory = AssociativeMemory(
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            num_slots=num_memory_slots,
            update_frequency=memory_update_frequency,
            use_surprise_gating=True,
            use_dgd=True,
        )
        
        # Key-Value-LR generator
        # Generates new memory entries and learning rate from current state
        self.kv_lr_gen = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 3),  # 2x input → 3x output (K, V, LR)
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        write_mode: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through the Titans layer.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            write_mode: Whether to write to memory
        
        Returns:
            output: Processed tensor (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            aux_info: Dictionary with auxiliary information (surprise, lr, etc.)
        """
        # Project input
        x_proj = self.input_proj(x)
        identity = x_proj
        
        # Generate query
        query = self.query_gen(x_proj)
        
        # Read from memory
        memory_out, surprise = self.memory(query, write_mode=False)
        
        # Combine input and memory readout
        combined = torch.cat([x_proj, memory_out], dim=-1)
        
        # Generate new key, value, and learning rate
        kv_lr = self.kv_lr_gen(combined)
        
        # Split into key, value, learning rate
        key_new, value_new, lr_logit = torch.chunk(kv_lr, 3, dim=-1)
        
        # Learning rate is sigmoid of lr_logit (0 to 1 range)
        # High lr during surprises, low lr during normal times
        lr = torch.sigmoid(lr_logit).mean(dim=-1, keepdim=True)  # (batch, 1)
        
        # Write to memory if in write mode
        if write_mode:
            # Use surprise as the target value for memory update
            # This creates a self-supervised learning signal
            self.memory._write_to_memory(key_new, value_new, surprise)
        
        # Generate output
        output = self.output_proj(combined)
        output = self.norm1(output + identity)
        
        # Auxiliary information
        aux_info = {
            'surprise': surprise,
            'learning_rate': lr,
            'memory_retrieval': memory_out,
        }
        
        return output, aux_info


class SelfModifyingTitans(nn.Module):
    """
    Multi-layer Self-Modifying Titans for sequence processing.
    
    Stacks multiple Titans layers with residual connections, creating
    a deep hierarchy of self-modifying memories.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for all layers
        num_memory_slots: Number of slots per memory
        num_layers: Number of Titans layers
        memory_update_frequency: Base update frequency for memories
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_memory_slots: int = 128,
        num_layers: int = 4,
        memory_update_frequency: int = 1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create layers
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_layers):
            layer = SelfModifyingTitansLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                num_memory_slots=num_memory_slots,
                memory_update_frequency=memory_update_frequency * (i + 1),  # Slower at deeper layers
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        # Final output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        write_mode: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through all Titans layers.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            write_mode: Whether to write to memories
        
        Returns:
            output: Final output (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            all_aux_info: Aggregated auxiliary info from all layers
        """
        all_surprises = []
        all_lrs = []
        current = x
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            current, aux_info = layer(current, write_mode=write_mode)
            all_surprises.append(aux_info['surprise'])
            all_lrs.append(aux_info['learning_rate'])
        
        # Final normalization
        output = self.output_norm(current)
        
        # Aggregate auxiliary information
        all_aux_info = {
            'surprises': all_surprises,  # List of surprise tensors per layer
            'learning_rates': all_lrs,   # List of lr tensors per layer
            'mean_surprise': torch.stack(all_surprises).mean(),
            'mean_lr': torch.stack(all_lrs).mean(),
        }
        
        return output, all_aux_info
    
    def reset_memories(self) -> None:
        """Reset all memories in all layers"""
        for layer in self.layers:
            layer.memory.reset()
    
    def get_memory_states(self) -> list:
        """Get states of all memories for checkpointing"""
        states = []
        for i, layer in enumerate(self.layers):
            states.append({
                'layer': i,
                'memory_state': layer.memory.get_memory_state(),
            })
        return states
    
    def load_memory_states(self, states: list) -> None:
        """Load memory states from checkpoint"""
        for i, state in enumerate(states):
            self.layers[i].memory.load_memory_state(state['memory_state'])
