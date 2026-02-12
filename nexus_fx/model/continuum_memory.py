"""
Continuum Memory System (CMS) - Multi-timescale knowledge hierarchy.

The CMS is a spectrum of MLP blocks operating at different update frequencies,
creating a knowledge cascade from fast microstructure patterns to slow macro regimes.

Theoretical Background:
    In traditional neural networks, all parameters update at the same rate. In NSAM,
    different parts of the network operate as nested optimization problems with
    different timescales. The CMS embodies this by creating memory levels that
    update at exponentially spaced intervals:
    
    Level 1 (Fastest):   Updates every 1 step     → Microstructure (ticks-minutes)
    Level 2:             Updates every 10 steps    → Intraday dynamics (hours-days)
    Level 3:             Updates every 100 steps   → Medium-term patterns (days-weeks)
    Level 4 (Slowest):   Updates every 1000 steps  → Macro regimes (weeks-months)
    
    Knowledge cascades: When fast blocks update, slow blocks retain previous knowledge.
    This creates anti-catastrophic-forgetting: macro knowledge persists even as
    microstructure adapts rapidly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ContinuumMemoryLevel(nn.Module):
    """
    A single level in the continuum memory hierarchy.
    
    Each level is an MLP block with:
    - Its own update frequency
    - Residual connections
    - Layer normalization
    - Knowledge cascade from faster levels
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for this level
        update_frequency: How often this level updates its parameters
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        update_frequency: int = 1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.update_frequency = update_frequency
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Projection if input_dim != hidden_dim
        self.input_proj = None
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Step counter for update scheduling
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))
        
        # Cached output from last update (for when not updating)
        self.register_buffer('cached_output', None)
        
    def forward(
        self,
        x: torch.Tensor,
        force_update: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with conditional updating.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            force_update: If True, force update regardless of frequency
        
        Returns:
            output: Processed tensor (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        """
        should_update = force_update or ((self.step_counter % self.update_frequency) == 0)
        
        if should_update or self.cached_output is None:
            # Compute new output
            if self.input_proj is not None:
                identity = self.input_proj(x)
            else:
                identity = x
            
            output = self.mlp(x) + identity
            
            # Cache for future non-update steps
            # Only cache if batch size is 1 or we're in eval mode
            if self.training and x.shape[0] == 1:
                self.cached_output = output.detach()
            
            self.step_counter += 1
            return output
        else:
            # Return cached output
            # This implements the "slow memory" concept: the level maintains
            # its previous knowledge without updating
            self.step_counter += 1
            
            # If shapes match, use cached; otherwise recompute
            if self.cached_output is not None and self.cached_output.shape == x.shape:
                return self.cached_output
            else:
                # Fallback: recompute if shapes don't match
                if self.input_proj is not None:
                    identity = self.input_proj(x)
                else:
                    identity = x
                return self.mlp(x) + identity
    
    def reset(self) -> None:
        """Reset step counter and cache"""
        self.step_counter.zero_()
        self.cached_output = None


class ContinuumMemorySystem(nn.Module):
    """
    A spectrum of MLP blocks at different update frequencies.
    
    This creates a hierarchy of memories matching forex market dynamics:
    - Fast levels capture microstructure (spread widening, momentum bursts)
    - Medium levels capture intraday patterns (session effects, news reactions)
    - Slow levels capture macro regimes (carry trade, risk-on/risk-off)
    
    Knowledge flows bidirectionally:
    - Bottom-up: Fast patterns inform slow regime detection
    - Top-down: Slow regimes modulate fast pattern interpretation
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for MLP blocks
        num_levels: Number of memory levels
        base_frequency: Update frequency for fastest level (default: 1)
        frequency_multiplier: Multiplier between levels (default: 10)
        hidden_dims: Optional list of hidden dims per level (default: all same)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_levels: int = 4,
        base_frequency: int = 1,
        frequency_multiplier: int = 10,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.base_frequency = base_frequency
        self.frequency_multiplier = frequency_multiplier
        
        # Determine hidden dims for each level
        if hidden_dims is None:
            hidden_dims = [hidden_dim] * num_levels
        else:
            assert len(hidden_dims) == num_levels
        self.hidden_dims = hidden_dims
        
        # Calculate update frequencies for each level
        self.update_frequencies = self._calculate_frequencies()
        
        # Create memory levels
        self.levels = nn.ModuleList()
        current_input_dim = input_dim
        
        for i, (freq, h_dim) in enumerate(zip(self.update_frequencies, hidden_dims)):
            level = ContinuumMemoryLevel(
                input_dim=current_input_dim,
                hidden_dim=h_dim,
                update_frequency=freq,
            )
            self.levels.append(level)
            current_input_dim = h_dim  # Output of this level feeds to next
        
        # Cross-level fusion: combine all levels for final output
        total_dim = sum(hidden_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
    def _calculate_frequencies(self) -> List[int]:
        """Calculate exponentially spaced update frequencies"""
        frequencies = []
        for i in range(self.num_levels):
            freq = self.base_frequency * (self.frequency_multiplier ** i)
            frequencies.append(freq)
        return frequencies
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_levels: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through all memory levels.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            return_all_levels: If True, return outputs from all levels
        
        Returns:
            If return_all_levels=False:
                Fused output from all levels (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            If return_all_levels=True:
                Tuple of (fused_output, level_outputs_list)
        """
        level_outputs = []
        current_input = x
        
        # Process through each level sequentially (knowledge cascade)
        for i, level in enumerate(self.levels):
            level_out = level(current_input)
            level_outputs.append(level_out)
            current_input = level_out  # Feed to next level
        
        # Fuse all levels
        # Each level captures patterns at different timescales
        fused = torch.cat(level_outputs, dim=-1)
        output = self.fusion(fused)
        
        if return_all_levels:
            return output, level_outputs
        else:
            return output
    
    def get_level_states(self) -> List[dict]:
        """Get state of each level for inspection"""
        states = []
        for i, level in enumerate(self.levels):
            states.append({
                'level': i,
                'update_frequency': level.update_frequency,
                'step_counter': level.step_counter.item(),
                'has_cached_output': level.cached_output is not None,
            })
        return states
    
    def reset(self) -> None:
        """Reset all memory levels"""
        for level in self.levels:
            level.reset()
    
    def get_slowest_level_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get output from the slowest (macro regime) level only.
        
        This is useful for regime detection, as the slowest level contains
        the most persistent, macro-scale knowledge.
        """
        current_input = x
        for level in self.levels:
            current_input = level(current_input)
        return current_input
