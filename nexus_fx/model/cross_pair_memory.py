"""
Cross-Pair Associative Memory - Learning currency pair correlations.

Forex pairs don't trade independently. EUR/USD movement creates associative
patterns in GBP/USD, USD/JPY, etc. This module learns these correlations
as nested associative memories.

Theoretical Background:
    In NSAM, memories can be composed hierarchically. Cross-pair memory
    treats each pair's hidden state as a key, and retrieves associated
    movements in other pairs. This captures:
    
    - Currency correlations (EUR/USD ↔ GBP/USD positive)
    - Inverse relationships (EUR/USD ↔ USD/JPY negative)
    - Safe-haven flows (Risk-off → JPY up, carry pairs down)
    - Macro context (Gold/yields affecting all USD pairs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .associative_memory import AssociativeMemory


class CrossPairMemory(nn.Module):
    """
    Learns correlations between currency pairs as associative memories.
    
    Also ingests macro features (yields, commodities, rates) as context keys
    that affect all pairs simultaneously.
    
    Args:
        num_pairs: Number of currency pairs
        pair_dim: Dimension of per-pair representations
        macro_dim: Dimension of macro feature encoding
        num_correlation_slots: Number of correlation pattern slots
    """
    
    def __init__(
        self,
        num_pairs: int,
        pair_dim: int,
        macro_dim: int,
        num_correlation_slots: int = 64,
    ):
        super().__init__()
        
        self.num_pairs = num_pairs
        self.pair_dim = pair_dim
        self.macro_dim = macro_dim
        
        # Pair-to-pair correlation memory
        # Key: pair_i state, Value: expected correlated movements in other pairs
        self.pair_correlation_memory = AssociativeMemory(
            key_dim=pair_dim,
            value_dim=pair_dim * num_pairs,
            num_slots=num_correlation_slots,
            update_frequency=5,  # Medium-term correlations
            use_surprise_gating=True,
        )
        
        # Macro-to-pairs memory
        # Key: macro state (yields, VIX, etc.), Value: expected pair responses
        self.macro_memory = AssociativeMemory(
            key_dim=macro_dim,
            value_dim=pair_dim * num_pairs,
            num_slots=num_correlation_slots,
            update_frequency=10,  # Slower, macro regimes
            use_surprise_gating=True,
        )
        
        # Fusion layer to combine correlation signals
        self.fusion = nn.Sequential(
            nn.Linear(pair_dim * num_pairs * 2, pair_dim * num_pairs),
            nn.LayerNorm(pair_dim * num_pairs),
            nn.GELU(),
            nn.Linear(pair_dim * num_pairs, pair_dim * num_pairs),
        )
        
        # Per-pair output projections
        self.pair_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pair_dim * 2, pair_dim),
                nn.LayerNorm(pair_dim),
            )
            for _ in range(num_pairs)
        ])
        
    def forward(
        self,
        pair_states: torch.Tensor,
        macro_state: torch.Tensor,
        write_mode: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass: compute cross-pair correlations.
        
        Args:
            pair_states: Per-pair hidden states (batch, num_pairs, pair_dim)
            macro_state: Macro feature encoding (batch, macro_dim)
            write_mode: Whether to write to correlation memories
        
        Returns:
            enriched_states: States enriched with correlation info (batch, num_pairs, pair_dim)
        """
        batch_size = pair_states.shape[0]
        
        # Flatten pair states for correlation lookup
        # Use mean across pairs as the query key
        pair_query = pair_states.mean(dim=1)  # (batch, pair_dim)
        
        # Retrieve pair-to-pair correlations
        pair_correlations, pair_surprise = self.pair_correlation_memory(
            query=pair_query,
            value_target=None,
            write_mode=False,
        )
        
        # Retrieve macro-driven correlations
        macro_correlations, macro_surprise = self.macro_memory(
            query=macro_state,
            value_target=None,
            write_mode=False,
        )
        
        # Fuse correlation signals
        fused_correlations = self.fusion(
            torch.cat([pair_correlations, macro_correlations], dim=-1)
        )
        
        # Reshape to (batch, num_pairs, pair_dim)
        fused_correlations = fused_correlations.view(batch_size, self.num_pairs, self.pair_dim)
        
        # Combine with original pair states
        enriched_states = []
        for i in range(self.num_pairs):
            pair_input = torch.cat([
                pair_states[:, i, :],
                fused_correlations[:, i, :],
            ], dim=-1)
            enriched = self.pair_outputs[i](pair_input)
            enriched_states.append(enriched)
        
        enriched_states = torch.stack(enriched_states, dim=1)
        
        # Write to memories if enabled
        if write_mode:
            # Write observed correlations
            actual_correlations = pair_states.view(batch_size, -1)
            self.pair_correlation_memory._write_to_memory(
                pair_query,
                actual_correlations,
                pair_surprise,
            )
            self.macro_memory._write_to_memory(
                macro_state,
                actual_correlations,
                macro_surprise,
            )
        
        return enriched_states
    
    def reset(self) -> None:
        """Reset all correlation memories"""
        self.pair_correlation_memory.reset()
        self.macro_memory.reset()
