"""
Session-Aware Frequency Gate - Adaptive memory activation.

Forex markets exhibit session-dependent dynamics. During London-NY overlap,
volatility spikes and fast memories should dominate. During Asian session,
slow memories better capture the ranging behavior.

This module gates memory levels based on active trading sessions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionFrequencyGate(nn.Module):
    """
    Gates memory levels based on active forex session.
    
    Session characteristics:
    - Sydney: Lowest volume, ranging
    - Tokyo: Medium volume, trend following
    - London: High volume, breakouts
    - New York: Highest volume, reversals
    - London-NY overlap: Extreme volatility, fast memories crucial
    
    Also responds to news events, which spike all frequencies.
    
    Args:
        num_memory_levels: Number of CMS levels to gate
        session_embedding_dim: Dimension of session embeddings
    """
    
    def __init__(
        self,
        num_memory_levels: int = 4,
        session_embedding_dim: int = 32,
    ):
        super().__init__()
        
        self.num_memory_levels = num_memory_levels
        self.session_embedding_dim = session_embedding_dim
        
        # Session encoder: maps session indicators to embeddings
        # Input: [is_sydney, is_tokyo, is_london, is_ny, is_overlap, is_news_event]
        self.session_encoder = nn.Sequential(
            nn.Linear(6, session_embedding_dim),
            nn.LayerNorm(session_embedding_dim),
            nn.GELU(),
            nn.Linear(session_embedding_dim, session_embedding_dim),
        )
        
        # Frequency gate generator
        # Outputs logits for each memory level's activation
        self.gate_generator = nn.Sequential(
            nn.Linear(session_embedding_dim, num_memory_levels * 2),
            nn.LayerNorm(num_memory_levels * 2),
            nn.GELU(),
            nn.Linear(num_memory_levels * 2, num_memory_levels),
        )
        
    def forward(
        self,
        session_indicators: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate frequency gates for memory levels.
        
        Args:
            session_indicators: Session state (batch, 6)
                [is_sydney, is_tokyo, is_london, is_ny, is_overlap, is_news_event]
        
        Returns:
            gates: Activation gates for each level (batch, num_memory_levels)
                Range: [0, 1] via sigmoid, higher = more active
        """
        # Encode session
        session_emb = self.session_encoder(session_indicators)
        
        # Generate gates
        gate_logits = self.gate_generator(session_emb)
        
        # Sigmoid to [0, 1] range
        gates = torch.sigmoid(gate_logits)
        
        # During news events (last indicator), boost all gates
        is_news = session_indicators[:, -1:].unsqueeze(-1)  # (batch, 1, 1)
        gates = gates + is_news * 0.5  # Boost by 0.5 during news
        gates = torch.clamp(gates, 0, 1)
        
        return gates
    
    def apply_gates(
        self,
        level_outputs: list,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply gates to memory level outputs.
        
        Args:
            level_outputs: List of tensors from each CMS level
            gates: Gates for each level (batch, num_memory_levels)
        
        Returns:
            gated_outputs: Weighted combination of levels (batch, ...)
        """
        # Normalize gates to sum to 1 (softmax across levels)
        gate_weights = F.softmax(gates, dim=-1)
        
        # Weight each level's output
        gated = []
        for i, output in enumerate(level_outputs):
            weight = gate_weights[:, i].unsqueeze(-1)
            # Expand weight to match output shape
            while weight.dim() < output.dim():
                weight = weight.unsqueeze(-1)
            gated.append(output * weight)
        
        # Sum weighted outputs
        gated_output = torch.stack(gated, dim=0).sum(dim=0)
        
        return gated_output
