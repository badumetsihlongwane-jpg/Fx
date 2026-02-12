"""
NEXUS-FX: Main model combining all components.

Integrates:
- Multi-timeframe feature processing
- Self-Modifying Titans for sequence processing
- Continuum Memory System for multi-scale persistence
- Cross-Pair Memory for correlation learning
- Session-aware frequency gating
- Regime detection
- Multi-task output heads
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from ..config import NexusFXConfig
from .associative_memory import AssociativeMemory
from .self_modifying_titans import SelfModifyingTitans
from .continuum_memory import ContinuumMemorySystem
from .cross_pair_memory import CrossPairMemory
from .session_gate import SessionFrequencyGate
from .regime_detector import RegimeDetector
from .output_heads import OutputHeads
from ..data import FeatureEngine, MacroFeatureEncoder, SessionClock


class NEXUSFX(nn.Module):
    """
    Full NEXUS-FX architecture.
    
    Forward pass flow:
    1. Feature engineering: OHLC → returns, volatility, technicals
    2. Macro encoding: calendar events, rates, yields → macro embedding
    3. Session detection: current time → session embedding
    4. Per-pair processing through Self-Modifying Titans
    5. Cross-pair correlation via CrossPairMemory
    6. Continuum Memory System for multi-scale persistence
    7. Session-gated frequency adjustment
    8. Regime detection (feeds back to CMS)
    9. Output heads: direction, volatility, regime, confidence
    
    Args:
        config: NexusFXConfig with all hyperparameters
    """
    
    def __init__(self, config: NexusFXConfig):
        super().__init__()
        
        self.config = config
        
        # Feature engineering (not trainable, pure computation)
        self.feature_engine = FeatureEngine(
            lookback_periods=config.lookback_periods,
            include_volume=config.include_volume,
        )
        
        # Macro feature encoder
        self.macro_encoder = MacroFeatureEncoder(
            pairs=config.pairs,
            include_calendar=config.include_macro,
            include_rates=config.include_macro,
            include_yields=config.include_macro,
            include_commodities=config.include_macro,
            include_sentiment=config.include_macro,
        )
        
        # Session clock
        self.session_clock = SessionClock()
        
        # Calculate input dimensions
        # Features per timeframe: OHLC (4) + technical features (~20)
        features_per_tf = 24  # Approximate
        num_timeframes = len(config.timeframes)
        total_feature_dim = features_per_tf * num_timeframes
        
        # Input projection: map all features to input_dim
        self.input_projection = nn.Sequential(
            nn.Linear(total_feature_dim, config.input_dim),
            nn.LayerNorm(config.input_dim),
            nn.GELU(),
        )
        
        # Macro projection
        macro_feature_dim = self.macro_encoder.feature_dim
        self.macro_projection = nn.Sequential(
            nn.Linear(macro_feature_dim, config.input_dim // 2),
            nn.LayerNorm(config.input_dim // 2),
            nn.GELU(),
        )
        
        # Session projection
        session_feature_dim = 19  # From session_clock.compute_session_features
        self.session_projection = nn.Sequential(
            nn.Linear(session_feature_dim, config.session_embedding_dim),
            nn.LayerNorm(config.session_embedding_dim),
            nn.GELU(),
        )
        
        # Per-pair Self-Modifying Titans
        self.titans_per_pair = nn.ModuleList([
            SelfModifyingTitans(
                input_dim=config.input_dim + config.input_dim // 2,  # features + macro
                hidden_dim=config.hidden_dim,
                num_memory_slots=config.num_memory_slots,
                num_layers=config.num_titans_layers,
            )
            for _ in range(config.num_pairs)
        ])
        
        # Cross-Pair Memory
        self.cross_pair_memory = CrossPairMemory(
            num_pairs=config.num_pairs,
            pair_dim=config.hidden_dim,
            macro_dim=config.input_dim // 2,
            num_correlation_slots=config.num_correlation_slots,
        )
        
        # Continuum Memory System
        self.continuum_memory = ContinuumMemorySystem(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_levels=config.num_cms_levels,
            base_frequency=config.cms_base_frequency,
            frequency_multiplier=config.cms_frequency_multiplier,
            hidden_dims=config.cms_hidden_dims,
        )
        
        # Session-aware Frequency Gate
        self.session_gate = SessionFrequencyGate(
            num_memory_levels=config.num_cms_levels,
            session_embedding_dim=config.session_embedding_dim,
        )
        
        # Regime Detector
        self.regime_detector = RegimeDetector(
            input_dim=config.hidden_dim,
            hidden_dim=config.regime_hidden_dim,
            num_regimes=config.num_regimes,
        )
        
        # Final fusion: combine all pair outputs
        self.pair_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_pairs, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        
        # Output heads
        self.output_heads = OutputHeads(
            input_dim=config.hidden_dim,
            num_direction_classes=config.num_direction_classes,
            predict_volatility=config.predict_volatility,
            predict_regime=config.predict_regime,
            output_confidence=config.output_confidence,
            num_regimes=config.num_regimes,
        )
    
    def forward(
        self,
        ohlc: torch.Tensor,
        volume: Optional[torch.Tensor],
        timestamps: torch.Tensor,
        macro_data: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NEXUS-FX.
        
        Args:
            ohlc: Multi-pair, multi-timeframe OHLC (batch, pairs, timeframes, seq, 4)
            volume: Volume data (batch, pairs, timeframes, seq)
            timestamps: Unix timestamps (batch, seq)
            macro_data: Optional macro feature data dict
        
        Returns:
            outputs: Dictionary with all predictions
        """
        batch_size, num_pairs, num_tf, seq_len, _ = ohlc.shape
        
        # 1. Feature Engineering per pair and timeframe
        pair_features = []
        for p in range(num_pairs):
            tf_features = []
            for tf in range(num_tf):
                ohlc_tf = ohlc[:, p, tf, :, :]  # (batch, seq, 4)
                vol_tf = volume[:, p, tf, :] if volume is not None else None
                
                features = self.feature_engine.compute_features(ohlc_tf, vol_tf)
                tf_features.append(features)
            
            # Concatenate timeframe features
            pair_feat = torch.cat(tf_features, dim=-1)  # (batch, seq, features)
            pair_features.append(pair_feat)
        
        # Stack pairs: (batch, num_pairs, seq, features)
        pair_features = torch.stack(pair_features, dim=1)
        
        # 2. Project features to input_dim
        pair_features = pair_features.view(batch_size * num_pairs, seq_len, -1)
        pair_features = self.input_projection(pair_features)
        pair_features = pair_features.view(batch_size, num_pairs, seq_len, -1)
        
        # 3. Macro feature encoding
        macro_features = self.macro_encoder.encode(
            timestamps=timestamps,
            calendar_data=macro_data.get('calendar') if macro_data else None,
            rates_data=macro_data.get('rates') if macro_data else None,
            yields_data=macro_data.get('yields') if macro_data else None,
            commodities_data=macro_data.get('commodities') if macro_data else None,
            sentiment_data=macro_data.get('sentiment') if macro_data else None,
        )
        macro_features = self.macro_projection(macro_features)  # (batch, seq, dim)
        
        # 4. Session detection
        session_features = self.session_clock.compute_session_features(timestamps)
        session_emb = self.session_projection(session_features)  # (batch, seq, dim)
        
        # Get session indicators for gating
        session_indicators = self.session_clock.detect_sessions(timestamps)
        
        # 5. Process each pair through Self-Modifying Titans
        pair_states = []
        for p in range(num_pairs):
            # Combine pair features with macro
            pair_input = torch.cat([
                pair_features[:, p, :, :],
                macro_features,
            ], dim=-1)  # (batch, seq, input_dim + macro_dim)
            
            # Process through Titans
            titans_out, aux_info = self.titans_per_pair[p](pair_input)
            
            # Take last timestep
            pair_state = titans_out[:, -1, :]  # (batch, hidden_dim)
            pair_states.append(pair_state)
        
        # Stack: (batch, num_pairs, hidden_dim)
        pair_states = torch.stack(pair_states, dim=1)
        
        # 6. Cross-Pair Memory (learn correlations)
        macro_state = macro_features[:, -1, :]  # Last timestep
        enriched_states = self.cross_pair_memory(pair_states, macro_state)
        
        # 7. Process through Continuum Memory System
        # Average across pairs for CMS input
        cms_input = enriched_states.mean(dim=1)  # (batch, hidden_dim)
        
        cms_output, level_outputs = self.continuum_memory(
            cms_input,
            return_all_levels=True
        )
        
        # 8. Session-aware frequency gating
        # Use last timestep's session indicators
        session_ind_last = session_indicators[:, -1, :]  # (batch, 6)
        gates = self.session_gate(session_ind_last)  # (batch, num_levels)
        
        # Apply gates to CMS levels
        gated_cms = self.session_gate.apply_gates(level_outputs, gates)
        
        # 9. Regime Detection from slowest CMS level
        slowest_level = level_outputs[-1]  # Slowest (macro) level
        regime_logits, regime_features = self.regime_detector(slowest_level)
        
        # 10. Fuse all pair states
        pair_states_flat = enriched_states.view(batch_size, -1)
        fused = self.pair_fusion(pair_states_flat)
        
        # Combine with CMS and regime features
        final_features = fused + gated_cms + regime_features
        
        # 11. Output heads
        outputs = self.output_heads(final_features)
        
        # Add regime prediction
        outputs['regime_logits'] = regime_logits
        
        return outputs
    
    def reset_memories(self) -> None:
        """Reset all memories (useful for online learning)"""
        for titans in self.titans_per_pair:
            titans.reset_memories()
        
        self.cross_pair_memory.reset()
        self.continuum_memory.reset()
