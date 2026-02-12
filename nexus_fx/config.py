"""
Configuration module for NEXUS-FX.

All hyperparameters and model configurations are defined here using
Python dataclasses for type safety and easy serialization.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NexusFXConfig:
    """
    Main configuration for the NEXUS-FX model.
    
    This configuration defines all hyperparameters for the nested associative
    memory architecture, including memory dimensions, update frequencies,
    and training parameters.
    """
    
    # ========== Model Dimensions ==========
    input_dim: int = 64
    """Base input dimension after feature encoding"""
    
    hidden_dim: int = 256
    """Hidden dimension for all internal representations"""
    
    num_memory_slots: int = 128
    """Number of key-value slots in each associative memory"""
    
    num_titans_layers: int = 4
    """Number of Self-Modifying Titans layers"""
    
    # ========== Continuum Memory System (CMS) ==========
    num_cms_levels: int = 4
    """Number of memory levels in the continuum (different timescales)"""
    
    cms_base_frequency: int = 1
    """Base update frequency (fastest level, updates every step)"""
    
    cms_frequency_multiplier: int = 10
    """Multiplier between adjacent memory levels (exponential scaling)"""
    
    cms_hidden_dims: Optional[List[int]] = None
    """Hidden dimensions for each CMS level (default: all use hidden_dim)"""
    
    # ========== Cross-Pair Memory ==========
    num_pairs: int = 4
    """Number of currency pairs in the dataset"""
    
    num_correlation_slots: int = 64
    """Number of slots for cross-pair correlation memory"""
    
    # ========== Session Awareness ==========
    session_embedding_dim: int = 32
    """Dimension of session embeddings (Tokyo/London/NY/Sydney)"""
    
    num_sessions: int = 4
    """Number of forex sessions"""
    
    # ========== Regime Detection ==========
    num_regimes: int = 4
    """Number of latent market regimes (trending/ranging/volatile/quiet)"""
    
    regime_hidden_dim: int = 128
    """Hidden dimension for regime detector"""
    
    # ========== Output Heads ==========
    num_direction_classes: int = 3
    """Direction prediction classes (up/neutral/down)"""
    
    predict_volatility: bool = True
    """Whether to predict future volatility"""
    
    predict_regime: bool = True
    """Whether to predict market regime"""
    
    output_confidence: bool = True
    """Whether to output confidence scores"""
    
    # ========== Training Parameters ==========
    learning_rate: float = 1e-4
    """Base learning rate for optimization"""
    
    batch_size: int = 32
    """Training batch size"""
    
    sequence_length: int = 512
    """Number of timesteps in each training sequence (5m candles)"""
    
    gradient_clip_norm: float = 1.0
    """Maximum gradient norm for clipping"""
    
    weight_decay: float = 0.01
    """L2 regularization weight"""
    
    num_epochs: int = 100
    """Number of training epochs"""
    
    warmup_steps: int = 1000
    """Learning rate warmup steps"""
    
    # ========== Optimizer Selection ==========
    optimizer_type: str = "delta_gd"
    """Optimizer type: 'delta_gd', 'multi_scale_momentum', or 'adam'"""
    
    use_dgd: bool = True
    """Whether to use Delta Gradient Descent principles"""
    
    # ========== Data Configuration ==========
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1H', '4H', '1D'])
    """Multi-timeframe resolutions to use"""
    
    pairs: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'])
    """Currency pairs to trade"""
    
    base_timeframe: str = '5m'
    """Base timeframe for data loading"""
    
    lookback_periods: int = 100
    """Number of periods to look back for feature computation"""
    
    # ========== Feature Engineering ==========
    include_technicals: bool = True
    """Include technical indicators (RSI, MACD, Bollinger, ATR, ADX)"""
    
    include_volume: bool = True
    """Include volume features (when available)"""
    
    include_macro: bool = True
    """Include macro features (rates, yields, calendar)"""
    
    include_session_features: bool = True
    """Include session detection features"""
    
    # ========== Associative Memory Parameters ==========
    memory_temperature: float = 1.0
    """Temperature for memory attention weights"""
    
    surprise_threshold: float = 0.5
    """Threshold for surprise-gated memory writing"""
    
    memory_decay: float = 0.99
    """Decay factor for memory slots"""
    
    # ========== Loss Weights ==========
    direction_loss_weight: float = 1.0
    """Weight for direction prediction loss"""
    
    volatility_loss_weight: float = 0.5
    """Weight for volatility prediction loss"""
    
    regime_loss_weight: float = 0.3
    """Weight for regime prediction loss"""
    
    calibration_loss_weight: float = 0.2
    """Weight for confidence calibration loss"""
    
    # ========== Evaluation ==========
    validation_split: float = 0.15
    """Fraction of data for validation"""
    
    test_split: float = 0.15
    """Fraction of data for testing"""
    
    # ========== Continual Learning ==========
    enable_continual_learning: bool = True
    """Enable continual learning mode"""
    
    continual_update_frequency: int = 100
    """How often to update slow memories in continual learning"""
    
    # ========== Miscellaneous ==========
    seed: int = 42
    """Random seed for reproducibility"""
    
    device: str = "cuda"
    """Device for training ('cuda' or 'cpu')"""
    
    num_workers: int = 4
    """Number of data loading workers"""
    
    log_interval: int = 100
    """How often to log training metrics"""
    
    checkpoint_interval: int = 1000
    """How often to save checkpoints"""
    
    def __post_init__(self):
        """Validate and auto-configure derived parameters"""
        if self.cms_hidden_dims is None:
            self.cms_hidden_dims = [self.hidden_dim] * self.num_cms_levels
        
        assert len(self.cms_hidden_dims) == self.num_cms_levels, \
            "cms_hidden_dims length must match num_cms_levels"
        
        assert len(self.pairs) == self.num_pairs, \
            "Number of pairs in list must match num_pairs"
        
        assert self.optimizer_type in ['delta_gd', 'multi_scale_momentum', 'adam'], \
            "Invalid optimizer type"
    
    def get_update_frequencies(self) -> List[int]:
        """
        Calculate update frequencies for each CMS level.
        
        Returns exponentially spaced update intervals:
        Level 0 (fastest): updates every 1 step
        Level 1: updates every 10 steps
        Level 2: updates every 100 steps
        Level 3 (slowest): updates every 1000 steps
        """
        frequencies = []
        for i in range(self.num_cms_levels):
            freq = self.cms_base_frequency * (self.cms_frequency_multiplier ** i)
            frequencies.append(freq)
        return frequencies
