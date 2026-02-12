"""
═══════════════════════════════════════════════════════════════════════════════
NEXUS-FX: Nested Exchange Universal Sequencer for ForeX
═══════════════════════════════════════════════════════════════════════════════

Consolidated Single-File Version for Google Colab / Kaggle

This file consolidates the entire NEXUS-FX codebase into a single importable module
for easy use in notebook environments like Google Colab and Kaggle.

NEXUS-FX is a nested associative memory architecture for forex trading that combines:
- Self-Modifying Titans for in-context adaptive learning
- Continuum Memory System for multi-timescale knowledge hierarchy
- Cross-Pair Memory for currency correlation learning  
- Session-aware frequency gating
- Regime detection
- Multi-task prediction heads

To use this file:
    1. Upload it to your Colab/Kaggle environment
    2. Import: `import nexus_fx_consolidated as nfx`
    3. Create model: `model = nfx.NEXUSFX(nfx.NexusFXConfig())`

Original repository: https://github.com/your-repo/Fx

═══════════════════════════════════════════════════════════════════════════════
"""

# ============================================================================
# IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import logging
import sys
import json
import math
import os



# ============= UTILITIES - LOGGING =============
"""
Logging utilities for NEXUS-FX.
"""



def setup_logger(name: str = 'nexus_fx', level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


class MetricsLogger:
    """
    Logs training metrics to file and/or console.
    """
    
    def __init__(self, log_file: str = 'metrics.jsonl'):
        self.log_file = log_file
        self.logger = setup_logger('metrics')
    
    def log(self, metrics: Dict[str, Any], step: int, epoch: int = 0) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Global step number
            epoch: Epoch number
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'epoch': epoch,
            **metrics
        }
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to console
        metrics_str = ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float)))
        self.logger.info(f"Step {step} - {metrics_str}")
    
    def log_summary(self, summary: str) -> None:
        """Log a summary string"""
        self.logger.info(summary)


# ============= UTILITIES - MARKET =============
"""
Utility functions for market analysis.
"""



def get_active_sessions(dt: datetime) -> List[str]:
    """
    Get list of active forex sessions for a given datetime.
    
    Args:
        dt: Datetime in UTC
    
    Returns:
        List of active session names
    """
    hour = dt.hour
    active = []
    
    # Sydney: 22:00-07:00 GMT
    if hour >= 22 or hour < 7:
        active.append('sydney')
    
    # Tokyo: 00:00-09:00 GMT
    if hour < 9:
        active.append('tokyo')
    
    # London: 08:00-17:00 GMT
    if 8 <= hour < 17:
        active.append('london')
    
    # New York: 13:00-22:00 GMT
    if 13 <= hour < 22:
        active.append('new_york')
    
    return active


def is_market_open(dt: datetime) -> bool:
    """Check if forex market is open"""
    # Forex is open 24/5
    weekday = dt.weekday()
    return weekday < 5  # Monday-Friday


def calculate_spread(pair: str, session: str = 'london') -> float:
    """
    Estimate typical spread for a currency pair.
    
    Args:
        pair: Currency pair (e.g., 'EURUSD')
        session: Trading session
    
    Returns:
        Spread in pips
    """
    # Typical spreads (in pips)
    base_spreads = {
        'EURUSD': 0.8,
        'GBPUSD': 1.0,
        'USDJPY': 0.9,
        'AUDUSD': 1.2,
    }
    
    spread = base_spreads.get(pair, 2.0)
    
    # Wider spreads during off-hours
    if session in ['sydney', 'tokyo']:
        spread *= 1.5
    
    return spread


def detect_session(hour: int) -> str:
    """
    Detect primary forex session for given hour.
    
    Args:
        hour: Hour in GMT (0-23)
    
    Returns:
        Primary session name
    """
    if 8 <= hour < 13:
        return 'london'
    elif 13 <= hour < 22:
        return 'new_york'
    elif hour < 9:
        return 'tokyo'
    else:
        return 'sydney'


# ============= CONFIGURATION =============
"""
Configuration module for NEXUS-FX.

All hyperparameters and model configurations are defined here using
Python dataclasses for type safety and easy serialization.
"""



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


# ============= DATA - PREPROCESSOR =============
"""
Preprocessor - Data normalization, alignment, and missing data handling.

Handles:
- Rolling z-score normalization (no lookahead)
- Missing data forward-fill with staleness indicators
- Cross-timeframe alignment
- Train/validation/test split (temporal order preserved)
"""



class Preprocessor:
    """
    Preprocesses forex data for training.
    
    All operations maintain temporal causality (no lookahead bias).
    
    Args:
        normalization_method: 'zscore' or 'minmax' or 'none'
        normalization_window: Rolling window for normalization
        fill_missing: Whether to forward-fill missing data
        max_staleness: Maximum staleness (in periods) before flagging
    """
    
    def __init__(
        self,
        normalization_method: str = 'zscore',
        normalization_window: int = 100,
        fill_missing: bool = True,
        max_staleness: int = 10,
    ):
        self.normalization_method = normalization_method
        self.normalization_window = normalization_window
        self.fill_missing = fill_missing
        self.max_staleness = max_staleness
        
        # Statistics for normalization (computed from training data)
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
    
    def fit(self, data: torch.Tensor) -> None:
        """
        Fit normalization parameters on training data.
        
        Args:
            data: Training data tensor (..., seq_len, features)
        """
        if self.normalization_method == 'zscore':
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True) + 1e-8
        elif self.normalization_method == 'minmax':
            self.min = data.min(dim=0, keepdim=True)[0]
            self.max = data.max(dim=0, keepdim=True)[0]
    
    def transform(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data tensor (..., seq_len, features)
        
        Returns:
            normalized_data: Normalized tensor
            staleness_flags: Binary flags for stale data (..., seq_len, features)
        """
        if self.normalization_method == 'zscore':
            if self.mean is None or self.std is None:
                raise ValueError("Must call fit() before transform()")
            normalized = (data - self.mean) / self.std
        elif self.normalization_method == 'minmax':
            if self.min is None or self.max is None:
                raise ValueError("Must call fit() before transform()")
            normalized = (data - self.min) / (self.max - self.min + 1e-8)
        else:
            normalized = data
        
        # Staleness flags (detect missing/unchanged data)
        staleness_flags = self._detect_staleness(data)
        
        return normalized, staleness_flags
    
    def fit_transform(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)
    
    def _detect_staleness(self, data: torch.Tensor) -> torch.Tensor:
        """
        Detect stale (unchanged) data points.
        
        Returns binary flags indicating staleness.
        """
        staleness = torch.zeros_like(data)
        
        # Check if data is unchanged for more than max_staleness periods
        for i in range(1, data.shape[-2]):
            unchanged = (data[..., i, :] == data[..., i-1, :]).float()
            
            # Accumulate staleness
            if i > 1:
                staleness[..., i, :] = (staleness[..., i-1, :] + 1) * unchanged
            else:
                staleness[..., i, :] = unchanged
        
        # Flag as stale if > max_staleness
        staleness_flags = (staleness > self.max_staleness).float()
        
        return staleness_flags
    
    def rolling_normalize(
        self,
        data: torch.Tensor,
        window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply rolling window normalization (no lookahead).
        
        Args:
            data: Data tensor (..., seq_len, features)
            window: Rolling window size (default: self.normalization_window)
        
        Returns:
            normalized: Rolling normalized data
        """
        if window is None:
            window = self.normalization_window
        
        normalized = torch.zeros_like(data)
        seq_len = data.shape[-2]
        
        for i in range(window, seq_len):
            # Use only past data for normalization
            window_data = data[..., i-window:i, :]
            
            if self.normalization_method == 'zscore':
                mean = window_data.mean(dim=-2, keepdim=True)
                std = window_data.std(dim=-2, keepdim=True) + 1e-8
                normalized[..., i, :] = (data[..., i, :] - mean.squeeze(-2)) / std.squeeze(-2)
            elif self.normalization_method == 'minmax':
                min_val = window_data.min(dim=-2, keepdim=True)[0]
                max_val = window_data.max(dim=-2, keepdim=True)[0]
                normalized[..., i, :] = (data[..., i, :] - min_val.squeeze(-2)) / (max_val.squeeze(-2) - min_val.squeeze(-2) + 1e-8)
            else:
                normalized[..., i, :] = data[..., i, :]
        
        return normalized
    
    def forward_fill_missing(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-fill missing (NaN) values.
        
        Args:
            data: Data tensor with potential NaNs (..., seq_len, features)
        
        Returns:
            filled_data: Data with NaNs filled
            fill_mask: Binary mask indicating which values were filled
        """
        filled = data.clone()
        fill_mask = torch.isnan(data).float()
        
        # Forward fill
        seq_len = data.shape[-2]
        for i in range(1, seq_len):
            nan_mask = torch.isnan(filled[..., i, :])
            filled[..., i, :] = torch.where(
                nan_mask,
                filled[..., i-1, :],
                filled[..., i, :]
            )
        
        # If still NaN at start, fill with zeros
        filled = torch.nan_to_num(filled, nan=0.0)
        
        return filled, fill_mask
    
    def temporal_split(
        self,
        data: torch.Tensor,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into train/val/test preserving temporal order.
        
        Args:
            data: Data tensor (..., seq_len, features)
            val_split: Fraction for validation
            test_split: Fraction for testing
        
        Returns:
            train_data, val_data, test_data
        """
        seq_len = data.shape[-2]
        
        train_end = int(seq_len * (1 - val_split - test_split))
        val_end = int(seq_len * (1 - test_split))
        
        train_data = data[..., :train_end, :]
        val_data = data[..., train_end:val_end, :]
        test_data = data[..., val_end:, :]
        
        return train_data, val_data, test_data
    
    def align_multi_timeframe(
        self,
        data_dict: Dict[str, torch.Tensor],
        base_timeframe: str = '5m',
    ) -> Dict[str, torch.Tensor]:
        """
        Align multiple timeframe tensors to base timeframe.
        
        Args:
            data_dict: Dict mapping timeframe -> tensor
            base_timeframe: Base timeframe to align to
        
        Returns:
            aligned_dict: Aligned tensors
        """
        if base_timeframe not in data_dict:
            raise ValueError(f"Base timeframe {base_timeframe} not in data_dict")
        
        base_len = data_dict[base_timeframe].shape[-2]
        aligned = {}
        
        for tf, tensor in data_dict.items():
            if tf == base_timeframe:
                aligned[tf] = tensor
            else:
                # Interpolate or repeat to match base length
                # This is a simplified version; in practice, use proper alignment
                tf_len = tensor.shape[-2]
                
                if tf_len < base_len:
                    # Repeat to match length
                    repeat_factor = base_len // tf_len + 1
                    repeated = tensor.repeat_interleave(repeat_factor, dim=-2)
                    aligned[tf] = repeated[..., :base_len, :]
                else:
                    # Downsample
                    indices = torch.linspace(0, tf_len - 1, base_len).long()
                    aligned[tf] = tensor[..., indices, :]
        
        return aligned


# ============= DATA - SESSION CLOCK =============
"""
Session Clock - Forex session detection and timing features.

Detects active trading sessions and computes session-related features:
- Sydney (22:00-07:00 GMT)
- Tokyo (00:00-09:00 GMT)
- London (08:00-17:00 GMT)
- New York (13:00-22:00 GMT)

Also tracks session overlaps which are high-volatility periods.
"""



class SessionClock:
    """
    Forex session detection and timing features.
    
    Generates features based on active trading sessions and their characteristics.
    """
    
    def __init__(self):
        # Session hours in GMT (24-hour format)
        self.sessions = {
            'sydney': (22, 7),    # 22:00-07:00 GMT
            'tokyo': (0, 9),      # 00:00-09:00 GMT
            'london': (8, 17),    # 08:00-17:00 GMT
            'new_york': (13, 22), # 13:00-22:00 GMT
        }
        
        # Session names for indexing
        self.session_names = ['sydney', 'tokyo', 'london', 'new_york']
    
    def detect_sessions(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Detect active sessions for given timestamps.
        
        Args:
            timestamps: Unix timestamps (batch, seq_len)
        
        Returns:
            session_indicators: Binary indicators (batch, seq_len, 6)
                [is_sydney, is_tokyo, is_london, is_ny, is_overlap, is_weekend]
        """
        batch_size, seq_len = timestamps.shape
        indicators = torch.zeros(batch_size, seq_len, 6)
        
        for b in range(batch_size):
            for t in range(seq_len):
                ts = timestamps[b, t].item()
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                
                hour = dt.hour
                weekday = dt.weekday()  # 0=Monday, 6=Sunday
                
                # Check each session
                is_sydney = self._is_in_session(hour, *self.sessions['sydney'])
                is_tokyo = self._is_in_session(hour, *self.sessions['tokyo'])
                is_london = self._is_in_session(hour, *self.sessions['london'])
                is_ny = self._is_in_session(hour, *self.sessions['new_york'])
                
                # Overlap detection (multiple sessions active)
                num_active = sum([is_sydney, is_tokyo, is_london, is_ny])
                is_overlap = float(num_active > 1)
                
                # Weekend detection
                is_weekend = float(weekday >= 5)  # Saturday or Sunday
                
                indicators[b, t, 0] = float(is_sydney)
                indicators[b, t, 1] = float(is_tokyo)
                indicators[b, t, 2] = float(is_london)
                indicators[b, t, 3] = float(is_ny)
                indicators[b, t, 4] = is_overlap
                indicators[b, t, 5] = is_weekend
        
        return indicators
    
    def _is_in_session(self, hour: int, start: int, end: int) -> bool:
        """Check if hour is within session"""
        if start < end:
            return start <= hour < end
        else:
            # Session crosses midnight (e.g., Sydney)
            return hour >= start or hour < end
    
    def compute_session_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute rich session features.
        
        Args:
            timestamps: Unix timestamps (batch, seq_len)
        
        Returns:
            features: Session features (batch, seq_len, feature_dim)
                - Session indicators (6 binary)
                - Time to session open/close (2 continuous)
                - Session volatility profile (4 continuous, one per session)
                - Day of week encoding (7 one-hot)
        """
        batch_size, seq_len = timestamps.shape
        
        # Session indicators
        session_indicators = self.detect_sessions(timestamps)
        
        # Time to next session change
        time_features = self._compute_time_features(timestamps)
        
        # Session volatility profiles (known characteristics)
        vol_profiles = self._get_session_volatility_profiles(session_indicators)
        
        # Day of week
        dow_features = self._encode_day_of_week(timestamps)
        
        # Concatenate all features
        features = torch.cat([
            session_indicators,  # 6
            time_features,       # 2
            vol_profiles,        # 4
            dow_features,        # 7
        ], dim=-1)
        
        return features
    
    def _compute_time_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute time-to-event features.
        
        Returns:
            time_features: (batch, seq_len, 2)
                - Hours to next session open
                - Hours to next session close
        """
        batch_size, seq_len = timestamps.shape
        time_features = torch.zeros(batch_size, seq_len, 2)
        
        for b in range(batch_size):
            for t in range(seq_len):
                ts = timestamps[b, t].item()
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                hour = dt.hour
                minute = dt.minute
                
                # Find next session transition
                # Simplified: distance to London open (most important) and NY close
                hours_to_london = (8 - hour) % 24
                hours_to_ny_close = (22 - hour) % 24
                
                time_features[b, t, 0] = hours_to_london + minute / 60
                time_features[b, t, 1] = hours_to_ny_close + minute / 60
        
        return time_features
    
    def _get_session_volatility_profiles(self, session_indicators: torch.Tensor) -> torch.Tensor:
        """
        Encode known volatility characteristics of each session.
        
        Historical volatility patterns:
        - Sydney: Low (0.3)
        - Tokyo: Medium-Low (0.5)
        - London: High (0.9)
        - New York: Very High (1.0)
        """
        batch_size, seq_len, _ = session_indicators.shape
        vol_profiles = torch.zeros(batch_size, seq_len, 4)
        
        # Volatility weights
        vol_weights = torch.tensor([0.3, 0.5, 0.9, 1.0])
        
        # Apply to active sessions
        vol_profiles = session_indicators[:, :, :4] * vol_weights.unsqueeze(0).unsqueeze(0)
        
        return vol_profiles
    
    def _encode_day_of_week(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        One-hot encoding of day of week.
        
        Monday=0, ..., Sunday=6
        """
        batch_size, seq_len = timestamps.shape
        dow_features = torch.zeros(batch_size, seq_len, 7)
        
        for b in range(batch_size):
            for t in range(seq_len):
                ts = timestamps[b, t].item()
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                weekday = dt.weekday()
                dow_features[b, t, weekday] = 1.0
        
        return dow_features
    
    def get_session_embedding(self, session_indicators: torch.Tensor) -> torch.Tensor:
        """
        Convert session indicators to learned embedding.
        
        This is a simple weighted sum; in practice, use a learned embedding layer.
        
        Args:
            session_indicators: (batch, seq_len, 6)
        
        Returns:
            session_embedding: (batch, seq_len, embedding_dim)
        """
        # Simple weighted combination as a placeholder
        # In the full model, this would be a learned embedding
        batch_size, seq_len, _ = session_indicators.shape
        
        # Weight matrix (6 sessions → 32 dim embedding)
        # This is a simplified version; use nn.Linear in practice
        embedding_dim = 32
        weights = torch.randn(6, embedding_dim) * 0.1
        
        session_embedding = torch.matmul(session_indicators, weights)
        
        return session_embedding


# ============= DATA - FEATURE ENGINE =============
"""
Feature Engine - Technical indicator computation.

Computes technical indicators from OHLC data without lookahead bias.
All indicators use only past data to ensure realistic backtesting.

Technical indicators computed:
- Returns (simple and log)
- Realized volatility (multiple estimators)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- ADX (Average Directional Index)
- Volume features (when available)
"""



class FeatureEngine:
    """
    Technical feature engineering for forex data.
    
    All computations are vectorized and avoid lookahead bias.
    
    Args:
        lookback_periods: Number of periods for rolling calculations
        include_volume: Whether to compute volume features
    """
    
    def __init__(
        self,
        lookback_periods: int = 100,
        include_volume: bool = True,
    ):
        self.lookback_periods = lookback_periods
        self.include_volume = include_volume
    
    def compute_features(
        self,
        ohlc: torch.Tensor,
        volume: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute all technical features from OHLC.
        
        Args:
            ohlc: OHLC tensor (..., seq_len, 4) [open, high, low, close]
            volume: Volume tensor (..., seq_len) optional
        
        Returns:
            features: Feature tensor (..., seq_len, num_features)
        """
        features = []
        
        # Extract OHLC components
        open_price = ohlc[..., 0]
        high_price = ohlc[..., 1]
        low_price = ohlc[..., 2]
        close_price = ohlc[..., 3]
        
        # 1. Returns
        returns = self.compute_returns(close_price)
        log_returns = self.compute_log_returns(close_price)
        features.extend([returns, log_returns])
        
        # 2. Volatility estimators
        realized_vol = self.compute_realized_volatility(returns)
        parkinson_vol = self.compute_parkinson_volatility(high_price, low_price)
        garman_klass_vol = self.compute_garman_klass_volatility(
            open_price, high_price, low_price, close_price
        )
        features.extend([realized_vol, parkinson_vol, garman_klass_vol])
        
        # 3. RSI
        rsi = self.compute_rsi(close_price)
        features.append(rsi)
        
        # 4. MACD
        macd, signal, histogram = self.compute_macd(close_price)
        features.extend([macd, signal, histogram])
        
        # 5. Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width, bb_position = self.compute_bollinger_bands(close_price)
        features.extend([bb_upper, bb_middle, bb_lower, bb_width, bb_position])
        
        # 6. ATR
        atr = self.compute_atr(high_price, low_price, close_price)
        features.append(atr)
        
        # 7. ADX
        adx = self.compute_adx(high_price, low_price, close_price)
        features.append(adx)
        
        # 8. Price momentum
        momentum = self.compute_momentum(close_price, periods=[5, 10, 20])
        features.extend(momentum)
        
        # 9. Volume features (if available)
        if volume is not None and self.include_volume:
            volume_features = self.compute_volume_features(volume, close_price)
            features.extend(volume_features)
        
        # Stack all features
        features_tensor = torch.stack(features, dim=-1)
        
        return features_tensor
    
    def compute_returns(self, prices: torch.Tensor) -> torch.Tensor:
        """Simple returns: (p_t - p_{t-1}) / p_{t-1}"""
        returns = torch.zeros_like(prices)
        returns[..., 1:] = (prices[..., 1:] - prices[..., :-1]) / (prices[..., :-1] + 1e-8)
        return returns
    
    def compute_log_returns(self, prices: torch.Tensor) -> torch.Tensor:
        """Log returns: log(p_t / p_{t-1})"""
        log_returns = torch.zeros_like(prices)
        log_returns[..., 1:] = torch.log((prices[..., 1:] + 1e-8) / (prices[..., :-1] + 1e-8))
        return log_returns
    
    def compute_realized_volatility(
        self,
        returns: torch.Tensor,
        window: int = 20,
    ) -> torch.Tensor:
        """Rolling standard deviation of returns"""
        vol = torch.zeros_like(returns)
        
        # Use unfold for efficient rolling window
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)
        
        batch_shape = returns.shape[:-1]
        seq_len = returns.shape[-1]
        
        for i in range(window, seq_len):
            window_data = returns[..., i-window:i]
            vol[..., i] = window_data.std(dim=-1)
        
        return vol
    
    def compute_parkinson_volatility(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        window: int = 20,
    ) -> torch.Tensor:
        """
        Parkinson volatility estimator (uses high-low range).
        More efficient than close-to-close volatility.
        """
        hl_ratio = torch.log((high + 1e-8) / (low + 1e-8))
        vol = torch.zeros_like(high)
        
        seq_len = high.shape[-1]
        for i in range(window, seq_len):
            window_data = hl_ratio[..., i-window:i]
            vol[..., i] = torch.sqrt((window_data ** 2).mean(dim=-1) / (4 * np.log(2)))
        
        return vol
    
    def compute_garman_klass_volatility(
        self,
        open_price: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        close: torch.Tensor,
        window: int = 20,
    ) -> torch.Tensor:
        """Garman-Klass volatility estimator (uses OHLC)"""
        hl = torch.log((high + 1e-8) / (low + 1e-8)) ** 2
        co = torch.log((close + 1e-8) / (open_price + 1e-8)) ** 2
        
        vol = torch.zeros_like(high)
        seq_len = high.shape[-1]
        
        for i in range(window, seq_len):
            hl_window = hl[..., i-window:i]
            co_window = co[..., i-window:i]
            vol[..., i] = torch.sqrt(0.5 * hl_window.mean(dim=-1) - (2 * np.log(2) - 1) * co_window.mean(dim=-1))
        
        return vol
    
    def compute_rsi(self, prices: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Relative Strength Index"""
        deltas = torch.zeros_like(prices)
        deltas[..., 1:] = prices[..., 1:] - prices[..., :-1]
        
        gains = torch.clamp(deltas, min=0)
        losses = torch.clamp(-deltas, min=0)
        
        rsi = torch.zeros_like(prices)
        seq_len = prices.shape[-1]
        
        for i in range(period, seq_len):
            avg_gain = gains[..., i-period:i].mean(dim=-1)
            avg_loss = losses[..., i-period:i].mean(dim=-1)
            
            rs = (avg_gain + 1e-8) / (avg_loss + 1e-8)
            rsi[..., i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_macd(
        self,
        prices: torch.Tensor,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MACD indicator"""
        # Exponential moving averages
        ema_fast = self.compute_ema(prices, fast_period)
        ema_slow = self.compute_ema(prices, slow_period)
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal = self.compute_ema(macd, signal_period)
        
        # Histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def compute_ema(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = torch.zeros_like(prices)
        ema[..., 0] = prices[..., 0]
        
        seq_len = prices.shape[-1]
        for i in range(1, seq_len):
            ema[..., i] = alpha * prices[..., i] + (1 - alpha) * ema[..., i-1]
        
        return ema
    
    def compute_bollinger_bands(
        self,
        prices: torch.Tensor,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bollinger Bands"""
        middle = torch.zeros_like(prices)
        std = torch.zeros_like(prices)
        
        seq_len = prices.shape[-1]
        for i in range(period, seq_len):
            window = prices[..., i-period:i]
            middle[..., i] = window.mean(dim=-1)
            std[..., i] = window.std(dim=-1)
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        width = upper - lower
        
        # Band position: where price is within bands (0 = lower, 1 = upper)
        position = (prices - lower) / (width + 1e-8)
        
        return upper, middle, lower, width, position
    
    def compute_atr(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        close: torch.Tensor,
        period: int = 14,
    ) -> torch.Tensor:
        """Average True Range"""
        # True range
        tr = torch.zeros_like(high)
        tr[..., 0] = high[..., 0] - low[..., 0]
        
        seq_len = high.shape[-1]
        for i in range(1, seq_len):
            tr[..., i] = torch.max(
                torch.stack([
                    high[..., i] - low[..., i],
                    torch.abs(high[..., i] - close[..., i-1]),
                    torch.abs(low[..., i] - close[..., i-1]),
                ], dim=0),
                dim=0
            )[0]
        
        # ATR is EMA of true range
        atr = self.compute_ema(tr, period)
        
        return atr
    
    def compute_adx(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        close: torch.Tensor,
        period: int = 14,
    ) -> torch.Tensor:
        """Average Directional Index (trend strength)"""
        # Simplified ADX calculation
        # In practice, would compute +DI, -DI, and DX
        
        # Use ATR as a proxy for now (simplified)
        atr = self.compute_atr(high, low, close, period)
        
        # Normalize to 0-100 range
        adx = 100 * torch.sigmoid(atr)
        
        return adx
    
    def compute_momentum(
        self,
        prices: torch.Tensor,
        periods: list = [5, 10, 20],
    ) -> list:
        """Price momentum over different periods"""
        momentum_features = []
        
        for period in periods:
            mom = torch.zeros_like(prices)
            mom[..., period:] = (prices[..., period:] - prices[..., :-period]) / (prices[..., :-period] + 1e-8)
            momentum_features.append(mom)
        
        return momentum_features
    
    def compute_volume_features(
        self,
        volume: torch.Tensor,
        prices: torch.Tensor,
        window: int = 20,
    ) -> list:
        """Volume-based features"""
        features = []
        
        # Volume moving average
        vol_ma = torch.zeros_like(volume)
        seq_len = volume.shape[-1]
        
        for i in range(window, seq_len):
            vol_ma[..., i] = volume[..., i-window:i].mean(dim=-1)
        
        # Relative volume
        rel_vol = volume / (vol_ma + 1e-8)
        
        # Volume-weighted price
        vwap = torch.zeros_like(prices)
        for i in range(window, seq_len):
            price_window = prices[..., i-window:i]
            vol_window = volume[..., i-window:i]
            vwap[..., i] = (price_window * vol_window).sum(dim=-1) / (vol_window.sum(dim=-1) + 1e-8)
        
        features.extend([vol_ma, rel_vol, vwap])
        
        return features


# ============= DATA - MACRO FEATURES =============
"""
Macro Feature Encoding - Economic calendar, rates, yields, commodities.

Encodes macro-fundamental data that affects forex markets:
- Economic calendar events (NFP, CPI, rate decisions)
- Interest rates (Fed, ECB, BoJ, RBA, BoE)
- Bond yields (US10Y, EU10Y, JP10Y, AU10Y)
- Commodities (Gold, Oil, DXY)
- Sentiment proxies (VIX, risk-on/risk-off)
"""



class MacroFeatureEncoder:
    """
    Encodes macro-fundamental features for forex trading.
    
    Works with limited data availability:
    - Economic calendar: time-to-event, expected/actual/previous values
    - Interest rates: current rates and differentials
    - Bond yields: current yields and spreads
    - Commodities: current prices
    - Sentiment: current VIX level, risk-on/off classification
    
    Args:
        pairs: List of currency pairs
        include_calendar: Whether to include economic calendar events
        include_rates: Whether to include interest rates
        include_yields: Whether to include bond yields
        include_commodities: Whether to include commodity prices
        include_sentiment: Whether to include sentiment indicators
    """
    
    def __init__(
        self,
        pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        include_calendar: bool = True,
        include_rates: bool = True,
        include_yields: bool = True,
        include_commodities: bool = True,
        include_sentiment: bool = True,
    ):
        self.pairs = pairs
        self.include_calendar = include_calendar
        self.include_rates = include_rates
        self.include_yields = include_yields
        self.include_commodities = include_commodities
        self.include_sentiment = include_sentiment
        
        # Extract currencies from pairs
        self.currencies = self._extract_currencies()
        
        # Feature dimension
        self.feature_dim = self._calculate_feature_dim()
    
    def _extract_currencies(self) -> set:
        """Extract unique currencies from pairs"""
        currencies = set()
        for pair in self.pairs:
            if len(pair) == 6:
                currencies.add(pair[:3])
                currencies.add(pair[3:])
        return currencies
    
    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension"""
        dim = 0
        
        if self.include_calendar:
            dim += 10  # Calendar event encoding
        
        if self.include_rates:
            dim += len(self.currencies) + len(self.pairs)  # Rates + differentials
        
        if self.include_yields:
            dim += len(self.currencies) + len(self.pairs)  # Yields + spreads
        
        if self.include_commodities:
            dim += 3  # Gold, Oil, DXY
        
        if self.include_sentiment:
            dim += 2  # VIX level, risk-on/off
        
        return dim
    
    def encode(
        self,
        timestamps: torch.Tensor,
        calendar_data: Optional[pd.DataFrame] = None,
        rates_data: Optional[Dict[str, float]] = None,
        yields_data: Optional[Dict[str, float]] = None,
        commodities_data: Optional[Dict[str, float]] = None,
        sentiment_data: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Encode macro features for given timestamps.
        
        Args:
            timestamps: Timestamps to encode for (batch, seq_len)
            calendar_data: DataFrame with economic events
            rates_data: Dict of current interest rates by currency
            yields_data: Dict of current bond yields by currency
            commodities_data: Dict of commodity prices
            sentiment_data: Dict of sentiment indicators
        
        Returns:
            macro_features: Encoded features (batch, seq_len, feature_dim)
        """
        batch_size, seq_len = timestamps.shape
        features = []
        
        # 1. Economic Calendar
        if self.include_calendar:
            calendar_features = self._encode_calendar(timestamps, calendar_data)
            features.append(calendar_features)
        
        # 2. Interest Rates
        if self.include_rates:
            rates_features = self._encode_rates(timestamps, rates_data)
            features.append(rates_features)
        
        # 3. Bond Yields
        if self.include_yields:
            yields_features = self._encode_yields(timestamps, yields_data)
            features.append(yields_features)
        
        # 4. Commodities
        if self.include_commodities:
            commodity_features = self._encode_commodities(timestamps, commodities_data)
            features.append(commodity_features)
        
        # 5. Sentiment
        if self.include_sentiment:
            sentiment_features = self._encode_sentiment(timestamps, sentiment_data)
            features.append(sentiment_features)
        
        # Concatenate all features
        if features:
            macro_features = torch.cat(features, dim=-1)
        else:
            macro_features = torch.zeros(batch_size, seq_len, 1)
        
        return macro_features
    
    def _encode_calendar(
        self,
        timestamps: torch.Tensor,
        calendar_data: Optional[pd.DataFrame],
    ) -> torch.Tensor:
        """
        Encode economic calendar events.
        
        Features:
        - Time to next major event (hours)
        - Event importance (0-3: low/medium/high/critical)
        - Expected impact direction (- to +)
        - Surprise factor (actual - expected, normalized)
        - Event type encoding (one-hot for NFP/CPI/Rate/GDP/Other)
        """
        batch_size, seq_len = timestamps.shape
        
        # Default: no events
        features = torch.zeros(batch_size, seq_len, 10)
        
        if calendar_data is not None and len(calendar_data) > 0:
            # Convert timestamps to datetime
            for b in range(batch_size):
                for t in range(seq_len):
                    ts = timestamps[b, t].item()
                    dt = datetime.fromtimestamp(ts)
                    
                    # Find next event
                    future_events = calendar_data[calendar_data['timestamp'] > dt]
                    if len(future_events) > 0:
                        next_event = future_events.iloc[0]
                        time_to_event = (next_event['timestamp'] - dt).total_seconds() / 3600
                        
                        features[b, t, 0] = min(time_to_event / 24, 10)  # Days to event, capped at 10
                        features[b, t, 1] = next_event.get('importance', 1) / 3  # Normalized
                        features[b, t, 2] = next_event.get('expected_direction', 0)
                        features[b, t, 3] = next_event.get('surprise', 0)
                        
                        # Event type one-hot
                        event_type = next_event.get('type', 'Other')
                        type_idx = {'NFP': 4, 'CPI': 5, 'Rate': 6, 'GDP': 7, 'Other': 8}.get(event_type, 8)
                        features[b, t, type_idx] = 1.0
        
        return features
    
    def _encode_rates(
        self,
        timestamps: torch.Tensor,
        rates_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode interest rates and differentials.
        
        Features:
        - Current rate for each currency
        - Rate differential for each pair
        """
        batch_size, seq_len = timestamps.shape
        
        # Map currencies to rates
        currency_map = {
            'USD': 'FED',
            'EUR': 'ECB',
            'GBP': 'BOE',
            'JPY': 'BOJ',
            'AUD': 'RBA',
        }
        
        num_currencies = len(self.currencies)
        num_pairs = len(self.pairs)
        
        features = torch.zeros(batch_size, seq_len, num_currencies + num_pairs)
        
        if rates_data is not None:
            # Currency rates
            for i, currency in enumerate(sorted(self.currencies)):
                rate_key = currency_map.get(currency, currency)
                rate = rates_data.get(rate_key, 0.0)
                features[:, :, i] = rate
            
            # Pair differentials
            for i, pair in enumerate(self.pairs):
                if len(pair) == 6:
                    base_curr = pair[:3]
                    quote_curr = pair[3:]
                    
                    base_rate = rates_data.get(currency_map.get(base_curr, base_curr), 0.0)
                    quote_rate = rates_data.get(currency_map.get(quote_curr, quote_curr), 0.0)
                    
                    differential = base_rate - quote_rate
                    features[:, :, num_currencies + i] = differential
        
        return features
    
    def _encode_yields(
        self,
        timestamps: torch.Tensor,
        yields_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode bond yields and spreads.
        
        Similar to rates encoding.
        """
        batch_size, seq_len = timestamps.shape
        
        yield_map = {
            'USD': 'US10Y',
            'EUR': 'EU10Y',
            'GBP': 'UK10Y',
            'JPY': 'JP10Y',
            'AUD': 'AU10Y',
        }
        
        num_currencies = len(self.currencies)
        num_pairs = len(self.pairs)
        
        features = torch.zeros(batch_size, seq_len, num_currencies + num_pairs)
        
        if yields_data is not None:
            # Currency yields
            for i, currency in enumerate(sorted(self.currencies)):
                yield_key = yield_map.get(currency, currency)
                yield_val = yields_data.get(yield_key, 0.0)
                features[:, :, i] = yield_val
            
            # Pair spreads
            for i, pair in enumerate(self.pairs):
                if len(pair) == 6:
                    base_curr = pair[:3]
                    quote_curr = pair[3:]
                    
                    base_yield = yields_data.get(yield_map.get(base_curr, base_curr), 0.0)
                    quote_yield = yields_data.get(yield_map.get(quote_curr, quote_curr), 0.0)
                    
                    spread = base_yield - quote_yield
                    features[:, :, num_currencies + i] = spread
        
        return features
    
    def _encode_commodities(
        self,
        timestamps: torch.Tensor,
        commodities_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode commodity prices.
        
        Features:
        - Gold (safe haven)
        - Oil (WTI or Brent)
        - DXY (US Dollar Index)
        """
        batch_size, seq_len = timestamps.shape
        features = torch.zeros(batch_size, seq_len, 3)
        
        if commodities_data is not None:
            features[:, :, 0] = commodities_data.get('Gold', 0.0) / 2000  # Normalize
            features[:, :, 1] = commodities_data.get('Oil', 0.0) / 100    # Normalize
            features[:, :, 2] = commodities_data.get('DXY', 0.0) / 100    # Normalize
        
        return features
    
    def _encode_sentiment(
        self,
        timestamps: torch.Tensor,
        sentiment_data: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        """
        Encode sentiment indicators.
        
        Features:
        - VIX level (volatility index)
        - Risk-on/risk-off classification
        """
        batch_size, seq_len = timestamps.shape
        features = torch.zeros(batch_size, seq_len, 2)
        
        if sentiment_data is not None:
            vix = sentiment_data.get('VIX', 15.0)
            features[:, :, 0] = vix / 50  # Normalize
            
            # Risk-on/off: -1 (risk-off) to +1 (risk-on)
            risk_sentiment = sentiment_data.get('risk_sentiment', 0.0)
            features[:, :, 1] = risk_sentiment
        
        return features


# ============= DATA - FOREX DATASET =============
"""
Forex Dataset - Multi-timeframe OHLC data loading and management.

This dataset handles:
- Loading 5-minute OHLC candles for multiple currency pairs
- Aggregating to multiple timeframes (15m, 1H, 4H, 1D)
- Temporal alignment across timeframes
- Streaming/online mode for live inference
- No lookahead bias in all operations
"""



class ForexDataset(Dataset):
    """
    Multi-timeframe forex dataset.
    
    Loads OHLC data at base timeframe and provides aligned multi-timeframe views.
    Designed to work with limited data: OHLC, volume (optional), timestamp.
    
    Args:
        data_path: Path to CSV files or DataFrame dict
        pairs: List of currency pair symbols
        base_timeframe: Base timeframe for data (e.g., '5m')
        target_timeframes: List of target timeframes to aggregate
        sequence_length: Length of sequences to return
        stride: Stride for sequence sampling
        mode: 'train', 'val', or 'test'
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        base_timeframe: str = '5m',
        target_timeframes: List[str] = ['5m', '15m', '1H', '4H', '1D'],
        sequence_length: int = 512,
        stride: int = 1,
        mode: str = 'train',
    ):
        super().__init__()
        
        self.pairs = pairs
        self.base_timeframe = base_timeframe
        self.target_timeframes = target_timeframes
        self.sequence_length = sequence_length
        self.stride = stride
        self.mode = mode
        
        # Load data
        if data_dict is not None:
            self.data = data_dict
        elif data_path is not None:
            self.data = self._load_from_path(data_path)
        else:
            # Generate synthetic data for demo/testing
            self.data = self._generate_synthetic_data()
        
        # Preprocess and align data
        self.aligned_data = self._preprocess_and_align()
        
        # Calculate valid indices for sequence extraction
        self.valid_indices = self._calculate_valid_indices()
        
    def _load_from_path(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load OHLC data from CSV files.
        
        Expected format: one CSV per pair with columns:
        [timestamp, open, high, low, close, volume (optional)]
        """
        data = {}
        data_path = Path(data_path)
        
        for pair in self.pairs:
            csv_path = data_path / f"{pair}_{self.base_timeframe}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                data[pair] = df
            else:
                print(f"Warning: {csv_path} not found, using synthetic data for {pair}")
                data[pair] = self._generate_pair_data(pair)
        
        return data
    
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic OHLC data for testing"""
        data = {}
        for pair in self.pairs:
            data[pair] = self._generate_pair_data(pair)
        return data
    
    def _generate_pair_data(self, pair: str, num_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic OHLC for one pair"""
        # Start from a base price
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.3000,
            'USDJPY': 110.00,
            'AUDUSD': 0.7500,
        }
        base_price = base_prices.get(pair, 1.0)
        
        # Generate random walk
        returns = np.random.randn(num_samples) * 0.0001  # Small returns
        prices = base_price * (1 + returns).cumprod()
        
        # Generate OHLC from prices (simplified)
        noise = np.random.randn(num_samples, 3) * base_price * 0.0002
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(noise[:, 0]),
            'low': prices - np.abs(noise[:, 1]),
            'close': prices + noise[:, 2],
            'volume': np.random.randint(100, 1000, num_samples),
        })
        
        # Add timestamp (5-minute intervals)
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        df['timestamp'] = pd.date_range(start=start_time, periods=num_samples, freq='5min')
        df = df.set_index('timestamp')
        
        return df
    
    def _preprocess_and_align(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Preprocess data and create aligned multi-timeframe views.
        
        Returns:
            aligned_data: Dict[pair][timeframe] -> DataFrame
        """
        aligned = {}
        
        for pair in self.pairs:
            aligned[pair] = {}
            base_df = self.data[pair].copy()
            
            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close']
            for col in required:
                assert col in base_df.columns, f"Missing column {col} in {pair}"
            
            # Store base timeframe
            aligned[pair][self.base_timeframe] = base_df
            
            # Aggregate to other timeframes
            for tf in self.target_timeframes:
                if tf == self.base_timeframe:
                    continue
                aligned[pair][tf] = self._aggregate_timeframe(base_df, tf)
        
        return aligned
    
    def _aggregate_timeframe(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Aggregate OHLC to target timeframe.
        
        Uses pandas resample with OHLC aggregation rules.
        """
        # Map timeframe strings to pandas freq
        tf_map = {
            '5m': '5min',
            '15m': '15min',
            '1H': '1h',
            '4H': '4h',
            '1D': '1D',
            '1W': '1W',
        }
        
        freq = tf_map.get(target_tf, target_tf)
        
        # Resample with OHLC rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }
        
        # Add volume if present
        if 'volume' in df.columns:
            agg_rules['volume'] = 'sum'
        
        resampled = df.resample(freq).agg(agg_rules).dropna()
        
        return resampled
    
    def _calculate_valid_indices(self) -> List[int]:
        """
        Calculate valid starting indices for sequences.
        
        Valid index = enough history before it for sequence_length.
        """
        # Use the base timeframe of the first pair to determine valid indices
        first_pair = self.pairs[0]
        base_df = self.aligned_data[first_pair][self.base_timeframe]
        
        max_idx = len(base_df) - self.sequence_length
        valid_indices = list(range(0, max_idx, self.stride))
        
        return valid_indices
    
    def __len__(self) -> int:
        """Number of valid sequences"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            sample: Dictionary containing:
                - ohlc: Multi-pair, multi-timeframe OHLC (pairs, timeframes, seq, 4)
                - volume: Volume data if available (pairs, timeframes, seq)
                - timestamps: Unix timestamps (seq,)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract OHLC for all pairs and timeframes
        ohlc_data = []
        volume_data = []
        
        for pair in self.pairs:
            pair_ohlc = []
            pair_volume = []
            
            for tf in self.target_timeframes:
                df = self.aligned_data[pair][tf]
                
                # Get corresponding indices in this timeframe
                # For higher timeframes, we need to find the aligned subset
                base_df = self.aligned_data[pair][self.base_timeframe]
                start_time = base_df.index[start_idx]
                end_time = base_df.index[end_idx - 1]
                
                # Extract data in this time range
                tf_data = df.loc[start_time:end_time]
                
                # Ensure we have enough data
                if len(tf_data) == 0:
                    # Fill with last available data
                    tf_data = df.iloc[-1:].copy()
                
                # Extract OHLC
                ohlc_array = tf_data[['open', 'high', 'low', 'close']].values
                
                # Pad or truncate to expected length
                # For higher timeframes, we'll have fewer samples
                expected_len = self._get_expected_length(tf)
                ohlc_array = self._pad_or_truncate(ohlc_array, expected_len)
                
                pair_ohlc.append(ohlc_array)
                
                # Volume
                if 'volume' in tf_data.columns:
                    vol_array = tf_data['volume'].values
                    vol_array = self._pad_or_truncate(vol_array, expected_len)
                    pair_volume.append(vol_array)
                else:
                    pair_volume.append(np.zeros(expected_len))
            
            ohlc_data.append(pair_ohlc)
            volume_data.append(pair_volume)
        
        # Convert to tensors
        ohlc_tensor = torch.tensor(ohlc_data, dtype=torch.float32)
        volume_tensor = torch.tensor(volume_data, dtype=torch.float32)
        
        # Get timestamps from base timeframe
        base_df = self.aligned_data[self.pairs[0]][self.base_timeframe]
        timestamps = base_df.index[start_idx:end_idx].astype(np.int64) // 10**9  # Unix timestamp
        timestamps_tensor = torch.tensor(timestamps.values, dtype=torch.long)
        
        return {
            'ohlc': ohlc_tensor,
            'volume': volume_tensor,
            'timestamps': timestamps_tensor,
        }
    
    def _get_expected_length(self, timeframe: str) -> int:
        """Calculate expected sequence length for a timeframe"""
        # Ratio relative to base timeframe
        ratios = {
            '5m': 1,
            '15m': 3,
            '1H': 12,
            '4H': 48,
            '1D': 288,
            '1W': 288 * 5,
        }
        
        base_ratio = ratios.get(self.base_timeframe, 1)
        tf_ratio = ratios.get(timeframe, 1)
        
        # Calculate expected length based on ratio
        if tf_ratio >= base_ratio:
            expected = max(1, self.sequence_length // (tf_ratio // base_ratio))
        else:
            expected = self.sequence_length
        
        return expected
    
    def _pad_or_truncate(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate array to target length"""
        if len(array) >= target_length:
            return array[:target_length]
        else:
            # Pad by repeating last value
            if len(array) == 0:
                return np.zeros((target_length,) + array.shape[1:])
            pad_length = target_length - len(array)
            if array.ndim == 1:
                padding = np.repeat(array[-1], pad_length)
            else:
                padding = np.repeat(array[-1:], pad_length, axis=0)
            return np.concatenate([array, padding], axis=0)


# ============= MODEL - ASSOCIATIVE MEMORY =============
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


# ============= MODEL - CONTINUUM MEMORY =============
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


# ============= MODEL - CROSS-PAIR MEMORY =============
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


# ============= MODEL - REGIME DETECTOR =============
"""
Regime Detector - Latent market state classification.

Detects whether the market is in trending, ranging, volatile, or quiet regimes.
Uses the slow CMS blocks as input, since regime = slow macro knowledge.

Regime detection feeds back into the model to modulate predictions.
"""



class RegimeDetector(nn.Module):
    """
    Detects latent market regimes from slow memory blocks.
    
    Regimes:
    0: Trending (directional movement, follow momentum)
    1: Ranging (mean reversion, fade extremes)
    2: Volatile (high uncertainty, reduce position size)
    3: Quiet (low volume, widen stops, reduce frequency)
    
    The regime is detected from the slowest CMS level, which contains
    the most persistent macro knowledge.
    
    Args:
        input_dim: Input dimension (from slow CMS level)
        hidden_dim: Hidden dimension for regime classifier
        num_regimes: Number of regime classes
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_regimes: int = 4,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        
        # Regime feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Regime classifier head
        self.classifier = nn.Linear(hidden_dim, num_regimes)
        
        # Regime embedding (for feeding back to the model)
        self.regime_embeddings = nn.Embedding(num_regimes, hidden_dim)
        
    def forward(
        self,
        slow_memory_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect market regime.
        
        Args:
            slow_memory_state: Output from slowest CMS level (batch, input_dim)
        
        Returns:
            regime_logits: Logits for each regime class (batch, num_regimes)
            regime_features: Regime feature embedding (batch, hidden_dim)
        """
        # Extract features
        regime_features = self.feature_extractor(slow_memory_state)
        
        # Classify regime
        regime_logits = self.classifier(regime_features)
        
        return regime_logits, regime_features
    
    def get_regime_embedding(
        self,
        regime_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get soft regime embedding based on regime probabilities.
        
        Args:
            regime_probs: Probabilities for each regime (batch, num_regimes)
        
        Returns:
            regime_emb: Soft regime embedding (batch, hidden_dim)
        """
        # Get all regime embeddings
        all_embeddings = self.regime_embeddings.weight  # (num_regimes, hidden_dim)
        
        # Soft combination based on probabilities
        regime_emb = torch.matmul(regime_probs, all_embeddings)  # (batch, hidden_dim)
        
        return regime_emb


# ============= MODEL - SESSION GATE =============
"""
Session-Aware Frequency Gate - Adaptive memory activation.

Forex markets exhibit session-dependent dynamics. During London-NY overlap,
volatility spikes and fast memories should dominate. During Asian session,
slow memories better capture the ranging behavior.

This module gates memory levels based on active trading sessions.
"""



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


# ============= MODEL - SELF-MODIFYING TITANS =============
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


# ============= MODEL - OUTPUT HEADS =============
"""
Output Heads - Multi-task prediction outputs.

NEXUS-FX predicts multiple targets simultaneously:
1. Direction (up/neutral/down classification)
2. Volatility (regression on future realized volatility)
3. Regime (current regime classification)
4. Confidence (uncertainty quantification)

Multi-task learning improves generalization and provides richer signals
for trading decisions.
"""



class OutputHeads(nn.Module):
    """
    Multi-task prediction heads for NEXUS-FX.
    
    Args:
        input_dim: Dimension of input features from backbone
        num_direction_classes: Number of direction classes (default: 3 for up/neutral/down)
        predict_volatility: Whether to predict volatility
        predict_regime: Whether to predict regime
        output_confidence: Whether to output confidence scores
        num_regimes: Number of regime classes
    """
    
    def __init__(
        self,
        input_dim: int,
        num_direction_classes: int = 3,
        predict_volatility: bool = True,
        predict_regime: bool = True,
        output_confidence: bool = True,
        num_regimes: int = 4,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_direction_classes = num_direction_classes
        self.predict_volatility = predict_volatility
        self.predict_regime = predict_regime
        self.output_confidence = output_confidence
        self.num_regimes = num_regimes
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )
        
        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_direction_classes),
        )
        
        # Volatility prediction head (regression)
        if predict_volatility:
            self.volatility_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
                nn.Softplus(),  # Ensure positive volatility predictions
            )
        
        # Regime prediction head
        if predict_regime:
            self.regime_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, num_regimes),
            )
        
        # Confidence head (predicts model uncertainty)
        if output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid(),  # Confidence in [0, 1]
            )
    
    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for all tasks.
        
        Args:
            features: Input features (batch, input_dim) or (batch, seq_len, input_dim)
        
        Returns:
            outputs: Dictionary containing:
                - direction_logits: (batch, num_direction_classes)
                - volatility: (batch, 1) if predict_volatility
                - regime_logits: (batch, num_regimes) if predict_regime
                - confidence: (batch, 1) if output_confidence
        """
        # Shared processing
        shared_features = self.shared(features)
        
        outputs = {}
        
        # Direction prediction
        outputs['direction_logits'] = self.direction_head(shared_features)
        
        # Volatility prediction
        if self.predict_volatility:
            outputs['volatility'] = self.volatility_head(shared_features)
        
        # Regime prediction
        if self.predict_regime:
            outputs['regime_logits'] = self.regime_head(shared_features)
        
        # Confidence prediction
        if self.output_confidence:
            outputs['confidence'] = self.confidence_head(shared_features)
        
        return outputs
    
    def get_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw outputs to predictions.
        
        Args:
            outputs: Raw outputs from forward pass
        
        Returns:
            predictions: Dictionary with:
                - direction_probs: Softmax probabilities
                - direction_class: Predicted class
                - volatility: Volatility prediction
                - regime_probs: Regime probabilities
                - regime_class: Predicted regime
                - confidence: Confidence score
        """
        predictions = {}
        
        # Direction
        direction_probs = F.softmax(outputs['direction_logits'], dim=-1)
        predictions['direction_probs'] = direction_probs
        predictions['direction_class'] = torch.argmax(direction_probs, dim=-1)
        
        # Volatility
        if 'volatility' in outputs:
            predictions['volatility'] = outputs['volatility']
        
        # Regime
        if 'regime_logits' in outputs:
            regime_probs = F.softmax(outputs['regime_logits'], dim=-1)
            predictions['regime_probs'] = regime_probs
            predictions['regime_class'] = torch.argmax(regime_probs, dim=-1)
        
        # Confidence
        if 'confidence' in outputs:
            predictions['confidence'] = outputs['confidence']
        
        return predictions


# ============= MODEL - NEXUS-FX (MAIN) =============
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


# ============= OPTIMIZER - DELTA GRADIENT DESCENT =============
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


# ============= OPTIMIZER - MULTI-SCALE MOMENTUM =============
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


# ============= TRAINING - LOSSES =============
"""
Multi-objective Loss Functions for NEXUS-FX.

Combines multiple prediction objectives:
1. Direction prediction (classification)
2. Volatility prediction (regression)
3. Regime prediction (classification)
4. Confidence calibration (alignment of predicted confidence with accuracy)

Each objective has an adaptive weight that adjusts during training.
"""



class NexusFXLoss(nn.Module):
    """
    Multi-objective loss for NEXUS-FX model.
    
    Combines:
    - Direction loss (cross-entropy)
    - Volatility loss (MSE)
    - Regime loss (cross-entropy)
    - Calibration loss (confidence vs accuracy alignment)
    
    Args:
        direction_weight: Weight for direction loss
        volatility_weight: Weight for volatility loss
        regime_weight: Weight for regime loss
        calibration_weight: Weight for calibration loss
        adaptive_weights: Whether to adapt weights during training
    """
    
    def __init__(
        self,
        direction_weight: float = 1.0,
        volatility_weight: float = 0.5,
        regime_weight: float = 0.3,
        calibration_weight: float = 0.2,
        adaptive_weights: bool = True,
    ):
        super().__init__()
        
        self.register_buffer('direction_weight', torch.tensor(direction_weight))
        self.register_buffer('volatility_weight', torch.tensor(volatility_weight))
        self.register_buffer('regime_weight', torch.tensor(regime_weight))
        self.register_buffer('calibration_weight', torch.tensor(calibration_weight))
        
        self.adaptive_weights = adaptive_weights
        
        # Track loss magnitudes for adaptive weighting
        self.register_buffer('direction_ema', torch.tensor(1.0))
        self.register_buffer('volatility_ema', torch.tensor(1.0))
        self.register_buffer('regime_ema', torch.tensor(1.0))
        self.register_buffer('calibration_ema', torch.tensor(1.0))
        
        self.ema_momentum = 0.9
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-objective loss.
        
        Args:
            outputs: Model outputs dict with keys:
                - direction_logits: (batch, 3)
                - volatility: (batch, 1)
                - regime_logits: (batch, num_regimes)
                - confidence: (batch, 1)
            targets: Target dict with keys:
                - direction: (batch,) with values in {0, 1, 2}
                - volatility: (batch,) target volatility
                - regime: (batch,) regime labels (optional)
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual losses for logging
        """
        losses = {}
        
        # 1. Direction loss (cross-entropy)
        if 'direction_logits' in outputs and 'direction' in targets:
            direction_loss = F.cross_entropy(
                outputs['direction_logits'],
                targets['direction'].long()
            )
            losses['direction'] = direction_loss
        
        # 2. Volatility loss (MSE)
        if 'volatility' in outputs and 'volatility' in targets:
            volatility_loss = F.mse_loss(
                outputs['volatility'].squeeze(-1),
                targets['volatility']
            )
            losses['volatility'] = volatility_loss
        
        # 3. Regime loss (cross-entropy)
        if 'regime_logits' in outputs and 'regime' in targets:
            regime_loss = F.cross_entropy(
                outputs['regime_logits'],
                targets['regime'].long()
            )
            losses['regime'] = regime_loss
        
        # 4. Calibration loss
        if 'confidence' in outputs and 'direction_logits' in outputs and 'direction' in targets:
            calibration_loss = self._compute_calibration_loss(
                outputs['direction_logits'],
                outputs['confidence'],
                targets['direction']
            )
            losses['calibration'] = calibration_loss
        
        # Update EMAs
        if self.training and self.adaptive_weights:
            self._update_loss_emas(losses)
        
        # Compute weighted sum
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        if 'direction' in losses:
            weight = self._get_adaptive_weight('direction')
            total_loss = total_loss + weight * losses['direction']
        
        if 'volatility' in losses:
            weight = self._get_adaptive_weight('volatility')
            total_loss = total_loss + weight * losses['volatility']
        
        if 'regime' in losses:
            weight = self._get_adaptive_weight('regime')
            total_loss = total_loss + weight * losses['regime']
        
        if 'calibration' in losses:
            weight = self._get_adaptive_weight('calibration')
            total_loss = total_loss + weight * losses['calibration']
        
        # Add total to dict
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def _compute_calibration_loss(
        self,
        direction_logits: torch.Tensor,
        confidence: torch.Tensor,
        direction_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calibration loss: predicted confidence should match actual accuracy.
        
        High confidence predictions should be more accurate than low confidence ones.
        """
        # Get predictions
        pred_classes = torch.argmax(direction_logits, dim=-1)
        correct = (pred_classes == direction_targets.long()).float()
        
        # Confidence should match correctness
        # If model is 90% confident, it should be right 90% of the time
        calibration_loss = F.mse_loss(confidence.squeeze(-1), correct)
        
        return calibration_loss
    
    def _update_loss_emas(self, losses: Dict[str, torch.Tensor]) -> None:
        """Update exponential moving averages of loss magnitudes"""
        if 'direction' in losses:
            self.direction_ema = (
                self.ema_momentum * self.direction_ema +
                (1 - self.ema_momentum) * losses['direction'].detach()
            )
        
        if 'volatility' in losses:
            self.volatility_ema = (
                self.ema_momentum * self.volatility_ema +
                (1 - self.ema_momentum) * losses['volatility'].detach()
            )
        
        if 'regime' in losses:
            self.regime_ema = (
                self.ema_momentum * self.regime_ema +
                (1 - self.ema_momentum) * losses['regime'].detach()
            )
        
        if 'calibration' in losses:
            self.calibration_ema = (
                self.ema_momentum * self.calibration_ema +
                (1 - self.ema_momentum) * losses['calibration'].detach()
            )
    
    def _get_adaptive_weight(self, loss_name: str) -> torch.Tensor:
        """
        Get adaptive weight for a loss component.
        
        Balances losses by normalizing by their typical magnitude.
        """
        if not self.adaptive_weights:
            if loss_name == 'direction':
                return self.direction_weight
            elif loss_name == 'volatility':
                return self.volatility_weight
            elif loss_name == 'regime':
                return self.regime_weight
            elif loss_name == 'calibration':
                return self.calibration_weight
        
        # Adaptive weighting based on EMA magnitudes
        # Normalize so all losses contribute roughly equally
        total_ema = (
            self.direction_ema + self.volatility_ema +
            self.regime_ema + self.calibration_ema
        )
        
        if loss_name == 'direction':
            base_weight = self.direction_weight
            adaptive_factor = total_ema / (self.direction_ema + 1e-8)
        elif loss_name == 'volatility':
            base_weight = self.volatility_weight
            adaptive_factor = total_ema / (self.volatility_ema + 1e-8)
        elif loss_name == 'regime':
            base_weight = self.regime_weight
            adaptive_factor = total_ema / (self.regime_ema + 1e-8)
        elif loss_name == 'calibration':
            base_weight = self.calibration_weight
            adaptive_factor = total_ema / (self.calibration_ema + 1e-8)
        else:
            return torch.tensor(1.0)
        
        return base_weight * adaptive_factor / 4  # Normalize by number of losses


# ============= TRAINING - EVALUATION =============
"""
Forex-specific evaluation metrics.

Metrics include:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Profit factor
- Per-regime performance
- Calibration metrics
"""



class NexusFXEvaluator:
    """
    Evaluator for forex trading performance.
    
    Computes both prediction accuracy metrics and trading performance metrics.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated statistics"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.returns = []
    
    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update with new batch of predictions.
        
        Args:
            predictions: Model predictions dict
            targets: Ground truth targets dict
            returns: Actual returns (optional, for trading metrics)
        """
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if 'confidence' in predictions:
            self.confidences.append(predictions['confidence'].cpu())
        
        if returns is not None:
            self.returns.append(returns.cpu())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Returns:
            metrics: Dictionary of metric name -> value
        """
        metrics = {}
        
        # Prediction accuracy metrics
        acc_metrics = self._compute_accuracy_metrics()
        metrics.update(acc_metrics)
        
        # Trading performance metrics
        if len(self.returns) > 0:
            trading_metrics = self._compute_trading_metrics()
            metrics.update(trading_metrics)
        
        # Calibration metrics
        if len(self.confidences) > 0:
            cal_metrics = self._compute_calibration_metrics()
            metrics.update(cal_metrics)
        
        return metrics
    
    def _compute_accuracy_metrics(self) -> Dict[str, float]:
        """Compute prediction accuracy metrics"""
        metrics = {}
        
        # Direction accuracy
        all_pred_classes = []
        all_target_classes = []
        
        for pred, target in zip(self.predictions, self.targets):
            if 'direction_logits' in pred and 'direction' in target:
                pred_class = torch.argmax(pred['direction_logits'], dim=-1)
                all_pred_classes.append(pred_class.cpu())
                all_target_classes.append(target['direction'].cpu())
        
        if all_pred_classes:
            pred_classes = torch.cat(all_pred_classes)
            target_classes = torch.cat(all_target_classes)
            
            accuracy = (pred_classes == target_classes).float().mean().item()
            metrics['direction_accuracy'] = accuracy
            
            # Per-class accuracy
            for i in range(3):  # Assuming 3 classes
                mask = target_classes == i
                if mask.sum() > 0:
                    class_acc = (pred_classes[mask] == target_classes[mask]).float().mean().item()
                    metrics[f'direction_accuracy_class_{i}'] = class_acc
        
        return metrics
    
    def _compute_trading_metrics(self) -> Dict[str, float]:
        """Compute trading performance metrics"""
        metrics = {}
        
        # Concatenate all returns
        all_returns = torch.cat(self.returns).numpy()
        
        # Cumulative returns
        cumulative_returns = np.cumprod(1 + all_returns) - 1
        total_return = cumulative_returns[-1]
        metrics['total_return'] = total_return
        
        # Sharpe ratio (annualized, assuming 5-min returns)
        # 252 trading days * 24 hours * 12 (5-min periods per hour)
        periods_per_year = 252 * 24 * 12
        sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(periods_per_year)
        metrics['sharpe_ratio'] = sharpe
        
        # Sortino ratio (only downside volatility)
        downside_returns = all_returns[all_returns < 0]
        if len(downside_returns) > 0:
            sortino = np.mean(all_returns) / (np.std(downside_returns) + 1e-8) * np.sqrt(periods_per_year)
            metrics['sortino_ratio'] = sortino
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + all_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        metrics['max_drawdown'] = max_drawdown
        
        # Win rate
        winning_trades = (all_returns > 0).sum()
        total_trades = len(all_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        metrics['win_rate'] = win_rate
        
        # Profit factor
        gross_profit = all_returns[all_returns > 0].sum()
        gross_loss = -all_returns[all_returns < 0].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        metrics['profit_factor'] = profit_factor
        
        return metrics
    
    def _compute_calibration_metrics(self) -> Dict[str, float]:
        """Compute calibration metrics"""
        metrics = {}
        
        # Confidence calibration
        all_confidences = torch.cat(self.confidences)
        all_pred_classes = []
        all_target_classes = []
        
        for pred, target in zip(self.predictions, self.targets):
            if 'direction_logits' in pred and 'direction' in target:
                pred_class = torch.argmax(pred['direction_logits'], dim=-1)
                all_pred_classes.append(pred_class.cpu())
                all_target_classes.append(target['direction'].cpu())
        
        if all_pred_classes:
            pred_classes = torch.cat(all_pred_classes)
            target_classes = torch.cat(all_target_classes)
            correct = (pred_classes == target_classes).float()
            
            # Expected Calibration Error (ECE)
            num_bins = 10
            bin_boundaries = torch.linspace(0, 1, num_bins + 1)
            ece = 0.0
            
            for i in range(num_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (all_confidences >= bin_lower) & (all_confidences < bin_upper)
                in_bin = in_bin.squeeze()
                
                if in_bin.sum() > 0:
                    bin_confidence = all_confidences[in_bin].mean()
                    bin_accuracy = correct[in_bin].mean()
                    ece += torch.abs(bin_confidence - bin_accuracy) * (in_bin.sum() / len(all_confidences))
            
            metrics['expected_calibration_error'] = ece.item()
        
        return metrics
    
    def get_performance_summary(self) -> str:
        """Get formatted performance summary"""
        metrics = self.compute_metrics()
        
        summary = "=== NEXUS-FX Performance Summary ===\n\n"
        
        summary += "Prediction Metrics:\n"
        summary += f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.4f}\n"
        
        summary += "\nTrading Metrics:\n"
        summary += f"  Total Return: {metrics.get('total_return', 0):.4f}\n"
        summary += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}\n"
        summary += f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}\n"
        summary += f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4f}\n"
        summary += f"  Win Rate: {metrics.get('win_rate', 0):.4f}\n"
        summary += f"  Profit Factor: {metrics.get('profit_factor', 0):.4f}\n"
        
        summary += "\nCalibration Metrics:\n"
        summary += f"  Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}\n"
        
        return summary


# ============= TRAINING - TRAINER =============
"""
Main training loop for NEXUS-FX with continual learning support.
"""




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


# ============================================================================
# MODULE METADATA & EXPORTS
# ============================================================================

__version__ = "1.0.0"
__author__ = "NEXUS-FX Team"
__all__ = [
    # Configuration
    'NexusFXConfig',
    
    # Model Components
    'NEXUSFX',
    'AssociativeMemory',
    'ContinuumMemorySystem',
    'ContinuumMemoryLevel',
    'CrossPairMemory',
    'RegimeDetector',
    'SessionFrequencyGate',
    'SelfModifyingTitans',
    'SelfModifyingTitansLayer',
    'OutputHeads',
    
    # Data
    'ForexDataset',
    'Preprocessor',
    'FeatureEngine',
    'MacroFeatureEncoder',
    'SessionClock',
    
    # Training
    'NexusFXTrainer',
    'NexusFXLoss',
    'NexusFXEvaluator',
    
    # Optimizers
    'DeltaGradientDescent',
    'MultiScaleMomentumMuon',
    
    # Utilities
    'setup_logger',
    'MetricsLogger',
    'get_active_sessions',
    'is_market_open',
    'calculate_spread',
    'detect_session',
]


def get_version():
    """Return the version of this module."""
    return __version__


def list_components():
    """List all available components in the consolidated module."""
    components = {
        'Models': ['NEXUSFX', 'AssociativeMemory', 'ContinuumMemorySystem', 
                   'CrossPairMemory', 'RegimeDetector', 'SessionFrequencyGate',
                   'SelfModifyingTitans', 'OutputHeads'],
        'Data': ['ForexDataset', 'Preprocessor', 'FeatureEngine', 
                 'MacroFeatureEncoder', 'SessionClock'],
        'Training': ['NexusFXTrainer', 'NexusFXLoss', 'NexusFXEvaluator'],
        'Optimizers': ['DeltaGradientDescent', 'MultiScaleMomentumMuon'],
        'Utilities': ['setup_logger', 'MetricsLogger', 'get_active_sessions',
                      'is_market_open', 'calculate_spread', 'detect_session'],
    }
    return components


def quick_start_example():
    """
    Return a quick start code example.
    """
    example = """
    # Quick Start Example for NEXUS-FX
    
    import nexus_fx_consolidated as nfx
    import torch
    
    # 1. Create configuration
    config = nfx.NexusFXConfig(
        pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        batch_size=32,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 2. Create dataset
    dataset = nfx.ForexDataset(
        data_path='/path/to/forex/data',  # Or None for synthetic data
        pairs=config.pairs,
        base_timeframe='5m',
        target_timeframes=config.timeframes,
        sequence_length=config.sequence_length,
    )
    
    # 3. Create model
    model = nfx.NEXUSFX(config)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 4. Create trainer
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    trainer = nfx.NexusFXTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        device=config.device
    )
    
    # 5. Train
    trainer.train()
    
    # 6. Evaluate
    evaluator = nfx.NexusFXEvaluator()
    # ... evaluation code ...
    metrics = evaluator.compute_metrics()
    print(metrics)
    """
    return example


# Print module info when imported
if __name__ != '__main__':
    print(f"NEXUS-FX v{__version__} - Consolidated module loaded successfully")
    print(f"Available components: {len(__all__)} classes and functions")
    print("Use help(nexus_fx_consolidated) for more information")

