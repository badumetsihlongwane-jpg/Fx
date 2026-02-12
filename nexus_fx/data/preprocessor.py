"""
Preprocessor - Data normalization, alignment, and missing data handling.

Handles:
- Rolling z-score normalization (no lookahead)
- Missing data forward-fill with staleness indicators
- Cross-timeframe alignment
- Train/validation/test split (temporal order preserved)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


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
