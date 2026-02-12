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

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


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
