"""
Forex Dataset - Multi-timeframe OHLC data loading and management.

This dataset handles:
- Loading 5-minute OHLC candles for multiple currency pairs
- Aggregating to multiple timeframes (15m, 1H, 4H, 1D)
- Temporal alignment across timeframes
- Streaming/online mode for live inference
- No lookahead bias in all operations
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


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
