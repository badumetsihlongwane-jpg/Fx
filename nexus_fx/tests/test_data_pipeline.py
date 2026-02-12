"""
Tests for data pipeline components.
"""

import torch
import pytest
import numpy as np

from nexus_fx.data import ForexDataset, FeatureEngine, SessionClock, Preprocessor


def test_forex_dataset_creation():
    """Test ForexDataset creation with synthetic data"""
    dataset = ForexDataset(
        data_dict=None,  # Will generate synthetic
        pairs=['EURUSD', 'GBPUSD'],
        base_timeframe='5m',
        target_timeframes=['5m', '15m', '1H'],
        sequence_length=100,
    )
    
    assert len(dataset) > 0
    assert len(dataset.pairs) == 2


def test_forex_dataset_getitem():
    """Test getting an item from ForexDataset"""
    dataset = ForexDataset(
        data_dict=None,
        pairs=['EURUSD'],
        base_timeframe='5m',
        target_timeframes=['5m', '15m'],
        sequence_length=50,
    )
    
    sample = dataset[0]
    
    assert 'ohlc' in sample
    assert 'volume' in sample
    assert 'timestamps' in sample
    
    # Check shapes
    assert sample['ohlc'].dim() == 4  # (pairs, timeframes, seq, 4)
    assert sample['volume'].dim() == 3  # (pairs, timeframes, seq)
    assert sample['timestamps'].dim() == 1  # (seq,)


def test_feature_engine_returns():
    """Test FeatureEngine return computation"""
    fe = FeatureEngine()
    
    # Create OHLC data
    ohlc = torch.randn(4, 100, 4).abs() + 1.0  # Positive prices
    
    # Compute features
    features = fe.compute_features(ohlc)
    
    assert features.shape[0] == 4  # Batch size
    assert features.shape[1] == 100  # Sequence length
    assert features.shape[2] > 0  # Has features


def test_session_clock_detection():
    """Test session detection"""
    sc = SessionClock()
    
    # Create timestamps (in Unix time)
    # Use a known time: 2024-01-01 12:00:00 UTC (Monday, NY session)
    timestamps = torch.tensor([[1704110400]], dtype=torch.long)
    
    # Detect sessions
    session_indicators = sc.detect_sessions(timestamps)
    
    assert session_indicators.shape == (1, 1, 6)
    assert session_indicators[0, 0, 3] == 1.0  # Should be in NY session


def test_preprocessor_normalization():
    """Test Preprocessor normalization"""
    prep = Preprocessor(normalization_method='zscore')
    
    # Create data
    data = torch.randn(100, 20)
    
    # Fit and transform
    normalized, staleness = prep.fit_transform(data)
    
    assert normalized.shape == data.shape
    assert staleness.shape == data.shape
    
    # Check that mean is close to 0 and std close to 1
    assert abs(normalized.mean().item()) < 0.1
    assert abs(normalized.std().item() - 1.0) < 0.1


def test_preprocessor_temporal_split():
    """Test temporal train/val/test split"""
    prep = Preprocessor()
    
    data = torch.randn(1000, 10)
    
    train, val, test = prep.temporal_split(data, val_split=0.2, test_split=0.2)
    
    assert train.shape[0] == 600
    assert val.shape[0] == 200
    assert test.shape[0] == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
