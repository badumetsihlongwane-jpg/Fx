"""
Tests for main NEXUS-FX model.
"""

import torch
import pytest

from nexus_fx.config import NexusFXConfig
from nexus_fx.model import NEXUSFX


def test_nexus_fx_creation():
    """Test NEXUSFX model creation"""
    config = NexusFXConfig(
        num_pairs=2,
        pairs=['EURUSD', 'GBPUSD'],
        timeframes=['5m', '15m'],
        input_dim=64,
        hidden_dim=128,
        num_titans_layers=2,
        num_cms_levels=3,
    )
    
    model = NEXUSFX(config)
    
    assert model.config == config
    assert len(model.titans_per_pair) == 2


def test_nexus_fx_forward():
    """Test forward pass through NEXUSFX"""
    config = NexusFXConfig(
        num_pairs=2,
        pairs=['EURUSD', 'GBPUSD'],
        timeframes=['5m', '15m'],
        input_dim=32,
        hidden_dim=64,
        num_titans_layers=2,
        num_cms_levels=2,
        sequence_length=50,
    )
    
    model = NEXUSFX(config)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    num_pairs = 2
    num_tf = 2
    seq_len = 50
    
    ohlc = torch.randn(batch_size, num_pairs, num_tf, seq_len, 4).abs() + 1.0
    volume = torch.randn(batch_size, num_pairs, num_tf, seq_len).abs()
    timestamps = torch.randint(1704110400, 1704200000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(ohlc, volume, timestamps, macro_data=None)
    
    # Check outputs
    assert 'direction_logits' in outputs
    assert 'volatility' in outputs
    assert 'regime_logits' in outputs
    
    # Check shapes
    assert outputs['direction_logits'].shape == (batch_size, 3)
    assert outputs['volatility'].shape == (batch_size, 1)


def test_nexus_fx_output_predictions():
    """Test getting predictions from outputs"""
    config = NexusFXConfig(
        num_pairs=1,
        pairs=['EURUSD'],
        timeframes=['5m'],
        input_dim=32,
        hidden_dim=64,
        sequence_length=20,
    )
    
    model = NEXUSFX(config)
    model.eval()
    
    # Create inputs
    batch_size = 1
    ohlc = torch.randn(batch_size, 1, 1, 20, 4).abs() + 1.0
    volume = torch.randn(batch_size, 1, 1, 20).abs()
    timestamps = torch.randint(1704110400, 1704200000, (batch_size, 20))
    
    # Get outputs
    with torch.no_grad():
        outputs = model(ohlc, volume, timestamps, macro_data=None)
        predictions = model.output_heads.get_predictions(outputs)
    
    # Check predictions
    assert 'direction_probs' in predictions
    assert 'direction_class' in predictions
    assert 'volatility' in predictions
    
    # Check that probabilities sum to 1
    assert torch.allclose(
        predictions['direction_probs'].sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-6
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
