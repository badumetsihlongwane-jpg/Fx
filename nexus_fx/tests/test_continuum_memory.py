"""
Tests for ContinuumMemorySystem module.
"""

import torch
import pytest

from nexus_fx.model.continuum_memory import ContinuumMemorySystem


def test_continuum_memory_creation():
    """Test ContinuumMemorySystem creation"""
    cms = ContinuumMemorySystem(
        input_dim=64,
        hidden_dim=128,
        num_levels=4,
        base_frequency=1,
        frequency_multiplier=10,
    )
    
    assert cms.num_levels == 4
    assert len(cms.levels) == 4
    assert cms.update_frequencies == [1, 10, 100, 1000]


def test_continuum_memory_forward():
    """Test forward pass through ContinuumMemorySystem"""
    cms = ContinuumMemorySystem(
        input_dim=64,
        hidden_dim=128,
        num_levels=4,
    )
    
    # Create input
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 64)
    
    # Forward pass
    output = cms(x)
    
    assert output.shape == (batch_size, seq_len, 128)


def test_continuum_memory_multiple_levels():
    """Test that all levels process data"""
    cms = ContinuumMemorySystem(
        input_dim=64,
        hidden_dim=128,
        num_levels=4,
    )
    
    x = torch.randn(4, 10, 64)
    
    # Get outputs from all levels
    output, level_outputs = cms(x, return_all_levels=True)
    
    assert len(level_outputs) == 4
    assert all(out.shape[0] == 4 for out in level_outputs)  # All have batch size 4


def test_continuum_memory_update_frequencies():
    """Test that different levels have different update frequencies"""
    cms = ContinuumMemorySystem(
        input_dim=64,
        hidden_dim=128,
        num_levels=3,
        base_frequency=1,
        frequency_multiplier=10,
    )
    
    # Check that levels have correct frequencies
    assert cms.levels[0].update_frequency == 1
    assert cms.levels[1].update_frequency == 10
    assert cms.levels[2].update_frequency == 100


def test_continuum_memory_reset():
    """Test CMS reset functionality"""
    cms = ContinuumMemorySystem(
        input_dim=64,
        hidden_dim=128,
        num_levels=4,
    )
    
    # Process some data
    x = torch.randn(4, 10, 64)
    cms(x)
    
    # Reset
    cms.reset()
    
    # Check that all levels are reset
    for level in cms.levels:
        assert level.step_counter == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
