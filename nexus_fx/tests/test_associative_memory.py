"""
Tests for AssociativeMemory module.
"""

import torch
import pytest

from nexus_fx.model.associative_memory import AssociativeMemory


def test_associative_memory_creation():
    """Test that AssociativeMemory can be created with valid parameters"""
    memory = AssociativeMemory(
        key_dim=64,
        value_dim=128,
        num_slots=32,
        update_frequency=1,
    )
    
    assert memory.key_dim == 64
    assert memory.value_dim == 128
    assert memory.num_slots == 32
    assert memory.keys.shape == (32, 64)
    assert memory.values.shape == (32, 128)


def test_associative_memory_forward():
    """Test forward pass through AssociativeMemory"""
    memory = AssociativeMemory(
        key_dim=64,
        value_dim=128,
        num_slots=32,
        update_frequency=1,
    )
    
    # Create query
    batch_size = 4
    query = torch.randn(batch_size, 64)
    
    # Forward pass (read only)
    retrieved, surprise = memory(query, write_mode=False)
    
    assert retrieved.shape == (batch_size, 128)
    assert surprise.shape == (batch_size,)


def test_associative_memory_write():
    """Test memory write mechanism"""
    memory = AssociativeMemory(
        key_dim=64,
        value_dim=128,
        num_slots=32,
        update_frequency=1,
        use_surprise_gating=True,
    )
    
    # Create query and target
    batch_size = 4
    query = torch.randn(batch_size, 64)
    target = torch.randn(batch_size, 128)
    
    # Get initial memory state
    initial_keys = memory.keys.clone()
    
    # Forward with write
    retrieved, surprise = memory(query, value_target=target, write_mode=True)
    
    # Memory should have changed
    assert not torch.allclose(memory.keys, initial_keys)
    assert memory.step_counter > 0


def test_associative_memory_update_frequency():
    """Test that update frequency is respected"""
    memory = AssociativeMemory(
        key_dim=64,
        value_dim=128,
        num_slots=32,
        update_frequency=5,  # Only update every 5 steps
    )
    
    query = torch.randn(1, 64)
    target = torch.randn(1, 128)
    
    initial_keys = memory.keys.clone()
    
    # Step 4 times - should not update
    for _ in range(4):
        memory(query, value_target=target, write_mode=True)
    
    # Memory should be unchanged
    assert torch.allclose(memory.keys, initial_keys)
    
    # Step 5th time - should update
    memory(query, value_target=target, write_mode=True)
    
    # Memory should have changed
    assert not torch.allclose(memory.keys, initial_keys)


def test_associative_memory_reset():
    """Test memory reset functionality"""
    memory = AssociativeMemory(
        key_dim=64,
        value_dim=128,
        num_slots=32,
        update_frequency=1,
    )
    
    # Write some data
    query = torch.randn(4, 64)
    target = torch.randn(4, 128)
    memory(query, value_target=target, write_mode=True)
    
    # Reset
    memory.reset()
    
    # Check that step counter is reset
    assert memory.step_counter == 0
    assert memory.slot_age.sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
