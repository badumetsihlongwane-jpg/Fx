"""
Quick Start Example for NEXUS-FX in Google Colab / Kaggle
==========================================================

This file demonstrates how to use the consolidated nexus_fx_consolidated.py
module in a notebook environment.

Instructions:
1. Upload nexus_fx_consolidated.py to your Colab/Kaggle environment
2. Run this script or copy the code to your notebook
3. Modify the configuration as needed
"""

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================
# Run this in a Colab cell:
# !pip install torch numpy pandas scikit-learn tqdm pyyaml matplotlib

# ============================================================================
# STEP 2: Import the Module
# ============================================================================
import nexus_fx_consolidated as nfx
import torch
from torch.utils.data import DataLoader

# ============================================================================
# STEP 3: Create Configuration
# ============================================================================
config = nfx.NexusFXConfig(
    # Currency pairs to trade
    pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    num_pairs=4,
    
    # Model architecture
    hidden_dim=256,
    num_titans_layers=4,
    num_cms_levels=4,  # Multi-timescale memory levels
    
    # Training parameters
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10,  # Start with fewer epochs for testing
    sequence_length=512,  # 512 5-minute candles
    
    # Device
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

print(f"Configuration created for {len(config.pairs)} pairs")
print(f"Update frequencies: {config.get_update_frequencies()}")

# ============================================================================
# STEP 4: Create Dataset
# ============================================================================
# Option A: Use synthetic data (for testing)
dataset = nfx.ForexDataset(
    pairs=config.pairs,
    base_timeframe='5m',
    target_timeframes=['5m', '15m', '1H', '4H', '1D'],
    sequence_length=config.sequence_length,
)

# Option B: Load your own data
# dataset = nfx.ForexDataset(
#     data_path='/path/to/your/forex/data',  # CSV files folder
#     pairs=config.pairs,
#     base_timeframe='5m',
#     target_timeframes=['5m', '15m', '1H', '4H', '1D'],
#     sequence_length=config.sequence_length,
# )

print(f"Dataset created with {len(dataset)} samples")

# Create data loader
train_loader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0,  # Use 0 in Colab/Kaggle
)

# ============================================================================
# STEP 5: Create Model
# ============================================================================
model = nfx.NEXUSFX(config)
model = model.to(config.device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {num_params:,} parameters")

# ============================================================================
# STEP 6: Create Optimizer
# ============================================================================
# Option A: Delta Gradient Descent (custom optimizer)
optimizer = nfx.DeltaGradientDescent(
    model.parameters(),
    lr=config.learning_rate,
)

# Option B: Multi-Scale Momentum (alternative)
# optimizer = nfx.MultiScaleMomentumMuon(
#     model.parameters(),
#     lr=config.learning_rate,
# )

print(f"Optimizer: {optimizer.__class__.__name__}")

# ============================================================================
# STEP 7: Create Trainer
# ============================================================================
# Note: NexusFXTrainer creates the optimizer internally based on config
trainer = nfx.NexusFXTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=None,  # You can add validation loader
    device=config.device,
)

print("Trainer initialized")

# ============================================================================
# STEP 8: Train the Model
# ============================================================================
print("\nStarting training...")
print("=" * 70)

# Train (uses config.num_epochs)
# To change the number of epochs, modify config.num_epochs before creating trainer
# config.num_epochs = 10  # Example: train for 10 epochs
trainer.train()

# ============================================================================
# STEP 9: Save the Model
# ============================================================================
# Save checkpoint
checkpoint_path = 'nexus_fx_checkpoint.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
}, checkpoint_path)

print(f"\nModel saved to {checkpoint_path}")

# ============================================================================
# STEP 10: Make Predictions (Example)
# ============================================================================
# Get a sample
sample = dataset[0]
ohlc = sample['ohlc'].unsqueeze(0).to(config.device)
volume = sample['volume'].unsqueeze(0).to(config.device)
timestamps = sample['timestamps'].unsqueeze(0).to(config.device)

# Make prediction
model.eval()
with torch.no_grad():
    outputs = model(ohlc, volume, timestamps, macro_data=None)

print("\nPrediction outputs:")
print(f"  - Direction logits shape: {outputs['direction_logits'].shape}")
print(f"  - Volatility prediction: {outputs['volatility'].shape}")
print(f"  - Regime logits: {outputs['regime_logits'].shape}")
print(f"  - Confidence: {outputs['confidence'].shape}")

# Interpret direction prediction
direction_probs = torch.softmax(outputs['direction_logits'], dim=-1)
direction = torch.argmax(direction_probs, dim=-1)
direction_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}

print(f"\nPredicted direction: {direction_map[direction.item()]}")
print(f"Confidence: {outputs['confidence'].item():.2%}")

print("\n" + "=" * 70)
print("✅ Quick start completed successfully!")
print("\nNext steps:")
print("  - Adjust hyperparameters in the config")
print("  - Load your own forex data")
print("  - Extend training epochs")
print("  - Implement backtesting using your predictions")
print("=" * 70)
