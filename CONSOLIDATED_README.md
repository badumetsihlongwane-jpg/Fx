# NEXUS-FX Consolidated Module

## Overview

`nexus_fx_consolidated.py` is a single-file version of the entire NEXUS-FX codebase, designed for easy use in notebook environments like Google Colab and Kaggle.

## File Statistics

- **Size**: 158 KB
- **Lines**: 4,681
- **Classes**: 22
- **Functions**: 129

## Contents

### Core Components

1. **Configuration** (`NexusFXConfig`)
   - Centralized hyperparameter configuration
   - 40+ configurable parameters

2. **Model Components**
   - `NEXUSFX` - Main model architecture
   - `AssociativeMemory` - L2-regression based memory
   - `ContinuumMemorySystem` - Multi-timescale memory hierarchy
   - `CrossPairMemory` - Currency correlation learning
   - `RegimeDetector` - Market regime classification
   - `SessionFrequencyGate` - Session-aware gating
   - `SelfModifyingTitans` - In-context adaptive layers
   - `OutputHeads` - Multi-task prediction outputs

3. **Data Pipeline**
   - `ForexDataset` - Multi-timeframe OHLC data loading
   - `Preprocessor` - Normalization and missing data handling
   - `FeatureEngine` - Technical indicator computation
   - `MacroFeatureEncoder` - Macro-fundamental features
   - `SessionClock` - Forex session detection

4. **Training**
   - `NexusFXTrainer` - Main training loop
   - `NexusFXLoss` - Multi-objective loss function
   - `NexusFXEvaluator` - Performance metrics

5. **Optimizers**
   - `DeltaGradientDescent` - L2-regression based optimizer
   - `MultiScaleMomentumMuon` - Multi-scale momentum

6. **Utilities**
   - Logging utilities
   - Market utilities
   - Session detection

## Usage

### In Google Colab

```python
# 1. Upload nexus_fx_consolidated.py to your Colab environment
from google.colab import files
files.upload()  # Select nexus_fx_consolidated.py

# 2. Import the module
import nexus_fx_consolidated as nfx

# 3. Check version
print(nfx.get_version())

# 4. List available components
components = nfx.list_components()
for category, items in components.items():
    print(f"{category}: {', '.join(items)}")

# 5. Create a model
config = nfx.NexusFXConfig(
    pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    batch_size=32,
    learning_rate=1e-4,
)

model = nfx.NEXUSFX(config)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### In Kaggle

```python
# 1. Add nexus_fx_consolidated.py as a dataset or upload it

# 2. Import
import nexus_fx_consolidated as nfx

# 3. Use it
config = nfx.NexusFXConfig()
model = nfx.NEXUSFX(config)
```

### Quick Start Example

```python
import nexus_fx_consolidated as nfx
import torch

# Configuration
config = nfx.NexusFXConfig(
    pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    batch_size=32,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Create dataset (will use synthetic data if no path provided)
dataset = nfx.ForexDataset(
    data_path=None,  # None = synthetic data for testing
    pairs=config.pairs,
    base_timeframe='5m',
    target_timeframes=config.timeframes,
    sequence_length=config.sequence_length,
)

# Create model
model = nfx.NEXUSFX(config)

# Create trainer
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

trainer = nfx.NexusFXTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    device=config.device
)

# Train
trainer.train()
```

## Testing

The module can be tested without importing PyTorch dependencies:

```bash
# Syntax check
python3 -m py_compile nexus_fx_consolidated.py

# Structure verification
python3 -c "import ast; ast.parse(open('nexus_fx_consolidated.py').read()); print('OK')"
```

## Benefits

1. **Single File**: Easy to share and upload to notebook environments
2. **Complete**: All NEXUS-FX functionality in one place
3. **Well-Organized**: Clear section markers and comprehensive docstrings
4. **Dependency Order**: Classes and functions ordered by dependencies
5. **Tested**: Syntax validation and structure verification passed

## Version

Current version: **1.0.0**

## Support

For issues or questions:
- Check the main repository: https://github.com/your-repo/Fx
- Review the docstrings: `help(nfx.NEXUSFX)`
- Get quick start code: `print(nfx.quick_start_example())`

