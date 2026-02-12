# NEXUS-FX: Nested EXchange Understanding System for Forex

**A hierarchical forex trading architecture based on nested associative memories operating at different timescales.**

NEXUS-FX implements the core insight from the HOPE/NSAM paper: treat the entire model as a hierarchy of nested optimization problems with associative memories at different update frequencies, specifically designed for forex markets.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Components](#components)
- [Training](#training)
- [Evaluation](#evaluation)
- [Backtesting](#backtesting)
- [Theoretical Background](#theoretical-background)
- [Citation](#citation)

## Overview

Traditional architectures (Transformers, LSTMs) fail in forex because they treat markets as stationary. NEXUS-FX mirrors how forex actually works: **nested timescales of knowledge** from tick-level microstructure up to macro regime shifts.

### Key Innovation

Everything is a nested associative memory. Pre-training weights, in-context hidden states, momentum—all are associative memories at different update frequencies matching market dynamics:

- **Fast memories** (update every step): Microstructure patterns, momentum bursts
- **Medium memories** (update every 10 steps): Intraday dynamics, session effects
- **Slow memories** (update every 100 steps): Medium-term trends
- **Slowest memories** (update every 1000 steps): Macro regimes, central bank cycles

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        NEXUS-FX                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ OHLC Data  │  │   Macro    │  │  Session   │           │
│  │ Multi-TF   │  │  Features  │  │   Clock    │           │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘           │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │  Feature   │  │   Macro    │  │  Session   │           │
│  │  Engine    │  │  Encoder   │  │ Projection │           │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘           │
│         │                │                │                 │
│         └────────────────┴────────────────┘                 │
│                          │                                   │
│         ┌────────────────┴────────────────┐                 │
│         │                                  │                 │
│         ▼                                  ▼                 │
│  ┌─────────────────┐            ┌─────────────────┐        │
│  │ Self-Modifying  │  ×N pairs  │  Cross-Pair     │        │
│  │    Titans       │────────────▶│    Memory       │        │
│  │ (per-pair seq)  │            │ (correlations)  │        │
│  └─────────────────┘            └────────┬────────┘        │
│                                           │                  │
│                                           ▼                  │
│                          ┌────────────────────────┐         │
│                          │  Continuum Memory      │         │
│                          │  System (CMS)          │         │
│                          │  - Level 1 (fast)      │         │
│                          │  - Level 2 (medium)    │         │
│                          │  - Level 3 (slow)      │         │
│                          │  - Level 4 (slowest)   │         │
│                          └─────────┬──────────────┘         │
│                                    │                         │
│                    ┌───────────────┴────────────┐           │
│                    │                             │           │
│                    ▼                             ▼           │
│          ┌──────────────────┐         ┌──────────────────┐ │
│          │ Session-Aware    │         │    Regime        │ │
│          │ Frequency Gate   │         │   Detector       │ │
│          └─────────┬────────┘         └────────┬─────────┘ │
│                    │                            │           │
│                    └────────────┬───────────────┘           │
│                                 │                            │
│                                 ▼                            │
│                    ┌────────────────────────┐               │
│                    │   Output Heads         │               │
│                    │  - Direction (↑↓→)     │               │
│                    │  - Volatility          │               │
│                    │  - Regime              │               │
│                    │  - Confidence          │               │
│                    └────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/badumetsihlongwane-jpg/Fx.git
cd Fx

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- ta >= 0.11.0 (technical analysis)
- tqdm, matplotlib, pyyaml

## Quick Start

```python
from nexus_fx.config import NexusFXConfig
from nexus_fx.model import NEXUSFX
from nexus_fx.data import ForexDataset
import torch

# Create configuration
config = NexusFXConfig(
    pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    timeframes=['5m', '15m', '1H', '4H', '1D'],
    hidden_dim=256,
    num_cms_levels=4,
)

# Create model
model = NEXUSFX(config)

# Create dataset (will use synthetic data if no path provided)
dataset = ForexDataset(
    pairs=config.pairs,
    base_timeframe='5m',
    target_timeframes=config.timeframes,
    sequence_length=512,
)

# Get a sample
sample = dataset[0]
ohlc = sample['ohlc']
volume = sample['volume']
timestamps = sample['timestamps']

# Forward pass
outputs = model(
    ohlc.unsqueeze(0),  # Add batch dimension
    volume.unsqueeze(0),
    timestamps.unsqueeze(0),
    macro_data=None,
)

print(outputs.keys())  # direction_logits, volatility, regime_logits, confidence
```

## Data Requirements

NEXUS-FX is designed to work with **limited, publicly available data**:

### Required Data
- **OHLC price data**: 5-minute candles as base (supports multi-timeframe aggregation)
- **Timestamps**: UTC timestamps for session detection

### Optional Data
- **Volume**: When available (treated as optional feature)
- **Economic calendar**: Scheduled macro events (NFP, CPI, rate decisions)
- **Interest rates**: Central bank rates (Fed, ECB, BOJ, RBA, BOE)
- **Bond yields**: US10Y, EU10Y, JP10Y, AU10Y
- **Commodities**: Gold (XAU/USD), Oil (WTI/Brent), DXY
- **Sentiment**: VIX, risk-on/risk-off indicators

### Data Format

CSV files with columns: `timestamp, open, high, low, close, volume (optional)`

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.1050,1.1055,1.1048,1.1052,1250
2024-01-01 00:05:00,1.1052,1.1058,1.1051,1.1056,1180
...
```

## Components

### 1. Associative Memory (`associative_memory.py`)

Core building block using L2-regression instead of dot-product attention.

### 2. Self-Modifying Titans (`self_modifying_titans.py`)

Generates own keys, values, and **learning rates** in-context.

### 3. Continuum Memory System (`continuum_memory.py`)

Spectrum of MLP blocks at exponentially spaced update frequencies.

### 4. Cross-Pair Memory (`cross_pair_memory.py`)

Learns currency correlations.

### 5. Session-Aware Frequency Gate (`session_gate.py`)

Adjusts memory activation by session.

### 6. Delta Gradient Descent (`delta_gd.py`)

Custom optimizer with adaptive decay.

### 7. Multi-Scale Momentum (`multi_scale_momentum.py`)

Maintains momentum at multiple timescales.

## Training

```bash
# Train on synthetic data (for testing)
python -m nexus_fx.scripts.train

# Train on real data
python -m nexus_fx.scripts.train --data_path /path/to/forex/data --batch_size 32 --num_epochs 100 --lr 1e-4
```

## Evaluation

```bash
# Evaluate on test set
python -m nexus_fx.scripts.evaluate --checkpoint checkpoints/best_model.pt --data_path /path/to/data
```

## Backtesting

```bash
# Run backtest
python -m nexus_fx.scripts.backtest --checkpoint checkpoints/best_model.pt --initial_capital 10000
```

## Running Tests

```bash
# Run all tests
pytest nexus_fx/tests/ -v

# Run specific test
pytest nexus_fx/tests/test_associative_memory.py -v
```

## License

MIT License

## Acknowledgments

This architecture is inspired by the theoretical insights from the HOPE/NSAM framework on treating neural networks as nested optimization problems.

---

**Built with ❤️ for the forex trading community**