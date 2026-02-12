# NEXUS-FX Quick Reference Guide

## Architecture at a Glance

**36 Python modules** implementing a novel forex trading architecture based on Nested Sequential Associative Memory (NSAM) principles.

## Core Innovation: Multi-Timescale Memories

Traditional ML models use a single learning rate. NEXUS-FX uses **nested memories** at different update frequencies matching forex market dynamics:

| Memory Level | Update Frequency | Timescale | Market Phenomena |
|--------------|------------------|-----------|------------------|
| **Level 1 (Fast)** | Every 1 step | Ticks-minutes | Spread widening, momentum bursts |
| **Level 2** | Every 10 steps | Hours-days | Session effects, intraday patterns |
| **Level 3** | Every 100 steps | Days-weeks | Trend following, mean reversion |
| **Level 4 (Slow)** | Every 1000 steps | Weeks-months | Central bank cycles, carry trades |

## Key Components

### 1. **Associative Memory** (`model/associative_memory.py`)
- Uses L2-regression instead of dot-product attention
- Update frequency control (core NSAM concept)
- Surprise-gated writing (learns more from prediction errors)
- Adaptive decay via DGD

```python
M(q) = argmin_v || v - Σ_i α_i * V_i ||^2
```

### 2. **Self-Modifying Titans** (`model/self_modifying_titans.py`)
- Generates own keys, values, AND learning rates in-context
- No weight updates needed for adaptation
- Automatically increases learning rate during volatility spikes

### 3. **Continuum Memory System** (`model/continuum_memory.py`)
- 4 MLP blocks at exponentially spaced update frequencies
- Knowledge cascade: fast → slow information flow
- Anti-catastrophic forgetting

### 4. **Cross-Pair Memory** (`model/cross_pair_memory.py`)
- Learns EUR/USD ↔ GBP/USD correlations
- Safe-haven flows (JPY in risk-off)
- Macro context affects all pairs simultaneously

### 5. **Session Gate** (`model/session_gate.py`)
- London-NY overlap → boost fast memories
- Asian session → boost slow memories
- News events → spike all frequencies

### 6. **Custom Optimizers**

**Delta GD** (`optim/delta_gd.py`):
```python
w ← w - lr * (grad + λ(w - w_ref))
# λ adapts based on gradient variance
```

**M3** (`optim/multi_scale_momentum.py`):
- Short-term momentum (β=0.9)
- Long-term momentum (β=0.999)
- Adaptive mixing

## Data Pipeline

Designed for **limited, publicly available data**:

**Required:**
- OHLC 5-minute candles
- Timestamps

**Optional (all handled gracefully if missing):**
- Volume
- Economic calendar (NFP, CPI, etc.)
- Interest rates (Fed, ECB, BOJ, RBA, BOE)
- Bond yields (US10Y, EU10Y, etc.)
- Commodities (Gold, Oil, DXY)
- Sentiment (VIX)

## Quick Start

```python
from nexus_fx.config import NexusFXConfig
from nexus_fx.model import NEXUSFX

# Create config
config = NexusFXConfig(
    pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    timeframes=['5m', '15m', '1H', '4H', '1D'],
    num_cms_levels=4,  # 4-level memory hierarchy
)

# Create model (15.4M parameters)
model = NEXUSFX(config)
```

## Training

```bash
# Train
python -m nexus_fx.scripts.train --data_path /path/to/data

# Evaluate
python -m nexus_fx.scripts.evaluate --checkpoint best_model.pt

# Backtest
python -m nexus_fx.scripts.backtest --checkpoint best_model.pt
```

## Testing

```bash
# Run all tests
pytest nexus_fx/tests/ -v
```

## Performance Metrics

The system computes:

**Prediction:**
- Direction accuracy
- Expected Calibration Error (ECE)

**Trading:**
- Sharpe ratio (annualized)
- Sortino ratio
- Maximum drawdown
- Win rate
- Profit factor

## Model Size

| Config | Parameters |
|--------|------------|
| Small (hidden=64, pairs=1) | ~180K |
| Medium (hidden=128, pairs=2) | ~2M |
| Large (hidden=256, pairs=4) | ~15M |

## Theoretical Foundation

**NSAM (Nested Sequential Associative Memory):**

Traditional neural networks: flat optimization problem
```
min L(θ)
```

NSAM: nested optimization
```
min L₁(θ₁, min L₂(θ₂, min L₃(θ₃, ...)))
     ↑fast    ↑medium  ↑slow
```

Each level optimizes at different timescales, creating a hierarchy that matches market structure.

## File Organization

```
nexus_fx/
├── model/          # 8 neural architecture components
├── data/           # 5 data pipeline modules  
├── optim/          # 2 custom optimizers
├── training/       # 3 training infrastructure modules
├── utils/          # 2 utility modules
├── scripts/        # 3 executable scripts
└── tests/          # 4 test suites
```

**Total: 36 Python files**

## Key Advantages

1. **Multi-timescale adaptation**: Fast adaptation to microstructure, slow persistence of macro regimes
2. **No catastrophic forgetting**: Slow memories preserve long-term knowledge
3. **Session awareness**: Different behaviors during Tokyo vs London-NY overlap
4. **Limited data requirements**: Works with just OHLC + timestamps
5. **Continual learning**: Can update on new data without full retraining

## Next Steps

1. Train on historical forex data
2. Backtest on out-of-sample period
3. Deploy for paper trading
4. Monitor calibration metrics
5. Gradually increase position sizing based on performance

## References

- NSAM framework: Nested optimization with associative memories
- HOPE paper: Hierarchical optimization via pairwise embeddings
- Delta GD: L2-regression based weight updates
- Multi-scale momentum: Muon optimizer family

---

Built for the forex trading community with ❤️
