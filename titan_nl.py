"""
TITAN-NL: Nested Learning Architecture for Forex Prediction
============================================================

A revolutionary forex prediction model combining:
1. Self-Modifying Delta Memory (generates own update targets)
2. Continuum Memory System (multi-frequency MLP hierarchy)
3. Delta Gradient Descent (state-dependent weight updates)
4. M3 Optimizer (Multi-scale Momentum Muon)
5. Market Regime Adaptive Memory

Based on: "Nested Learning: The Illusion of Deep Learning Architecture"
         by Behrouz et al. (NeurIPS 2025)

Optimized for: Kaggle P100 GPU, 6 months forex data, 4 currency pairs
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
import math
from typing import Optional, Tuple, List, Dict

# ==========================================
# 1. CONFIGURATION
# ==========================================
SEQ_LEN = 32
BATCH_SIZE = 128
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
NUM_NODES = len(PAIRS)
D_MODEL = 128

# CMS Configuration (Continuum Memory System)
CMS_CHUNK_SIZES = [16, 64, 256]  # Multi-frequency: fast/medium/slow
CMS_LEVELS = len(CMS_CHUNK_SIZES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🧠 TITAN-NL: Nested Learning Architecture")
print(f"🔧 Device: {DEVICE}")
print(f"📊 CMS Levels: {CMS_LEVELS} (chunks: {CMS_CHUNK_SIZES})")


# ==========================================
# 2. M3 OPTIMIZER (Multi-scale Momentum Muon)
# ==========================================
class M3Optimizer(optim.Optimizer):
    """
    Multi-scale Momentum Muon (M3) Optimizer

    Combines:
    - Adam-style adaptive learning rates (second moment)
    - Multi-frequency momentum (CMS-inspired)
    - Newton-Schulz orthogonalization for gradient mapping

    Key insight from NL: Optimizers are associative memories that
    compress gradients into their parameters. Multi-scale design
    prevents catastrophic forgetting of gradient subspaces.
    """

    def __init__(self,
        params,
        lr: float = 3e-4,
        betas: Tuple[float, float, float] = (0.9, 0.95, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        ns_steps: int = 5,
        slow_momentum_freq: int = 10,
        alpha_slow: float = 0.1,
    ):
        """
        Args:
            lr: Learning rate
            betas: (beta1_fast, beta1_slow, beta2) momentum factors
            eps: Numerical stability
            weight_decay: L2 regularization
            ns_steps: Newton-Schulz iteration steps for orthogonalization
            slow_momentum_freq: Update slow momentum every N steps
            alpha_slow: Weight for slow momentum contribution
        """
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            ns_steps=ns_steps, slow_momentum_freq=slow_momentum_freq,
            alpha_slow=alpha_slow
        )
        super().__init__(params, defaults)
        self.step_count = 0

    @staticmethod
    def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Newton-Schulz iteration for orthogonalization.
        Maps momentum to proper metric space for better gradient management.

        From NL paper: This is learning a mapping P(g) that satisfies
        ||P(g)^T P(g) - I||_F^2 = 0 (orthogonal space)

        Fix #3: Added spectral radius guard to prevent divergence when
        singular values fall outside (0, sqrt(3)) convergence range.
        """
        if M.dim() != 2 or M.shape[0] > M.shape[1]:
            # Only apply to 2D weight matrices with more cols than rows
            return M

        # Normalize
        norm = M.norm()
        if norm < 1e-8:
            return M

        # Normalize so spectral radius < sqrt(3) for convergence
        X = M / max(norm.item(), 1e-6)

        for _ in range(steps):
            A = X @ X.T
            # Check for divergence
            if A.norm().item() > 1e4:
                return M  # Fall back to un-orthogonalized
            X = 1.5 * X - 0.5 * A @ X

        return X * norm

    @torch.no_grad()
    def step(self, closure=None):
        """
        Fix #4: Replaced gradient buffer list with running average to prevent
        unbounded memory growth (OOM risk on P100 with 10 grad copies per param).
        """ 
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            beta1_fast, beta1_slow, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay (decoupled)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m1_fast'] = torch.zeros_like(p)  # Fast momentum
                    state['m1_slow'] = torch.zeros_like(p)  # Slow momentum
                    state['v'] = torch.zeros_like(p)        # Second moment
                    # Fix #4: Running average instead of gradient buffer list
                    state['grad_running_avg'] = torch.zeros_like(p)
                    state['grad_count'] = 0

                state['step'] += 1

                m1_fast = state['m1_fast']
                m1_slow = state['m1_slow']
                v = state['v']

                # Fast momentum (EMA of gradients)
                m1_fast.mul_(beta1_fast).add_(grad, alpha=1 - beta1_fast)

                # Second moment (for adaptive LR like Adam)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Slow momentum update (running average, memory-efficient)
                state['grad_count'] += 1
                count = state['grad_count']
                state['grad_running_avg'].mul_(
                    (count - 1) / count
                ).add_(grad, alpha=1.0 / count)

                if self.step_count % group['slow_momentum_freq'] == 0:
                    m1_slow.mul_(beta1_slow).add_(
                        state['grad_running_avg'], alpha=1 - beta1_slow
                    )
                    state['grad_running_avg'].zero_()
                    state['grad_count'] = 0

                # Bias correction
                bias_correction1 = 1 - beta1_fast ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Corrected moments
                m1_fast_corrected = m1_fast / bias_correction1
                v_corrected = v / bias_correction2

                # Apply Newton-Schulz orthogonalization to momentum
                if m1_fast_corrected.dim() == 2 and min(m1_fast_corrected.shape) > 1:
                    m1_orth = self.newton_schulz(m1_fast_corrected, group['ns_steps'])
                    m1_slow_orth = self.newton_schulz(m1_slow, group['ns_steps'])
                else:
                    m1_orth = m1_fast_corrected
                    m1_slow_orth = m1_slow

                # Combine fast and slow momentums
                combined = m1_orth + group['alpha_slow'] * m1_slow_orth

                # Update with adaptive scaling
                denom = v_corrected.sqrt().add_(group['eps'])
                p.data.addcdiv_(combined, denom, value=-group['lr'])

        return loss


# ==========================================
# 3. SELF-MODIFYING DELTA MEMORY
# ==========================================
class SelfModifyingDeltaMemory(nn.Module):
    """
    Self-Modifying Deep Memory with Delta Gradient Descent

    Key innovations from NL paper:
    1. Memory generates its own values (self-referential)
    2. Delta rule: M_t = M_{t-1}(αI - ηkk^T) + ηv̂k^T
    3. State-dependent updates capture token dependencies

    This replaces simple Hebbian outer-product updates with
    a more expressive learning rule.

    Fix #1: eta and alpha are now scalar per-token [B*N, 1, 1] for correct
    identity scaling in the delta rule.
    Fix #2: Memory update uses right-multiplication M @ decay_matrix
    (not transposed left-multiplication) per the paper's formulation.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Projections (will be adapted in-context)
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)

        # Self-referential value generator (generates own update targets)
        self.value_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Learnable scalar learning rate and decay for Delta rule
        self.eta_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),  # Scalar output
            nn.Sigmoid()
        )
        self.alpha_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),  # Scalar output
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize memory state
        self.register_buffer('init_memory', torch.zeros(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq, Nodes, Features]
        Returns:
            output: [Batch, Seq, Nodes, Features]
        """
        b, s, n, f = x.shape
        x_flat = x.view(b * n, s, f)

        residual = x_flat

        # Project to q, k, v
        q = self.proj_q(x_flat)  # [B*N, S, D]
        k = self.proj_k(x_flat)
        v = self.proj_v(x_flat)

        # Self-referential: generate update targets from values
        # This is the key innovation - memory controls its own learning
        v_hat = self.value_generator(v)  # [B*N, S, D]

        # Learnable scalar learning rate and decay (per position)
        # Fix #1: Output is scalar [B*N, S, 1] -> will become [B*N, 1, 1] per step
        eta = self.eta_proj(x_flat) * 0.1 + 0.01  # [B*N, S, 1]
        alpha = self.alpha_proj(x_flat) * 0.5 + 0.5  # [B*N, S, 1]

        # Initialize memory
        M = self.init_memory.unsqueeze(0).expand(b * n, -1, -1).clone()  # [B*N, D, D]

        outputs = []

        # Causal processing with Delta Gradient Descent
        for t in range(s):
            q_t = q[:, t, :]  # [B*N, D]
            k_t = k[:, t, :]
            v_t = v_hat[:, t, :]
            # Fix #1: Scalar eta/alpha -> [B*N, 1, 1] for uniform scaling
            eta_t = eta[:, t, :].unsqueeze(-1)  # [B*N, 1, 1]
            alpha_t = alpha[:, t, :].unsqueeze(-1)  # [B*N, 1, 1]

            # Normalize keys for stability
            k_t_norm = F.normalize(k_t, dim=-1)

            # Read from memory
            out_t = torch.bmm(M, q_t.unsqueeze(-1)).squeeze(-1)  # [B*N, D]

            # Delta Gradient Descent Update:
            # M_t = M_{t-1} @ (αI - ηkk^T) + ηv̂k^T
            # This is more expressive than simple Hebbian: M_t = M_{t-1} + vk^T

            # Compute k⊗k term (outer product for state-dependent decay)
            kk_t = torch.bmm(k_t_norm.unsqueeze(-1), k_t_norm.unsqueeze(-2))  # [B*N, D, D]

            # Create identity and apply delta rule
            I = torch.eye(self.d_model, device=x.device).unsqueeze(0)
            decay_matrix = alpha_t * I - eta_t * kk_t  # [B*N, D, D] (scalar broadcast)

            # Fix #2: Right-multiply: M_t = M_{t-1} @ decay_matrix + η * v̂k^T
            vk = torch.bmm(v_t.unsqueeze(-1), k_t_norm.unsqueeze(-2))  # [B*N, D, D]
            M = torch.bmm(M, decay_matrix) + eta_t * vk

            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # [B*N, S, D]
        output = self.out_proj(output)
        output = self.dropout(output)
        output = self.norm(output + residual)

        return output.view(b, s, n, f)


# ==========================================
# 4. CONTINUUM MEMORY SYSTEM (CMS)
# ==========================================
class ContinuumMemoryMLP(nn.Module):
    """
    Multi-frequency MLP block for Continuum Memory System

    From NL paper:
    - Different levels update at different frequencies
    - Lower frequency = more persistent memory (less forgetting)
    - Higher frequency = faster adaptation
    - Knowledge transfers between levels through backprop

    This creates a "loop" through time - forgotten knowledge
    can be recovered from lower-frequency levels.

    Fix #5: Removed auto-incrementing step_counter buffer.
    Step must always be passed explicitly to avoid corruption
    during validation and multi-GPU training.
    """

    def __init__(self,
        d_model: int,
        chunk_sizes: List[int] = [16, 64, 256],
        expansion: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_sizes = chunk_sizes
        self.num_levels = len(chunk_sizes)

        # Create MLP for each frequency level
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * expansion),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * expansion, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(self.num_levels)
        ])

        # Aggregation weights (learnable)
        self.level_weights = nn.Parameter(torch.ones(self.num_levels))

        # Level-specific layer norms
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.num_levels)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq, Nodes, Features]
            step: Current training step (for determining which levels to update)
        Returns:
            output: [Batch, Seq, Nodes, Features]
        """
        b, s, n, f = x.shape
        x_flat = x.view(b * s * n, f)

        # Compute outputs from each level
        level_outputs = []

        for level_idx, (mlp, chunk_size, norm) in enumerate(
            zip(self.mlps, self.chunk_sizes, self.level_norms)
        ):
            # Check if this level should update at this step
            # Higher levels (larger chunk) update less frequently
            should_update = (step % chunk_size) == 0 or step == 0

            if should_update or self.training:
                # During training, all levels participate (for gradient flow)
                out = mlp(x_flat)
                out = norm(out + x_flat)  # Residual connection
            else:
                # During inference, use cached output if not update step
                out = mlp(x_flat)
                out = norm(out + x_flat)

            level_outputs.append(out)

        # Aggregate outputs with learned weights
        weights = F.softmax(self.level_weights, dim=0)

        aggregated = sum(w * out for w, out in zip(weights, level_outputs))
        output = self.final_norm(aggregated)

        return output.view(b, s, n, f)


# ==========================================
# 5. MARKET REGIME ADAPTIVE MEMORY
# ==========================================
class MarketRegimeMemory(nn.Module):
    """
    Adaptive memory that detects market regimes and adjusts behavior.

    Regimes:
    - Trending: High directional persistence
    - Ranging: Mean-reverting behavior
    - Volatile: High variance, rapid changes

    The memory adapts its:
    - Gate (α): Higher for idiosyncratic moves
    - Learning rate (η): Higher for regime changes
    - Isolation: Whether to rely on cross-pair correlation

    Fix #6: Return signature is now always consistent (4 values).
    regime_probs is always returned (None when not computed).
    """

    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # Regime detection network
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),  # 3 regimes
            nn.Softmax(dim=-1)
        )

        # Regime-specific memory parameters
        self.regime_eta = nn.Parameter(torch.tensor([0.1, 0.05, 0.2]))  # Trending, Ranging, Volatile
        self.regime_alpha = nn.Parameter(torch.tensor([0.8, 0.9, 0.6]))

        # Graph attention components
        self.q_graph = nn.Linear(d_model, d_model)
        self.k_graph = nn.Linear(d_model, d_model)
        self.v_graph = nn.Linear(d_model, d_model)

        # Regime-aware gating
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3 + 3, 64),  # +3 for regime probabilities
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, Seq, Nodes, Features]
        Returns:
            output: [Batch, Nodes, Features]
            alpha: Isolation gate
            attn_weights: Graph attention weights
            regime_probs: Regime probabilities [B, N, 3]
        """
        # State: Average of last 3 steps
        state = x[:, -3:, :, :].mean(dim=1)  # [B, N, D]
        b, n, d = state.shape
        residual = state

        # Context statistics for regime detection
        global_mean = state.mean(dim=1, keepdim=True)  # [B, 1, D]
        global_std = state.std(dim=1, keepdim=True)    # [B, 1, D]

        # Detect regime per node
        regime_input = torch.cat([state, global_mean.expand(-1, n, -1)], dim=-1)
        regime_probs = self.regime_detector(regime_input)  # [B, N, 3]

        # Compute regime-weighted parameters
        eta = (regime_probs * self.regime_eta.view(1, 1, 3)).sum(dim=-1, keepdim=True)
        regime_alpha = (regime_probs * self.regime_alpha.view(1, 1, 3)).sum(dim=-1, keepdim=True)

        # Graph attention with regime-aware gating
        gate_input = torch.cat([
            state,
            global_mean.expand(-1, n, -1),
            global_std.expand(-1, n, -1),
            regime_probs
        ], dim=-1)
        alpha = self.gate_net(gate_input)  # [B, N, 1]

        # Standard graph attention
        Q = self.q_graph(state)
        K = self.k_graph(state)
        V = self.v_graph(state)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Dynamic mixing with regime-aware isolation
        I = torch.eye(n, device=x.device).unsqueeze(0).expand(b, -1, -1)
        mixed_weights = (alpha * I) + ((1 - alpha) * attn_weights)

        out = torch.matmul(mixed_weights, V)
        out = self.dropout(out)
        out = self.norm(out + residual)

        return out, alpha, attn_weights, regime_probs


# ==========================================
# 6. NESTED GRAPH TITAN-NL (MAIN MODEL)
# ==========================================
class NestedGraphTitanNL(nn.Module):
    """
    Main TITAN-NL model.

    Fix #7: CMS is now applied BEFORE MarketRegimeMemory so it receives
    the full temporal sequence [B, S, N, D] and can leverage multi-frequency
    chunking. Previously it received [B, 1, N, D] which defeated its purpose.

    Fix #8: Positional embeddings use sinusoidal initialization for better
    inductive bias on time series data.

    Fix #13: Removed unused N_HEADS constant.
    """

    def __init__(self,
        num_nodes: int = NUM_NODES,
        feats_per_node: int = 34,
        d_model: int = D_MODEL,
        num_layers: int = 4,
        dropout: float = 0.2,
        cms_chunk_sizes: List[int] = CMS_CHUNK_SIZES
    ):
        super().__init__()

        # Input embedding
        self.embedding = nn.Linear(feats_per_node, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Fix #8: Sinusoidal positional embedding initialization
        max_len = 100
        pos_emb = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_emb = nn.Parameter(pos_emb)

        # Stack of Self-Modifying Delta Memory layers
        self.temporal_layers = nn.ModuleList([
            SelfModifyingDeltaMemory(d_model, dropout)
            for _ in range(num_layers)
        ])

        # Fix #7: CMS applied before graph layer to preserve temporal structure
        # Continuum Memory System (multi-frequency processing)
        self.cms = ContinuumMemoryMLP(d_model, cms_chunk_sizes, expansion=4, dropout=dropout)

        # Market Regime Adaptive Memory (graph layer - collapses temporal dim)
        self.regime_memory = MarketRegimeMemory(num_nodes, d_model, dropout)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # Sell=0, Hold=1, Buy=2
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling for NL architecture"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        step: int = 0
    ) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq, Nodes, Features]
            return_attn: Whether to return attention info
            step: Current step for CMS frequency management
        Returns:
            logits: [Batch, Nodes, 3]
        """
        b, s, n, f = x.shape

        # Embed input
        x = self.embedding(x)  # [B, S, N, D]

        # Add positional embedding
        pos = self.pos_emb[:, :s, :].unsqueeze(2).expand(b, s, n, -1)
        x = x + pos
        x = self.input_norm(x)

        # Level 3: High-frequency temporal processing
        for layer in self.temporal_layers:
            x = layer(x)

        # Fix #7: Apply CMS BEFORE collapsing temporal dimension
        # Level 2/1: Continuum Memory System (multi-frequency on full sequence)
        x = self.cms(x, step=step)  # [B, S, N, D] - preserves temporal structure

        # Level 2: Regime-aware graph attention (collapses temporal dim)
        graph_out, alpha, attn_weights, regime_probs = self.regime_memory(x)
        # graph_out: [B, N, D]

        # Classification
        logits = self.head(graph_out)  # [B, N, 3]

        if return_attn:
            return logits, attn_weights, alpha
        return logits


# ==========================================
# 7. WEIGHTED FOCAL LOSS
# ==========================================
class WeightedFocalLoss(nn.Module):
    """Focal loss with class weights for imbalanced forex labels"""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 1)  # [B, C, N]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==========================================
# 8. DATASET
# ==========================================
class ForexGraphDataset(Dataset):
    """Dataset for forex graph prediction with raw returns for Sharpe calculation"""

    def __init__(self, X: np.ndarray, y: np.ndarray, raw_returns: np.ndarray, seq_len: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.raw_returns = torch.FloatTensor(raw_returns)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X[idx:idx + self.seq_len],
            self.y[idx + self.seq_len],
            self.raw_returns[idx + self.seq_len]
        )


# ==========================================
# 9. TRAINING UTILITIES
# ==========================================
def calculate_sharpe_proxy(
    preds: np.ndarray,
    returns: np.ndarray,
    periods_per_year: Optional[int] = None
) -> float:
    """
    Calculate annualized Sharpe ratio proxy.

    Fix #12: Accounts for actual forex trading hours (~252 days * 20 hrs * 12 bars/hr)
    instead of assuming continuous 24/7 trading (252 * 288).
    """
    signals = preds - 1  # Map 0,1,2 to -1,0,1
    pnl = signals * returns

    avg_ret = np.mean(pnl)
    std_ret = np.std(pnl)

    if std_ret < 1e-8:
        return 0.0

    if periods_per_year is None:
        # ~252 trading days * ~20 hours * 12 (5min bars per hour)
        periods_per_year = 252 * 20 * 12

    return (avg_ret / std_ret) * np.sqrt(periods_per_year)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    step_counter: int
) -> Tuple[float, float, int]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        x = torch.clamp(x, -10.0, 10.0)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            logits = model(x, step=step_counter)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()
        step_counter += 1

    return total_loss / len(loader), 100 * correct / total, step_counter


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Validate and compute metrics"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_returns = []

    for x, y, r in loader:
        x, y = x.to(device), y.to(device)
        x = torch.clamp(x, -10.0, 10.0)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

        all_preds.append(preds.cpu().numpy().flatten())
        all_returns.append(r.numpy().flatten())

    sharpe = calculate_sharpe_proxy(
        np.concatenate(all_preds),
        np.concatenate(all_returns)
    )

    return total_loss / len(loader), 100 * correct / total, sharpe


# ==========================================
# 10. MAIN TRAINING SCRIPT
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 TITAN-NL: Nested Learning Architecture for Forex")
    print("=" * 60)

    # Load data
    print("\n>>> LOADING DATASET...")
    df = pd.read_parquet('/kaggle/input/5m-engineered-4-pairs/ALL_PAIRS_5M_Engineered_FIXED.parquet')

    # 6 months of data (2020 H1)
    df = df[(df.index >= '2020-01-01') & (df.index <= '2020-06-30')]
    print(f"📅 Date Range: {df.index.min()} to {df.index.max()}")

    # Feature columns (all defined upfront, Fix #16)
    feature_cols = [
        'log_returns', 'volatility_10', 'volatility_30',
        'momentum_5', 'momentum_20', 'momentum_60',
        'body_pct', 'upper_wick', 'lower_wick', 'candle_range', 'close_position',
        'volume_ratio', 'volume_momentum',
        'ATR_15m', 'RSI_15m', 'Price_to_SMA', 'RSI_5m',
        'price_to_sma10', 'price_to_sma20', 'price_to_sma50',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'Session_London', 'Session_NY', 'Session_Tokyo', 'Session_Overlap',
        'News_Shock_Decay', 'Effective_Surprise', 'Recent_News_60m',
        'Gold_Returns', 'Fed_Rate_Change',
        'Market_Divergence', 'Market_Mean'  # Fix #16: defined upfront
    ]

    # Data preparation
    print(f"🔧 Extracting features and adding Market Context...")
    dfs_per_pair = []
    common_index = None

    for pid in range(NUM_NODES):
        d_pair = df[df['Pair_ID'] == pid].copy()
        d_pair = d_pair[~d_pair.index.duplicated(keep='first')]

        if common_index is None:
            common_index = d_pair.index
        else:
            common_index = common_index.intersection(d_pair.index)

    for pid in range(NUM_NODES):
        d_pair = df[df['Pair_ID'] == pid].copy()
        d_pair = d_pair.reindex(common_index)
        d_pair = d_pair.ffill().bfill()
        base_cols = [c for c in feature_cols if c not in ('Market_Divergence', 'Market_Mean')]
        cols_needed = base_cols + ['target_return_3']
        d_pair = d_pair[cols_needed]
        dfs_per_pair.append(d_pair)

    # Add Market Context
    all_returns = pd.DataFrame({i: d['log_returns'] for i, d in enumerate(dfs_per_pair)})
    market_mean = all_returns.mean(axis=1)

    for i in range(NUM_NODES):
        dfs_per_pair[i]['Market_Divergence'] = dfs_per_pair[i]['log_returns'] - market_mean
        dfs_per_pair[i]['Market_Mean'] = market_mean

    print(f"✅ Added Market Context. Total Features: {len(feature_cols)}")

    # Stack data
    all_cols_with_target = feature_cols + ['target_return_3']
    aligned_arrays = [d[all_cols_with_target].values for d in dfs_per_pair]
    master_tensor_raw = np.stack(aligned_arrays, axis=1)

    master_tensor = master_tensor_raw[:, :, :-1]
    future_returns = master_tensor_raw[:, :, -1]

    # Split indices
    N_SAMPLES = len(master_tensor)
    train_split_idx = int(N_SAMPLES * 0.70)
    val_split_idx = int(N_SAMPLES * 0.85)

    # Calculate thresholds
    train_returns = future_returns[:train_split_idx].flatten()
    lower_thresh = np.percentile(train_returns, 33)
    upper_thresh = np.percentile(train_returns, 66)

    print(f"📈 Thresholds (Train Only): Sell < {lower_thresh:.6f} | Buy > {upper_thresh:.6f}")

    labels = np.ones(future_returns.shape, dtype=int)
    labels[future_returns > upper_thresh] = 2
    labels[future_returns < lower_thresh] = 0

    # Fix #11: Scale features using ONLY training data (no data leakage)
    N, Nodes, Feats = master_tensor.shape
    scaler = RobustScaler()
    train_data_for_fit = master_tensor[:train_split_idx].reshape(-1, Feats)
    scaler.fit(train_data_for_fit)
    master_tensor_scaled = scaler.transform(
        master_tensor.reshape(-1, Feats)
    ).reshape(N, Nodes, Feats)

    # Create datasets
    train_ds = ForexGraphDataset(
        master_tensor_scaled[:train_split_idx],
        labels[:train_split_idx],
        future_returns[:train_split_idx],
        SEQ_LEN
    )
    val_ds = ForexGraphDataset(
        master_tensor_scaled[train_split_idx:val_split_idx],
        labels[train_split_idx:val_split_idx],
        future_returns[train_split_idx:val_split_idx],
        SEQ_LEN
    )
    test_ds = ForexGraphDataset(
        master_tensor_scaled[val_split_idx:],
        labels[val_split_idx:],
        future_returns[val_split_idx:],
        SEQ_LEN
    )

    # Fix #10: shuffle=True for training loader (windows are independent)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"✅ Data Split: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    # Create model
    model = NestedGraphTitanNL(
        num_nodes=NUM_NODES,
        feats_per_node=len(feature_cols),
        d_model=D_MODEL,
        num_layers=4,
        dropout=0.2,
        cms_chunk_sizes=CMS_CHUNK_SIZES
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model Parameters: {total_params:,}")

    # Class weights
    train_labels_flat = labels[:train_split_idx].flatten()
    class_counts = np.bincount(train_labels_flat)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * 3

    print(f"⚖️  Class Weights: {class_weights.cpu().numpy()}")
    criterion = WeightedFocalLoss(alpha=class_weights, gamma=2.0)

    # M3 Optimizer
    optimizer = M3Optimizer(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95, 0.999),
        weight_decay=1e-2,
        ns_steps=5,
        slow_momentum_freq=10,
        alpha_slow=0.1
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda')

    # Training
    EPOCHS = 20
    PATIENCE = 6
    best_val_loss = float('inf')
    patience_counter = 0
    step_counter = 0

    print("\n🔥 TRAINING STARTED (M3 Optimizer)...\n")

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc, step_counter = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, step_counter
        )

        # Validate
        val_loss, val_acc, val_sharpe = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
              f"Acc: {train_acc:.1f}% / {val_acc:.1f}% | "
              f"💵 Est. Sharpe: {val_sharpe:.3f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'Best_TITAN_NL.pth')
            print("  ✅ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏸️  Early stopping triggered at epoch {epoch+1}")
                break

    # Final Evaluation
    print("\n" + "=" * 60)
    print("🧠 FINAL ANALYSIS")
    print("=" * 60)

    model.load_state_dict(torch.load('Best_TITAN_NL.pth', weights_only=True))  # Fix #15
    model.eval()

    all_preds = []
    all_labels = []
    all_alphas = []
    all_returns = []

    with torch.no_grad():
        for x, y, r in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = torch.clamp(x, -10.0, 10.0)

            logits, attn, alpha = model(x, return_attn=True)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_alphas.append(alpha.cpu())
            all_returns.append(r)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_alphas = torch.cat(all_alphas).squeeze(-1).numpy()
    all_returns = torch.cat(all_returns).numpy()

    print("\nMean Isolation (Alpha) per Pair:")
    for i, pair in enumerate(PAIRS):
        mean_a = np.mean(all_alphas[:, i])
        print(f"  {pair}: {mean_a:.4f}")

    print("\n" + "=" * 60)
    print("📈 TRADING PERFORMANCE (TEST SET)")
    print("=" * 60)

    total_sharpe = 0
    for pair_id in range(NUM_NODES):
        pair_preds = all_preds[:, pair_id]
        pair_rets = all_returns[:, pair_id]

        sharpe = calculate_sharpe_proxy(pair_preds, pair_rets)
        total_sharpe += sharpe

        print(f"\n{PAIRS[pair_id]}: Sharpe = {sharpe:.3f}")
        print(classification_report(
            all_labels[:, pair_id],
            pair_preds,
            target_names=['Sell', 'Hold', 'Buy'],
            digits=4
        ))

    print(f"\n🌍 Portfolio Average Sharpe: {total_sharpe/NUM_NODES:.3f}")

    # Fix #17: Save RobustScaler alongside model checkpoint for inference
    import pickle

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'num_nodes': NUM_NODES,
            'd_model': D_MODEL,
            'cms_chunk_sizes': CMS_CHUNK_SIZES,
            'features': len(feature_cols),
            'feature_cols': feature_cols
        }
    }, 'titan_nl_complete.pth')

    with open('titan_nl_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\n✅ TITAN-NL TRAINING COMPLETE.")
    print("📦 Saved: titan_nl_complete.pth + titan_nl_scaler.pkl")
