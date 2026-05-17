#!/usr/bin/env python
"""
Positional Embedding Ablation Experiment
=========================================
Compares three Transformer variants on US CPI inflation forecasting:

  1. NoPE        – Transformer with NO positional encoding
  2. AbsolutePE  – Fixed sinusoidal absolute PE (Vaswani et al. 2017)
  3. RelativePE  – Learned relative position bias per head (T5-style, Shaw et al. 2018)

Research question
-----------------
Does positional encoding help a Transformer forecast inflation, and if so,
does *how* position is encoded (absolute vs relative) matter?

Architecture (held fixed across all three variants for fair comparison)
-----------------------------------------------------------------------
  d_model = 32, n_heads = 2, n_layers = 1
  dim_feedforward = 64 (2 × d_model)
  Readout: mean-pooling over sequence → Linear(32, 1)

The only difference between variants is whether and how position information
enters the model:
  NoPE       – attention sees content only; self-attention is permutation-equivariant
  AbsolutePE – token at position t receives a fixed sine/cosine encoding of t
  RelativePE – pre-softmax attention logit A[i,j] gets a learned bias b_h(i−j)
               per head h; position is encoded as distance, not absolute index

Data & evaluation (mirrors main.py rolling-window setup)
---------------------------------------------------------
  Full series: 1960-01 to 2020-06 (SA and/or NA monthly CPI % change)
  Train:       1960-01 to 1989-12  (fixed; hyperparams selected on this split)
  Test:        1995-01 to 2015-06  (rolling window; model re-trained each origin)
  Lags (p):    24 (fixed; no IC selection to isolate PE effect)
  Horizons:    h = 1 … 12 months ahead (iterated one-step forecasts)

Outputs  →  results/pe_experiment/
-----------------------------------------
  forecasts_<dtype>.pkl    cached rolling-window predictions
  msfe_<dtype>.csv         MSFE per variant × horizon
  mafe_<dtype>.csv         MAFE per variant × horizon
  pe_msfe_<dtype>.pdf      Line plot: MSFE by horizon for all 3 variants
  pe_relative_msfe_<dtype>.pdf   MSFE ratio relative to NoPE baseline

Usage
-----
  python experiment_pe.py               # SA data, default settings
  python experiment_pe.py --na          # NA data
  python experiment_pe.py --both        # both SA and NA
  python experiment_pe.py --rerun       # ignore cached forecasts, re-run
  python experiment_pe.py --epochs 200  # override training epochs
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Add project root to path so we can import existing modules ────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data import download_data, make_lstm_sequence
from models import train_model, DEVICE

# ══════════════════════════════════════════════════════════════════════════════
# Default experiment configuration
# ══════════════════════════════════════════════════════════════════════════════

D_MODEL    = 32      # must be divisible by N_HEADS
N_HEADS    = 2
N_LAYERS   = 1
DIM_FF     = D_MODEL * 2   # 64
MAX_LAG    = 24      # sequence length / lag order (fixed, no IC selection)
LR         = 0.001
EPOCHS     = 50
TRAIN_END  = "1989-12"
TEST_START = "1995-01"
TEST_END   = "2015-06"
H_MAX      = 12
SEED       = 42
OUT_DIR    = Path("results/pe_experiment")


# ══════════════════════════════════════════════════════════════════════════════
# Model variant 1 – No positional encoding
# ══════════════════════════════════════════════════════════════════════════════

class NoPETransformer(nn.Module):
    """
    Transformer encoder with absolutely no positional information.

    The standard multi-head self-attention is permutation-equivariant: if you
    shuffle the input tokens the output (after mean-pooling) is identical.
    This variant is therefore a pure *bag-of-lags* model — it sees which lag
    values are present but not which lag they correspond to.
    """
    def __init__(self, n_hidden: int = D_MODEL):
        super().__init__()
        d = n_hidden
        self.input_proj = nn.Linear(1, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=N_HEADS, dim_feedforward=d * 2,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.fc = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)          # (B, seq, 1)
        x = self.input_proj(x)           # (B, seq, d) — NO positional signal
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Model variant 2 – Sinusoidal absolute positional encoding
# ══════════════════════════════════════════════════════════════════════════════

class AbsolutePETransformer(nn.Module):
    """
    Transformer encoder with fixed (parameter-free) sinusoidal absolute PE.

    Position index t gets encoding PE[t, 2k]   = sin(t / 10000^(2k/d))
                                    PE[t, 2k+1] = cos(t / 10000^(2k/d))
    Added to each token *after* the input projection, before self-attention.

    This matches the existing TransformerModel in models.py exactly.
    """
    def __init__(self, n_hidden: int = D_MODEL):
        super().__init__()
        d = n_hidden
        self.input_proj = nn.Linear(1, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=N_HEADS, dim_feedforward=d * 2,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.fc = nn.Linear(d, 1)

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int,
                       device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        i   = torch.arange(0, d_model, 2, device=device).float()
        div = 10_000 ** (i / d_model)
        pe  = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        return pe.unsqueeze(0)           # (1, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self._sinusoidal_pe(x.size(1), x.size(2), x.device)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Model variant 3 – Relative positional encoding (learned bias in attention)
# ══════════════════════════════════════════════════════════════════════════════

class _RelativeAttention(nn.Module):
    """
    Multi-head self-attention with T5-style learned relative position bias.

    For each attention head h and query–key pair (i, j), a scalar bias
    b_h(i − j) is added to the pre-softmax logit A[h, i, j].  The bias
    depends *only* on the signed distance (i − j), not on i or j separately,
    making the model invariant to absolute position (shift-equivariant).

    Distances are clipped to [−(max_len−1), max_len−1] so the embedding
    table has size (2·max_len − 1) entries.

    Reference: Raffel et al. (2020) "Exploring the Limits of Transfer Learning
    with a Unified Text-to-Text Transformer" (T5), Section 3.4.
    """
    def __init__(self, d_model: int, nhead: int, max_len: int):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead   = nhead
        self.d_head  = d_model // nhead
        self.scale   = self.d_head ** -0.5
        self.max_len = max_len

        # Embedding table: one scalar per (relative distance, head)
        self.rel_bias = nn.Embedding(2 * max_len - 1, nhead)

        # Standard QKV projections (no bias, following T5 convention)
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh   = self.nhead, self.d_head

        # Project and reshape to (B, H, L, Dh)
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        # Content-based attention scores: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Relative position bias
        # delta[i, j] = i − j, clipped to [−(M−1), M−1], shifted to [0, 2M−2]
        pos   = torch.arange(L, device=x.device)
        delta = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(
            -(self.max_len - 1), self.max_len - 1
        ) + (self.max_len - 1)                                # (L, L) long tensor
        bias  = self.rel_bias(delta).permute(2, 0, 1).unsqueeze(0)  # (1, H, L, L)

        attn = torch.softmax(scores + bias, dim=-1)           # (B, H, L, L)
        out  = torch.matmul(attn, v)                          # (B, H, L, Dh)
        out  = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class _RelativeEncoderLayer(nn.Module):
    """Single Transformer encoder layer using relative attention (no absolute PE)."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, max_len: int):
        super().__init__()
        self.attn  = _RelativeAttention(d_model, nhead, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class RelativePETransformer(nn.Module):
    """
    Transformer encoder with learned relative position bias.

    The input projection adds NO positional signal to token embeddings.
    Instead, position is captured entirely within the attention mechanism:
    the attention score between query i and key j is shifted by a learned
    bias that depends on the signed distance (i − j).

    This gives the model two key properties absent from AbsolutePE:
    (a) *Translation equivariance*: shifting the sequence does not change
        relative distances, so the model generalises across starting points.
    (b) *Explicit distance weighting*: the model can learn that nearby lags
        (small |i−j|) are more informative without hard-coding it.
    """
    def __init__(self, n_hidden: int = D_MODEL, max_len: int = MAX_LAG + 1):
        super().__init__()
        d = n_hidden
        self.input_proj = nn.Linear(1, d)
        self.encoder    = _RelativeEncoderLayer(
            d_model=d, nhead=N_HEADS, dim_feedforward=d * 2, max_len=max_len,
        )
        self.fc = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)           # (B, seq, d) — no absolute PE
        x = self.encoder(x)              # position enters only via attention bias
        return self.fc(x.mean(dim=1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Variant registry
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS: dict[str, type] = {
    "NoPE":       NoPETransformer,
    "AbsolutePE": AbsolutePETransformer,
    "RelativePE": RelativePETransformer,
}

VARIANT_DISPLAY = {
    "NoPE":       "No PE",
    "AbsolutePE": "Absolute PE (sinusoidal)",
    "RelativePE": "Relative PE (learned bias)",
}

COLORS = {
    "NoPE":       "#d62728",   # red
    "AbsolutePE": "#1f77b4",   # blue
    "RelativePE": "#2ca02c",   # green
}

LINESTYLES = {
    "NoPE":       "--",
    "AbsolutePE": "-",
    "RelativePE": "-.",
}


# ══════════════════════════════════════════════════════════════════════════════
# Forecasting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iterated_forecast(model: nn.Module, context: np.ndarray,
                       p: int, h_max: int, device: str) -> np.ndarray:
    """
    Generate h = 1 … h_max iterated one-step-ahead forecasts.

    context : recent observations, oldest first, length >= p.
    p       : sequence / lag length fed to the model.

    The model receives the sequence [y_{t-p}, …, y_{t-1}] (oldest first),
    predicts y_t, appends it to the buffer, and repeats.
    """
    buf = list(context[-p:])           # oldest → newest, length = p
    forecasts: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(h_max):
            seq = np.array(buf[-p:], dtype=np.float32)
            xt  = torch.tensor(seq, device=device).unsqueeze(0)  # (1, p)
            pred = model(xt).item()
            forecasts.append(pred)
            buf.append(pred)
    return np.array(forecasts)


def rolling_window_forecast(
    variant_name: str,
    y: pd.Series,
    epochs: int,
    device: str,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Rolling-window real-time forecasting for one Transformer variant.

    At each origin t in [TEST_START, TEST_END]:
      1. Fit variant on all observations strictly before t.
      2. Iteratively generate h = 1 … H_MAX forecasts.

    Returns
    -------
    forecasts : (N_origins, H_MAX) array
    origins   : DatetimeIndex of forecast origins
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ModelClass = VARIANTS[variant_name]
    test_idx   = y[TEST_START:TEST_END].index
    N          = len(test_idx)
    forecasts  = np.full((N, H_MAX), np.nan)

    for i, date in enumerate(test_idx):
        y_avail = y[:date].iloc[:-1].values.astype(np.float32)

        # Build supervised dataset with oldest-first sequences
        X, Y_ = make_lstm_sequence(y_avail, MAX_LAG)

        model = ModelClass()
        train_model(model, X, Y_, lr=LR, epochs=epochs, device=device)

        fc = _iterated_forecast(model, y_avail, MAX_LAG, H_MAX, device)
        forecasts[i, :len(fc)] = fc

        if (i + 1) % 60 == 0 or (i + 1) == N:
            print(f"    [{variant_name}]  {i+1:3d}/{N} origins", flush=True)

    return forecasts, test_idx


def compute_errors(
    forecasts: np.ndarray,
    y: pd.Series,
    origins: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align h-step-ahead forecasts with realised values.

    Returns
    -------
    msfe : (H_MAX,)  mean squared forecast error per horizon
    mafe : (H_MAX,)  mean absolute forecast error per horizon
    """
    y_arr = y.values
    y_idx = y.index
    msfe  = np.full(H_MAX, np.nan)
    mafe  = np.full(H_MAX, np.nan)

    for h in range(1, H_MAX + 1):
        sq, ab = [], []
        for i, orig in enumerate(origins):
            ti = y_idx.get_indexer([orig], method="nearest")[0] + h
            if ti >= len(y_arr):
                continue
            fc = forecasts[i, h - 1]
            if np.isnan(fc):
                continue
            err = fc - y_arr[ti]
            sq.append(err ** 2)
            ab.append(abs(err))
        if sq:
            msfe[h - 1] = float(np.mean(sq))
            mafe[h - 1] = float(np.mean(ab))

    return msfe, mafe


# ══════════════════════════════════════════════════════════════════════════════
# Parameter count utility
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_param_summary() -> None:
    print("\n── Parameter counts (d_model={}, nhead={}) ──────────────────".format(
        D_MODEL, N_HEADS))
    for name, Cls in VARIANTS.items():
        m = Cls()
        print(f"  {VARIANT_DISPLAY[name]:<35s} {count_params(m):>6,} params")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════

def print_msfe_table(
    msfe_dict: dict[str, np.ndarray],
    mafe_dict: dict[str, np.ndarray],
    dtype_label: str,
) -> None:
    vnames = list(VARIANTS.keys())
    col_w  = 14

    header = f"{'h':>3}" + "".join(f"{VARIANT_DISPLAY[v]:>{col_w}}" for v in vnames)
    sep    = "-" * len(header)
    print(f"\n{'='*60}")
    print(f"MSFE — {dtype_label.upper()} data | test: {TEST_START} – {TEST_END}")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for h in range(1, H_MAX + 1):
        row = f"{h:>3}"
        for v in vnames:
            val = msfe_dict[v][h - 1] if v in msfe_dict else np.nan
            row += f"{val:>{col_w}.5f}"
        print(row)

    print(f"\n{'='*60}")
    print(f"MSFE ratio relative to NoPE baseline")
    print(f"{'='*60}")
    header2 = f"{'h':>3}" + "".join(
        f"{VARIANT_DISPLAY[v]:>{col_w}}" for v in vnames if v != "NoPE"
    )
    print(header2)
    print(sep)
    base = msfe_dict.get("NoPE", np.ones(H_MAX))
    for h in range(1, H_MAX + 1):
        row = f"{h:>3}"
        for v in vnames:
            if v == "NoPE":
                continue
            ratio = msfe_dict[v][h - 1] / base[h - 1] if base[h - 1] > 0 else np.nan
            row  += f"{ratio:>{col_w}.4f}"
        print(row)
    print()


def save_csv_tables(
    msfe_dict: dict[str, np.ndarray],
    mafe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
) -> None:
    horizons = list(range(1, H_MAX + 1))

    msfe_rows = [{"h": h, **{v: msfe_dict[v][h - 1] for v in VARIANTS}} for h in horizons]
    mafe_rows = [{"h": h, **{v: mafe_dict[v][h - 1] for v in VARIANTS}} for h in horizons]

    pd.DataFrame(msfe_rows).to_csv(out_dir / f"msfe_{dtype_label}.csv", index=False)
    pd.DataFrame(mafe_rows).to_csv(out_dir / f"mafe_{dtype_label}.csv", index=False)
    print(f"  Tables saved → {out_dir}/msfe_{dtype_label}.csv, mafe_{dtype_label}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_msfe_by_horizon(
    msfe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
) -> None:
    """MSFE per horizon for all three variants (absolute values)."""
    horizons = np.arange(1, H_MAX + 1)
    fig, ax  = plt.subplots(figsize=(8, 4.5))

    for vname in VARIANTS:
        ax.plot(
            horizons, msfe_dict[vname],
            color=COLORS[vname], ls=LINESTYLES[vname], marker="o", ms=4,
            label=VARIANT_DISPLAY[vname],
        )

    ax.set_xlabel("Forecast horizon h (months)", fontsize=11)
    ax.set_ylabel("MSFE", fontsize=11)
    ax.set_title(
        f"MSFE by Horizon — Positional Embedding Ablation\n"
        f"({dtype_label.upper()} data, test: {TEST_START}–{TEST_END})",
        fontsize=11,
    )
    ax.set_xticks(horizons)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = out_dir / f"pe_msfe_{dtype_label}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {out}")


def plot_relative_msfe(
    msfe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
) -> None:
    """MSFE ratio relative to NoPE baseline — shows how much PE helps."""
    horizons = np.arange(1, H_MAX + 1)
    base     = msfe_dict["NoPE"]
    fig, ax  = plt.subplots(figsize=(8, 4.5))

    ax.axhline(1.0, color="grey", ls=":", lw=1.2, label="NoPE baseline (ratio = 1)")

    for vname in ["AbsolutePE", "RelativePE"]:
        ratio = msfe_dict[vname] / base
        ax.plot(
            horizons, ratio,
            color=COLORS[vname], ls=LINESTYLES[vname], marker="o", ms=4,
            label=VARIANT_DISPLAY[vname],
        )

    ax.set_xlabel("Forecast horizon h (months)", fontsize=11)
    ax.set_ylabel("MSFE / MSFE(NoPE)", fontsize=11)
    ax.set_title(
        f"Relative MSFE vs No-PE Baseline\n"
        f"({dtype_label.upper()} data, test: {TEST_START}–{TEST_END})",
        fontsize=11,
    )
    ax.set_xticks(horizons)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = out_dir / f"pe_relative_msfe_{dtype_label}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {out}")


def plot_rolling_msfe_over_time(
    forecasts_dict: dict[str, np.ndarray],
    y: pd.Series,
    origins: pd.DatetimeIndex,
    out_dir: Path,
    dtype_label: str,
    window: int = 12,
) -> None:
    """
    Rolling-window MSFE at h=1 over the test period.
    Shows whether PE advantages are concentrated in particular sub-periods.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for vname, fc in forecasts_dict.items():
        sq_errs = []
        for i, orig in enumerate(origins):
            ti = y.index.get_indexer([orig], method="nearest")[0] + 1
            if ti < len(y):
                err = fc[i, 0] - y.iloc[ti]
                sq_errs.append(err ** 2)
            else:
                sq_errs.append(np.nan)
        roll = pd.Series(sq_errs, index=origins).rolling(window, min_periods=1).mean()
        ax.plot(
            origins, roll,
            color=COLORS[vname], ls=LINESTYLES[vname], lw=1.0,
            label=VARIANT_DISPLAY[vname],
        )

    ax.set_xlabel("Forecast origin", fontsize=11)
    ax.set_ylabel(f"MSFE ({window}m rolling, h=1)", fontsize=11)
    ax.set_title(
        f"Rolling MSFE over Time — Positional Embedding Ablation\n"
        f"({dtype_label.upper()} data)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = out_dir / f"pe_rolling_msfe_{dtype_label}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-data-type experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_for_series(y: pd.Series, dtype_label: str, epochs: int,
                   device: str, rerun: bool) -> None:
    cache_file = OUT_DIR / f"forecasts_{dtype_label}.pkl"

    # ── Load or compute rolling forecasts ─────────────────────────────────────
    if cache_file.exists() and not rerun:
        print(f"\nLoading cached forecasts ← {cache_file}")
        with open(cache_file, "rb") as f:
            forecasts_dict, origins = pickle.load(f)
    else:
        forecasts_dict: dict[str, np.ndarray] = {}
        origins = None

        for vname in VARIANTS:
            print(f"\n── {VARIANT_DISPLAY[vname]} [{dtype_label.upper()}] ──")
            fc, orig = rolling_window_forecast(vname, y, epochs, device)
            forecasts_dict[vname] = fc
            if origins is None:
                origins = orig

        with open(cache_file, "wb") as f:
            pickle.dump((forecasts_dict, origins), f)
        print(f"\nForecasts cached → {cache_file}")

    # ── Compute errors ────────────────────────────────────────────────────────
    msfe_dict: dict[str, np.ndarray] = {}
    mafe_dict: dict[str, np.ndarray] = {}
    for vname, fc in forecasts_dict.items():
        msfe, mafe = compute_errors(fc, y, origins)
        msfe_dict[vname] = msfe
        mafe_dict[vname] = mafe

    # ── Report ────────────────────────────────────────────────────────────────
    print_msfe_table(msfe_dict, mafe_dict, dtype_label)
    save_csv_tables(msfe_dict, mafe_dict, OUT_DIR, dtype_label)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_msfe_by_horizon(msfe_dict, OUT_DIR, dtype_label)
    plot_relative_msfe(msfe_dict, OUT_DIR, dtype_label)
    plot_rolling_msfe_over_time(forecasts_dict, y, origins, OUT_DIR, dtype_label)


# ══════════════════════════════════════════════════════════════════════════════
# CLI and entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Positional embedding ablation for inflation forecasting Transformers."
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--sa",   action="store_true", default=True,
                     help="Run on seasonally adjusted data (default)")
    grp.add_argument("--na",   action="store_true", default=False,
                     help="Run on non-adjusted data only")
    grp.add_argument("--both", action="store_true", default=False,
                     help="Run on both SA and NA data")
    p.add_argument("--rerun",  action="store_true", default=False,
                   help="Ignore cached forecasts and re-run from scratch")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help=f"Training epochs per origin fit (default: {EPOCHS})")
    p.add_argument("--device", default="auto",
                   help="Device: 'auto' | 'cpu' | 'cuda' | 'mps'")
    p.add_argument("--cache-dir", default=".",
                   help="Directory where cpi_cache.csv is stored")
    return p.parse_args()


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return DEVICE
    return raw


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    epochs = args.epochs

    # Determine which series to run
    if args.both:
        dtypes = ["sa", "na"]
    elif args.na:
        dtypes = ["na"]
    else:
        dtypes = ["sa"]

    # Setup output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Banner
    print("\n" + "=" * 65)
    print("  Positional Embedding Ablation — Inflation Forecasting")
    print("=" * 65)
    print(f"  Variants  : {', '.join(VARIANT_DISPLAY.values())}")
    print(f"  d_model   : {D_MODEL}  |  n_heads : {N_HEADS}  |  n_layers : {N_LAYERS}")
    print(f"  Lags (p)  : {MAX_LAG}")
    print(f"  Epochs    : {epochs}  |  LR : {LR}")
    print(f"  Test      : {TEST_START} – {TEST_END}")
    print(f"  Device    : {device}")
    print(f"  Output    : {OUT_DIR}/")

    print_param_summary()

    # Load data
    print("Loading CPI data …")
    y_sa, y_na = download_data(cache_dir=args.cache_dir)

    series_map = {"sa": y_sa, "na": y_na}

    for dtype_label in dtypes:
        print(f"\n{'='*65}")
        print(f"  Data series: {dtype_label.upper()}")
        print(f"{'='*65}")
        run_for_series(
            series_map[dtype_label], dtype_label,
            epochs=epochs, device=device, rerun=args.rerun,
        )

    print(f"\nAll done. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
