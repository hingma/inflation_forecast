#!/usr/bin/env python
"""
Comprehensive Model Comparison Experiment
==========================================
Jointly evaluates all 6 models on US CPI inflation rolling-window forecasting:

  Baseline models (from models.py)
  ---------------------------------
  AR         – AutoRegressive linear model  (no hidden layer)
  NN         – One hidden layer, ReLU activation
  LSTM       – Single-layer LSTM, final hidden state → scalar

  Transformer PE variants (from experiment_pe.py)
  ------------------------------------------------
  TF-NoPE    – Transformer, no positional encoding  (permutation-equivariant)
  TF-AbsPE   – Transformer, sinusoidal absolute PE  (≡ TransformerModel in models.py)
  TF-RelPE   – Transformer, learned relative position bias  (T5-style)

Research questions
------------------
1. Do any Transformer PE variants beat LSTM and/or classical NN baselines?
2. At which horizons (h=1…12) is the PE choice most consequential?
3. Does any single model dominate across the full h=1…12 range?

Architecture & hyperparameters
-------------------------------
Fixed to match config.yml user_params for fair comparison:
  AR:   p=24, lr=0.003, epochs=50
  NN:   p=24, n_hidden=20, lr=0.001, epochs=50
  LSTM: p=24, n_hidden=50, lr=0.001, epochs=50
  TF-*: p=24, n_hidden=32, lr=0.001, epochs=50

Evaluation methodology
-----------------------
  Rolling-window real-time forecasting (mirrors main.py):
    Training: 1960-01 → 1989-12 (fixed initial window, grows each origin)
    Test:     1995-01 → 2015-06 (N≈245 origins)
    Lag order: p=24 (fixed; no IC selection)
    Horizons:  h = 1 … 12 months ahead (iterated one-step forecasts)

  Metrics:
    MSFE  = mean squared forecast error  (per horizon)
    MAFE  = mean absolute forecast error (per horizon)
    RMSFE = MSFE_model / MSFE_AR         (< 1 → model beats AR baseline)

Outputs  →  results/all_models/
-------------------------------
  forecasts_<dtype>.pkl         cached rolling forecasts (re-run with --rerun)
  msfe_<dtype>.csv              MSFE table: models × horizons
  mafe_<dtype>.csv              MAFE table: models × horizons
  rmsfe_<dtype>.csv             RMSFE table relative to AR baseline
  summary_<dtype>.csv           horizon-averaged MSFE / MAFE / RMSFE per model
  all_msfe_horizon_<dtype>.pdf  line plot: MSFE vs h for all 6 models
  all_rmsfe_vs_ar_<dtype>.pdf   RMSFE vs h (ratio relative to AR baseline)
  all_msfe_bars_<dtype>.pdf     grouped bar chart at h=1,3,6,12
  all_rolling_msfe_<dtype>.pdf  12-month rolling MSFE at h=1 over time

Usage
-----
  python experiment_all_models.py               # SA data, default settings
  python experiment_all_models.py --na          # NA data only
  python experiment_all_models.py --both        # SA + NA
  python experiment_all_models.py --rerun       # discard cache, re-run
  python experiment_all_models.py --epochs 200  # override epochs for all NN models
  python experiment_all_models.py --subset 60   # run on first 60 test origins (smoke test)
  python experiment_all_models.py --rolling-start 2000-01 --rolling-end 2010-06
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data import download_data, make_lag_matrix, make_lstm_sequence
from models import ARModel, NNModel, LSTMModel, train_model, DEVICE
from experiment_pe import NoPETransformer, AbsolutePETransformer, RelativePETransformer
from forecasting import sarima_forecast

# ══════════════════════════════════════════════════════════════════════════════
# Experiment configuration
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_END  = "1989-12"
TEST_START = "1995-01"
TEST_END   = "2015-06"
H_MAX      = 12
SEED       = 42
OUT_DIR    = Path("results/all_models")

# ── Per-model hyperparameters (matches config.yml user_params) ────────────────
_AR_PARAMS   = dict(max_lag=24, lr=0.003, epochs=50)
_NN_PARAMS   = dict(max_lag=24, n_hidden=20, lr=0.001, epochs=50)
_LSTM_PARAMS = dict(max_lag=24, n_hidden=50, lr=0.001, epochs=100)
_TF_PARAMS   = dict(max_lag=24, n_hidden=32, lr=0.001, epochs=50)


# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════
# Each entry:
#   display  – label for tables and plots
#   group    – "baseline" or "transformer" (for visual grouping)
#   seq      – True → oldest-first sequence input (LSTM / Transformer)
#              False → flat lag-1-first vector input (AR / NN)
#   factory  – callable(p: int) → nn.Module
#   params   – hyperparameter dict

REGISTRY: dict[str, dict] = {
    "AR": dict(
        display="AR",
        group="baseline",
        seq=False,
        factory=lambda p: ARModel(p),
        params=_AR_PARAMS,
    ),
    "NN": dict(
        display="NN",
        group="baseline",
        seq=False,
        factory=lambda p: NNModel(p, n_hidden=_NN_PARAMS["n_hidden"]),
        params=_NN_PARAMS,
    ),
    "LSTM": dict(
        display="LSTM",
        group="baseline",
        seq=True,
        factory=lambda p: LSTMModel(n_hidden=_LSTM_PARAMS["n_hidden"]),
        params=_LSTM_PARAMS,
    ),
    # ── Statistical baseline ──────────────────────────────────────────────────
    # SARIMA has no factory/seq/epochs; rolling_window_forecast dispatches on
    # kind="statistical" and calls sarima_forecast() directly.
    "SARIMA": dict(
        display="SARIMA",
        group="baseline",
        seq=False,
        kind="statistical",
        params={},
    ),
    "TF-NoPE": dict(
        display="TF-NoPE",
        group="transformer",
        seq=True,
        factory=lambda p: NoPETransformer(n_hidden=_TF_PARAMS["n_hidden"]),
        params=_TF_PARAMS,
    ),
    "TF-AbsPE": dict(
        display="TF-AbsPE",
        group="transformer",
        seq=True,
        factory=lambda p: AbsolutePETransformer(n_hidden=_TF_PARAMS["n_hidden"]),
        params=_TF_PARAMS,
    ),
    "TF-RelPE": dict(
        display="TF-RelPE",
        group="transformer",
        seq=True,
        factory=lambda p: RelativePETransformer(
            n_hidden=_TF_PARAMS["n_hidden"],
            max_len=_TF_PARAMS["max_lag"] + 1,
        ),
        params=_TF_PARAMS,
    ),
}

# Visual styles — warm tones for baselines, cool tones for Transformers
STYLES: dict[str, dict] = {
    "AR":       dict(color="#e41a1c", ls="--", marker="s", lw=1.8),  # red
    "NN":       dict(color="#ff7f00", ls="--", marker="^", lw=1.8),  # orange
    "LSTM":     dict(color="#984ea3", ls="--", marker="D", lw=1.8),  # purple
    "SARIMA":   dict(color="#8c564b", ls="-.", marker="P", lw=1.8),  # brown
    "TF-NoPE":  dict(color="#999999", ls=":",  marker="o", lw=2.0),  # gray
    "TF-AbsPE": dict(color="#377eb8", ls="-",  marker="o", lw=2.0),  # blue
    "TF-RelPE": dict(color="#4daf4a", ls="-.", marker="o", lw=2.0),  # green
}

MODEL_KEYS = list(REGISTRY.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Forecasting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iterated_forecast_flat(model: nn.Module, context: np.ndarray,
                            p: int, h_max: int, device: str) -> np.ndarray:
    """
    Iterated h-step forecast for flat-input models (AR, NN).
    Input convention: x = [lag1, lag2, …, lagp]  (most-recent first).
    context: recent observations, oldest first, length >= p.
    """
    buf = list(context[-p:])           # oldest → newest
    forecasts: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(h_max):
            x   = np.array(buf[-p:][::-1], dtype=np.float32)   # lag-1 first
            xt  = torch.tensor(x, device=device).unsqueeze(0)   # (1, p)
            pred = model(xt).item()
            forecasts.append(pred)
            buf.append(pred)
    return np.array(forecasts)


def _iterated_forecast_seq(model: nn.Module, context: np.ndarray,
                           p: int, h_max: int, device: str) -> np.ndarray:
    """
    Iterated h-step forecast for sequence-input models (LSTM, Transformers).
    Input convention: sequence fed oldest → newest.
    context: recent observations, oldest first, length >= p.
    """
    buf = list(context[-p:])           # oldest → newest
    forecasts: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(h_max):
            seq  = np.array(buf[-p:], dtype=np.float32)
            xt   = torch.tensor(seq, device=device).unsqueeze(0)  # (1, p)
            pred = model(xt).item()
            forecasts.append(pred)
            buf.append(pred)
    return np.array(forecasts)


def iterated_forecast(model_key: str, model: nn.Module,
                      context: np.ndarray, p: int,
                      h_max: int, device: str) -> np.ndarray:
    """Dispatch to the correct iterated forecast function based on input type."""
    if REGISTRY[model_key]["seq"]:
        return _iterated_forecast_seq(model, context, p, h_max, device)
    else:
        return _iterated_forecast_flat(model, context, p, h_max, device)


# ══════════════════════════════════════════════════════════════════════════════
# Rolling-window forecast runner
# ══════════════════════════════════════════════════════════════════════════════

def rolling_window_forecast(
    model_key: str,
    y: pd.Series,
    epochs_override: int | None,
    device: str,
    n_origins: int | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    For each origin t in the test period:
      1. Fit model on all observations strictly before t.
      2. Generate h = 1 … H_MAX iterated forecasts.

    Parameters
    ----------
    model_key       : key in REGISTRY
    y               : full inflation series
    epochs_override : if provided, overrides the registry epochs
    n_origins       : if provided, cap the number of origins (for smoke tests)

    Returns
    -------
    forecasts : (N_origins, H_MAX)
    origins   : DatetimeIndex
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    meta       = REGISTRY[model_key]
    is_stat    = meta.get("kind") == "statistical"
    params     = meta["params"]
    p          = params.get("max_lag", 0)
    lr         = params.get("lr", 0.0)
    epochs     = epochs_override if epochs_override is not None else params.get("epochs", 0)
    use_seq    = meta.get("seq", False)

    test_idx  = y[TEST_START:TEST_END].index
    if n_origins is not None:
        test_idx = test_idx[:n_origins]

    N         = len(test_idx)
    forecasts = np.full((N, H_MAX), np.nan)

    for i, date in enumerate(test_idx):
        y_avail = y[:date].iloc[:-1].values.astype(np.float32)

        if is_stat:
            # Statistical model: direct multi-step forecast, no training loop
            fc = sarima_forecast(y_avail, H_MAX)
        else:
            # Neural model: build lag matrix, train, then iterate
            if use_seq:
                X, Y_ = make_lstm_sequence(y_avail, p)
            else:
                X, Y_ = make_lag_matrix(y_avail, p)
            model = meta["factory"](p)
            train_model(model, X, Y_, lr=lr, epochs=epochs, device=device)
            fc = iterated_forecast(model_key, model, y_avail, p, H_MAX, device)

        forecasts[i, :len(fc)] = fc

        if (i + 1) % 60 == 0 or (i + 1) == N:
            print(f"    [{model_key:<10s}]  {i+1:3d}/{N} origins", flush=True)

    return forecasts, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# Error computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_errors(
    forecasts: np.ndarray,
    y: pd.Series,
    origins: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """
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
            ti  = y_idx.get_indexer([orig], method="nearest")[0] + h
            if ti >= len(y_arr):
                continue
            fc  = forecasts[i, h - 1]
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
# Tables
# ══════════════════════════════════════════════════════════════════════════════

def _divider(width: int = 80) -> str:
    return "─" * width


def print_full_table(
    msfe_dict: dict[str, np.ndarray],
    mafe_dict: dict[str, np.ndarray],
    dtype_label: str,
) -> None:
    col = 11
    keys = MODEL_KEYS

    def _header_row(prefix: str) -> str:
        return prefix + "".join(f"{REGISTRY[k]['display']:>{col}}" for k in keys)

    print(f"\n{'='*80}")
    print(f"MSFE  —  {dtype_label.upper()} data  |  test: {TEST_START} – {TEST_END}")
    print("=" * 80)
    print(_header_row(f"{'h':>3} "))
    print(_divider())
    for h in range(1, H_MAX + 1):
        row = f"{h:>3} "
        for k in keys:
            v = msfe_dict[k][h - 1] if k in msfe_dict else np.nan
            row += f"{v:>{col}.5f}"
        print(row)

    ar_base = msfe_dict.get("AR", np.ones(H_MAX))
    print(f"\n{'='*80}")
    print("MSFE ratio vs AR baseline  (< 1.00 = beats AR)")
    print("=" * 80)
    print(_header_row(f"{'h':>3} "))
    print(_divider())
    for h in range(1, H_MAX + 1):
        row = f"{h:>3} "
        for k in keys:
            v = (msfe_dict[k][h - 1] / ar_base[h - 1]
                 if k in msfe_dict and ar_base[h - 1] > 0 else np.nan)
            mark = " *" if (not np.isnan(v) and v < 1.0 and k != "AR") else "  "
            row += f"{v:>{col - 2}.4f}{mark}"
        print(row)
    print("  (* = beats AR baseline)")

    print(f"\n{'='*80}")
    print("Horizon-averaged MSFE / MAFE / RMSFE")
    print("=" * 80)
    print(f"{'Model':<14} {'mean MSFE':>12} {'mean MAFE':>12} {'mean RMSFE':>12}")
    print(_divider(52))
    ar_mean = float(np.nanmean(ar_base))
    for k in keys:
        m_msfe = float(np.nanmean(msfe_dict[k])) if k in msfe_dict else np.nan
        m_mafe = float(np.nanmean(mafe_dict[k])) if k in mafe_dict else np.nan
        m_rmsfe = m_msfe / ar_mean if ar_mean > 0 else np.nan
        print(f"{REGISTRY[k]['display']:<14} {m_msfe:>12.5f} {m_mafe:>12.5f} {m_rmsfe:>12.4f}")
    print()


def save_csv_tables(
    msfe_dict: dict[str, np.ndarray],
    mafe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
) -> None:
    horizons = list(range(1, H_MAX + 1))
    ar_base  = msfe_dict.get("AR", np.ones(H_MAX))

    msfe_rows  = [{"h": h, **{k: msfe_dict[k][h-1] for k in MODEL_KEYS}} for h in horizons]
    mafe_rows  = [{"h": h, **{k: mafe_dict[k][h-1] for k in MODEL_KEYS}} for h in horizons]
    rmsfe_rows = [{"h": h, **{k: msfe_dict[k][h-1] / ar_base[h-1]
                               for k in MODEL_KEYS}} for h in horizons]

    ar_mean = float(np.nanmean(ar_base))
    summary_rows = [{
        "model":      k,
        "display":    REGISTRY[k]["display"],
        "mean_msfe":  float(np.nanmean(msfe_dict[k])),
        "mean_mafe":  float(np.nanmean(mafe_dict[k])),
        "mean_rmsfe": float(np.nanmean(msfe_dict[k])) / ar_mean,
    } for k in MODEL_KEYS]

    pd.DataFrame(msfe_rows).to_csv(out_dir / f"msfe_{dtype_label}.csv", index=False)
    pd.DataFrame(mafe_rows).to_csv(out_dir / f"mafe_{dtype_label}.csv", index=False)
    pd.DataFrame(rmsfe_rows).to_csv(out_dir / f"rmsfe_{dtype_label}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / f"summary_{dtype_label}.csv", index=False)

    print(f"  Tables → {out_dir}/ [msfe, mafe, rmsfe, summary]_{dtype_label}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _group_legend_handles() -> list:
    """Return two patch handles marking the visual groups."""
    return [
        mpatches.Patch(color="#e41a1c", label="Baseline (AR / NN / LSTM)"),
        mpatches.Patch(color="#377eb8", label="Transformer PE variants"),
    ]


def _draw_msfe_lines(ax: plt.Axes, msfe_dict: dict[str, np.ndarray],
                    dtype_label: str) -> None:
    """Shared helper: draw MSFE-by-horizon lines onto an existing Axes."""
    horizons = np.arange(1, H_MAX + 1)
    for k in MODEL_KEYS:
        st = STYLES[k]
        ax.plot(horizons, msfe_dict[k],
                color=st["color"], ls=st["ls"], marker=st["marker"],
                lw=st["lw"], ms=5, label=REGISTRY[k]["display"])
    ax.set_xlabel("Forecast horizon h (months)", fontsize=11)
    ax.set_ylabel("MSFE", fontsize=11)
    ax.set_title(
        f"MSFE by Forecast Horizon — All Models\n"
        f"({dtype_label.upper()} data,  test: {TEST_START}–{TEST_END})",
        fontsize=11,
    )
    ax.set_xticks(horizons)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)


def _draw_forecast_path(
    ax: plt.Axes,
    forecasts_dict: dict[str, np.ndarray],
    y: pd.Series,
    origins: pd.DatetimeIndex,
    snapshot_date: str,
    dtype_label: str = "",
) -> None:
    """
    Draw 12 months of context + actual future vs model forecast paths
    from the origin nearest to `snapshot_date`.
    """
    snap_ts   = pd.Timestamp(snapshot_date)
    i_snap    = origins.get_indexer([snap_ts], method="nearest")[0]
    snap_ts   = origins[i_snap]                          # actual origin used
    ti_origin = y.index.get_indexer([snap_ts], method="nearest")[0]

    # 12-month look-back context shown in grey
    ctx_start = max(0, ti_origin - 12)
    ctx_dates = y.index[ctx_start : ti_origin + 1]
    ctx_vals  = y.values[ctx_start : ti_origin + 1]

    # Actual future values h = 1 … H_MAX
    fut_end   = min(ti_origin + H_MAX + 1, len(y))
    fut_dates = y.index[ti_origin + 1 : fut_end]
    fut_vals  = y.values[ti_origin + 1 : fut_end]
    n_fut     = len(fut_dates)

    if n_fut == 0:
        ax.set_title(f"No future data after {snap_ts.strftime('%Y-%m')}")
        return

    # Context line (grey)
    ax.plot(ctx_dates, ctx_vals, color="grey", lw=1.2, alpha=0.5)
    # Actual future (black, thicker)
    ax.plot(fut_dates, fut_vals, color="black", lw=2.2, label="Actual", zorder=5)
    # Origin marker
    ax.axvline(snap_ts, color="grey", ls=":", lw=1.0, alpha=0.7)

    # Model forecast paths
    for k in MODEL_KEYS:
        fc = forecasts_dict[k][i_snap, :n_fut]
        st = STYLES[k]
        ax.plot(fut_dates, fc,
                color=st["color"], ls=st["ls"], marker=st["marker"],
                lw=st["lw"], ms=4, label=REGISTRY[k]["display"])

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Inflation (%)", fontsize=11)
    ax.set_title(
        f"12-Step Forecast from {snap_ts.strftime('%Y-%m')}\n"
        f"({dtype_label} — grey = context, black = actual)",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_msfe_by_horizon(
    msfe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
) -> None:
    """
    Plot MSFE per horizon for all models and save as a figure.
    """
    fig, ax_msfe = plt.subplots(figsize=(9, 5))
    _draw_msfe_lines(ax_msfe, msfe_dict, dtype_label)
    fig.tight_layout()
    out = out_dir / f"all_msfe_horizon_{dtype_label}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")


def plot_forecast_path(
    forecasts_dict: dict[str, np.ndarray],
    y: pd.Series,
    origins: pd.DatetimeIndex,
    snapshot_date: str,
    out_dir: Path,
    dtype_label: str,
) -> None:
    """
    Plot 12-step forecast paths vs actual data from `snapshot_date` and save as a figure.
    """
    fig, ax_path = plt.subplots(figsize=(9, 5))
    _draw_forecast_path(ax_path, forecasts_dict, y, origins, snapshot_date, dtype_label)
    fig.tight_layout()
    out = out_dir / f"all_forecast_path_{snapshot_date}_{dtype_label}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")


def plot_rmsfe_vs_ar(
    msfe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
) -> None:
    """MSFE ratio relative to AR baseline; dashed line at 1 = parity with AR."""
    horizons = np.arange(1, H_MAX + 1)
    ar_base  = msfe_dict["AR"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="black", ls=":", lw=1.2, label="AR baseline (ratio = 1)")

    for k in MODEL_KEYS:
        if k == "AR":
            continue
        ratio = msfe_dict[k] / ar_base
        st    = STYLES[k]
        ax.plot(horizons, ratio,
                color=st["color"], ls=st["ls"], marker=st["marker"],
                lw=st["lw"], ms=5, label=REGISTRY[k]["display"])

    ax.set_xlabel("Forecast horizon h (months)", fontsize=11)
    ax.set_ylabel("MSFE  /  MSFE(AR)", fontsize=11)
    ax.set_title(
        f"Relative MSFE vs AR Baseline — All Models\n"
        f"({dtype_label.upper()} data,  test: {TEST_START}–{TEST_END})",
        fontsize=11,
    )
    ax.set_xticks(horizons)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhspan(0, 1.0, alpha=0.04, color="green")    # shaded "beats AR" region
    fig.tight_layout()

    out = out_dir / f"all_rmsfe_vs_ar_{dtype_label}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")


def plot_msfe_bars(
    msfe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
    highlight_horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> None:
    """
    Grouped bar chart of MSFE at key horizons.
    Baseline group (AR/NN/LSTM) and Transformer group (NoPE/AbsPE/RelPE)
    are colour-coded and separated by a gap.
    """
    n_models = len(MODEL_KEYS)
    n_h      = len(highlight_horizons)
    fig, axes = plt.subplots(1, n_h, figsize=(4 * n_h, 4.5), sharey=False)
    if n_h == 1:
        axes = [axes]

    group_gap  = 0.15           # extra space between baseline and transformer groups
    bar_width  = 0.55
    baseline_keys    = [k for k in MODEL_KEYS if REGISTRY[k]["group"] == "baseline"]
    transformer_keys = [k for k in MODEL_KEYS if REGISTRY[k]["group"] == "transformer"]
    ordered_keys     = baseline_keys + transformer_keys

    x_pos = {}
    pos = 0.0
    for i, k in enumerate(ordered_keys):
        if i == len(baseline_keys):        # insert gap before transformers
            pos += group_gap
        x_pos[k] = pos
        pos += bar_width + 0.1

    for ax, h in zip(axes, highlight_horizons):
        for k in ordered_keys:
            val = msfe_dict[k][h - 1]
            st  = STYLES[k]
            bar = ax.bar(x_pos[k], val, width=bar_width,
                         color=st["color"], alpha=0.85, edgecolor="white", lw=0.6)

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(
            [REGISTRY[k]["display"] for k in ordered_keys],
            rotation=35, ha="right", fontsize=8.5,
        )
        ax.set_title(f"h = {h}", fontsize=11)
        ax.set_ylabel("MSFE" if h == highlight_horizons[0] else "")
        ax.grid(True, axis="y", alpha=0.3)

        # Draw divider between baseline / transformer groups
        mid = (x_pos[baseline_keys[-1]] + x_pos[transformer_keys[0]]) / 2
        ax.axvline(mid, color="grey", ls=":", lw=0.9, alpha=0.7)

    fig.suptitle(
        f"MSFE at Selected Horizons — {dtype_label.upper()} data\n"
        f"(test: {TEST_START}–{TEST_END})",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out = out_dir / f"all_msfe_bars_{dtype_label}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


_NBER_RECESSIONS = [("2001-03", "2001-11"), ("2007-12", "2009-06")]


def _rolling_sq_errors(
    forecasts_dict: dict[str, np.ndarray],
    y: pd.Series,
    origins: pd.DatetimeIndex,
    window: int,
) -> dict[str, pd.Series]:
    """
    Shared helper: compute the rolling-window squared error at h=1 for each
    model and return a dict of smoothed pd.Series indexed by origin date.
    """
    result = {}
    for k, fc in forecasts_dict.items():
        sq_errs = []
        for i, orig in enumerate(origins):
            ti = y.index.get_indexer([orig], method="nearest")[0] + 1
            sq_errs.append((fc[i, 0] - y.iloc[ti]) ** 2 if ti < len(y) else np.nan)
        result[k] = pd.Series(sq_errs, index=origins).rolling(window, min_periods=1).mean()
    return result


def _format_month(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def _parse_origin_bound(value: str | None, fallback: pd.Timestamp, is_end: bool) -> pd.Timestamp:
    if value is None:
        return fallback

    ts = pd.Timestamp(value)
    if is_end and len(value) == 7 and value[4] == "-":
        return ts + pd.offsets.MonthEnd(0)
    return ts


def _slice_rolling_origin_range(
    rolls: dict[str, pd.Series],
    origins: pd.DatetimeIndex,
    origin_start: str | None,
    origin_end: str | None,
) -> tuple[dict[str, pd.Series], pd.DatetimeIndex, str]:
    """
    Restrict already-computed rolling MSFE series to a plotted origin range.
    The rolling values keep their full-history window; only the displayed dates change.
    """
    if origin_start is None and origin_end is None:
        return rolls, origins, ""

    start_ts = _parse_origin_bound(origin_start, origins.min(), is_end=False)
    end_ts = _parse_origin_bound(origin_end, origins.max(), is_end=True)
    if start_ts > end_ts:
        raise ValueError(
            f"rolling origin start ({origin_start}) must be <= end ({origin_end})"
        )

    mask = (origins >= start_ts) & (origins <= end_ts)
    plot_origins = origins[mask]
    if len(plot_origins) == 0:
        raise ValueError(
            "No forecast origins fall in rolling plot range "
            f"{_format_month(start_ts)} to {_format_month(end_ts)}. "
            f"Available origins are {_format_month(origins.min())} "
            f"to {_format_month(origins.max())}."
        )

    sliced = {k: roll.loc[plot_origins] for k, roll in rolls.items()}
    suffix = f"_{_format_month(plot_origins[0])}_to_{_format_month(plot_origins[-1])}"
    return sliced, plot_origins, suffix


def _set_origin_xlim(ax: plt.Axes, origins: pd.DatetimeIndex) -> None:
    if len(origins) == 1:
        center = origins[0]
        ax.set_xlim(center - pd.DateOffset(days=15), center + pd.DateOffset(days=15))
    else:
        ax.set_xlim(origins[0], origins[-1])


def plot_rolling_msfe_over_time(
    forecasts_dict: dict[str, np.ndarray],
    y: pd.Series,
    origins: pd.DatetimeIndex,
    out_dir: Path,
    dtype_label: str,
    window: int = 12,
    origin_start: str | None = None,
    origin_end: str | None = None,
) -> None:
    """
    Produces two figures:

    Figure 1 — split panel  (all_rolling_msfe_{dtype}.pdf)
        Top: baseline models (AR / NN / LSTM)
        Bottom: Transformer PE variants
        Makes within-group comparison easy without visual clutter.

    Figure 2 — combined single panel  (all_rolling_msfe_combined_{dtype}.pdf)
        All 6 models on one axes for direct cross-group comparison.

    Both shade NBER recessions to contextualise error spikes.
    """
    rolls = _rolling_sq_errors(forecasts_dict, y, origins, window)
    rolls, plot_origins, range_suffix = _slice_rolling_origin_range(
        rolls, origins, origin_start, origin_end,
    )
    range_title = ""
    if range_suffix:
        range_title = (
            f"\n(origins: {_format_month(plot_origins[0])}"
            f"–{_format_month(plot_origins[-1])})"
        )

    # ── Figure 1: split baseline / transformer panels ──────────────────────────
    fig1, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
        constrained_layout=True,
    )
    for k, roll in rolls.items():
        st  = STYLES[k]
        ax  = ax_top if REGISTRY[k]["group"] == "baseline" else ax_bot
        ax.plot(plot_origins, roll, color=st["color"], ls=st["ls"], lw=1.1,
                label=REGISTRY[k]["display"])

    for ax, title in [(ax_top, "Baseline models  (AR / NN / LSTM)"),
                      (ax_bot,  "Transformer PE variants")]:
        ax.set_ylabel(f"MSFE ({window}m rolling)", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=10, pad=3)
        for s, e in _NBER_RECESSIONS:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15, zorder=0)
        _set_origin_xlim(ax, plot_origins)

    ax_bot.set_xlabel("Forecast origin", fontsize=10)
    fig1.suptitle(
        f"Rolling MSFE at h=1 — {dtype_label.upper()}  (shaded = NBER recessions)"
        f"{range_title}",
        fontsize=11,
    )
    out1 = out_dir / f"all_rolling_msfe_{dtype_label}{range_suffix}.pdf"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"  Figure → {out1}")

    # ── Figure 2: all models on one axes ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 4.5))
    for k, roll in rolls.items():
        st = STYLES[k]
        ax2.plot(plot_origins, roll, color=st["color"], ls=st["ls"], lw=st["lw"] * 0.9,
                 label=REGISTRY[k]["display"])
    for s, e in _NBER_RECESSIONS:
        ax2.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15, zorder=0)
    _set_origin_xlim(ax2, plot_origins)

    ax2.set_xlabel("Forecast origin", fontsize=10)
    ax2.set_ylabel(f"MSFE ({window}m rolling, h=1)", fontsize=10)
    ax2.set_title(
        f"Rolling MSFE — All Models Combined — {dtype_label.upper()} data\n"
        f"(shaded = NBER recessions){range_title}",
        fontsize=11,
    )
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = out_dir / f"all_rolling_msfe_combined_{dtype_label}{range_suffix}.pdf"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"  Figure → {out2}")


def plot_mafe_heatmap(
    mafe_dict: dict[str, np.ndarray],
    out_dir: Path,
    dtype_label: str,
) -> None:
    """
    Heatmap: models (rows) × horizons (columns), cell colour = MAFE.
    Quickly reveals which model × horizon combinations are strongest.
    """
    data = np.array([mafe_dict[k] for k in MODEL_KEYS])   # (6, 12)
    labels = [REGISTRY[k]["display"] for k in MODEL_KEYS]
    horizons = [str(h) for h in range(1, H_MAX + 1)]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="MAFE", shrink=0.8)

    ax.set_xticks(range(H_MAX))
    ax.set_xticklabels(horizons, fontsize=9)
    ax.set_yticks(range(len(MODEL_KEYS)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Horizon h (months)", fontsize=10)
    ax.set_title(
        f"MAFE Heatmap — {dtype_label.upper()} data\n"
        f"(darker = larger error)",
        fontsize=11,
    )

    # Annotate each cell with its value
    for i in range(len(MODEL_KEYS)):
        for j in range(H_MAX):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=6.5, color="black")

    # Horizontal separator between baseline and transformer groups
    n_base = sum(1 for k in MODEL_KEYS if REGISTRY[k]["group"] == "baseline")
    ax.axhline(n_base - 0.5, color="white", lw=2.5)

    fig.tight_layout()
    out = out_dir / f"all_mafe_heatmap_{dtype_label}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Parameter summary
# ══════════════════════════════════════════════════════════════════════════════

def print_model_summary(p: int = 24) -> None:
    print("\n── Model parameter counts (p={}) ──────────────────────────────".format(p))
    for k, meta in REGISTRY.items():
        if meta.get("kind") == "statistical":
            print(f"  {meta['display']:<12}  {meta['group']:<12}  "
                  f"seq=N/A    statistical  (SARIMA(1,1,1)(0,0,1)[12])")
        else:
            m     = meta["factory"](p)
            n_par = sum(x.numel() for x in m.parameters() if x.requires_grad)
            print(f"  {meta['display']:<12}  {meta['group']:<12}  "
                  f"seq={str(meta['seq']):<5}  {n_par:>6,} params")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Per-series runner
# ══════════════════════════════════════════════════════════════════════════════

def run_for_series(
    y: pd.Series,
    dtype_label: str,
    epochs_override: int | None,
    device: str,
    rerun: bool,
    n_origins: int | None,
    snapshot_date: str | None = None,
    rolling_start: str | None = None,
    rolling_end: str | None = None,
) -> None:
    cache = OUT_DIR / f"forecasts_{dtype_label}.pkl"

    # ── Rolling-window forecasts ───────────────────────────────────────────────
    if cache.exists() and not rerun:
        print(f"\n  Loading cached forecasts ← {cache}")
        with open(cache, "rb") as f:
            forecasts_dict, origins = pickle.load(f)
    else:
        forecasts_dict: dict[str, np.ndarray] = {}
        origins = None

        for k in MODEL_KEYS:
            print(f"\n── {REGISTRY[k]['display']} [{dtype_label.upper()}] ──")
            fc, orig = rolling_window_forecast(
                k, y, epochs_override, device, n_origins,
            )
            forecasts_dict[k] = fc
            if origins is None:
                origins = orig

        with open(cache, "wb") as f:
            pickle.dump((forecasts_dict, origins), f)
        print(f"\n  Forecasts cached → {cache}")

    # ── Errors ────────────────────────────────────────────────────────────────
    msfe_dict: dict[str, np.ndarray] = {}
    mafe_dict: dict[str, np.ndarray] = {}
    for k, fc in forecasts_dict.items():
        msfe, mafe = compute_errors(fc, y, origins)
        msfe_dict[k] = msfe
        mafe_dict[k] = mafe

    # ── Tables ────────────────────────────────────────────────────────────────
    print_full_table(msfe_dict, mafe_dict, dtype_label)
    save_csv_tables(msfe_dict, mafe_dict, OUT_DIR, dtype_label)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_msfe_by_horizon(
        msfe_dict, OUT_DIR, dtype_label,
    )
    plot_forecast_path(
        forecasts_dict, y, origins, snapshot_date, OUT_DIR, dtype_label,
    )
    plot_rmsfe_vs_ar(msfe_dict, OUT_DIR, dtype_label)
    plot_msfe_bars(msfe_dict, OUT_DIR, dtype_label)
    plot_rolling_msfe_over_time(
        forecasts_dict, y, origins, OUT_DIR, dtype_label,
        origin_start=rolling_start,
        origin_end=rolling_end,
    )
    plot_mafe_heatmap(mafe_dict, OUT_DIR, dtype_label)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive model comparison: baselines vs Transformer PE variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--sa",   action="store_true", default=True,
                     help="Seasonally adjusted data (default)")
    grp.add_argument("--na",   action="store_true", default=False,
                     help="Non-adjusted data only")
    grp.add_argument("--both", action="store_true", default=True,
                     help="Both SA and NA data")
    p.add_argument("--rerun",  action="store_true", default=False,
                   help="Ignore cache; re-run all rolling-window forecasts")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training epochs for all NN-based models")
    p.add_argument("--subset", type=int, default=None, metavar="N",
                   help="Run on only the first N test origins (smoke test)")
    p.add_argument("--device", default="auto",
                   help="Compute device: 'auto' | 'cpu' | 'cuda' | 'mps'")
    p.add_argument("--cache-dir", default=".",
                   help="Directory containing cpi_cache.csv")
    p.add_argument(
        "--snapshot", default="2007-03", metavar="YYYY-MM",
        help="Origin date for the forecast-path panel in the MSFE figure "
             "(nearest available origin is used; default: 2007-03)",
    )
    p.add_argument(
        "--rolling-start", default="1995-01", metavar="YYYY-MM",
        help="First forecast origin to display in rolling-MSFE plots",
    )
    p.add_argument(
        "--rolling-end", default="2015-06", metavar="YYYY-MM",
        help="Last forecast origin to display in rolling-MSFE plots",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = DEVICE if args.device == "auto" else args.device

    if args.both:
        dtypes = ["sa", "na"]
    elif args.na:
        dtypes = ["na"]
    else:
        dtypes = ["sa"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Banner
    print("\n" + "=" * 70)
    print("  Comprehensive Model Comparison — Inflation Forecasting")
    print("=" * 70)
    print(f"  Models    : {', '.join(MODEL_KEYS)}")
    print(f"  Test      : {TEST_START} – {TEST_END}")
    print(f"  Device    : {device}")
    if args.epochs:
        print(f"  Epochs    : {args.epochs}  (override)")
    if args.subset:
        print(f"  Origins   : first {args.subset} only  (smoke test)")
    print(f"  Snapshot  : {args.snapshot}  (forecast-path panel)")
    if args.rolling_start or args.rolling_end:
        print(
            "  Rolling   : "
            f"{args.rolling_start or 'first origin'} – {args.rolling_end or 'last origin'}"
        )
    print(f"  Output    : {OUT_DIR}/")

    print_model_summary()

    print("Loading CPI data …")
    y_sa, y_na = download_data(cache_dir=args.cache_dir)
    series_map = {"sa": y_sa, "na": y_na}

    for dtype_label in dtypes:
        print(f"\n{'='*70}")
        print(f"  Series: {dtype_label.upper()}")
        print(f"{'='*70}")
        run_for_series(
            series_map[dtype_label],
            dtype_label,
            epochs_override=args.epochs,
            device=device,
            rerun=args.rerun,
            n_origins=args.subset,
            snapshot_date=args.snapshot,
            rolling_start=args.rolling_start,
            rolling_end=args.rolling_end,
        )

    print(f"\nAll done. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
