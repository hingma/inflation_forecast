"""
Epoch sufficiency analysis: show that the epoch counts configured in
experiment_all_models.py yield validation MSE that is negligibly different
from fully-converged (MAX_EPOCHS) validation MSE.

For each neural model in the registry:
  1. Reserve the last VAL_FRAC of the 1960-1989 training window as a
     held-out validation set; train on the remaining earlier observations.
  2. Train for MAX_EPOCHS, recording train and val MSE at every epoch.
  3. Repeat for N_SEEDS random seeds, then report median + IQR curves.
  4. Mark the configured epoch count with a vertical dashed line and
     annotate the % gap:
         gap = 100 × (val_MSE[config] - val_MSE[MAX]) / val_MSE[MAX]
     A small positive gap (< ~5 %) means the configured epochs are sufficient.

Models and their configured epochs (from experiment_all_models.py):
  AR        – 50 epochs,  lr=0.003
  NN        – 50 epochs,  lr=0.001
  LSTM      – 100 epochs, lr=0.001
  TF-NoPE   – 50 epochs,  lr=0.001
  TF-AbsPE  – 50 epochs,  lr=0.001
  TF-RelPE  – 50 epochs,  lr=0.001

Outputs → results/epoch_sufficiency/
  val_mse_curves.pdf          – 2×3 panel figure (train + val curves per model)
  val_gap_bars.pdf            – bar chart of % gap per model
  epoch_sufficiency.csv       – numeric summary table

Usage:
    python epoch_sufficiency.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from data import load_cached, make_lag_matrix, make_lstm_sequence
from models import ARModel, NNModel, LSTMModel, get_device
from experiment_pe import NoPETransformer, AbsolutePETransformer, RelativePETransformer

# ── Config ────────────────────────────────────────────────────────────────────

LAG       = 24
VAL_FRAC  = 0.20      # last 20 % of 1960-1989 used as validation
MAX_EPOCHS = 200
N_SEEDS   = 10
TRAIN_END = "1989-12"

OUT_DIR = Path("results/epoch_sufficiency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Registry: mirrors experiment_all_models.py exactly ───────────────────────
#    (name, factory, configured_epochs, lr, uses_seq_input)

REGISTRY = [
    ("AR",       lambda: ARModel(p=LAG),                                          100,  0.003, False),
    ("NN",       lambda: NNModel(p=LAG, n_hidden=20),                             50,  0.001, False),
    ("LSTM",     lambda: LSTMModel(n_hidden=50),                                  100, 0.001, True),
    ("TF-NoPE",  lambda: NoPETransformer(n_hidden=32),                            50,  0.001, True),
    ("TF-AbsPE", lambda: AbsolutePETransformer(n_hidden=32),                      100,  0.001, True),
    ("TF-RelPE", lambda: RelativePETransformer(n_hidden=32, max_len=LAG + 1),     50,  0.001, True),
]

# Visual styles from experiment_all_models.py
STYLES = {
    "AR":       dict(color="#e41a1c"),
    "NN":       dict(color="#ff7f00"),
    "LSTM":     dict(color="#984ea3"),
    "TF-NoPE":  dict(color="#999999"),
    "TF-AbsPE": dict(color="#377eb8"),
    "TF-RelPE": dict(color="#4daf4a"),
}


# ── Data preparation ──────────────────────────────────────────────────────────

def make_train_val(y: np.ndarray, seq: bool) -> tuple:
    """
    Build train/val splits.
    Returns (X_tr, Y_tr, X_val, Y_val) as float32 arrays.
    The val set is the last VAL_FRAC of the full lag-matrix rows.
    """
    if seq:
        X, Y = make_lstm_sequence(y, LAG)
    else:
        X, Y = make_lag_matrix(y, LAG)

    n_val = max(1, int(len(Y) * VAL_FRAC))
    return X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]


# ── Training loop with per-epoch eval ────────────────────────────────────────

def train_with_eval(
    model: nn.Module,
    X_tr: np.ndarray, Y_tr: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    lr: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train for MAX_EPOCHS; evaluate both train and val MSE each epoch.
    Returns (train_losses, val_losses) arrays of shape (MAX_EPOCHS,).
    """
    model = model.to(device)
    Xtr = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y_tr,  dtype=torch.float32, device=device)
    Xv  = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yv  = torch.tensor(Y_val, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    mse = nn.MSELoss()

    train_losses = np.empty(MAX_EPOCHS)
    val_losses   = np.empty(MAX_EPOCHS)

    for ep in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = mse(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        train_losses[ep] = loss.item()

        model.eval()
        with torch.no_grad():
            val_losses[ep] = mse(model(Xv), Yv).item()

    return train_losses, val_losses


def run_model(name: str, factory, config_ep: int, lr: float,
              seq: bool, y_train: np.ndarray,
              device: str) -> dict:
    """Run N_SEEDS seeds; return dict with loss matrices and % gap stats."""
    X_tr, Y_tr, X_val, Y_val = make_train_val(y_train, seq)

    tr_mat  = np.empty((N_SEEDS, MAX_EPOCHS))
    val_mat = np.empty((N_SEEDS, MAX_EPOCHS))

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = factory()
        tr_losses, val_losses = train_with_eval(
            model, X_tr, Y_tr, X_val, Y_val, lr, device
        )
        tr_mat[seed]  = tr_losses
        val_mat[seed] = val_losses

        val_at_config = val_losses[config_ep - 1]
        val_at_max    = val_losses[-1]
        gap = (val_at_config - val_at_max) / (val_at_max + 1e-12) * 100
        print(f"  {name} seed {seed:2d}: "
              f"val@{config_ep}={val_at_config:.5f}  "
              f"val@{MAX_EPOCHS}={val_at_max:.5f}  "
              f"gap={gap:+.2f}%")

    # Per-seed gap at configured epochs
    gaps = (val_mat[:, config_ep - 1] - val_mat[:, -1]) / (val_mat[:, -1] + 1e-12) * 100

    return dict(
        tr_mat=tr_mat,
        val_mat=val_mat,
        gaps=gaps,
        config_ep=config_ep,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(results: dict):
    """
    2 × 3 panel figure: one panel per model.
    Each panel shows the median train and val MSE curves with IQR shading,
    a vertical line at the configured epoch count, and the gap annotation.
    """
    epochs = np.arange(1, MAX_EPOCHS + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    fig.suptitle(
        "Validation MSE vs Epoch  (CPI inflation 1960–1989, p=24)\n"
        f"Train/val split: {100*(1-VAL_FRAC):.0f}%/{100*VAL_FRAC:.0f}%  |  "
        f"{N_SEEDS} random seeds  |  vertical line = configured epoch count",
        fontsize=10,
    )

    for ax, (name, factory, config_ep, lr, seq) in zip(axes, REGISTRY):
        r   = results[name]
        c   = STYLES[name]["color"]

        tr_med  = np.median(r["tr_mat"],  axis=0)
        val_med = np.median(r["val_mat"], axis=0)
        val_q25 = np.percentile(r["val_mat"], 25, axis=0)
        val_q75 = np.percentile(r["val_mat"], 75, axis=0)

        # Training curve
        ax.plot(epochs, tr_med, color=c, lw=1.0, ls="--", alpha=0.5,
                label="Train MSE (median)")
        # Val curve + IQR
        ax.fill_between(epochs, val_q25, val_q75, color=c, alpha=0.18)
        ax.plot(epochs, val_med, color=c, lw=2.0, ls="-",
                label="Val MSE (median)")

        # Vertical line at configured epoch count
        val_at_config = np.median(r["val_mat"][:, config_ep - 1])
        val_at_max    = np.median(r["val_mat"][:, -1])
        gap           = (val_at_config - val_at_max) / (val_at_max + 1e-12) * 100

        ax.axvline(config_ep, color=c, lw=1.5, ls=":",
                   label=f"Config ep={config_ep}")

        # Mark the val MSE at configured epoch
        ax.scatter([config_ep], [val_at_config], color=c, s=40, zorder=5)

        # Annotate gap
        ax.annotate(
            f"gap = {gap:+.1f}%",
            xy=(config_ep, val_at_config),
            xytext=(config_ep + MAX_EPOCHS * 0.05, val_at_config),
            fontsize=8, color=c,
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8),
        )

        ax.set_title(
            f"{name}  (config={config_ep} epochs, lr={lr})",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("MSE", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(1, MAX_EPOCHS)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "val_mse_curves.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")


def plot_gap_bars(results: dict):
    """
    Bar chart showing the median % gap per model.
    Error bars show the IQR across seeds.
    """
    names      = [row[0] for row in REGISTRY]
    med_gaps   = [float(np.median(results[n]["gaps"])) for n in names]
    q25_gaps   = [float(np.percentile(results[n]["gaps"], 25)) for n in names]
    q75_gaps   = [float(np.percentile(results[n]["gaps"], 75)) for n in names]
    colors     = [STYLES[n]["color"] for n in names]

    yerr_lo = [m - q25 for m, q25 in zip(med_gaps, q25_gaps)]
    yerr_hi = [q75 - m for m, q75 in zip(med_gaps, q75_gaps)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    bars = ax.bar(x, med_gaps, color=colors, alpha=0.8, edgecolor="white",
                  yerr=[yerr_lo, yerr_hi], capsize=4, error_kw=dict(lw=1.2))

    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(5,  color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.text(len(names) - 0.5, 5.2, "5 % threshold", fontsize=7.5,
            color="grey", ha="right")

    # Annotate bar values
    for bar, v in zip(bars, med_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.15 if v >= 0 else -0.4),
                f"{v:+.2f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Val MSE gap: (MSE@config − MSE@max) / MSE@max × 100", fontsize=9)
    ax.set_title(
        f"Epoch sufficiency: % validation-MSE gap at configured epoch counts\n"
        f"(negative = over-trained relative to {MAX_EPOCHS}-epoch baseline; "
        f"error bars = IQR over {N_SEEDS} seeds)",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "val_gap_bars.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")


def plot_combined_curves(results: dict):
    """
    Single-axes figure: all models' val MSE curves overlaid.

    Each curve is normalised to its value at epoch 1 (relative MSE) so that
    models with different absolute error scales can be compared on one axis.
    A vertical marker (dot + label) is placed at each model's configured
    epoch count.  IQR shading is drawn around the median curve.
    """
    epochs = np.arange(1, MAX_EPOCHS + 1)

    fig, (ax_abs, ax_rel) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"wspace": 0.30},
    )
    fig.suptitle(
        "Validation MSE vs Epoch — all models in one figure  "
        f"(CPI 1960–1989, p=24,  {N_SEEDS} seeds)\n"
        "Left: raw val MSE   |   Right: normalised to epoch-1 val MSE  "
        "(dot = configured epoch count)",
        fontsize=10,
    )

    abs_vals = []
    rel_vals = []

    for name, factory, config_ep, lr, seq in REGISTRY:
        r   = results[name]
        c   = STYLES[name]["color"]
        ls  = ":" if name.startswith("TF-") else "--"

        val_med = np.median(r["val_mat"], axis=0)
        val_q25 = np.percentile(r["val_mat"], 25, axis=0)
        val_q75 = np.percentile(r["val_mat"], 75, axis=0)

        # ── Left panel: raw val MSE ───────────────────────────────────────────
        abs_vals.extend([val_q25, val_q75, val_med])
        ax_abs.fill_between(epochs, val_q25, val_q75, color=c, alpha=0.10)
        ax_abs.plot(epochs, val_med, color=c, lw=1.6, ls=ls,
                    label=f"{name} (ep={config_ep})")
        ax_abs.scatter([config_ep], [val_med[config_ep - 1]],
                       color=c, s=55, zorder=5, linewidths=0)

        # ── Right panel: normalised val MSE ───────────────────────────────────
        norm       = val_med[0] + 1e-12
        rel_med    = val_med / norm
        rel_q25    = val_q25 / norm
        rel_q75    = val_q75 / norm
        rel_vals.extend([rel_q25, rel_q75, rel_med])
        ax_rel.fill_between(epochs, rel_q25, rel_q75, color=c, alpha=0.10)
        ax_rel.plot(epochs, rel_med, color=c, lw=1.6, ls=ls,
                    label=f"{name} (ep={config_ep})")
        ax_rel.scatter([config_ep], [rel_med[config_ep - 1]],
                       color=c, s=55, zorder=5, linewidths=0)
        # Annotate the dot with the normalised value
        ax_rel.annotate(
            f"{rel_med[config_ep - 1]:.3f}",
            xy=(config_ep, rel_med[config_ep - 1]),
            xytext=(config_ep + 6, rel_med[config_ep - 1]),
            fontsize=6.5, color=c, va="center",
        )

    for ax, ylabel, title in [
        (ax_abs, "Validation MSE",            "Raw validation MSE"),
        (ax_rel, "Val MSE / Val MSE (epoch 1)", "Normalised validation MSE"),
    ]:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(1, MAX_EPOCHS)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Zoom both y-axes to the dense region (robust to extreme outliers).
    def _set_robust_ylim(ax, values):
        vals = np.concatenate([np.ravel(v) for v in values])
        y_lo, y_hi = np.percentile(vals, [1, 98])
        span = max(y_hi - y_lo, 1e-12)
        pad = 0.08 * span
        ax.set_ylim(y_lo - pad, y_hi + pad)

    _set_robust_ylim(ax_abs, abs_vals)
    _set_robust_ylim(ax_rel, rel_vals)

    fig.tight_layout()
    out = OUT_DIR / "val_mse_combined.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def save_summary(results: dict):
    rows = []
    for name, factory, config_ep, lr, seq in REGISTRY:
        r   = results[name]
        val_config = r["val_mat"][:, config_ep - 1]
        val_max    = r["val_mat"][:, -1]
        gaps       = r["gaps"]
        rows.append({
            "model":         name,
            "config_epochs": config_ep,
            "val_MSE@config (median)": float(np.median(val_config)),
            f"val_MSE@{MAX_EPOCHS} (median)":   float(np.median(val_max)),
            "gap_median_%":  float(np.median(gaps)),
            "gap_q25_%":     float(np.percentile(gaps, 25)),
            "gap_q75_%":     float(np.percentile(gaps, 75)),
        })
    df = pd.DataFrame(rows)
    out = OUT_DIR / "epoch_sufficiency.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Table  → {out}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    sa, _ = load_cached()
    y_train = sa[:TRAIN_END].values.astype(np.float32)
    print(f"Training observations (1960–1989): {len(y_train)}")
    print(f"Val set size: {max(1, int((len(y_train) - LAG) * VAL_FRAC))} samples\n")

    results = {}
    for name, factory, config_ep, lr, seq in REGISTRY:
        print(f"── {name}  (config={config_ep} ep, lr={lr}, "
              f"{'seq' if seq else 'flat'} input) ──")
        results[name] = run_model(name, factory, config_ep, lr, seq, y_train, device)

    plot_curves(results)
    plot_combined_curves(results)
    plot_gap_bars(results)
    df = save_summary(results)

    # Console summary
    print(f"\n── Epoch sufficiency summary (val MSE gap at configured epochs) ──")
    print(f"{'Model':<14} {'Config ep':>10} "
          f"{'Val@config':>12} {f'Val@{MAX_EPOCHS}':>12} "
          f"{'Gap median':>12} {'Gap IQR':>16}")
    print("-" * 80)
    for _, row in df.iterrows():
        q25 = row["gap_q25_%"]
        q75 = row["gap_q75_%"]
        print(
            f"{row['model']:<14} {int(row['config_epochs']):>10} "
            f"{row['val_MSE@config (median)']:>12.5f} "
            f"{row[f'val_MSE@{MAX_EPOCHS} (median)']:>12.5f} "
            f"{row['gap_median_%']:>+11.2f}% "
            f"[{q25:+.2f}%, {q75:+.2f}%]"
        )
    print("\nInterpretation: gap ≈ 0 % → configured epochs are sufficient;")
    print("  gap < 5 % → negligible extra gain from training longer.")


if __name__ == "__main__":
    main()
