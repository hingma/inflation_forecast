"""
Convergence detection: automatically find how many epochs each model needs
to converge on the actual CPI inflation training data.

Convergence criterion (applied per seed):
  After every epoch t >= MIN_EPOCH, compute the relative loss improvement
  over the previous WINDOW epochs:
      rel_improv(t) = (loss[t - WINDOW] - loss[t]) / loss[t - WINDOW]
  The model is considered converged at the first epoch where this falls
  below TOL.  If it never drops below TOL within MAX_EPOCHS, the
  convergence epoch is recorded as MAX_EPOCHS (marked as "no convergence").

Model registry mirrors experiment_all_models.py exactly:
  NN        – one hidden layer,            n_hidden=20
  LSTM      – single-layer LSTM,           n_hidden=50
  TF-NoPE   – Transformer, no PE,          n_hidden=32
  TF-AbsPE  – Transformer, sinusoidal PE,  n_hidden=32
  TF-RelPE  – Transformer, relative PE,    n_hidden=32

Usage:
    python convergence_proof.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data import load_cached, make_lag_matrix, make_lstm_sequence
from models import NNModel, LSTMModel, get_device
from experiment_pe import NoPETransformer, AbsolutePETransformer, RelativePETransformer

# ── Config ────────────────────────────────────────────────────────────────────

LAG           = 24
N_HIDDEN_NN   = 20
N_HIDDEN_LSTM = 50
N_HIDDEN_TF   = 32
LR            = 0.001
MAX_EPOCHS    = 500      # upper bound; training stops early once converged
N_SEEDS       = 50

TRAIN_END     = "1989-12"

# Convergence detection parameters
WINDOW    = 10      # look-back window (epochs) for relative improvement
TOL       = 1e-4    # convergence threshold: rel improvement < 0.01%
MIN_EPOCH = 20      # don't check before this epoch (avoid noise early on)

OUT_DIR = Path("results/convergence_proof")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual styles from experiment_all_models.py
STYLES = {
    "NN":       dict(color="#ff7f00", ls="--"),
    "LSTM":     dict(color="#984ea3", ls="--"),
    "TF-NoPE":  dict(color="#999999", ls=":"),
    "TF-AbsPE": dict(color="#377eb8", ls="-"),
    "TF-RelPE": dict(color="#4daf4a", ls="-."),
}


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_model(name: str) -> nn.Module:
    if name == "NN":
        return NNModel(p=LAG, n_hidden=N_HIDDEN_NN)
    elif name == "LSTM":
        return LSTMModel(n_hidden=N_HIDDEN_LSTM)
    elif name == "TF-NoPE":
        return NoPETransformer(n_hidden=N_HIDDEN_TF)
    elif name == "TF-AbsPE":
        return AbsolutePETransformer(n_hidden=N_HIDDEN_TF)
    elif name == "TF-RelPE":
        return RelativePETransformer(n_hidden=N_HIDDEN_TF, max_len=LAG + 1)
    else:
        raise ValueError(f"Unknown model: {name}")


# ── Training + convergence detection ─────────────────────────────────────────

def detect_convergence(losses: np.ndarray) -> int:
    """
    Return the first 1-indexed epoch at which relative improvement drops
    below TOL, or MAX_EPOCHS if convergence was not detected.
    """
    for t in range(WINDOW + MIN_EPOCH - 1, len(losses)):
        ref = losses[t - WINDOW]
        if ref <= 0:
            continue
        rel_improv = (ref - losses[t]) / ref
        if rel_improv < TOL:
            return t + 1   # 1-indexed
    return len(losses)     # did not converge within MAX_EPOCHS


def train_one_seed(name: str,
                   X: np.ndarray, Y: np.ndarray,
                   device: str) -> tuple[np.ndarray, int]:
    """
    Train for MAX_EPOCHS, record per-epoch loss, detect convergence epoch.
    Returns (losses array, convergence_epoch).
    """
    model = _build_model(name).to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

    losses = np.empty(MAX_EPOCHS, dtype=np.float64)
    model.train()
    for ep in range(MAX_EPOCHS):
        opt.zero_grad()
        loss = criterion(model(Xt), Yt)
        loss.backward()
        opt.step()
        losses[ep] = loss.item()

    conv_ep = detect_convergence(losses)
    return losses, conv_ep


def run_all_seeds(name: str,
                  X: np.ndarray, Y: np.ndarray,
                  device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run N_SEEDS independent trainings.
    Returns:
      loss_mat   : (N_SEEDS, MAX_EPOCHS) MSE per epoch
      conv_epochs: (N_SEEDS,)            convergence epoch per seed
    """
    loss_mat    = np.empty((N_SEEDS, MAX_EPOCHS))
    conv_epochs = np.empty(N_SEEDS, dtype=int)
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        losses, conv_ep = train_one_seed(name, X, Y, device)
        loss_mat[seed]    = losses
        conv_epochs[seed] = conv_ep
        converged = conv_ep < MAX_EPOCHS
        tag = f"epoch {conv_ep}" if converged else f">{MAX_EPOCHS} (no conv)"
        print(f"  {name} seed {seed:2d}: converged @ {tag}  "
              f"| final loss = {losses[-1]:.6f}")
    return loss_mat, conv_epochs


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_loss_panels(models_cfg, results):
    """
    One panel per model.  Shows:
      - individual seed loss curves (thin, semi-transparent)
      - median loss curve (thick)
      - a dot on each seed curve at its detected convergence epoch
      - a vertical shaded band spanning the IQR of convergence epochs
      - a dashed line at the median convergence epoch
    """
    epochs = np.arange(1, MAX_EPOCHS + 1)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    fig.suptitle(
        "Training-loss convergence detection  (CPI inflation 1960–1989, p=24)\n"
        f"Criterion: rel. improvement < {TOL:.0e} over {WINDOW}-epoch window  "
        f"({N_SEEDS} random seeds per model)",
        fontsize=10,
    )

    for ax, (name, _, _) in zip(axes, models_cfg):
        loss_mat, conv_epochs = results[name]
        med  = np.median(loss_mat, axis=0)
        c    = STYLES[name]["color"]
        ls   = STYLES[name]["ls"]

        # Individual seed curves
        for s in range(N_SEEDS):
            ax.plot(epochs, loss_mat[s], color=c, lw=0.5, alpha=0.25)

        # Median curve
        ax.plot(epochs, med, color=c, ls=ls, lw=2.0, label="Median MSE", zorder=4)

        # Convergence epoch markers on each seed curve
        for s in range(N_SEEDS):
            ep = conv_epochs[s]
            ax.scatter(ep, loss_mat[s, ep - 1],
                       color=c, s=18, zorder=5, linewidths=0)

        # Shaded IQR band of convergence epochs
        q25_ep = int(np.percentile(conv_epochs, 25))
        q75_ep = int(np.percentile(conv_epochs, 75))
        med_ep = int(np.median(conv_epochs))
        ax.axvspan(q25_ep, q75_ep, alpha=0.12, color=c,
                   label=f"Conv. IQR [{q25_ep}–{q75_ep}]")
        ax.axvline(med_ep, color=c, lw=1.2, ls="--",
                   label=f"Median conv. = {med_ep}")

        n_no_conv = int(np.sum(conv_epochs >= MAX_EPOCHS))
        subtitle = f"median={med_ep}  IQR=[{q25_ep},{q75_ep}]"
        # if n_no_conv:
        #     subtitle += f"  ({n_no_conv} seeds: no conv.)"

        ax.set_title(f"{name}\n{subtitle}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("MSE (training)" if name == "NN" else "", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.set_xlim(1, MAX_EPOCHS)

    fig.tight_layout()
    out = OUT_DIR / "convergence_curves.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")


def _plot_convergence_boxplot(models_cfg, results):
    """
    Boxplot comparing the distribution of convergence epochs across models.
    """
    names = [name for name, _, _ in models_cfg]
    data  = [results[name][1] for name in names]   # list of (N_SEEDS,) arrays
    colors = [STYLES[name]["color"] for name in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for element in ("whiskers", "caps", "fliers"):
        for item, c in zip(
            bp[element],
            [c for c in colors for _ in range(2 if element != "fliers" else 1)],
        ):
            item.set_color(c)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Convergence epoch", fontsize=10)
    ax.set_title(
        f"Epochs needed to converge  (criterion: rel. improv. < {TOL:.0e} "
        f"over {WINDOW} epochs,  {N_SEEDS} seeds)",
        fontsize=10,
    )
    ax.axhline(50,  color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(100, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.text(len(names) + 0.45, 51,  "50",  fontsize=7, color="grey", va="bottom")
    ax.text(len(names) + 0.45, 101, "100", fontsize=7, color="grey", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "convergence_epochs_boxplot.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")

    sa, _ = load_cached()
    y_train = sa[:TRAIN_END].values.astype(np.float32)
    print(f"Training observations: {len(y_train)}")

    X_flat, Y_flat = make_lag_matrix(y_train, LAG)
    X_seq,  Y_seq  = make_lstm_sequence(y_train, LAG)

    models_cfg = [
        ("NN",       X_flat, Y_flat),
        ("LSTM",     X_seq,  Y_seq),
        ("TF-NoPE",  X_seq,  Y_seq),
        ("TF-AbsPE", X_seq,  Y_seq),
        ("TF-RelPE", X_seq,  Y_seq),
    ]

    results = {}
    for name, X, Y in models_cfg:
        print(f"\n── {name}  ({N_SEEDS} seeds × {MAX_EPOCHS} epochs) ──")
        results[name] = run_all_seeds(name, X, Y, device)

    _plot_loss_panels(models_cfg, results)
    _plot_convergence_boxplot(models_cfg, results)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n── Convergence epoch summary (criterion: rel. improv < {TOL:.0e} "
          f"over {WINDOW} epochs) ──")
    hdr = (f"{'Model':<14} {'min':>6} {'q25':>6} {'median':>8} "
           f"{'q75':>6} {'max':>6} {'no-conv':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, _, _ in models_cfg:
        _, conv_epochs = results[name]
        no_conv = int(np.sum(conv_epochs >= MAX_EPOCHS))
        print(
            f"{name:<14} "
            f"{conv_epochs.min():>6d} "
            f"{int(np.percentile(conv_epochs, 25)):>6d} "
            f"{int(np.median(conv_epochs)):>8d} "
            f"{int(np.percentile(conv_epochs, 75)):>6d} "
            f"{conv_epochs.max():>6d} "
            f"{no_conv:>9d}"
        )


if __name__ == "__main__":
    main()
