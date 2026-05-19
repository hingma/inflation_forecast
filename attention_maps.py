#!/usr/bin/env python
"""
Attention Map Comparison for Transformer PE Variants
====================================================
Extracts multi-head self-attention weights from three Transformer variants:

  TF-NoPE   – no positional encoding (permutation-equivariant)
  TF-AbsPE  – sinusoidal absolute PE (Vaswani et al. 2017)
  TF-RelPE  – learned relative position bias (T5-style, Shaw et al. 2018)

For each variant × snapshot date the script produces:
  attention_<variant>_<date>_<label>_avg.pdf    head-averaged heatmap (p × p)
  attention_<variant>_<date>_<label>_heads.pdf  per-head heatmap row
  attention_<variant>_<date>_<label>.csv        raw weights (head, query, key)
  attention_compare_<date>_<label>.pdf          3-variant side-by-side comparison

Model loading mirrors lrp.py: only data strictly before each snapshot date is
used for training (real-time evaluation), and trained models are cached under
results/models/lrp/ and reused across runs.

Usage
-----
  python attention_maps.py                                   # default dates, SA
  python attention_maps.py --na                              # non-adjusted data
  python attention_maps.py --both                            # SA + NA
  python attention_maps.py --dates 2005-10 2008-11 2000-01  # custom dates
  python attention_maps.py --rerun                           # retrain models
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from data import download_data
from lrp import load_or_train_lrp_model, lrp_target_timestamp
from model_selection import PAPER_BEST_PARAMS
from models import DEVICE as AUTO_DEVICE

OUT_DIR = Path("results/attention_maps")

TF_VARIANTS = ["tf_nope", "tf_abspe", "tf_relpe"]
TF_DISPLAY  = {
    "tf_nope":  "TF-NoPE",
    "tf_abspe": "TF-AbsPE (Sinusoidal)",
    "tf_relpe": "TF-RelPE (Learned Bias)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Attention extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_std_attn(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """
    Extract per-head attention weights from NoPETransformer or AbsolutePETransformer.

    Both use nn.TransformerEncoder → nn.TransformerEncoderLayer → nn.MultiheadAttention.
    PyTorch's TransformerEncoderLayer calls self_attn with need_weights=False by
    default (for speed), so we register a forward hook on the encoder layer that
    intercepts its input and re-runs just the attention sub-module with weights on.

    Returns: (n_heads, p, p)  —  attn[h, i, j] = weight head h places on key j
                                                   when processing query i.
    """
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)  # (1, p)
    captured: dict = {}

    def _hook(module, inputs, output):
        src = inputs[0]           # (1, p, d_model) — post-PE token embeddings
        with torch.no_grad():
            _, w = module.self_attn(
                src, src, src,
                need_weights=True,
                average_attn_weights=False,   # keep all heads separate
            )
        captured["weights"] = w.squeeze(0).detach().cpu().numpy()  # (n_heads, p, p)

    handle = model.encoder.layers[0].register_forward_hook(_hook)
    with torch.no_grad():
        model(xt)
    handle.remove()
    return captured["weights"]


def _extract_relpe_attn(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """
    Extract per-head attention weights from RelativePETransformer.

    The custom _RelativeAttention module computes softmax(Q·Kᵀ/√dₕ + bias) but
    returns only the attended values.  We hook its forward, capture the input
    embedding, and recompute Q, K, and the relative bias from the module's own
    stored parameters — identical arithmetic to the live forward pass.

    Returns: (n_heads, p, p)
    """
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    captured: dict = {}

    def _hook(module, inputs, output):
        x_in = inputs[0]                          # (1, p, d_model)
        B, L, D = x_in.shape
        H, Dh = module.nhead, module.d_head
        with torch.no_grad():
            q = module.q_proj(x_in).view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)
            k = module.k_proj(x_in).view(B, L, H, Dh).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale  # (B,H,L,L)

            pos   = torch.arange(L, device=x_in.device)
            delta = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(
                -(module.max_len - 1), module.max_len - 1
            ) + (module.max_len - 1)                                   # (L,L)
            bias  = module.rel_bias(delta).permute(2, 0, 1).unsqueeze(0)  # (1,H,L,L)

            attn = torch.softmax(scores + bias, dim=-1)
        captured["weights"] = attn.squeeze(0).detach().cpu().numpy()  # (H,L,L)

    handle = model.encoder.attn.register_forward_hook(_hook)
    with torch.no_grad():
        model(xt)
    handle.remove()
    return captured["weights"]


def extract_attention(model_type: str, model: nn.Module,
                      x: np.ndarray, device: str) -> np.ndarray:
    """Dispatch to the correct extraction function. Returns (n_heads, p, p)."""
    if model_type == "tf_relpe":
        return _extract_relpe_attn(model, x, device)
    return _extract_std_attn(model, x, device)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _lag_ticks(p: int, max_ticks: int = 8) -> tuple[list[int], list[str]]:
    """Sparse tick positions and labels: 'lag-p' (oldest) … 'lag-1' (newest)."""
    stride    = max(1, p // max_ticks)
    positions = list(range(0, p, stride))
    labels    = [f"lag-{p - i}" for i in positions]
    return positions, labels


def _heatmap(ax: plt.Axes, data: np.ndarray, p: int,
             vmin: float, vmax: float, cbar_label: str = "Attention") -> None:
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    pos, labs = _lag_ticks(p)
    ax.set_xticks(pos);  ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(pos);  ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("Key (attended to)", fontsize=9)
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)


def plot_attention_avg(attn: np.ndarray, model_type: str, label: str,
                       month_tag: str, out_dir: Path) -> None:
    """Head-averaged (p × p) heatmap."""
    avg = attn.mean(axis=0)
    p   = avg.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, avg, p, vmin=0.0, vmax=float(avg.max()))
    ax.set_ylabel("Query (attending)", fontsize=9)
    ax.set_title(
        f"{TF_DISPLAY[model_type]}  —  {label.upper()}  —  {month_tag}\n"
        f"Head-averaged attention  (n_heads={attn.shape[0]})",
        fontsize=10,
    )
    fig.tight_layout()
    out = out_dir / f"attention_{model_type}_{month_tag}_{label}_avg.pdf"
    fig.savefig(out, dpi=150);  plt.close(fig)
    print(f"  → {out}")


def plot_attention_heads(attn: np.ndarray, model_type: str, label: str,
                         month_tag: str, out_dir: Path) -> None:
    """One panel per head, arranged in a single row."""
    n_heads, p, _ = attn.shape
    vmax = float(attn.max())
    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5.2),
                             squeeze=False)
    for h, ax in enumerate(axes[0]):
        _heatmap(ax, attn[h], p, vmin=0.0, vmax=vmax)
        ax.set_title(f"Head {h + 1}", fontsize=10)
        if h > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Query (attending)", fontsize=9)
    fig.suptitle(
        f"{TF_DISPLAY[model_type]}  —  {label.upper()}  —  {month_tag}\n"
        "Per-head attention weights",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / f"attention_{model_type}_{month_tag}_{label}_heads.pdf"
    fig.savefig(out, dpi=150);  plt.close(fig)
    print(f"  → {out}")


def save_attention_csv(attn: np.ndarray, model_type: str, label: str,
                       month_tag: str, out_dir: Path) -> None:
    """Flat CSV: one row per (head, query_lag, key_lag)."""
    n_heads, p, _ = attn.shape
    _, labs = _lag_ticks(p, max_ticks=p)   # all labels
    all_labs = [f"lag-{p - i}" for i in range(p)]
    rows = [
        {"head": h + 1, "query": all_labs[i], "key": all_labs[j],
         "weight": float(attn[h, i, j])}
        for h in range(n_heads)
        for i in range(p)
        for j in range(p)
    ]
    out = out_dir / f"attention_{model_type}_{month_tag}_{label}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  → {out}")


def plot_comparison(attn_dict: dict[str, np.ndarray], label: str,
                    month_tag: str, out_dir: Path) -> None:
    """
    Three-panel side-by-side figure: head-averaged attention for NoPE, AbsPE, RelPE.
    Shared colour scale so differences are directly comparable.
    """
    avgs = {mt: attn_dict[mt].mean(axis=0) for mt in TF_VARIANTS}
    vmax = float(max(a.max() for a in avgs.values()))
    p    = next(iter(avgs.values())).shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, mt in zip(axes, TF_VARIANTS):
        _heatmap(ax, avgs[mt], p, vmin=0.0, vmax=vmax, cbar_label="")
        ax.set_title(TF_DISPLAY[mt], fontsize=11)
    axes[0].set_ylabel("Query (attending)", fontsize=9)

    fig.suptitle(
        f"Head-averaged self-attention — {label.upper()} — {month_tag}\n"
        f"(shared colour scale,  n_heads={next(iter(attn_dict.values())).shape[0]})",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = out_dir / f"attention_compare_{month_tag}_{label}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight");  plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-date runner
# ══════════════════════════════════════════════════════════════════════════════

def _lrp_cache_paths(models_dir: Path, model_type: str, label: str,
                     target_date: pd.Timestamp) -> tuple[Path, Path]:
    """Mirror of lrp._lrp_model_paths — used only for --rerun cleanup."""
    tag = target_date.strftime("%Y-%m")
    d   = models_dir / "lrp"
    return d / f"{model_type}_{label}_{tag}.pt", d / f"{model_type}_{label}_{tag}_meta.json"


def run_for_date(
    month: str,
    y: pd.Series,
    label: str,
    best_params: dict,
    device: str,
    models_dir: Path,
    out_dir: Path,
    rerun: bool,
) -> None:
    """Extract and save attention maps for all 3 TF variants at one snapshot date."""
    target_date = lrp_target_timestamp(month)
    if target_date not in y.index:
        print(f"  Skipping {month}: not in {label} series.")
        return

    print(f"\n── {month}  [{label.upper()}] ──────────────────────────────────")
    month_tag  = target_date.strftime("%Y-%m")
    attn_dict: dict[str, np.ndarray] = {}

    for mt in TF_VARIANTS:
        params = best_params.get(mt)
        if not params:
            print(f"  No params for {mt}, skipping.")
            continue

        if rerun:
            for path in _lrp_cache_paths(models_dir, mt, label, target_date):
                path.unlink(missing_ok=True)

        try:
            model, meta = load_or_train_lrp_model(
                mt, y, params, label, target_date, models_dir, device,
            )
            p = meta["p"]
        except ValueError as exc:
            print(f"  Skipping {mt} @ {month}: {exc}")
            continue

        y_avail = y[:target_date].iloc[:-1].values.astype(np.float32)
        x = y_avail[-p:]                                  # (p,) oldest → newest

        attn = extract_attention(mt, model, x, device)   # (n_heads, p, p)
        attn_dict[mt] = attn

        plot_attention_avg(attn, mt, label, month_tag, out_dir)
        plot_attention_heads(attn, mt, label, month_tag, out_dir)
        save_attention_csv(attn, mt, label, month_tag, out_dir)

    if len(attn_dict) == len(TF_VARIANTS):
        plot_comparison(attn_dict, label, month_tag, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attention map comparison: NoPE vs AbsPE vs RelPE Transformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--sa",   action="store_true", default=True,
                     help="Seasonally adjusted data (default)")
    grp.add_argument("--na",   action="store_true", default=False,
                     help="Non-adjusted data")
    grp.add_argument("--both", action="store_true", default=False,
                     help="Both SA and NA")
    p.add_argument(
        "--dates", nargs="+",
        default=["2005-10", "2008-11", "1989-10", "2000-01"],
        metavar="YYYY-MM",
        help="Snapshot dates to process (space-separated)",
    )
    p.add_argument("--device",    default="auto",
                   help="'auto' | 'cpu' | 'cuda' | 'mps'")
    p.add_argument("--cache-dir", default=".", metavar="DIR",
                   help="Directory containing cpi_cache.csv")
    p.add_argument("--rerun", action="store_true", default=False,
                   help="Delete cached models and retrain from scratch")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = AUTO_DEVICE if args.device == "auto" else args.device

    if args.both:
        dtypes = ["sa", "na"]
    elif args.na:
        dtypes = ["na"]
    else:
        dtypes = ["sa"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Transformer Attention Map Comparison")
    print("=" * 65)
    print(f"  Variants : {', '.join(TF_DISPLAY.values())}")
    print(f"  Dates    : {', '.join(args.dates)}")
    print(f"  Device   : {device}")
    print(f"  Output   : {OUT_DIR}/")

    print("\nLoading CPI data …")
    y_sa, y_na = download_data(cache_dir=args.cache_dir)
    series_map = {"sa": y_sa, "na": y_na}

    best_params = {k: PAPER_BEST_PARAMS[k] for k in TF_VARIANTS}

    for label in dtypes:
        y = series_map[label]
        print(f"\n{'='*65}")
        print(f"  Series: {label.upper()}")
        print(f"{'='*65}")
        for month in args.dates:
            run_for_date(
                month, y, label, best_params, device,
                models_dir, OUT_DIR, rerun=args.rerun,
            )

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
