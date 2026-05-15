"""
Plotting utilities reproducing the paper's key figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_STYLES = {
    "rw":          dict(label="RW",          color="green",      ls="-"),
    "ar":          dict(label="AR",          color="red",        ls="--"),
    "nn":          dict(label="NN",          color="blue",       ls="--"),
    "lstm":        dict(label="LSTM",        color="cyan",       ls="-"),
    "transformer": dict(label="Transformer", color="darkorange", ls="-"),
    "ms_ar":       dict(label="MS-AR",       color="purple",     ls="--"),
    "sarima":      dict(label="SARIMA",      color="orange",     ls="--"),
}


# ── Table helpers ─────────────────────────────────────────────────────────────

def print_error_table(
    msfe_dict: dict[str, np.ndarray],
    mafe_dict: dict[str, np.ndarray],
    title: str = "Real-time forecast errors",
):
    models = [m for m in MODEL_STYLES if m in msfe_dict]
    h_max = max(len(v) for v in msfe_dict.values())
    print(f"\n{'='*80}")
    print(f"TABLE: {title}")
    print(f"{'='*80}")
    header_msfe = "".join(f"{m.upper():>10}" for m in models if m != "rw")
    header_mafe = "".join(f"{m.upper():>10}" for m in models if m not in ("rw",))
    print(f"\n{'MSFE':>4}", f"{'RW':>10}", header_msfe, "   |   MAFE", header_mafe)
    print("-" * 80)
    for h in range(1, h_max + 1):
        row = f"h={h:2d}  "
        rw_val = msfe_dict.get("rw", [np.nan] * h_max)[h - 1]
        row += f"{rw_val:10.3f}"
        for m in models:
            if m == "rw":
                continue
            val = msfe_dict[m][h - 1] if len(msfe_dict[m]) >= h else np.nan
            row += f"{val:10.3f}"
        # MAFE (skip RW as paper does)
        row += "   |   "
        for m in models:
            if m == "rw":
                continue
            val = mafe_dict.get(m, [np.nan] * h_max)[h - 1] if m in mafe_dict else np.nan
            row += f"{val:10.3f}"
        print(row)
    print("=" * 80)


# ── Figure 7: MSFE over time ──────────────────────────────────────────────────

def _plot_msfe_panel(
    ax: plt.Axes,
    model: str,
    sq_err_dict: dict[str, np.ndarray],
    origins: pd.DatetimeIndex,
):
    se = sq_err_dict[model]            # (N_test,) squared errors at h=1
    roll = pd.Series(se, index=origins).rolling(12).mean()
    ax.plot(origins, roll, label=MODEL_STYLES[model]["label"],
            color=MODEL_STYLES[model]["color"], linewidth=0.8)
    ax.set_title(MODEL_STYLES[model]["label"])
    ax.set_ylabel("MSFE (12m rolling)")


def plot_msfe_over_time(
    sq_err_dict: dict[str, np.ndarray],
    origins: pd.DatetimeIndex,
    save: bool = True,
):
    """Rolling 12-month MSFE for each model (all lines on one figure -- single panel)."""
    plt.figure(figsize=(10, 5))
    models = [m for m in MODEL_STYLES if m in sq_err_dict and m != "rw"]
    for m in models:
        se = sq_err_dict[m]          # (N_test,) squared errors at h=1
        roll = pd.Series(se, index=origins).rolling(12).mean()
        st = MODEL_STYLES[m]
        plt.plot(origins, roll, label=st["label"], color=st["color"], linewidth=0.9)
    plt.title("MSFE over time (h=1 step ahead)\n(12-month rolling average)")
    plt.ylabel("MSFE (12m rolling)")
    plt.xlabel("Forecast origin")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "fig7_msfe_over_time.png", dpi=150)
    plt.show()


# ── Figure 8/9/10: Real-time forecast path ────────────────────────────────────

def _plot_forecast_path_panel(
    ax: plt.Axes,
    fc_dict: dict[str, np.ndarray],
    y: pd.Series,
    origin: pd.Timestamp,
    h_max: int = 12,
):
    future_idx = pd.date_range(origin, periods=h_max + 1, freq="MS")[1:]
    actual_window = y[origin: future_idx[-1]].iloc[:h_max]

    ax.plot(actual_window.index, actual_window.values,
            "r-", linewidth=1.5, label="Data")
    for m, fc in fc_dict.items():
        st = MODEL_STYLES.get(m, dict(label=m, color="black", ls="--"))
        ax.plot(future_idx[:len(fc)], fc, color=st["color"],
                ls=st["ls"], linewidth=1, label=st["label"])
    ax.set_title(f"Forecast from {origin.strftime('%Y-%m')}")


def plot_forecast_path(
    fc_dict: dict[str, np.ndarray],   # model → (h_max,) h-step forecasts
    y: pd.Series,
    origin: pd.Timestamp,
    h_max: int = 12,
    save: bool = True,
    fname: str = "forecast_path.png",
):
    """Plot actual vs all model forecasts from one origin (Figures 8–10)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_forecast_path_panel(ax, fc_dict, y, origin, h_max)
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / fname, dpi=150)
    plt.show()


def plot_combined_results(
    sq_err_dict: dict[str, np.ndarray],
    origins: pd.DatetimeIndex,
    forecast_snapshots: list[tuple[pd.Timestamp, dict[str, np.ndarray]]],
    y: pd.Series,
    h_max: int = 12,
    save: bool = True,
    fname: str = "combined_results.png",
):
    """
    Show rolling MSFE for all models in one figure,
    and optionally sample forecast paths.
    """
    import matplotlib.pyplot as plt

    # ── Draw all MSFE lines in a single panel ──
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [m for m in MODEL_STYLES if m in sq_err_dict and m != "rw"]
    for m in models:
        msfe = sq_err_dict[m]
        # 12-month rolling average, ignore nan
        if len(msfe) >= 12:
            roll = pd.Series(msfe).rolling(12, min_periods=1).mean()
        else:
            roll = pd.Series(msfe)
        style = MODEL_STYLES[m]
        ax.plot(origins, roll, label=style.get("label", m), color=style.get("color", "black"), ls=style.get("ls", "-"), linewidth=1)
    ax.set_title("MSFE over time (h=1 step ahead)\n(12-month rolling average)")
    ax.set_ylabel("MSFE (12m rolling)")
    ax.set_xlabel("Forecast origin")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(FIG_DIR / fname, dpi=150)
    plt.show()


# ── Sensitivity analysis (Figures 11–12) ─────────────────────────────────────

def plot_sensitivity(
    param_name: str,
    param_values: list,
    mean_rmsfe: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    model_type: str,
    save: bool = True,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(param_values, mean_rmsfe, "o", ms=5, color="steelblue")
    ax.vlines(param_values, ci_lo, ci_hi, color="steelblue", linewidth=1.5)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Test RMSFE")
    ax.set_title(f"{model_type.upper()} sensitivity – {param_name}")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / f"sensitivity_{model_type}_{param_name}.png", dpi=150)
    plt.show()


# ── LRP bar plots (Figures 13–14) ─────────────────────────────────────────────

def plot_lrp(
    relevances: np.ndarray,
    y_input: np.ndarray,
    y_pred: float,
    model_type: str,
    p: int,
    save: bool = True,
    fname_suffix: str = "",
):
    """
    Recreate the LRP bar plots.
    relevances: (p,) array, index 0 = lag-1 (most recent).
    y_input:    (p,) input values in same order.
    """
    lags = np.arange(-p + 1, 1)        # -p+1, ..., 0  (0 = lag-1)
    # For NN/AR: relevances[0] = lag-1, relevances[-1] = lag-p → align
    # For LSTM: relevances[0] = lag-p, relevances[-1] = lag-1 → reverse
    if model_type in ("lstm", "transformer"):
        rel = relevances[::-1]          # now index 0 = lag-1
    else:
        rel = relevances

    colors = ["tomato" if r > 0 else "steelblue" for r in rel]
    alphas = np.abs(rel) / (np.abs(rel).max() + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    # Actual input series (black line)
    ax.plot(lags, y_input[::-1] if model_type not in ("lstm", "transformer") else y_input,
            "k-", linewidth=1.2)
    # Predicted value (dashed)
    ax.axhline(y_pred, color="k", ls="--", linewidth=0.8, alpha=0.6)
    # Relevance bars (coloured background bands)
    for j, (lag, r, a, c) in enumerate(zip(lags, rel, alphas, colors)):
        ax.axvspan(lag - 0.5, lag + 0.5, alpha=float(a) * 0.6, color=c, zorder=0)

    ax.set_xlabel("Lags")
    ax.set_ylabel("Inflation value")
    ax.set_title(f"LRP – {model_type.upper()}")
    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / f"lrp_{model_type}{fname_suffix}.png", dpi=150)
    plt.show()
