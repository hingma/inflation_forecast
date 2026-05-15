"""
Main experiment runner.

Reproduces the paper's Tables 1 & 2 and Figures 7–14.

Configuration is read from config.yml (default) or a file passed via --config.
CLI flags override any value in the YAML file.

Usage
-----
    python main.py                          # use config.yml
    python main.py --config my.yml          # use custom config
    python main.py --device cpu             # override device from YAML
    python main.py --data-type sa           # override data.type from YAML
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import yaml

import numpy as np
import pandas as pd

# ── Setup ──────────────────────────────────────────────────────────────────────
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
(RESULTS / "figures").mkdir(exist_ok=True)

MODEL_TYPES_NN   = ["ar", "nn", "lstm", "transformer"]
MODEL_TYPES_BENCH = ["rw", "sarima", "ms_ar"]
ALL_MODELS       = MODEL_TYPES_NN + MODEL_TYPES_BENCH


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yml") -> dict:
    """Load YAML config; return empty dict if file not found."""
    p = Path(path)
    if not p.exists():
        print(f"[warn] Config file '{path}' not found — using defaults.", file=sys.stderr)
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _get(cfg: dict, *keys, default=None):
    """Nested key lookup with default."""
    for k in keys:
        if not isinstance(cfg, dict) or k not in cfg:
            return default
        cfg = cfg[k]
    return cfg


class Cfg:
    """
    Flat configuration object built from YAML + CLI overrides.
    CLI values that are not None / False override YAML values.
    """
    def __init__(self, yaml_cfg: dict, cli):
        c = yaml_cfg

        # ── data ──
        self.data_type   = cli.data_type or _get(c, "data", "type", default="both")
        self.cache_dir   = _get(c, "data", "cache_dir", default=".")

        # ── device ──
        raw_dev = cli.device or _get(c, "device", default="auto")
        if raw_dev == "auto":
            from models import DEVICE
            self.device = DEVICE
        else:
            self.device = raw_dev

        # ── selection ──
        self.use_self_defined_params = (cli.use_self_defined_params
                                        or _get(c, "selection", "use_self_defined_params", default=False))
        self.use_paper_params = (cli.use_paper_params
                                 or _get(c, "selection", "use_paper_params", default=False))
        self.quick            = (cli.quick
                                 or _get(c, "selection", "quick", default=False))
        self.skip_if_saved    = _get(c, "selection", "skip_if_saved", default=True)
        self.n_cv             = _get(c, "selection", "n_cv",      default=20)
        self.val_frac         = _get(c, "selection", "val_frac",  default=0.10)
        self.top_frac         = _get(c, "selection", "top_frac",  default=0.10)
        self.user_params      = _get(c, "user_params", default={})

        # ── forecasting ──
        self.h_max       = _get(c, "forecasting", "h_max",       default=12)
        self.train_end   = _get(c, "forecasting", "train_end",   default="1989-12")
        self.test_start  = _get(c, "forecasting", "test_start",  default="1990-01")
        self.test_end    = _get(c, "forecasting", "test_end",    default="2020-06")

        # ── lrp / sensitivity ──
        self.lrp_enabled  = not cli.no_lrp  and _get(c, "lrp",         "enabled", default=True)
        self.sens_enabled = not cli.no_sensitivity and _get(c, "sensitivity", "enabled", default=True)
        self.sens_n_cv    = _get(c, "sensitivity", "n_cv", default=20)

        # ── results ──
        self.results_dir    = Path(_get(c, "results", "dir", default="results"))
        self.save_figures   = _get(c, "results", "save_figures", default=True)
        self.save_tables    = _get(c, "results", "save_tables",  default=True)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Inflation forecasting replication (Almosova & Andresen 2023)."
    )
    p.add_argument("--config", default="config.yml",
                   help="Path to YAML config file (default: config.yml)")
    # CLI overrides — all optional; None/False means "use YAML value"
    p.add_argument("--quick",                   action="store_true", default=False)
    p.add_argument("--use-self-defined-params", action="store_true", default=False,
                   help="Use user_params from config (highest priority, overrides YAML)")
    p.add_argument("--use-paper-params",        action="store_true", default=False,
                   help="Use Tables 3-5 best params directly (overrides YAML)")
    p.add_argument("--data-type",  default=None, choices=["sa", "na", "both"])
    p.add_argument("--device",     default=None,
                   help="'cpu' | 'cuda' | 'mps' | 'auto'  (overrides YAML)")
    p.add_argument("--no-lrp",         action="store_true", default=False)
    p.add_argument("--no-sensitivity", action="store_true", default=False)
    return p.parse_args()


# ── Helper ─────────────────────────────────────────────────────────────────────
def run_experiment(y: pd.Series, label: str, cfg: "Cfg", best_params_all: dict):
    """Full pipeline for one inflation series (SA or NA)."""
    from model_selection import select_hyperparams
    from forecasting import rolling_window_forecast, compute_errors, H_MAX
    from plots import print_error_table, plot_combined_results

    y_train = y[cfg.train_end if False else "1960-01": cfg.train_end]

    # ── 1. Model selection ────────────────────────────────────────────────────
    best_params: dict[str, dict] = {}
    params_file = cfg.results_dir / f"best_params_{label}.json"

    if cfg.use_self_defined_params:
        best_params = {mt: dict(p) for mt, p in cfg.user_params.items()}
        print(f"\nUsing user-defined hyperparameters:")
        for mt, p in best_params.items():
            print(f"  {mt.upper():5s}: {p}")
    elif cfg.use_paper_params:
        from model_selection import PAPER_BEST_PARAMS
        best_params = dict(PAPER_BEST_PARAMS)
        print(f"\nUsing paper's best hyperparameters:")
        for mt, p in best_params.items():
            print(f"  {mt.upper():5s}: {p}")
    elif cfg.skip_if_saved and params_file.exists():
        print(f"\nLoading saved hyperparameters for [{label}] from {params_file}")
        with open(params_file) as f:
            best_params = json.load(f)
    else:
        print(f"\n{'='*60}")
        print(f"Hyperparameter selection [{label}]")
        print(f"{'='*60}")
        for mt in MODEL_TYPES_NN:
            best_params[mt] = select_hyperparams(
                mt, y_train.values,
                quick=cfg.quick,
                top_frac=cfg.top_frac,
                n_cv=cfg.n_cv,
                val_frac=cfg.val_frac,
                device=cfg.device,
                verbose=True,
            )
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=2)

    best_params_all[label] = best_params

    # ── 1b. Train and save final models (reused by LRP) ──────────────────────
    print(f"\nFinal model training [{label}] …")
    models_dir = cfg.results_dir / "models"
    for mt in MODEL_TYPES_NN:
        params = best_params.get(mt, {})
        if not params:
            continue
        model_path = models_dir / f"{mt}_{label}.pt"
        meta_path  = models_dir / f"{mt}_{label}_meta.json"
        model, _ = _load_model(model_path, meta_path, cfg.device)
        if model is not None:
            print(f"  {mt.upper()}: loaded ← {model_path}")
        else:
            _train_and_save_model(
                mt, y_train.values.astype(np.float32),
                params, model_path, meta_path, cfg.device,
            )

    # ── 2. Rolling-window forecasting ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Rolling-window real-time forecasting [{label}]")
    print(f"{'='*60}")

    fc_cache = cfg.results_dir / f"forecasts_{label}.pkl"
    if fc_cache.exists():
        print(f"Loading cached forecasts from {fc_cache}")
        with open(fc_cache, "rb") as f:
            all_fc, origins = pickle.load(f)
    else:
        all_fc: dict[str, np.ndarray] = {}
        origins = None
        for mt in ALL_MODELS:
            print(f"\n  Model: {mt.upper()}")
            params = best_params.get(mt, {})
            fc, orig = rolling_window_forecast(
                mt, params, y,
                train_end=cfg.train_end,
                test_start=cfg.test_start,
                test_end=cfg.test_end,
                h_max=cfg.h_max,
                device=cfg.device,
                verbose=True,
            )
            all_fc[mt] = fc
            if origins is None:
                origins = orig
        with open(fc_cache, "wb") as f:
            pickle.dump((all_fc, origins), f)

    # ── 3. Compute MSFE / MAFE ────────────────────────────────────────────────
    msfe_dict: dict[str, np.ndarray] = {}
    mafe_dict: dict[str, np.ndarray] = {}
    for mt in ALL_MODELS:
        if mt in all_fc:
            ms, ma = compute_errors(all_fc[mt], y, origins, cfg.h_max)
            msfe_dict[mt] = ms
            mafe_dict[mt] = ma

    print_error_table(msfe_dict, mafe_dict,
                      title=f"Real-time forecast errors – {label.upper()} data")

    if cfg.save_tables:
        rows = []
        for h in range(1, cfg.h_max + 1):
            row = {"h": h}
            for mt in ALL_MODELS:
                if mt in msfe_dict:
                    row[f"msfe_{mt}"] = round(msfe_dict[mt][h - 1], 4)
                    row[f"mafe_{mt}"] = round(mafe_dict.get(mt, [np.nan]*cfg.h_max)[h - 1], 4)
            rows.append(row)
        out = cfg.results_dir / f"errors_{label}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Errors saved → {out}")

    # ── 4. Figure 7: MSFE over time ───────────────────────────────────────────
    sq_err_dict = {}
    for mt in ALL_MODELS:
        if mt in all_fc and mt != "rw":
            se = []
            for i, orig in enumerate(origins):
                target_pos = y.index.get_indexer([orig], method="nearest")[0] + 1
                if target_pos < len(y):
                    fc = all_fc[mt][i, 0]
                    se.append((fc - y.iloc[target_pos]) ** 2)
                else:
                    se.append(np.nan)
            sq_err_dict[mt] = np.array(se)
    # ── 5. Combined figure: MSFE and sample forecast paths ────────────────────
    sample_dates = ["2007-03-01", "2010-07-01", "1996-02-01"]
    forecast_snapshots = []
    for date_str in sample_dates:
        date = pd.Timestamp(date_str)
        if date not in origins:
            continue
        idx = origins.get_indexer([date], method="nearest")[0]
        fc_at_date = {mt: all_fc[mt][idx] for mt in ALL_MODELS if mt in all_fc}
        forecast_snapshots.append((date, fc_at_date))
    plot_combined_results(
        sq_err_dict,
        origins,
        forecast_snapshots,
        y,
        h_max=cfg.h_max,
        save=cfg.save_figures,
        fname=f"combined_results_{label}.png",
    )

    return best_params


# ── Sensitivity analysis ───────────────────────────────────────────────────────
def run_sensitivity(y_train: np.ndarray, best_params: dict,
                    label: str, cfg: "Cfg"):
    """Replicate Figures 11–12: RMSFE vs single hyperparameter."""
    from data import make_lag_matrix, make_lstm_sequence, select_lags
    from models import build_model, train_model, model_msfe
    from plots import plot_sensitivity

    sens_cache = cfg.results_dir / f"sensitivity_{label}.pkl"

    if sens_cache.exists():
        print(f"  Loading cached sensitivity results ← {sens_cache}")
        with open(sens_cache, "rb") as f:
            results = pickle.load(f)
        for mt, param_name, vals, means, lo, hi in results:
            plot_sensitivity(param_name, vals,
                             np.array(means), np.array(lo), np.array(hi),
                             mt + "_" + label, save=cfg.save_figures)
        return

    n_cv  = cfg.sens_n_cv
    T     = len(y_train)
    n_val = max(1, int(T * 0.10))

    def cv_score(mt, params_override):
        params = {**best_params.get(mt, {}), **params_override}
        scores = []
        np.random.seed(0)
        for _ in range(n_cv):
            val_start = np.random.randint(T // 4, T - n_val)
            y_tr = np.concatenate([y_train[:val_start], y_train[val_start + n_val:]])
            y_va = y_train[val_start: val_start + n_val]
            p = select_lags(y_tr, params.get("infc"), params["max_lag"])
            if mt in ("lstm", "transformer"):
                X_tr, Y_tr = make_lstm_sequence(y_tr, p)
                X_full, Y_full = make_lstm_sequence(np.concatenate([y_tr, y_va]), p)
            else:
                X_tr, Y_tr = make_lag_matrix(y_tr, p)
                X_full, Y_full = make_lag_matrix(np.concatenate([y_tr, y_va]), p)
            X_va = X_full[-len(y_va):]
            Y_va = Y_full[-len(y_va):]
            try:
                m = build_model(mt, p, params.get("n_hidden", 50))
                m = train_model(m, X_tr, Y_tr, lr=params["lr"],
                                epochs=params["epochs"], device=cfg.device)
                scores.append(np.sqrt(model_msfe(m, X_va, Y_va, cfg.device)))
            except Exception:
                pass
        return np.array(scores) if scores else np.array([np.nan])

    grid = [
        ("hidden_units",  "n_hidden", [10, 20, 30, 50, 75, 100, 120, 140]),
        ("learning_rate", "lr",       [0.001, 0.003, 0.005, 0.010, 0.030, 0.050, 0.075, 0.100]),
        ("n_epochs",      "epochs",   [100, 250, 500, 750, 1000, 1250, 1500, 2000]),
    ]

    results = []
    for mt in ["nn", "lstm", "transformer"]:
        if not best_params.get(mt):
            continue
        for param_name, key, vals in grid:
            means, lo, hi = [], [], []
            for v in vals:
                sc = cv_score(mt, {key: v})
                means.append(np.nanmean(sc))
                lo.append(np.nanpercentile(sc, 5))
                hi.append(np.nanpercentile(sc, 95))
            results.append((mt, param_name, vals, means, lo, hi))

    with open(sens_cache, "wb") as f:
        pickle.dump(results, f)
    print(f"  Sensitivity results saved → {sens_cache}")

    for mt, param_name, vals, means, lo, hi in results:
        plot_sensitivity(param_name, vals,
                         np.array(means), np.array(lo), np.array(hi),
                         mt + "_" + label, save=cfg.save_figures)


# ── Model persistence helpers ─────────────────────────────────────────────────
def _load_model(model_path: Path, meta_path: Path, device: str):
    """Return (model, meta) from disk, or (None, None) if not found."""
    import torch
    from models import build_model
    if not (model_path.exists() and meta_path.exists()):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    model = build_model(meta["model_type"], meta["p"], meta["n_hidden"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return model, meta


def _train_and_save_model(mt: str, y_train: np.ndarray, params: dict,
                          model_path: Path, meta_path: Path, device: str):
    """Train model on full training data, save weights + metadata, return (model, meta)."""
    import torch
    from data import make_lag_matrix, make_lstm_sequence, select_lags
    from models import build_model, train_model

    p        = select_lags(y_train, params.get("infc"), params["max_lag"])
    n_hidden = params.get("n_hidden", 50)

    if mt in ("lstm", "transformer"):
        X, Y = make_lstm_sequence(y_train, p)
    else:
        X, Y = make_lag_matrix(y_train, p)

    model = build_model(mt, p, n_hidden)
    model = train_model(model, X, Y, lr=params["lr"],
                        epochs=params["epochs"], device=device)

    torch.save(model.state_dict(), model_path)
    meta = {"model_type": mt, "p": p, "n_hidden": n_hidden}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Model trained and saved → {model_path}")
    return model, meta


# ── LRP analysis ──────────────────────────────────────────────────────────────
def run_lrp(y: pd.Series, best_params: dict, label: str, cfg: "Cfg"):
    """Load or train best models on full training set, then run LRP analysis."""
    import torch
    from data import TRAIN_START, TRAIN_END, make_lag_matrix, make_lstm_sequence
    from lrp import compute_lrp
    from plots import plot_lrp

    y_train    = y[TRAIN_START:TRAIN_END].values.astype(np.float32)
    models_dir = cfg.results_dir / "models"

    for mt in ["ar", "nn", "lstm", "transformer"]:
        params = best_params.get(mt)
        if params is None:
            continue

        model_path = models_dir / f"{mt}_{label}.pt"
        meta_path  = models_dir / f"{mt}_{label}_meta.json"

        # ── Training phase: reuse saved weights or train from scratch ─────────
        model, meta = _load_model(model_path, meta_path, cfg.device)
        if model is not None:
            print(f"  Loaded saved model ← {model_path}")
        else:
            model, meta = _train_and_save_model(
                mt, y_train, params, model_path, meta_path, cfg.device
            )

        p = meta["p"]

        # ── Prediction / LRP phase ────────────────────────────────────────────
        if mt in ("lstm", "transformer"):
            X, _ = make_lstm_sequence(y_train, p)
        else:
            X, _ = make_lag_matrix(y_train, p)

        for i, idx in enumerate([len(X) // 2, len(X) - 1]):
            x_inp = X[idx]
            with torch.no_grad():
                xt = torch.tensor(x_inp, dtype=torch.float32).unsqueeze(0)
                if mt in ("lstm", "transformer"):
                    xt = xt.unsqueeze(-1)
                y_pred = model(xt).item()
            rel    = compute_lrp(model, x_inp, mt)
            x_plot = x_inp[::-1] if mt in ("lstm", "transformer") else x_inp
            plot_lrp(rel, x_plot, y_pred, mt, p, save=cfg.save_figures,
                     fname_suffix=f"_{label}_ex{i+1}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    cli      = parse_args()
    yaml_cfg = load_config(cli.config)
    cfg      = Cfg(yaml_cfg, cli)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "figures").mkdir(exist_ok=True)
    (cfg.results_dir / "models").mkdir(exist_ok=True)

    print(f"\nInflation Forecasting Replication")
    print(f"  config       : {cli.config}")
    print(f"  device       : {cfg.device}")
    print(f"  data         : {cfg.data_type}")
    print(f"  self params  : {cfg.use_self_defined_params}")
    print(f"  paper params : {cfg.use_paper_params}")
    print(f"  lrp          : {cfg.lrp_enabled}")
    print(f"  sensitivity  : {cfg.sens_enabled}")

    from data import download_data

    print("\nDownloading / loading CPI data …")
    y_sa, y_na = download_data(cache_dir=cfg.cache_dir)
    print(f"  SA: {y_sa.index[0].strftime('%Y-%m')} – {y_sa.index[-1].strftime('%Y-%m')} "
          f"({len(y_sa)} obs)")
    print(f"  NA: {y_na.index[0].strftime('%Y-%m')} – {y_na.index[-1].strftime('%Y-%m')} "
          f"({len(y_na)} obs)")

    datasets = {}
    if cfg.data_type in ("sa", "both"):
        datasets["sa"] = y_sa
    if cfg.data_type in ("na", "both"):
        datasets["na"] = y_na

    best_params_all = {}

    for label, y in datasets.items():
        run_experiment(y, label, cfg, best_params_all)
        best_params = best_params_all[label]

        if cfg.sens_enabled:
            print(f"\nSensitivity analysis [{label}] …")
            run_sensitivity(
                y["1960-01": cfg.train_end].values.astype(np.float32),
                best_params, label, cfg,
            )

        if cfg.lrp_enabled:
            print(f"\nLRP analysis [{label}] …")
            run_lrp(y, best_params, label, cfg)

    print(f"\nDone. Results saved in {cfg.results_dir}/")


if __name__ == "__main__":
    main()
