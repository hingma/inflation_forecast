"""
Two-stage hyperparameter selection (Section 2 / Figure 1 of the paper):

Stage 1 – Pre-training:
  Train every hyperparameter combination on the full training set.
  Record RMSFE on the training set.

Stage 2 – Monte-Carlo cross-validation:
  Take the top-10 % candidates from Stage 1.
  For each, do N_CV random 90/10 train/validation splits of the training set.
  Record mean validation RMSFE.
  Return the single best set of hyperparameters.
"""
import itertools
import numpy as np
from typing import Any

from data import make_lag_matrix, make_lstm_sequence, select_lags
from models import build_model, train_model, model_msfe


# ── Paper's best hyperparameters (Tables 3–5, SA data) ───────────────────────
# AR  : Table 3, rank-1 — infc=None, max_lag=24, LR=0.003, epochs=500
# NN  : Table 4, rank-1 — n=20, infc=bic, max_lag=12, LR=0.001, epochs=500
# LSTM: Table 5, rank-1 — n=50, infc=None, max_lag=24, LR=0.001, epochs=2000
PAPER_BEST_PARAMS: dict[str, dict] = {
    "ar":          {"infc": None,   "max_lag": 24, "lr": 0.003, "epochs": 500},
    "nn":          {"n_hidden": 20, "infc": "bic", "max_lag": 12, "lr": 0.001, "epochs": 500},
    "lstm":        {"n_hidden": 50, "infc": None,  "max_lag": 24, "lr": 0.001, "epochs": 2000},
    "transformer": {"n_hidden": 32, "infc": None,  "max_lag": 24, "lr": 0.001, "epochs": 1000},
    "tf_nope":     {"n_hidden": 32, "infc": None,  "max_lag": 24, "lr": 0.001, "epochs": 1000},
    "tf_abspe":    {"n_hidden": 32, "infc": None,  "max_lag": 24, "lr": 0.001, "epochs": 1000},
    "tf_relpe":    {"n_hidden": 32, "infc": None,  "max_lag": 24, "lr": 0.001, "epochs": 1000},
}


# ── Hyperparameter grids (paper's Tables 3–5) ─────────────────────────────────

GRIDS: dict[str, dict[str, list]] = {
    "ar": {
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.003, 0.010, 0.050, 0.100, 0.300],
        "epochs":   [500, 1000, 1500, 2000, 5000, 9000],
    },
    "nn": {
        "n_hidden": [10, 20, 50, 100, 120],
        "infc":     [None, "bic", "hqic", "aic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.003, 0.010, 0.050],
        "epochs":   [500, 1000, 1500, 2000],
    },
    "lstm": {
        "n_hidden": [20, 50, 100],
        "infc":     [None],
        "max_lag":  [24],
        "lr":       [0.001, 0.010, 0.050, 0.100, 0.300],
        "epochs":   [500, 1000, 1500, 2000],
    },
    "transformer": {
        "n_hidden": [16, 24, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.003, 0.010, 0.050],
        "epochs":   [500, 1000, 1500, 2000],
    },
    "tf_nope": {
        "n_hidden": [16, 24, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.003, 0.010, 0.050],
        "epochs":   [500, 1000, 1500, 2000],
    },
    "tf_abspe": {
        "n_hidden": [16, 24, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.003, 0.010, 0.050],
        "epochs":   [500, 1000, 1500, 2000],
    },
    "tf_relpe": {
        "n_hidden": [16, 24, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.003, 0.010, 0.050],
        "epochs":   [500, 1000, 1500, 2000],
    },
}

# Smaller grid for quick smoke-test (set QUICK=True in main.py)
QUICK_GRIDS: dict[str, dict[str, list]] = {
    "ar": {
        "infc":    [None, "bic"],
        "max_lag": [12, 24],
        "lr":      [0.001, 0.010, 0.100],
        "epochs":  [500, 1000],
    },
    "nn": {
        "n_hidden": [10, 50],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.010],
        "epochs":   [500, 1000],
    },
    "lstm": {
        "n_hidden": [20, 50, 100],
        "infc":     [None],
        "max_lag":  [24],
        "lr":       [0.001, 0.010, 0.050],
        "epochs":   [500, 1000],
    },
    "transformer": {
        "n_hidden": [16, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.010],
        "epochs":   [500, 1000],
    },
    "tf_nope": {
        "n_hidden": [16, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.010],
        "epochs":   [500, 1000],
    },
    "tf_abspe": {
        "n_hidden": [16, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.010],
        "epochs":   [500, 1000],
    },
    "tf_relpe": {
        "n_hidden": [16, 32],
        "infc":     [None, "bic"],
        "max_lag":  [12, 24],
        "lr":       [0.001, 0.010],
        "epochs":   [500, 1000],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _param_combos(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


_SEQ_MODELS = ("lstm", "transformer", "tf_nope", "tf_abspe", "tf_relpe")


def _make_features(y: np.ndarray, p: int, model_type: str):
    if model_type in _SEQ_MODELS:
        return make_lstm_sequence(y, p)
    return make_lag_matrix(y, p)


def _fit_and_score(model_type: str, params: dict,
                   y_train: np.ndarray, y_val: np.ndarray,
                   device: str = "cpu") -> float:
    """Train on y_train, score RMSFE on y_val."""
    p = select_lags(y_train, params.get("infc"), params["max_lag"])
    n_hidden = params.get("n_hidden", 50)

    # Build training features from y_train only
    X_tr, Y_tr = _make_features(y_train, p, model_type)

    # Build validation features; the context includes train tail for lags
    y_full = np.concatenate([y_train, y_val])
    X_va, Y_va = _make_features(y_full, p, model_type)
    X_va = X_va[-len(y_val):]
    Y_va = Y_va[-len(y_val):]

    model = build_model(model_type, p, n_hidden, max_len=p + 1)
    model = train_model(model, X_tr, Y_tr, lr=params["lr"],
                        epochs=params["epochs"], device=device)
    msfe = model_msfe(model, X_va, Y_va, device)
    return float(np.sqrt(max(msfe, 0)))  # RMSFE


# ── Two-stage selection ───────────────────────────────────────────────────────

def select_hyperparams(
    model_type: str,
    y_train: np.ndarray,
    quick: bool = False,
    top_frac: float = 0.10,
    n_cv: int = 20,
    val_frac: float = 0.10,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Two-stage hyperparameter selection on the training set.

    Returns the single best parameter dict for model_type.
    """
    grid = QUICK_GRIDS[model_type] if quick else GRIDS[model_type]
    combos = _param_combos(grid)
    if verbose:
        print(f"  [{model_type.upper()}] Stage 1: {len(combos)} candidates …")

    # ── Stage 1: train on full training set, score on training set ────────────
    stage1 = []
    for i, params in enumerate(combos):
        p = select_lags(y_train, params.get("infc"), params["max_lag"])
        X, Y = _make_features(y_train, p, model_type)
        model = build_model(model_type, p, params.get("n_hidden", 50), max_len=p + 1)
        model = train_model(model, X, Y, lr=params["lr"],
                            epochs=params["epochs"], device=device)
        rmsfe = float(np.sqrt(model_msfe(model, X, Y, device)))
        stage1.append((rmsfe, params))
        if verbose and (i + 1) % max(1, len(combos) // 5) == 0:
            print(f"    {i+1}/{len(combos)} done, best so far: {min(s[0] for s in stage1):.4f}")

    stage1.sort(key=lambda t: t[0])
    n_top = max(1, int(len(combos) * top_frac))
    top_combos = [p for _, p in stage1[:n_top]]
    if verbose:
        print(f"  [{model_type.upper()}] Stage 2: CV over top {n_top} candidates …")

    # ── Stage 2: Monte-Carlo CV ───────────────────────────────────────────────
    T = len(y_train)
    n_val = max(1, int(T * val_frac))

    best_rmsfe = np.inf
    best_params = top_combos[0]

    for params in top_combos:
        cv_rmsfe = []
        for _ in range(n_cv):
            # Random contiguous validation block inside training set
            val_start = np.random.randint(T // 4, T - n_val)
            y_tr = np.concatenate([y_train[:val_start],
                                   y_train[val_start + n_val:]])
            y_va = y_train[val_start: val_start + n_val]
            try:
                rmsfe = _fit_and_score(model_type, params, y_tr, y_va, device)
                cv_rmsfe.append(rmsfe)
            except Exception:
                pass
        if cv_rmsfe:
            mean_rmsfe = float(np.mean(cv_rmsfe))
            if mean_rmsfe < best_rmsfe:
                best_rmsfe = mean_rmsfe
                best_params = params

    if verbose:
        print(f"  [{model_type.upper()}] Best CV RMSFE={best_rmsfe:.4f}, params={best_params}")
    return best_params
