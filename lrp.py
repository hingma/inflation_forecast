"""
Layer-wise Relevance Propagation (LRP) for NN and LSTM models.

For NN: standard ε-LRP rule propagated through the two linear layers.
For LSTM: gradient × input (GI) as a tractable approximation of the
          Arras et al. (2017) LSTM-LRP, sufficient to reproduce the
          qualitative patterns shown in the paper's Figures 13–14.

Both return an array of length p (one relevance score per input lag),
where index 0 = lag-1 (most recent) and index p-1 = lag-p (oldest).
"""
import numpy as np
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from models import NNModel, LSTMModel, TransformerModel


EPS = 1e-9


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


# ── NN LRP (ε-rule) ───────────────────────────────────────────────────────────

def lrp_nn(model: NNModel, x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    ε-LRP for the two-layer NN.

    Parameters
    ----------
    model : trained NNModel
    x     : (p,) input vector [lag-1, ..., lag-p]

    Returns
    -------
    R_input : (p,) relevance for each input lag
    """
    model.eval()
    device = _model_device(model)
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)  # (1, p)

        # Forward pass – save activations
        z1 = model.fc1(xt)                     # (1, n_hidden) pre-activation
        h  = model.relu(z1)                    # (1, n_hidden) post-ReLU
        z2 = model.fc2(h)                      # (1, 1)

        out = z2.squeeze().item()

        # Layer 2 → hidden: R_n ← R_out · (h_n w2_n) / (Σ h_m w2_m + ε)
        w2 = model.fc2.weight.data.squeeze()   # (n_hidden,)
        b2 = model.fc2.bias.data.squeeze()
        h_np  = h.squeeze().cpu().numpy()
        w2_np = w2.cpu().numpy()

        num2 = h_np * w2_np
        den2 = num2.sum() + b2.item() + eps
        R_hidden = num2 / den2 * out           # (n_hidden,)

        # Layer 1 → input: R_i ← Σ_n R_n · (x_i w1_ni) / (Σ_j x_j w1_nj + ε)
        w1 = model.fc1.weight.data.cpu().numpy()     # (n_hidden, p)
        b1 = model.fc1.bias.data.cpu().numpy()       # (n_hidden,)
        x_np = x.astype(np.float64)

        R_input = np.zeros(len(x_np))
        for n in range(len(R_hidden)):
            num1 = x_np * w1[n]
            den1 = num1.sum() + b1[n] + eps
            R_input += R_hidden[n] * num1 / den1

    return R_input.astype(np.float32)


# ── LSTM LRP (gradient × input approximation) ────────────────────────────────

def lrp_lstm(model: LSTMModel, x: np.ndarray) -> np.ndarray:
    """
    Gradient × input relevance for LSTM (Arras et al. 2017 approximation).

    Parameters
    ----------
    model : trained LSTMModel
    x     : (p,) input sequence [oldest, ..., most recent]  (lag-p, ..., lag-1)

    Returns
    -------
    R_input : (p,) relevance per time step.
              Index 0 = oldest lag, index p-1 = most recent lag.
              To align with paper's plotting convention (lag-1 on right),
              reverse this array before plotting.
    """
    model.eval()
    device = _model_device(model)
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    xt.requires_grad_(True)

    out = model(xt)
    out.backward()

    grad = xt.grad.detach().squeeze().cpu().numpy()   # (p,)
    inp  = x                                    # (p,)
    gi   = grad * inp                           # gradient × input

    return gi.astype(np.float32)


# ── AR LRP (analytical – AR weights are the relevances) ──────────────────────

def lrp_ar(model, x: np.ndarray) -> np.ndarray:
    """
    For a linear AR model the relevance of each input is simply w_i · x_i.
    Normalised so they sum to the prediction.
    """
    with torch.no_grad():
        w = model.fc.weight.data.squeeze().cpu().numpy()  # (p,)
        b = model.fc.bias.data.item()
    R = w * x
    return R.astype(np.float32)


# ── Transformer LRP (gradient × input approximation) ─────────────────────────

def lrp_transformer(model: TransformerModel, x: np.ndarray) -> np.ndarray:
    """
    Gradient × input relevance for Transformer.

    Parameters
    ----------
    model : trained TransformerModel
    x     : (p,) input sequence [oldest, ..., most recent]  (lag-p, ..., lag-1)

    Returns
    -------
    R_input : (p,) relevance per time step, oldest first.
    """
    model.eval()
    device = _model_device(model)
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    xt.requires_grad_(True)
    out = model(xt)
    out.backward()
    grad = xt.grad.detach().squeeze().cpu().numpy()  # (p,)
    return (grad * x).astype(np.float32)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def compute_lrp(model: nn.Module, x: np.ndarray,
                model_type: str) -> np.ndarray:
    """
    Compute input-level relevances for a single input vector x.

    For AR/NN:          x = [lag-1, ..., lag-p]
    For LSTM/Transformer: x = [lag-p, ..., lag-1]  (oldest first)
    """
    if model_type == "ar":
        return lrp_ar(model, x)
    elif model_type == "nn":
        return lrp_nn(model, x)
    elif model_type == "lstm":
        return lrp_lstm(model, x)
    elif model_type == "transformer":
        return lrp_transformer(model, x)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── Per-target LRP model persistence ──────────────────────────────────────────

def lrp_target_timestamp(month: str) -> pd.Timestamp:
    """Return the month-end timestamp used by the CPI series index."""
    return pd.Period(month, freq="M").to_timestamp("M")


def _lrp_model_paths(models_dir: Path, model_type: str, label: str,
                     target_date: pd.Timestamp) -> tuple[Path, Path]:
    """LRP uses per-target models, separate from the full-sample models."""
    month_tag = target_date.strftime("%Y-%m")
    lrp_dir = models_dir / "lrp"
    return (
        lrp_dir / f"{model_type}_{label}_{month_tag}.pt",
        lrp_dir / f"{model_type}_{label}_{month_tag}_meta.json",
    )


def _load_lrp_model(model_path: Path, meta_path: Path, device: str):
    """Return (model, meta) from disk, or (None, None) if not found."""
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


def _train_and_save_lrp_model(model_type: str, y_train: np.ndarray, params: dict,
                              target_date: pd.Timestamp, model_path: Path,
                              meta_path: Path, device: str):
    """Train the model available at one target date and persist it for LRP."""
    from data import make_lag_matrix, make_lstm_sequence, select_lags
    from models import build_model, train_model

    p = select_lags(y_train, params.get("infc"), params["max_lag"])
    n_hidden = params.get("n_hidden", 50)

    if len(y_train) <= p:
        raise ValueError(
            f"Not enough observations before {target_date.strftime('%Y-%m')} "
            f"to train {model_type.upper()} with p={p}"
        )

    if model_type in ("lstm", "transformer"):
        X, Y = make_lstm_sequence(y_train, p)
    else:
        X, Y = make_lag_matrix(y_train, p)

    model = build_model(model_type, p, n_hidden)
    model = train_model(model, X, Y, lr=params["lr"],
                        epochs=params["epochs"], device=device)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    meta = {
        "model_type": model_type,
        "p": p,
        "n_hidden": n_hidden,
        "target_date": target_date.strftime("%Y-%m"),
        "trained_through": (target_date - pd.offsets.MonthEnd(1)).strftime("%Y-%m"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Model trained and saved -> {model_path}")
    return model, meta


def load_or_train_lrp_model(model_type: str, y: pd.Series, params: dict,
                            label: str, target_date: pd.Timestamp,
                            models_dir: Path, device: str):
    """
    Load the model corresponding exactly to one LRP target month.

    If no such model exists, train on observations strictly before target_date,
    save under results/models/lrp/, and return the new model.
    """
    model_path, meta_path = _lrp_model_paths(models_dir, model_type, label, target_date)
    model, meta = _load_lrp_model(model_path, meta_path, device)
    if model is not None:
        print(f"  Loaded target-specific model <- {model_path}")
        return model, meta

    y_train = y[:target_date].iloc[:-1].values.astype(np.float32)
    return _train_and_save_lrp_model(
        model_type, y_train, params, target_date, model_path, meta_path, device
    )


def make_lrp_input_for_date(y: pd.Series, target_date: pd.Timestamp,
                            p: int, model_type: str) -> np.ndarray:
    """Build the single input vector that predicts target_date."""
    y_avail = y[:target_date].iloc[:-1].values.astype(np.float32)
    if len(y_avail) < p:
        raise ValueError(
            f"Not enough observations before {target_date.strftime('%Y-%m')} "
            f"to build an LRP input with p={p}"
        )
    x_recent = y_avail[-p:]
    if model_type in ("lstm", "transformer"):
        return x_recent.astype(np.float32)
    return x_recent[::-1].astype(np.float32)


# ── Aggregate LRP over multiple test observations ─────────────────────────────

def aggregate_lrp(model: nn.Module, X: np.ndarray,
                  model_type: str) -> np.ndarray:
    """
    Average absolute relevances over all rows of X.
    Returns array of shape (p,).
    """
    Rs = np.stack([compute_lrp(model, X[i], model_type) for i in range(len(X))])
    return Rs.mean(axis=0)
