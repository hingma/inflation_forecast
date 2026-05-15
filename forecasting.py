"""
Rolling-window real-time forecasting (Section 2.1 of the paper).

At each month t in the test period [1990:01, 2020:06]:
  1. Use ALL data available up to t−1 to (re)train the model.
  2. Iterate the one-step-ahead model to produce ĥ=1,...,12 forecasts.
  3. Record the forecasts against the realised values.

Benchmark models:
  RW    – ŷ_{t+h} = y_{t-1}  (random walk, n=1 case)
  SARIMA(1,1,1)(0,0,1)[12] – fitted with statsmodels MLE
  MS-AR – 2-state Markov switching AR, switching variance, via statsmodels
"""
import numpy as np
import pandas as pd
import warnings
from typing import Callable

import torch

from data import make_lag_matrix, make_lstm_sequence, select_lags
from models import ARModel, NNModel, LSTMModel, TransformerModel, build_model, train_model


H_MAX = 12


# ── Iterated h-step forecasting ───────────────────────────────────────────────

def _iter_forecast_nn(model, context: np.ndarray, p: int,
                      h_max: int = H_MAX, device: str = "cpu") -> np.ndarray:
    """
    Iterated forecast for AR / NN.
    context: recent y values (at least p). lag-1 = context[-1].
    Returns array of length h_max.
    """
    buf = list(context[-p:])          # most recent p values, oldest first
    forecasts = []
    model.eval()
    with torch.no_grad():
        for _ in range(h_max):
            x = np.array(buf[-p:][::-1], dtype=np.float32)  # [lag1, lag2, ..., lagp]
            Xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(Xt).item()
            forecasts.append(pred)
            buf.append(pred)
    return np.array(forecasts)


def _iter_forecast_lstm(model, context: np.ndarray, p: int,
                        h_max: int = H_MAX, device: str = "cpu") -> np.ndarray:
    """
    Iterated forecast for LSTM.
    Sequence is fed oldest→newest; we slide the window forward.
    """
    buf = list(context[-p:])          # oldest first
    forecasts = []
    model.eval()
    with torch.no_grad():
        for _ in range(h_max):
            seq = np.array(buf[-p:], dtype=np.float32)       # oldest→newest
            Xt = torch.tensor(seq, dtype=torch.float32,
                              device=device).unsqueeze(0).unsqueeze(-1)
            pred = model(Xt).item()
            forecasts.append(pred)
            buf.append(pred)
    return np.array(forecasts)


# ── Benchmark forecasters ─────────────────────────────────────────────────────

def rw_forecast(y_past: np.ndarray, h_max: int = H_MAX) -> np.ndarray:
    """Random walk: ŷ_{t+h} = y_{t-1} for all h."""
    return np.full(h_max, float(y_past[-1]))


def sarima_forecast(y_past: np.ndarray, h_max: int = H_MAX) -> np.ndarray:
    """SARIMA(1,1,1)(0,0,1)[12] fitted by MLE."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = SARIMAX(
                y_past,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            fc = res.forecast(steps=h_max)
            return np.asarray(fc, dtype=float)
        except Exception:
            return np.full(h_max, np.nan)


def ms_ar_forecast(y_past: np.ndarray, h_max: int = H_MAX,
                   ar_order: int = 1) -> np.ndarray:
    """
    Markov-switching AR: 2 states, switching variance, fixed AR params.
    Iterates one-step-ahead forecast h times (u_t=0 assumption).
    Expected value over regime probabilities at the last observation.
    """
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = MarkovAutoregression(
                y_past,
                k_regimes=2,
                order=ar_order,
                switching_variance=True,
                switching_ar=False,
            ).fit(disp=False, maxiter=200)

            names = res.model.param_names
            params = res.params

            # Regime-probability-weighted intercept at last obs
            smp = res.smoothed_marginal_probabilities
            p_state = (smp.values[-1] if hasattr(smp, "values") else smp[-1])  # (2,)
            intercepts = np.array([params[names.index(f"const[{k}]")]
                                   for k in range(2)])
            intercept = float(p_state @ intercepts)

            # AR coefficient (constant across regimes)
            ar_coef = float(params[names.index("ar.L1")])

            context = list(y_past[-ar_order:])
            forecasts = []
            for _ in range(h_max):
                pred = intercept + ar_coef * context[-1]
                forecasts.append(pred)
                context.append(pred)
            return np.array(forecasts, dtype=float)
        except Exception:
            return np.full(h_max, np.nan)


# ── Per-origin one-step training + h-step forecast ───────────────────────────

def _nn_forecast_at_origin(model_type: str, params: dict,
                           y_avail: np.ndarray,
                           device: str = "cpu") -> np.ndarray:
    p = select_lags(y_avail, params.get("infc"), params["max_lag"])
    seq_model = model_type in ("lstm", "transformer")
    if seq_model:
        X, Y = make_lstm_sequence(y_avail, p)
    else:
        X, Y = make_lag_matrix(y_avail, p)

    model = build_model(model_type, p, params.get("n_hidden", 50))
    model = train_model(model, X, Y, lr=params["lr"],
                        epochs=params["epochs"], device=device)

    if seq_model:
        return _iter_forecast_lstm(model, y_avail, p, H_MAX, device)
    else:
        return _iter_forecast_nn(model, y_avail, p, H_MAX, device)


# ── Main rolling-window engine ────────────────────────────────────────────────

def rolling_window_forecast(
    model_type: str,
    best_params: dict,
    y: pd.Series,
    train_end: str = "1989-12",
    test_start: str = "1990-01",
    test_end: str = "2020-06",
    h_max: int = H_MAX,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    For each origin t in [test_start, test_end]:
      - fit on y[:t-1]
      - generate h=1..h_max step-ahead forecasts

    Returns
    -------
    forecasts : (N_test, h_max) array, forecasts[i, h-1] = ĥ-step from origin i
    origins   : DatetimeIndex of forecast origins
    """
    test_idx = y[test_start:test_end].index
    N = len(test_idx)
    forecasts = np.full((N, h_max), np.nan)

    for i, date in enumerate(test_idx):
        # All data strictly before this month
        y_avail = y[:date].iloc[:-1].values.astype(np.float32)

        if model_type == "rw":
            fc = rw_forecast(y_avail, h_max)
        elif model_type == "sarima":
            fc = sarima_forecast(y_avail, h_max)
        elif model_type == "ms_ar":
            fc = ms_ar_forecast(y_avail, h_max)
        else:
            fc = _nn_forecast_at_origin(model_type, best_params, y_avail, device)

        forecasts[i, :len(fc)] = fc

        if verbose and (i + 1) % 50 == 0:
            print(f"    {i+1}/{N} origins processed")

    return forecasts, test_idx


# ── Error computation ─────────────────────────────────────────────────────────

def compute_errors(
    forecasts: np.ndarray,
    y: pd.Series,
    origins: pd.DatetimeIndex,
    h_max: int = H_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align h-step-ahead forecasts with realised values.

    Returns
    -------
    msfe : (h_max,)   mean squared forecast error per horizon
    mafe : (h_max,)   mean absolute forecast error per horizon
    """
    y_arr = y.values
    y_idx = y.index

    msfe = np.full(h_max, np.nan)
    mafe = np.full(h_max, np.nan)

    for h in range(1, h_max + 1):
        sq_errs, abs_errs = [], []
        for i, orig in enumerate(origins):
            # Realised value is h months after the origin
            target_idx = y_idx.get_indexer([orig], method="nearest")[0] + h
            if target_idx >= len(y_arr):
                continue
            fc = forecasts[i, h - 1]
            if np.isnan(fc):
                continue
            actual = y_arr[target_idx]
            sq_errs.append((fc - actual) ** 2)
            abs_errs.append(abs(fc - actual))
        if sq_errs:
            msfe[h - 1] = float(np.mean(sq_errs))
            mafe[h - 1] = float(np.mean(abs_errs))

    return msfe, mafe
