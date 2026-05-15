"""
Data loading and preprocessing for inflation forecasting.
Downloads monthly US CPI inflation from FRED (1960:01-2020:06).
"""
import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_START = "1960-01"
TRAIN_END   = "1989-12"
TEST_START  = "1990-01"
TEST_END    = "2020-06"

FRED_SA = "CPALTT01USM661S"   # SA monthly % change
FRED_NA = "CPALTT01USM657N"   # NA monthly % change
FALLBACK_SA = "CPIAUCSL"      # SA CPI index (fallback)
FALLBACK_NA = "CPIAUCNS"      # NA CPI index (fallback)


# ── Download ──────────────────────────────────────────────────────────────────

def _download_fred(series_id: str, api_key: str | None = None) -> pd.Series:
    """Download a FRED series.  API key used if provided, else direct CSV URL."""
    if api_key:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        s = fred.get_series(series_id, observation_start="1959-12-01",
                            observation_end="2020-07-01")
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        return s.dropna()

    # Public CSV endpoint – no API key required
    import io
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True,
                     na_values=[".", ""])
    s = df.iloc[:, 0].dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    s = s.loc[:"2020-07-01"]
    return s


def _to_monthly_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change() * 100


def download_data(api_key: str | None = None,
                  cache_dir: str | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Return (sa_inflation, na_inflation) as monthly % change Series,
    covering 1960:01-2020:06.

    Tries the paper's exact FRED series first; falls back to standard
    CPIAUCSL/CPIAUCNS and computes MoM % change.
    """
    cache = Path(cache_dir or ".") / "cpi_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index)
        return df["sa"], df["na"]

    def _series_or_pct(raw: pd.Series) -> pd.Series:
        """If |mean| < 5 assume already %-change; otherwise compute MoM pct change."""
        return raw if raw.abs().mean() < 5 else _to_monthly_pct_change(raw)

    try:
        sa_raw = _download_fred(FRED_SA, api_key)   # index level → needs pct_change
        na_raw = _download_fred(FRED_NA, api_key)   # already MoM% → use directly
        sa = _series_or_pct(sa_raw)
        na = _series_or_pct(na_raw)
    except Exception:
        sa_idx = _download_fred(FALLBACK_SA, api_key)
        na_idx = _download_fred(FALLBACK_NA, api_key)
        sa = _to_monthly_pct_change(sa_idx)
        na = _to_monthly_pct_change(na_idx)

    sa = sa.loc[TRAIN_START:TEST_END].dropna()
    na = na.loc[TRAIN_START:TEST_END].dropna()
    sa.name = "sa"
    na.name = "na"

    pd.DataFrame({"sa": sa, "na": na}).to_csv(cache)
    return sa, na


def load_cached(path: str = "cpi_cache.csv") -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["sa"], df["na"]


# ── Feature engineering ───────────────────────────────────────────────────────

def make_lag_matrix(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised dataset from univariate series.

    Returns
    -------
    X : (T-p, p)  each row = [y_{t-1}, y_{t-2}, ..., y_{t-p}]  (lag-1 first)
    Y : (T-p,)    each element = y_t
    """
    T = len(y)
    X = np.zeros((T - p, p), dtype=np.float32)
    Y = np.zeros(T - p, dtype=np.float32)
    for t in range(p, T):
        X[t - p] = y[t - p: t][::-1]          # lag-1, lag-2, ..., lag-p
        Y[t - p] = y[t]
    return X, Y


def make_lstm_sequence(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build LSTM dataset: sequences fed oldest→newest.

    Returns
    -------
    X : (T-p, p)  each row = [y_{t-p}, ..., y_{t-1}]  (oldest first)
    Y : (T-p,)    each element = y_t
    """
    T = len(y)
    X = np.zeros((T - p, p), dtype=np.float32)
    Y = np.zeros(T - p, dtype=np.float32)
    for t in range(p, T):
        X[t - p] = y[t - p: t]                 # oldest → newest
        Y[t - p] = y[t]
    return X, Y


# ── Lag selection via information criteria ────────────────────────────────────

def _ar_ic(y: np.ndarray, p: int, ic: str) -> float:
    """Fit AR(p) by OLS and return the requested information criterion."""
    from statsmodels.tsa.ar_model import AutoReg
    try:
        res = AutoReg(y, lags=p, old_names=False).fit()
        return getattr(res, ic)
    except Exception:
        return np.inf


def select_lags(y: np.ndarray, criterion: str | None, max_lag: int) -> int:
    """
    Return optimal lag order.
    criterion in {None, 'bic', 'aic', 'hqic'}; None → max_lag.
    """
    if criterion is None:
        return max_lag
    scores = [_ar_ic(y, p, criterion) for p in range(1, max_lag + 1)]
    return int(np.argmin(scores)) + 1
