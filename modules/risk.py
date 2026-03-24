"""Risk analytics computations for backtested strategies.

All functions operate on daily return series (floats).  The convention
matches engine/backtest.py: index 0 of ``returns`` is always 0.0 (the
first backtest day has no prior return), so every function skips it.

No Streamlit or Plotly imports here — this module is pure computation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from engine.backtest import BacktestResult


# ── Value-at-Risk & Expected Shortfall ───────────────────────────────────────

def compute_var(returns: list[float], confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at the given confidence level.

    Returns the loss (positive number) not exceeded on ``confidence`` fraction
    of days.  E.g. VaR(0.95) = 0.02 means only 5% of days had losses > 2%.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.  Index 0 (= 0.0) is skipped.
    confidence : float
        Confidence level in (0, 1).  Default 0.95.

    Returns
    -------
    float
        VaR as a positive number (loss magnitude).  NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    if len(r) < 10:
        return float("nan")
    return float(-np.percentile(r, (1.0 - confidence) * 100))


def compute_cvar(returns: list[float], confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) at the given confidence level.

    The average loss on the worst ``(1 - confidence)`` fraction of days.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.  Index 0 (= 0.0) is skipped.
    confidence : float
        Confidence level in (0, 1).  Default 0.95.

    Returns
    -------
    float
        CVaR as a positive number (loss magnitude).  NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    if len(r) < 10:
        return float("nan")
    var_threshold = -compute_var(r.tolist(), confidence)  # negative (loss side)
    tail = r[r <= var_threshold]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


# ── Distribution statistics ───────────────────────────────────────────────────

def compute_tail_stats(returns: list[float]) -> dict[str, float]:
    """Descriptive statistics of the return distribution.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.  Index 0 (= 0.0) is skipped.

    Returns
    -------
    dict[str, float]
        Keys:
        - ``skewness``      : Fisherian skewness (0 = symmetric).
        - ``excess_kurtosis``: Excess kurtosis (0 = normal tails).
        - ``tail_ratio``    : |95th pct| / |5th pct|.  > 1 = fatter right tail.
        - ``best_day``      : Best single-day return (as a decimal).
        - ``worst_day``     : Worst single-day return (as a decimal, negative).
        - ``pct_positive``  : Fraction of days with a positive return.
    """
    r = np.array(returns[1:], dtype=float)
    if len(r) < 4:
        nan = float("nan")
        return dict(
            skewness=nan,
            excess_kurtosis=nan,
            tail_ratio=nan,
            best_day=nan,
            worst_day=nan,
            pct_positive=nan,
        )

    skewness = float(stats.skew(r, bias=False))
    excess_kurtosis = float(stats.kurtosis(r, fisher=True, bias=False))  # Fisher → excess

    p95 = float(np.percentile(r, 95))
    p05 = float(np.percentile(r, 5))
    tail_ratio = abs(p95) / abs(p05) if abs(p05) > 1e-10 else float("nan")

    return dict(
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        tail_ratio=tail_ratio,
        best_day=float(r.max()),
        worst_day=float(r.min()),
        pct_positive=float(np.mean(r > 0)),
    )


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_correlation_matrix(results: list[BacktestResult]) -> pd.DataFrame:
    """Pairwise Pearson correlation of daily returns across strategies.

    Returns aligned to the intersection of all result date sets so every
    series has the same length.

    Parameters
    ----------
    results : list[BacktestResult]
        Results to correlate (strategies + benchmark).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix indexed and columned by ``strategy_name``.
        Empty DataFrame if fewer than 2 results are provided.
    """
    if len(results) < 2:
        return pd.DataFrame()

    # Build a DataFrame: index = dates (intersection), columns = strategy names
    series: dict[str, pd.Series] = {}
    for r in results:
        s = pd.Series(r.returns, index=r.dates, name=r.strategy_name)
        series[r.strategy_name] = s

    df = pd.DataFrame(series).dropna()
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    return df.iloc[1:].corr(method="pearson")   # skip day-0 zeros


# ── Market beta & alpha ───────────────────────────────────────────────────────

def compute_beta_alpha(
    returns: list[float],
    benchmark_returns: list[float],
) -> tuple[float, float]:
    """OLS market beta and annualised Jensen's alpha vs the benchmark.

    Parameters
    ----------
    returns : list[float]
        Daily strategy returns.  Index 0 skipped.
    benchmark_returns : list[float]
        Daily benchmark returns, same length and alignment as ``returns``.

    Returns
    -------
    tuple[float, float]
        ``(beta, annualised_alpha)``.  Both NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    b = np.array(benchmark_returns[1:], dtype=float)

    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]

    if min_len < 30:
        return float("nan"), float("nan")

    result = stats.linregress(b, r)
    beta = float(result.slope)
    alpha_daily = float(result.intercept)
    alpha_annualised = alpha_daily * 252
    return beta, alpha_annualised


def compute_r_squared(
    returns: list[float],
    benchmark_returns: list[float],
) -> float:
    """R² of strategy returns regressed against benchmark returns.

    Parameters
    ----------
    returns : list[float]
        Daily strategy returns.  Index 0 skipped.
    benchmark_returns : list[float]
        Daily benchmark returns, same alignment.

    Returns
    -------
    float
        Coefficient of determination in [0, 1].  NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    b = np.array(benchmark_returns[1:], dtype=float)

    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]

    if min_len < 30:
        return float("nan")

    result = stats.linregress(b, r)
    return float(result.rvalue ** 2)


# ── Drawdown detail ───────────────────────────────────────────────────────────

def compute_drawdown_stats(drawdown: list[float]) -> dict[str, float]:
    """Extended drawdown statistics from a pre-computed drawdown series.

    Parameters
    ----------
    drawdown : list[float]
        Daily peak-to-trough drawdown series (values ≤ 0), as produced by
        ``engine.metrics.compute_drawdown``.

    Returns
    -------
    dict[str, float]
        Keys:
        - ``max_drawdown``       : Worst single drawdown (≤ 0).
        - ``avg_drawdown``       : Mean drawdown on days spent underwater (≤ 0).
        - ``pct_time_underwater``: Fraction of days with drawdown < 0.
        - ``n_drawdown_periods`` : Number of distinct drawdown episodes.
    """
    dd = np.array(drawdown, dtype=float)
    if len(dd) == 0:
        nan = float("nan")
        return dict(
            max_drawdown=nan,
            avg_drawdown=nan,
            pct_time_underwater=nan,
            n_drawdown_periods=nan,
        )

    underwater = dd < -1e-9
    underwater_vals = dd[underwater]

    max_dd = float(dd.min())
    avg_dd = float(underwater_vals.mean()) if len(underwater_vals) > 0 else 0.0
    pct_underwater = float(underwater.mean())

    # Count distinct episodes (transitions from above to below waterline)
    n_periods = 0
    in_period = False
    for val in dd:
        if val < -1e-9:
            if not in_period:
                n_periods += 1
                in_period = True
        else:
            in_period = False

    return dict(
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        pct_time_underwater=pct_underwater,
        n_drawdown_periods=float(n_periods),
    )
