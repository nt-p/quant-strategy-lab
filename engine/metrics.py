"""Performance metrics for backtested strategies.

All metrics operate on daily equity / return series.  The convention
throughout is that index 0 is the first day of the backtest (return = 0,
equity = initial_capital).  Metric functions therefore skip index 0.
"""

from __future__ import annotations

import math

import numpy as np


# ── Public API ────────────────────────────────────────────────────────────────

def compute_metrics(
    equity: list[float],
    returns: list[float],
    benchmark_returns: list[float],
) -> dict[str, float]:
    """Compute the full suite of performance metrics.

    Parameters
    ----------
    equity : list[float]
        Daily portfolio value, index 0 = initial_capital.
    returns : list[float]
        Daily portfolio returns, index 0 = 0.0.
    benchmark_returns : list[float]
        Daily benchmark (SPY) returns, aligned to the same dates.
        Used for excess-return and information-ratio calculations.

    Returns
    -------
    dict[str, float]
        Keys: total_return, cagr, ann_vol, sharpe, sortino,
              max_drawdown, max_drawdown_duration, calmar,
              win_rate, excess_return, information_ratio.
        Values are floats; percentages are expressed as decimals (0.12 = 12%).
    """
    r = np.array(returns[1:], dtype=float)   # skip day-0 zero
    eq = np.array(equity, dtype=float)
    b = np.array(benchmark_returns[1:], dtype=float)

    # Pad benchmark to match if lengths differ (alignment is caller's job,
    # but we guard here to avoid crashes)
    min_len = min(len(r), len(b))
    r_aligned = r[:min_len]
    b_aligned = b[:min_len]

    n_days = len(r)
    years = n_days / 252.0

    total_return = float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0
    cagr = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    ann_vol = _annualised_vol(r)
    sharpe = _sharpe(r)
    sortino = _sortino(r)

    dd = compute_drawdown(eq.tolist())
    max_dd = float(min(dd)) if dd else 0.0
    max_dd_dur = _max_drawdown_duration(dd)

    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    win_rate = float(np.mean(r > 0)) if len(r) > 0 else 0.0

    bench_total = float(np.prod(1.0 + b_aligned) - 1.0) if len(b_aligned) > 0 else 0.0
    excess_return = total_return - bench_total
    information_ratio = _information_ratio(r_aligned, b_aligned)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration": float(max_dd_dur),
        "calmar": calmar,
        "win_rate": win_rate,
        "excess_return": excess_return,
        "information_ratio": information_ratio,
    }


def compute_drawdown(equity: list[float]) -> list[float]:
    """Compute daily peak-to-trough drawdown series.

    Parameters
    ----------
    equity : list[float]
        Daily portfolio values.

    Returns
    -------
    list[float]
        Drawdown at each point: (equity - running_peak) / running_peak.
        Values are <= 0.
    """
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (eq - peak) / peak, 0.0)
    return dd.tolist()


def compute_rolling_sharpe(returns: list[float], window: int = 60) -> list[float]:
    """60-day rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.
    window : int
        Rolling window in trading days.

    Returns
    -------
    list[float]
        NaN for the warm-up period, then annualised Sharpe.
    """
    r = np.array(returns, dtype=float)
    out = np.full(len(r), float("nan"))
    for i in range(window, len(r) + 1):
        chunk = r[i - window : i]
        std = chunk.std(ddof=1)
        if std > 1e-10:
            out[i - 1] = chunk.mean() / std * math.sqrt(252)
    return out.tolist()


def compute_rolling_vol(returns: list[float], window: int = 60) -> list[float]:
    """60-day rolling annualised volatility.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.
    window : int
        Rolling window in trading days.

    Returns
    -------
    list[float]
        NaN for the warm-up period, then annualised vol.
    """
    r = np.array(returns, dtype=float)
    out = np.full(len(r), float("nan"))
    for i in range(window, len(r) + 1):
        out[i - 1] = r[i - window : i].std(ddof=1) * math.sqrt(252)
    return out.tolist()


# ── Private helpers ───────────────────────────────────────────────────────────

def _annualised_vol(r: np.ndarray) -> float:
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(252))


def _sharpe(r: np.ndarray) -> float:
    if len(r) < 2:
        return float("nan")
    std = r.std(ddof=1)
    if std < 1e-10:
        return float("nan")
    return float(r.mean() / std * math.sqrt(252))


def _sortino(r: np.ndarray) -> float:
    if len(r) < 2:
        return float("nan")
    neg = r[r < 0]
    if len(neg) < 2:
        return float("nan")
    downside_std = neg.std(ddof=1)
    if downside_std < 1e-10:
        return float("nan")
    return float(r.mean() / downside_std * math.sqrt(252))


def _max_drawdown_duration(drawdown: list[float]) -> int:
    """Length in days of the longest period spent below the previous peak."""
    max_dur = 0
    current = 0
    for dd in drawdown:
        if dd < 0:
            current += 1
            max_dur = max(max_dur, current)
        else:
            current = 0
    return max_dur


def _information_ratio(r: np.ndarray, b: np.ndarray) -> float:
    if len(r) < 2 or len(b) < 2:
        return float("nan")
    excess = r - b
    std = excess.std(ddof=1)
    if std < 1e-10:
        return float("nan")
    return float(excess.mean() / std * math.sqrt(252))
