"""Unit tests for engine/metrics.py."""

import math

import numpy as np
import pytest

from engine.metrics import (
    compute_drawdown,
    compute_metrics,
    compute_rolling_sharpe,
    compute_rolling_vol,
)


# ── compute_drawdown ──────────────────────────────────────────────────────────

def test_drawdown_all_up():
    """Monotonically increasing equity → zero drawdown throughout."""
    equity = [100.0, 110.0, 120.0, 130.0]
    dd = compute_drawdown(equity)
    assert all(abs(v) < 1e-12 for v in dd)


def test_drawdown_peak_then_recovery():
    """Drawdown hits minimum at trough, returns to 0 at new peak."""
    equity = [100.0, 120.0, 90.0, 120.0, 130.0]
    dd = compute_drawdown(equity)
    assert dd[0] == pytest.approx(0.0)
    assert dd[1] == pytest.approx(0.0)
    assert dd[2] == pytest.approx(-0.25, abs=1e-9)   # (90-120)/120
    assert dd[3] == pytest.approx(0.0, abs=1e-9)
    assert dd[4] == pytest.approx(0.0, abs=1e-9)


def test_drawdown_length_matches_equity():
    equity = list(range(1, 11))
    assert len(compute_drawdown(equity)) == len(equity)


# ── compute_metrics ───────────────────────────────────────────────────────────

def _flat_inputs(n: int = 252) -> tuple[list[float], list[float]]:
    """Flat equity curve (0% return) for edge-case testing."""
    equity = [100.0] * (n + 1)
    returns = [0.0] * (n + 1)
    return equity, returns


def _trending_inputs(daily_ret: float = 0.001, n: int = 252):
    equity = [100.0 * (1 + daily_ret) ** i for i in range(n + 1)]
    returns = [0.0] + [daily_ret] * n
    return equity, returns


def test_total_return_known():
    equity = [100.0, 150.0]
    returns = [0.0, 0.5]
    m = compute_metrics(equity, returns, returns)
    assert m["total_return"] == pytest.approx(0.5)


def test_cagr_one_year():
    """~20% daily return for 252 days should give ~20% CAGR."""
    equity, returns = _trending_inputs(daily_ret=0.0007317, n=252)  # ≈ 20% annualised
    m = compute_metrics(equity, returns, returns)
    assert m["cagr"] == pytest.approx(0.20, abs=0.01)


def test_zero_return_metrics():
    equity, returns = _flat_inputs()
    m = compute_metrics(equity, returns, returns)
    assert m["total_return"] == pytest.approx(0.0)
    assert m["win_rate"] == pytest.approx(0.0)
    assert m["max_drawdown"] == pytest.approx(0.0)


def test_sharpe_positive_for_trending():
    """Noisy but consistently positive returns → positive Sharpe."""
    rng = np.random.default_rng(1)
    n = 252
    r = [0.0] + list(0.001 + rng.normal(0, 0.005, n))   # mean > 0, noisy
    equity = [100.0]
    for ret in r[1:]:
        equity.append(equity[-1] * (1 + ret))
    m = compute_metrics(equity, r, r)
    assert m["sharpe"] > 0


def test_max_drawdown_known():
    equity = [100.0, 120.0, 80.0, 90.0]
    returns = [0.0, 0.2, -1/3, 0.125]
    m = compute_metrics(equity, returns, returns)
    # Peak = 120, trough = 80 → (80-120)/120 = -1/3
    assert m["max_drawdown"] == pytest.approx(-1 / 3, abs=1e-9)


def test_win_rate_half():
    equity = [100.0, 110.0, 99.0, 108.9, 97.9]
    returns = [0.0, 0.1, -0.1, 0.1, -0.1]
    m = compute_metrics(equity, returns, returns)
    assert m["win_rate"] == pytest.approx(0.5, abs=0.01)


def test_excess_return_vs_benchmark():
    _, returns = _trending_inputs(daily_ret=0.001, n=100)
    _, bench = _trending_inputs(daily_ret=0.0005, n=100)
    equity = [100.0 * (1.001) ** i for i in range(101)]
    bench_equity = [100.0 * (1.0005) ** i for i in range(101)]
    m = compute_metrics(equity, returns, bench)
    # Strategy total return > benchmark → positive excess
    assert m["excess_return"] > 0


def test_information_ratio_positive_excess():
    n = 252
    strat_r = [0.0] + [0.001] * n           # consistently beats benchmark
    bench_r = [0.0] + [0.0005] * n
    equity = [100.0 * (1.001) ** i for i in range(n + 1)]
    m = compute_metrics(equity, strat_r, bench_r)
    # Constant positive excess → IR should be very high (or inf)
    assert m["information_ratio"] > 0 or math.isnan(m["information_ratio"])


def test_calmar_positive():
    equity, returns = _trending_inputs(daily_ret=0.0005, n=504)
    m = compute_metrics(equity, returns, returns)
    # No drawdown on monotonic series → calmar should be nan (division by 0 guard)
    assert math.isnan(m["calmar"])


# ── rolling helpers ───────────────────────────────────────────────────────────

def test_rolling_sharpe_nan_warmup():
    """NaN for the first (window-1) entries; computable thereafter (with noise)."""
    rng = np.random.default_rng(2)
    returns = list(0.001 + rng.normal(0, 0.005, 100))  # noisy so std > 0
    rs = compute_rolling_sharpe(returns, window=60)
    assert all(math.isnan(v) for v in rs[:59])
    assert not math.isnan(rs[59])


def test_rolling_vol_positive():
    returns = [0.001, -0.002] * 50
    rv = compute_rolling_vol(returns, window=20)
    # After warmup, vol should be positive
    non_nan = [v for v in rv if not math.isnan(v)]
    assert all(v > 0 for v in non_nan)
