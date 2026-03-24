"""Walk-forward backtest engine.

Architecture
------------
- get_weights() is called at the *close* of each rebalance date using all
  price data up to and including that date (no lookahead).
- The resulting weights take effect on the *next* trading day.
- Between rebalances the portfolio drifts with market returns (no implicit
  daily rebalancing).
- Transaction costs are not modelled (Phase 3 scope).

Weight drift
------------
After each day's return, the effective weight of each position changes:
    w_i(t) = w_i(t-1) * (1 + r_i(t)) / (1 + R_portfolio(t))
This drift is tracked precisely so that turnover at each rebalance event
reflects the true cost of getting back to target weights.
"""

from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategies.base import RebalanceFrequency, StrategyBase

from .metrics import (
    compute_drawdown,
    compute_metrics,
    compute_rolling_sharpe,
    compute_rolling_vol,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Full output of a single strategy backtest.

    All list fields are aligned to the same daily date index.
    """

    strategy_id: str
    strategy_name: str
    category: str                              # StrategyCategory.value or "benchmark"

    dates: list[pd.Timestamp]                  # trading days in the backtest window
    equity: list[float]                        # daily portfolio value ($ terms)
    returns: list[float]                       # daily portfolio returns (day-0 = 0)
    drawdown: list[float]                      # daily peak-to-trough drawdown (≤ 0)
    rolling_sharpe: list[float]                # 60-day rolling annualised Sharpe
    rolling_vol: list[float]                   # 60-day rolling annualised vol

    weights_history: dict[str, list[float]]    # {ticker: [daily weights]}
    metrics: dict[str, float]                  # summary statistics
    turnovers: list[float] = field(default_factory=list)  # turnover at each rebalance


# ── Public API ────────────────────────────────────────────────────────────────

def run_backtest(
    strategy: StrategyBase,
    prices: pd.DataFrame,
    universe: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float = 100_000.0,
    benchmark_returns: list[float] | None = None,
    fundamentals: dict[str, dict] | None = None,
) -> BacktestResult:
    """Run a walk-forward backtest for one strategy.

    Parameters
    ----------
    strategy : StrategyBase
        Instantiated strategy.  Must have get_weights() implemented.
    prices : pd.DataFrame
        Full OHLCV history in long format (DATA_START → today, all universe tickers).
        The engine slices this before passing to get_weights() to prevent lookahead.
    universe : list[str]
        Tickers to include in the portfolio.
    start_date, end_date : pd.Timestamp
        Backtest window.  Equity curve covers this window.
        Strategies may use data *before* start_date for signal generation.
    initial_capital : float
        Starting portfolio value in dollars.
    benchmark_returns : list[float] | None
        Aligned daily benchmark returns, used for information ratio.
        If None, excess metrics will be NaN.
    fundamentals : dict[str, dict] or None
        Optional fundamental data keyed by ticker (see DataSource.fetch_fundamentals).
        Passed to get_weights() only for strategies whose signature declares the
        ``fundamentals`` parameter; older strategies receive only (prices, date, universe).

    Returns
    -------
    BacktestResult
    """
    # Pre-compute whether this strategy's get_weights accepts fundamentals
    _strategy_accepts_fundamentals = (
        "fundamentals" in inspect.signature(strategy.get_weights).parameters
    )
    # ── 0. Training (ML strategies) ───────────────────────────────
    if strategy.requires_training():
        train_prices = prices[prices["date"] < start_date]
        strategy.train(train_prices, start_date)

    # ── 1. Build the close-price pivot for the backtest window ────
    window_prices = prices[
        (prices["date"] >= start_date) & (prices["date"] <= end_date)
    ].copy()

    # Keep only universe tickers that have price data
    available = [t for t in universe if t in window_prices["ticker"].unique()]
    if not available:
        raise ValueError("None of the universe tickers have price data in the backtest window.")

    close = (
        window_prices[window_prices["ticker"].isin(available)]
        .pivot_table(values="close", index="date", columns="ticker")
        .sort_index()
    )

    trading_days: list[pd.Timestamp] = close.index.tolist()
    if len(trading_days) < 2:
        raise ValueError("Backtest window contains fewer than 2 trading days.")

    # Daily returns (NaN on first row)
    daily_rets = close.pct_change()

    # ── 2. Determine rebalance schedule ───────────────────────────
    rebalance_dates: set[pd.Timestamp] = set(
        _get_rebalance_dates(trading_days, strategy.rebalance_frequency)
    )

    # ── 3. Walk-forward loop ──────────────────────────────────────
    equity = initial_capital
    current_weights: dict[str, float] = {}
    prev_weights: dict[str, float] = {}

    equity_curve: list[float] = []
    returns_list: list[float] = []
    daily_weights: list[dict[str, float]] = []
    turnovers: list[float] = []

    for i, today in enumerate(trading_days):
        # ── Rebalance: use signals from previous close ─────────────
        # Day 0: rebalance using day-0 close (initial allocation)
        # Day i>0: rebalance if yesterday was a rebalance date
        should_rebalance = (
            i == 0
            or trading_days[i - 1] in rebalance_dates
        )

        if should_rebalance:
            signal_date = trading_days[i - 1] if i > 0 else today
            prices_slice = prices[prices["date"] <= signal_date]
            if _strategy_accepts_fundamentals:
                new_weights = strategy.get_weights(
                    prices_slice, signal_date, available, fundamentals=fundamentals
                )
            else:
                new_weights = strategy.get_weights(prices_slice, signal_date, available)
            # Cash handling: remainder below 1.0 → CASH; above 1.0 → normalise down + warn.
            total_w = sum(new_weights.values())
            if total_w > 1.0 + 1e-6:
                warnings.warn(
                    f"Strategy '{strategy.strategy_id}' returned weights summing to "
                    f"{total_w:.6f} (> 1.0). Normalising down to 1.0.",
                    stacklevel=2,
                )
                new_weights = {t: w / total_w for t, w in new_weights.items()}
                cash_weight = 0.0
            elif total_w > 1e-9:
                cash_weight = 1.0 - total_w
            else:
                # All-cash allocation
                new_weights = {}
                cash_weight = 1.0
            if cash_weight > 1e-9:
                new_weights["CASH"] = cash_weight

            # Turnover = half the sum of absolute weight changes
            if current_weights:
                all_tickers = set(current_weights) | set(new_weights)
                turnover = sum(
                    abs(new_weights.get(t, 0.0) - current_weights.get(t, 0.0))
                    for t in all_tickers
                ) / 2.0
                turnovers.append(turnover)

            prev_weights = dict(current_weights)
            current_weights = {t: w for t, w in new_weights.items() if w > 1e-9}

        # ── Record equity on day 0 (no return yet) ────────────────
        if i == 0:
            equity_curve.append(equity)
            returns_list.append(0.0)
            daily_weights.append(dict(current_weights))
            continue

        # ── Apply today's market returns ───────────────────────────
        row = daily_rets.loc[today]
        port_ret = 0.0
        next_weights: dict[str, float] = {}

        for ticker, w in current_weights.items():
            if ticker == "CASH":
                # Cash earns 0% — dollar value unchanged, weight drifts with portfolio
                next_weights["CASH"] = w
                continue
            r = float(row.get(ticker, float("nan")))
            if math.isnan(r):
                r = 0.0
            port_ret += w * r
            next_weights[ticker] = w * (1.0 + r)

        equity *= 1.0 + port_ret
        equity_curve.append(equity)
        returns_list.append(port_ret)

        # ── Drift weights ──────────────────────────────────────────
        total = sum(next_weights.values())
        if total > 1e-9:
            current_weights = {t: w / total for t, w in next_weights.items()}
        else:
            current_weights = {}

        daily_weights.append(dict(current_weights))

    # ── 4. Build weights_history {ticker: [daily_weight]} ─────────
    # Include CASH only if the strategy held cash on at least one day
    has_cash = any("CASH" in day_w for day_w in daily_weights)
    tracked = list(available) + (["CASH"] if has_cash else [])
    weights_history: dict[str, list[float]] = {t: [] for t in tracked}
    for day_w in daily_weights:
        for t in tracked:
            weights_history[t].append(day_w.get(t, 0.0))

    # ── 5. Metrics ────────────────────────────────────────────────
    bench_rets = benchmark_returns if benchmark_returns is not None else [0.0] * len(returns_list)
    metrics = compute_metrics(equity_curve, returns_list, bench_rets)

    # Average monthly turnover (annualised fraction)
    if turnovers:
        rebal_per_year = {
            RebalanceFrequency.NEVER: 1.0,
            RebalanceFrequency.WEEKLY: 52.0,
            RebalanceFrequency.MONTHLY: 12.0,
            RebalanceFrequency.QUARTERLY: 4.0,
        }.get(strategy.rebalance_frequency, 12.0)
        metrics["avg_monthly_turnover"] = float(np.mean(turnovers)) * (rebal_per_year / 12.0)
    else:
        metrics["avg_monthly_turnover"] = 0.0

    return BacktestResult(
        strategy_id=strategy.strategy_id,
        strategy_name=strategy.name,
        category=strategy.category.value,
        dates=trading_days,
        equity=equity_curve,
        returns=returns_list,
        drawdown=compute_drawdown(equity_curve),
        rolling_sharpe=compute_rolling_sharpe(returns_list),
        rolling_vol=compute_rolling_vol(returns_list),
        weights_history=weights_history,
        metrics=metrics,
        turnovers=turnovers,
    )


# ── Rebalance schedule helpers ────────────────────────────────────────────────

def _get_rebalance_dates(
    trading_days: list[pd.Timestamp],
    freq: RebalanceFrequency,
) -> list[pd.Timestamp]:
    """Return the subset of trading_days on which rebalancing occurs.

    The first trading day is always included so the strategy can set
    its initial allocation.

    Parameters
    ----------
    trading_days : list[pd.Timestamp]
        All trading days in the backtest window, in order.
    freq : RebalanceFrequency
        Desired rebalance cadence.

    Returns
    -------
    list[pd.Timestamp]
        Sorted rebalance dates (subset of trading_days).
    """
    if not trading_days:
        return []

    first = trading_days[0]

    if freq == RebalanceFrequency.NEVER:
        return [first]

    if freq == RebalanceFrequency.WEEKLY:
        return [trading_days[i] for i in range(0, len(trading_days), 5)]

    if freq == RebalanceFrequency.MONTHLY:
        seen: set[tuple[int, int]] = set()
        result: list[pd.Timestamp] = []
        for d in trading_days:
            key = (d.year, d.month)
            if key not in seen:
                seen.add(key)
                result.append(d)
        return result

    if freq == RebalanceFrequency.QUARTERLY:
        seen_q: set[tuple[int, int]] = set()
        result_q: list[pd.Timestamp] = []
        for d in trading_days:
            key = (d.year, (d.month - 1) // 3)
            if key not in seen_q:
                seen_q.add(key)
                result_q.append(d)
        return result_q

    return [first]
