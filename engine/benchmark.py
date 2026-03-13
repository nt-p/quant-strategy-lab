"""S&P 500 benchmark — buy-and-hold SPY.

The benchmark is always computed regardless of the user's asset selection.
If SPY is not in the loaded prices, it is fetched via the cache layer.
"""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult
from .metrics import compute_drawdown, compute_metrics, compute_rolling_sharpe, compute_rolling_vol


def run_benchmark(
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Compute buy-and-hold SPY equity curve as the benchmark.

    Parameters
    ----------
    prices : pd.DataFrame
        Full OHLCV data in long format.  SPY is extracted from here if present.
        If SPY is missing, it is fetched from the cache / yfinance.
    start_date, end_date : pd.Timestamp
        Backtest window.
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    BacktestResult
        strategy_id = "benchmark_spy", category = "benchmark".
    """
    spy = prices[prices["ticker"] == "SPY"].copy()

    if spy.empty:
        spy = _fetch_spy(start_date, end_date)

    window = (
        spy[(spy["date"] >= start_date) & (spy["date"] <= end_date)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if window.empty:
        raise ValueError("No SPY data available in the selected backtest window.")

    dates = window["date"].tolist()
    close = window["close"].to_numpy(dtype=float)
    first_close = close[0]

    # Buy-and-hold: equity tracks SPY price linearly
    equity = (close / first_close * initial_capital).tolist()
    returns = [0.0] + [
        float(equity[i] / equity[i - 1] - 1.0) for i in range(1, len(equity))
    ]

    metrics = compute_metrics(equity, returns, returns)  # benchmark vs itself
    metrics["avg_monthly_turnover"] = 0.0

    return BacktestResult(
        strategy_id="benchmark_spy",
        strategy_name="S&P 500 (SPY)",
        category="benchmark",
        dates=dates,
        equity=equity,
        returns=returns,
        drawdown=compute_drawdown(equity),
        rolling_sharpe=compute_rolling_sharpe(returns),
        rolling_vol=compute_rolling_vol(returns),
        weights_history={"SPY": [1.0] * len(dates)},
        metrics=metrics,
    )


def _fetch_spy(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch SPY OHLCV data via the cache layer."""
    from data.factory import get_data_source
    from data.universe import DATA_START, get_end_date

    provider = get_data_source()
    return provider.fetch_ohlcv(["SPY"], start=DATA_START, end=get_end_date())
