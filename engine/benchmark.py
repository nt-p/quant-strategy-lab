"""Market benchmark — buy-and-hold of a configurable benchmark ticker.

Defaults to SPY (S&P 500) for US markets.
Pass benchmark_ticker="STW.AX" for the S&P/ASX 200 benchmark.

If the benchmark ticker is not in the loaded prices DataFrame it is
fetched automatically via the cache / yfinance.
"""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult
from .metrics import compute_drawdown, compute_metrics, compute_rolling_sharpe, compute_rolling_vol

_BENCHMARK_DISPLAY_NAMES: dict[str, str] = {
    "SPY": "S&P 500 (SPY)",
    "STW.AX": "S&P/ASX 200 (STW.AX)",
}


def run_benchmark(
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float = 100_000.0,
    benchmark_ticker: str = "SPY",
) -> BacktestResult:
    """Compute buy-and-hold equity curve for the benchmark ticker.

    Parameters
    ----------
    prices : pd.DataFrame
        Full OHLCV data in long format.  The benchmark ticker is extracted
        from here if present; otherwise it is fetched from cache / yfinance.
    start_date, end_date : pd.Timestamp
        Backtest window.
    initial_capital : float
        Starting portfolio value.
    benchmark_ticker : str
        yfinance ticker to use as the benchmark.  Defaults to "SPY".

    Returns
    -------
    BacktestResult
        strategy_id = "benchmark_{safe_ticker}", category = "benchmark".
    """
    bench_data = prices[prices["ticker"] == benchmark_ticker].copy()

    if bench_data.empty:
        bench_data = _fetch_benchmark(benchmark_ticker, start_date, end_date)

    window = (
        bench_data[(bench_data["date"] >= start_date) & (bench_data["date"] <= end_date)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if window.empty:
        raise ValueError(
            f"No data available for benchmark '{benchmark_ticker}' in the selected window."
        )

    dates = window["date"].tolist()
    close = window["close"].to_numpy(dtype=float)
    first_close = close[0]

    # Buy-and-hold: equity tracks benchmark price linearly
    equity = (close / first_close * initial_capital).tolist()
    returns = [0.0] + [
        float(equity[i] / equity[i - 1] - 1.0) for i in range(1, len(equity))
    ]

    metrics = compute_metrics(equity, returns, returns)  # benchmark vs itself
    metrics["avg_monthly_turnover"] = 0.0

    safe_ticker = benchmark_ticker.replace(".", "_").replace("-", "_").lower()
    display_name = _BENCHMARK_DISPLAY_NAMES.get(benchmark_ticker, benchmark_ticker)

    return BacktestResult(
        strategy_id=f"benchmark_{safe_ticker}",
        strategy_name=display_name,
        category="benchmark",
        dates=dates,
        equity=equity,
        returns=returns,
        drawdown=compute_drawdown(equity),
        rolling_sharpe=compute_rolling_sharpe(returns),
        rolling_vol=compute_rolling_vol(returns),
        weights_history={benchmark_ticker: [1.0] * len(dates)},
        metrics=metrics,
    )


def _fetch_benchmark(
    benchmark_ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch benchmark OHLCV data via the cache layer."""
    from data.factory import get_data_source
    from data.universe import DATA_START, get_end_date

    provider = get_data_source()
    return provider.fetch_ohlcv([benchmark_ticker], start=DATA_START, end=get_end_date())
