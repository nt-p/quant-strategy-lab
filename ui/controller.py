"""Sidebar action handlers — portfolio loading and backtest execution.

Two entry points:
- handle_portfolio_run()   → called from app.py (Portfolio Hub)
- handle_sidebar_actions() → called from pages/1_Strategy_Lab.py (Strategy Lab)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data.factory import get_data_source
from data.universe import BENCHMARK_TICKER, DATA_START, get_end_date
from engine.backtest import run_backtest
from engine.benchmark import _BENCHMARK_DISPLAY_NAMES, run_benchmark
from engine.metrics import compute_metrics
from strategies.registry import discover_strategies
from ui.sidebar import render_portfolio_builder_sidebar, render_strategy_sidebar

_PRICES_KEY = "prices"
_FUNDAMENTALS_KEY = "fundamentals"
_LOADED_CONFIG_KEY = "loaded_config"
_RESULTS_KEY = "backtest_results"


def _load_prices(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    provider = get_data_source()
    end_str = get_end_date()

    failed: list[str] = []
    succeeded: list[str] = []
    frames: list[pd.DataFrame] = []

    progress = st.progress(0, text="Fetching data…")
    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Fetching {ticker}…")
        df = provider.fetch_ohlcv([ticker], start=DATA_START, end=end_str)
        if df.empty:
            failed.append(ticker)
        else:
            frames.append(df)
            succeeded.append(ticker)
    progress.empty()

    if failed:
        st.warning(
            f"Could not fetch: {', '.join(failed)}. "
            "Check that the symbols are valid yfinance tickers."
        )

    if not frames:
        st.error("No data loaded. Check your ticker symbols and try again.")
        return

    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    st.session_state[_PRICES_KEY] = prices

    progress2 = st.progress(0, text="Fetching fundamentals…")
    fundamentals: dict[str, dict] = {}
    for i, ticker in enumerate(succeeded):
        progress2.progress((i + 1) / len(succeeded), text=f"Fundamentals: {ticker}…")
        try:
            result = provider.fetch_fundamentals([ticker])
            fundamentals.update(result)
        except Exception:
            pass
    progress2.empty()

    st.session_state[_FUNDAMENTALS_KEY] = fundamentals
    st.session_state[_LOADED_CONFIG_KEY] = {
        "tickers": succeeded,
        "start_date": start_date,
        "end_date": end_date,
    }
    st.session_state.pop(_RESULTS_KEY, None)


def _run_backtest(
    prices: pd.DataFrame,
    strategy_ids: list[str],
    universe: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    benchmark_ticker: str = "SPY",
    fundamentals: dict[str, dict] | None = None,
) -> None:
    bench_label = _BENCHMARK_DISPLAY_NAMES.get(benchmark_ticker, benchmark_ticker)
    status = st.status("Running backtest…", expanded=True)

    with status:
        st.write(f"Running {bench_label} benchmark…")
        try:
            benchmark = run_benchmark(
                prices, start_date, end_date, initial_capital,
                benchmark_ticker=benchmark_ticker,
            )
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")
            return

        bench_returns = benchmark.returns
        all_strategies = {s.strategy_id: s for s in discover_strategies()}
        strategy_results = []

        for sid in strategy_ids:
            strategy = all_strategies.get(sid)
            if strategy is None:
                st.warning(f"Strategy '{sid}' not found — skipped.")
                continue
            st.write(f"Running {strategy.name}…")
            try:
                result = run_backtest(
                    strategy=strategy,
                    prices=prices,
                    universe=universe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    benchmark_returns=bench_returns,
                    fundamentals=fundamentals,
                )
                strategy_results.append(result)
            except Exception as exc:
                st.warning(f"{strategy.name} failed: {exc}")

        status.update(label="Backtest complete.", state="complete", expanded=False)

    if not strategy_results:
        st.error("All strategies failed. Check your selections.")
        return

    st.session_state[_RESULTS_KEY] = {
        "strategy_results": strategy_results,
        "benchmark": benchmark,
        "initial_capital": initial_capital,
    }


def handle_portfolio_run() -> bool:
    """Render the Portfolio Hub sidebar and handle Run Portfolio.

    Fetches price data, computes portfolio + benchmark returns, stores to
    session state, then reruns.  Returns True when an action was handled
    so the caller can ``st.stop()``.
    """
    sel = render_portfolio_builder_sidebar()

    tickers: list[str] = sel["tickers"]
    weights: dict[str, float] = sel["weights"]
    start_date: pd.Timestamp = sel["start_date"]
    end_date: pd.Timestamp   = sel["end_date"]
    benchmark_ticker: str    = sel["benchmark_ticker"]
    run: bool                = sel["run"]

    if not run:
        return False

    if not tickers:
        st.error("Select at least one asset before running.")
        return True

    # ── 1. Fetch OHLCV ────────────────────────────────────────────────────────
    with st.spinner(f"Loading data for {', '.join(tickers)}…"):
        _load_prices(tickers, start_date, end_date)

    prices: pd.DataFrame | None = st.session_state.get(_PRICES_KEY)
    if prices is None:
        return True   # _load_prices already showed an error

    # ── 2. Normalise weights to fractions summing to 1 ────────────────────────
    raw_pct = {t: weights.get(t, 0.0) for t in tickers}
    total_pct = sum(raw_pct.values())
    if total_pct <= 0:
        st.error("Portfolio weights must sum to more than 0%.")
        return True
    norm_weights = {t: w / total_pct for t, w in raw_pct.items()}

    # ── 3. Build wide-format daily close prices (window only) ─────────────────
    window = prices[
        (prices["date"] >= start_date) & (prices["date"] <= end_date)
    ].copy()

    close_wide: pd.DataFrame = (
        window.pivot_table(values="close", index="date", columns="ticker")
        .sort_index()
    )

    # Drop tickers with no data in the window
    available = [t for t in tickers if t in close_wide.columns]
    if not available:
        st.error("No price data available for the selected tickers in this date range.")
        return True

    close_wide = close_wide[available].dropna(how="all")

    # ── 4. Compute per-holding daily returns ───────────────────────────────────
    returns_wide: pd.DataFrame = close_wide.pct_change().iloc[1:]  # drop day-0 NaN row

    # ── 5. Compute weighted portfolio return series ───────────────────────────
    avail_weights = {t: norm_weights.get(t, 0.0) for t in available}
    total_avail = sum(avail_weights.values())
    if total_avail > 0:
        avail_weights = {t: w / total_avail for t, w in avail_weights.items()}

    weight_series = pd.Series(avail_weights)
    portfolio_returns: pd.Series = returns_wide[list(weight_series.index)].fillna(0.0).dot(weight_series)

    # ── 6. Fetch benchmark returns ────────────────────────────────────────────
    provider = get_data_source()
    bench_df = provider.fetch_ohlcv([benchmark_ticker], start=DATA_START, end=get_end_date())

    if bench_df.empty:
        # Fall back to first available ticker as benchmark if fetch fails
        bench_returns = portfolio_returns * 0.0
    else:
        bench_window = bench_df[
            (bench_df["date"] >= start_date) & (bench_df["date"] <= end_date)
        ]
        bench_close = (
            bench_window.pivot_table(values="close", index="date", columns="ticker")
            .sort_index()
        )
        if benchmark_ticker in bench_close.columns:
            bench_returns = bench_close[benchmark_ticker].pct_change().iloc[1:]
        else:
            bench_returns = portfolio_returns * 0.0

    # Align benchmark to portfolio dates
    bench_returns = bench_returns.reindex(portfolio_returns.index).fillna(0.0)

    # ── 7. Store everything to session state ──────────────────────────────────
    st.session_state["portfolio"] = {
        "holdings": norm_weights,
        "start_date": start_date,
        "end_date": end_date,
        "benchmark_ticker": benchmark_ticker,
        "name": "My Portfolio",
    }
    st.session_state["returns"]            = returns_wide
    st.session_state["portfolio_returns"]  = portfolio_returns
    st.session_state["benchmark_returns"]  = bench_returns
    # Keep legacy keys so Strategy Lab still works
    st.session_state[_LOADED_CONFIG_KEY] = {
        "tickers": available,
        "start_date": start_date,
        "end_date": end_date,
    }
    # Clear any stale backtest results when portfolio changes
    st.session_state.pop(_RESULTS_KEY, None)

    st.rerun()
    return True


def handle_sidebar_actions() -> bool:
    """Render the Strategy Lab sidebar and handle Run Backtest.

    Returns True when an action was handled so the caller can ``st.stop()``.
    """
    sel = render_strategy_sidebar()

    strategy_ids: list[str]  = sel["selected_strategy_ids"]
    initial_capital: float   = sel["initial_capital"]
    benchmark_ticker: str    = sel["benchmark_ticker"]
    run: bool                = sel["run"]

    if not run:
        return False

    prices: pd.DataFrame | None = st.session_state.get(_PRICES_KEY)
    fundamentals: dict[str, dict] | None = st.session_state.get(_FUNDAMENTALS_KEY)
    portfolio = st.session_state.get("portfolio")

    if not strategy_ids:
        st.error("Enable at least one strategy before running.")
        return True

    if prices is None or portfolio is None:
        st.error("Load a portfolio in Portfolio Hub before running a backtest.")
        return True

    cfg = st.session_state.get(_LOADED_CONFIG_KEY, {})
    universe: list[str] = cfg.get("tickers", list(portfolio.get("holdings", {}).keys()))
    start_date: pd.Timestamp = portfolio["start_date"]
    end_date: pd.Timestamp   = portfolio["end_date"]

    _run_backtest(
        prices, strategy_ids, universe, start_date, end_date,
        initial_capital, benchmark_ticker=benchmark_ticker,
        fundamentals=fundamentals,
    )
    st.rerun()
    return True
