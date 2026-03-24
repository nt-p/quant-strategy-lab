"""Quant Strategy Lab — Streamlit entry point.

Run with: streamlit run app.py

Session state keys
------------------
prices              pd.DataFrame        full OHLCV (DATA_START → today, all loaded tickers)
fundamentals        dict[str, dict]     fundamental data per ticker (fetched alongside OHLCV)
loaded_config       dict                {tickers, start_date, end_date}
backtest_results    list[BacktestResult]  last run's results (strategies + benchmark)
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Quant Strategy Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# All imports must come after set_page_config
from data.factory import get_data_source  # noqa: E402
from data.universe import DATA_START, get_end_date  # noqa: E402
from engine.backtest import run_backtest  # noqa: E402
from engine.benchmark import run_benchmark  # noqa: E402
from strategies.registry import discover_strategies  # noqa: E402
from ui.data_preview import render_data_preview  # noqa: E402
from ui.results import render_results, render_todays_allocation  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.theme import inject_css  # noqa: E402

_PRICES_KEY = "prices"
_FUNDAMENTALS_KEY = "fundamentals"
_LOADED_CONFIG_KEY = "loaded_config"
_RESULTS_KEY = "backtest_results"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_prices(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Fetch OHLCV for all tickers (full history) and store in session state."""
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

    # Fetch fundamentals for all successfully loaded tickers
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
    # Invalidate any cached backtest results when data changes
    st.session_state.pop(_RESULTS_KEY, None)


# ── Backtest execution ────────────────────────────────────────────────────────

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
    """Run the benchmark + selected strategies and store results."""
    from engine.benchmark import _BENCHMARK_DISPLAY_NAMES  # local import to avoid cycle

    bench_label = _BENCHMARK_DISPLAY_NAMES.get(benchmark_ticker, benchmark_ticker)

    # ── Benchmark (always first) ──────────────────────────────────
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

        # ── Strategies ────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()
    st.markdown(
        """
        <div style='padding: 0.5rem 0 0.15rem 0;'>
            <h1 style='font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;
                       background: linear-gradient(90deg, #4ecdc4 0%, #81a4e8 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-family: "DM Sans", sans-serif;'>
                Quant Strategy Lab
            </h1>
            <p style='color: #636b78; font-size: 0.70rem; margin: 0.25rem 0 0 0;
                      letter-spacing: 2.5px; font-weight: 600;
                      font-family: "DM Sans", sans-serif;'>
                BACKTEST &nbsp;·&nbsp; COMPARE &nbsp;·&nbsp; ALLOCATE
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    sel = render_sidebar()

    tickers = sel["tickers"]
    start_date = sel["start_date"]
    end_date = sel["end_date"]
    initial_capital = sel["initial_capital"]
    benchmark_ticker = sel["benchmark_ticker"]
    load_data = sel["load_data"]
    run = sel["run"]
    strategy_ids = sel["selected_strategy_ids"]

    # ── Handle "Load Data" ────────────────────────────────────────
    if load_data:
        if not tickers:
            st.error("Select at least one asset before loading data.")
        else:
            with st.spinner(f"Loading data for {', '.join(tickers)}…"):
                _load_prices(tickers, start_date, end_date)
            st.rerun()
        return

    # ── Retrieve session state ────────────────────────────────────
    prices: pd.DataFrame | None = st.session_state.get(_PRICES_KEY)
    fundamentals: dict[str, dict] | None = st.session_state.get(_FUNDAMENTALS_KEY)

    if prices is None:
        st.info("Pick assets and a date range in the sidebar, then click **Load Data**.")
        st.markdown(
            """
### How it works
1. **Pick assets** — choose a curated group or type any yfinance ticker
2. **Set a date range** — from 2000 to today
3. **Load Data** — fetches and caches OHLCV prices; inspect the preview below
4. **Toggle strategies** — mix passive, quant, and ML approaches
5. **Run Backtest** — walk-forward engine produces equity curves, metrics, and drawdowns
            """
        )
        return

    # ── Handle "Run Backtest" ─────────────────────────────────────
    cfg = st.session_state.get(_LOADED_CONFIG_KEY, {})
    universe = cfg.get("tickers", tickers)

    if run:
        if not strategy_ids:
            st.error("Enable at least one strategy before running.")
        else:
            _run_backtest(
                prices, strategy_ids, universe, start_date, end_date,
                initial_capital, benchmark_ticker=benchmark_ticker,
                fundamentals=fundamentals,
            )
            st.rerun()
        return

    # ── Render ───────────────────────────────────────────────────
    results_state = st.session_state.get(_RESULTS_KEY)

    if results_state:
        strategy_results = results_state["strategy_results"]
        benchmark = results_state["benchmark"]
        initial_capital = results_state["initial_capital"]

        # Show results above the data preview
        render_results(
            strategy_results=strategy_results,
            benchmark=benchmark,
            initial_capital=initial_capital,
        )

        st.divider()

        # Today's Allocation — live weights as of the most recent price date
        all_strategies = {s.strategy_id: s for s in discover_strategies()}
        active_strategies = [
            all_strategies[r.strategy_id]
            for r in strategy_results
            if r.strategy_id in all_strategies
        ]
        render_todays_allocation(
            strategy_results=strategy_results,
            strategies=active_strategies,
            prices=prices,
            universe=universe,
            initial_capital=initial_capital,
        )

        st.divider()

    # Always show the data preview below
    render_data_preview(prices, start_date, end_date, fundamentals=fundamentals)


if __name__ == "__main__":
    main()
