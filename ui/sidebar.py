"""Sidebar UI component.

Renders (in order):
0. Market selector — US Market / Australian Market / Both Markets
1. Asset picker — curated groups + manual ticker entry
2. Date range slider — 2000-01-01 to today
3. Load Data button — fetches & caches OHLCV; gates everything below
4. Strategy toggles — auto-discovered from strategies/, disabled until data loaded
5. Initial capital input — disabled until data loaded
6. Run Backtest button — disabled until data loaded
"""

from datetime import date

import pandas as pd
import streamlit as st

from data.universe import (
    AU_BENCHMARK_TICKER,
    AU_CURATED_UNIVERSE,
    BENCHMARK_TICKER,
    CURATED_UNIVERSE,
)
from strategies.base import StrategyCategory
from strategies.registry import discover_strategies

_CATEGORY_LABELS = {
    StrategyCategory.PASSIVE: "Passive / Traditional",
    StrategyCategory.QUANTITATIVE: "Quantitative / Active",
    StrategyCategory.ML: "ML / Advanced",
}

# Badge HTML for each category — rendered via unsafe_allow_html in markdown
_CATEGORY_BADGE_HTML = {
    StrategyCategory.PASSIVE: (
        "<span style='background:#1a2a3c;color:#63b3ed;padding:1px 6px;"
        "border-radius:3px;font-size:0.65em;font-weight:700;letter-spacing:0.8px;'>"
        "PASSIVE</span>"
    ),
    StrategyCategory.QUANTITATIVE: (
        "<span style='background:#2a1e00;color:#f6ad55;padding:1px 6px;"
        "border-radius:3px;font-size:0.65em;font-weight:700;letter-spacing:0.8px;'>"
        "QUANT</span>"
    ),
    StrategyCategory.ML: (
        "<span style='background:#221535;color:#b794f4;padding:1px 6px;"
        "border-radius:3px;font-size:0.65em;font-weight:700;letter-spacing:0.8px;'>"
        "ML</span>"
    ),
}

_BENCHMARK_LABELS: dict[str, str] = {
    BENCHMARK_TICKER: "S&P 500 (SPY)",
    AU_BENCHMARK_TICKER: "S&P/ASX 200 (STW.AX)",
}

# Session state keys
_PRICES_KEY = "prices"
_LOADED_CONFIG_KEY = "loaded_config"


def _data_is_loaded() -> bool:
    return st.session_state.get(_PRICES_KEY) is not None


def _config_is_stale(
    tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp
) -> bool:
    """Return True if the loaded data no longer matches the current sidebar selections."""
    if not _data_is_loaded():
        return False
    cfg = st.session_state.get(_LOADED_CONFIG_KEY, {})
    return (
        cfg.get("tickers") != tickers
        or cfg.get("start_date") != start_date
        or cfg.get("end_date") != end_date
    )


def render_portfolio_sidebar() -> None:
    """Render a read-only portfolio workbench sidebar for non-home pages.

    Shows: current data/backtest status, top performer, and a hint to go to
    the home page to rerun.  Calls ``st.sidebar.*`` directly — no return value.
    """
    st.sidebar.markdown(
        """
        <div style='padding: 0.6rem 0 0.9rem 0; border-bottom: 1px solid #252836;
                    margin-bottom: 0.25rem;'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #4ecdc4;
                        letter-spacing: -0.3px; font-family: "DM Sans", sans-serif;'>
                ⚡ Quant Strategy Lab
            </div>
            <div style='font-size: 0.62rem; color: #636b78; letter-spacing: 2.5px;
                        font-weight: 600; margin-top: 4px; font-family: "DM Sans", sans-serif;'>
                BACKTEST &nbsp;·&nbsp; COMPARE &nbsp;·&nbsp; ALLOCATE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Portfolio Status ──────────────────────────────────────────
    st.sidebar.header("Portfolio Status")

    cfg = st.session_state.get(_LOADED_CONFIG_KEY)
    if cfg:
        tickers: list[str] = cfg.get("tickers", [])
        start: pd.Timestamp = cfg.get("start_date")
        end: pd.Timestamp = cfg.get("end_date")
        ticker_str = "  ".join(tickers) if len(tickers) <= 8 else "  ".join(tickers[:8]) + f"  +{len(tickers)-8}"
        st.sidebar.success(f"✓ {len(tickers)} asset(s) loaded")
        st.sidebar.markdown(
            f"<p style='font-size:0.75rem;color:#a0a8b8;margin:0;font-family:\"DM Sans\",sans-serif;'>"
            f"{ticker_str}</p>",
            unsafe_allow_html=True,
        )
        if start and end:
            st.sidebar.caption(
                f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
            )
    else:
        st.sidebar.info("No data loaded yet.")

    # ── Last Backtest ─────────────────────────────────────────────
    st.sidebar.header("Last Backtest")

    results_state = st.session_state.get("backtest_results")
    if results_state:
        strategy_results = results_state.get("strategy_results", [])
        benchmark = results_state.get("benchmark")
        initial_capital = results_state.get("initial_capital", 0.0)

        st.sidebar.markdown(
            f"<p style='font-size:0.80rem;color:#c4cad6;margin:0 0 0.4rem 0;"
            f"font-family:\"DM Sans\",sans-serif;'>"
            f"{len(strategy_results)} strateg{'y' if len(strategy_results) == 1 else 'ies'} run"
            f"<br>${initial_capital:,.0f} initial capital</p>",
            unsafe_allow_html=True,
        )

        # Top performer by total return
        if strategy_results:
            best = max(strategy_results, key=lambda r: r.metrics.get("total_return", float("-inf")))
            best_ret = best.metrics.get("total_return", float("nan"))
            sign = "+" if best_ret >= 0 else ""
            st.sidebar.markdown(
                "<p style='font-size:0.70rem;color:#636b78;font-weight:600;"
                "text-transform:uppercase;letter-spacing:1.2px;margin:0.5rem 0 0.1rem 0;"
                "font-family:\"DM Sans\",sans-serif;'>Top Performer</p>",
                unsafe_allow_html=True,
            )
            st.sidebar.markdown(
                f"<p style='font-size:0.82rem;color:#4ecdc4;margin:0;"
                f"font-family:\"JetBrains Mono\",monospace;font-weight:600;'>"
                f"{best.strategy_name}</p>"
                f"<p style='font-size:0.80rem;color:#68d391;margin:0;"
                f"font-family:\"JetBrains Mono\",monospace;'>{sign}{best_ret:.1%}</p>",
                unsafe_allow_html=True,
            )

        # Benchmark reference
        if benchmark:
            bench_ret = benchmark.metrics.get("total_return", float("nan"))
            b_sign = "+" if bench_ret >= 0 else ""
            st.sidebar.markdown(
                f"<p style='font-size:0.75rem;color:#636b78;margin:0.4rem 0 0 0;"
                f"font-family:\"DM Sans\",sans-serif;'>"
                f"Benchmark ({benchmark.strategy_name}): "
                f"<span style='font-family:\"JetBrains Mono\",monospace;color:#94a3b8;'>"
                f"{b_sign}{bench_ret:.1%}</span></p>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("No backtest run yet.")

    # ── Navigation hint ───────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.caption("To load data or rerun the backtest, return to the **home page**.")


def render_sidebar() -> dict:
    """Render the full sidebar and return user selections.

    Returns
    -------
    dict
        Keys:
        - tickers : list[str]
        - start_date : pd.Timestamp
        - end_date : pd.Timestamp
        - selected_strategy_ids : list[str]
        - initial_capital : float
        - benchmark_ticker : str
        - load_data : bool  — True on the frame the Load Data button is clicked
        - run : bool        — True on the frame the Run Backtest button is clicked
    """
    st.sidebar.markdown(
        """
        <div style='padding: 0.6rem 0 0.9rem 0; border-bottom: 1px solid #252836;
                    margin-bottom: 0.25rem;'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #4ecdc4;
                        letter-spacing: -0.3px; font-family: "DM Sans", sans-serif;'>
                ⚡ Quant Strategy Lab
            </div>
            <div style='font-size: 0.62rem; color: #636b78; letter-spacing: 2.5px;
                        font-weight: 600; margin-top: 4px; font-family: "DM Sans", sans-serif;'>
                BACKTEST &nbsp;·&nbsp; COMPARE &nbsp;·&nbsp; ALLOCATE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 0. Market Selector ───────────────────────────────────────
    st.sidebar.header("Market")

    MARKET_US = "🇺🇸 US Market"
    MARKET_AU = "🇦🇺 Australian Market"
    MARKET_BOTH = "🌐 Both Markets"

    market = st.sidebar.radio(
        "Select market",
        options=[MARKET_US, MARKET_AU, MARKET_BOTH],
        label_visibility="collapsed",
    )

    if market == MARKET_US:
        active_universe = CURATED_UNIVERSE
        benchmark_ticker = BENCHMARK_TICKER
    elif market == MARKET_AU:
        active_universe = AU_CURATED_UNIVERSE
        benchmark_ticker = AU_BENCHMARK_TICKER
    else:
        active_universe = {**CURATED_UNIVERSE, **AU_CURATED_UNIVERSE}
        benchmark_ticker = None  # resolved below via selectbox

    # ── Benchmark (shown only for Both Markets) ──────────────────
    if market == MARKET_BOTH:
        st.sidebar.header("Benchmark")
        benchmark_ticker = st.sidebar.selectbox(
            "Benchmark index",
            options=[BENCHMARK_TICKER, AU_BENCHMARK_TICKER],
            format_func=lambda t: _BENCHMARK_LABELS.get(t, t),
        )
    else:
        # Display the auto-selected benchmark as an info caption
        st.sidebar.caption(f"Benchmark: {_BENCHMARK_LABELS.get(benchmark_ticker, benchmark_ticker)}")

    # ── 1. Asset Picker ──────────────────────────────────────────
    st.sidebar.header("Assets")

    group_choice = st.sidebar.selectbox(
        "Quick-add group",
        options=["(none)"] + list(active_universe.keys()),
    )

    default_tickers: list[str] = []
    if group_choice != "(none)":
        default_tickers = active_universe[group_choice]

    manual_input = st.sidebar.text_input(
        "Add tickers (comma-separated)",
        placeholder="e.g. BHP.AX, CBA.AX  or  AAPL, MSFT",
    )
    manual_tickers = [t.strip().upper() for t in manual_input.split(",") if t.strip()]

    tickers = list(dict.fromkeys(default_tickers + manual_tickers))

    if tickers:
        st.sidebar.caption(f"Selected: {', '.join(tickers)}")
    else:
        st.sidebar.warning("Select at least one asset.")

    # ── 2. Date Range ────────────────────────────────────────────
    st.sidebar.header("Date Range")

    start_min = date(2000, 1, 1)
    end_max = date.today()

    date_range = st.sidebar.slider(
        "Backtest window",
        min_value=start_min,
        max_value=end_max,
        value=(date(2015, 1, 1), end_max),
        format="YYYY-MM-DD",
    )
    start_date = pd.Timestamp(date_range[0], tz="UTC")
    end_date = pd.Timestamp(date_range[1], tz="UTC")

    # ── 3. Load Data Button ──────────────────────────────────────
    st.sidebar.header("Data")

    stale = _config_is_stale(tickers, start_date, end_date)
    data_ready = _data_is_loaded() and not stale

    if stale:
        st.sidebar.warning("Selections changed — reload data.")
    elif data_ready:
        cfg = st.session_state[_LOADED_CONFIG_KEY]
        prices_df: pd.DataFrame = st.session_state[_PRICES_KEY]
        st.sidebar.success(
            f"✓ {len(cfg['tickers'])} asset(s) loaded "
            f"({len(prices_df):,} rows)"
        )

    load_data = st.sidebar.button(
        "Reload Data" if data_ready else "Load Data",
        disabled=not tickers,
        use_container_width=True,
        help="Fetch OHLCV data for the selected assets.",
    )

    # ── 4. Strategy Toggles (disabled until data loaded) ─────────
    st.sidebar.header("Strategies")

    all_strategies = discover_strategies()
    selected_strategy_ids: list[str] = []

    if not all_strategies:
        st.sidebar.info("No strategies found. Add files to strategies/ to get started.")
    else:
        if not data_ready:
            st.sidebar.caption("Load data first to enable strategy selection.")

        for category in StrategyCategory:
            group = [s for s in all_strategies if s.category == category]
            if not group:
                continue

            badge = _CATEGORY_BADGE_HTML[category]
            st.sidebar.markdown(
                f"<p style='font-size:0.75rem;font-weight:600;margin:0.8rem 0 0.3rem 0;"
                f"color:#a0a8b8;letter-spacing:0.2px;'>{_CATEGORY_LABELS[category]} {badge}</p>",
                unsafe_allow_html=True,
            )
            for strategy in group:
                checked = st.sidebar.checkbox(
                    strategy.name,
                    value=(category == StrategyCategory.PASSIVE and data_ready),
                    disabled=not data_ready,
                    help=strategy.description,
                    key=f"strategy_{strategy.strategy_id}",
                )
                if checked:
                    selected_strategy_ids.append(strategy.strategy_id)

    # ── 5. Capital ───────────────────────────────────────────────
    st.sidebar.header("Capital")
    initial_capital = st.sidebar.number_input(
        "Initial capital ($)",
        min_value=1_000,
        max_value=100_000_000,
        value=100_000,
        step=10_000,
        disabled=not data_ready,
    )

    # ── 6. Run Backtest Button ───────────────────────────────────
    st.sidebar.markdown("---")
    run = st.sidebar.button(
        "Run Backtest",
        type="primary",
        disabled=not data_ready or not selected_strategy_ids,
        use_container_width=True,
        help=(
            "Load data and select at least one strategy first."
            if not data_ready
            else "Run the walk-forward backtest."
        ),
    )

    return {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "selected_strategy_ids": selected_strategy_ids,
        "initial_capital": float(initial_capital),
        "benchmark_ticker": benchmark_ticker,
        "load_data": load_data,
        "run": run,
    }
