"""Sidebar UI component.

Renders (in order):
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

from data.universe import CURATED_UNIVERSE
from strategies.base import StrategyCategory
from strategies.registry import discover_strategies

_CATEGORY_LABELS = {
    StrategyCategory.PASSIVE: "Passive / Traditional",
    StrategyCategory.QUANTITATIVE: "Quantitative / Active",
    StrategyCategory.ML: "ML / Advanced",
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
        - load_data : bool  — True on the frame the Load Data button is clicked
        - run : bool        — True on the frame the Run Backtest button is clicked
    """
    st.sidebar.title("QuantScope")
    st.sidebar.caption("Strategy Racing Dashboard")

    # ── 1. Asset Picker ──────────────────────────────────────────
    st.sidebar.header("Assets")

    group_choice = st.sidebar.selectbox(
        "Quick-add group",
        options=["(none)"] + list(CURATED_UNIVERSE.keys()),
    )

    default_tickers: list[str] = []
    if group_choice != "(none)":
        default_tickers = CURATED_UNIVERSE[group_choice]

    manual_input = st.sidebar.text_input(
        "Add tickers (comma-separated)",
        placeholder="e.g. AAPL, MSFT, ASML",
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

            st.sidebar.subheader(_CATEGORY_LABELS[category])
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
        "load_data": load_data,
        "run": run,
    }
