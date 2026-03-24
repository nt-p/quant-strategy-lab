"""Page 1: Strategy Lab — run walk-forward backtests and compare strategies."""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Strategy Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from strategies.registry import discover_strategies  # noqa: E402
from ui.controller import _PRICES_KEY, _RESULTS_KEY, handle_sidebar_actions  # noqa: E402
from ui.results import render_results, render_todays_allocation  # noqa: E402
from ui.sidebar import require_portfolio  # noqa: E402
from ui.theme import inject_css  # noqa: E402

inject_css()

st.markdown(
    """
    <div style='padding: 0.5rem 0 0.15rem 0;'>
        <h1 style='font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;
                   background: linear-gradient(90deg, #4ecdc4 0%, #81a4e8 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-family: "DM Sans", sans-serif;'>
            Strategy Lab
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

# ── Guard: must have a portfolio loaded from Portfolio Hub ────────────────────
require_portfolio()

# ── Sidebar: strategy toggles + Run Backtest ──────────────────────────────────
if handle_sidebar_actions():
    st.stop()

# ── Read data from session state (set by Portfolio Hub) ───────────────────────
prices: pd.DataFrame | None = st.session_state.get(_PRICES_KEY)
portfolio = st.session_state.get("portfolio")

cfg = st.session_state.get("loaded_config", {})
tickers: list[str] = cfg.get("tickers", list((portfolio or {}).get("holdings", {}).keys()))
universe = tickers

start_date: pd.Timestamp = portfolio["start_date"]
end_date: pd.Timestamp   = portfolio["end_date"]

# ── Render results if a backtest has been run ─────────────────────────────────
results_state = st.session_state.get(_RESULTS_KEY)

if not results_state:
    st.info(
        "Select strategies in the sidebar and click **Run Backtest** to compare them "
        "against your portfolio's assets.",
        icon="⚡",
    )
    st.markdown(
        """
### How it works
1. Your portfolio assets and date range (from Portfolio Hub) are used as the backtest universe
2. Toggle strategies by category in the sidebar
3. Set initial capital and click **Run Backtest**
4. Walk-forward equity curves, metrics, and drawdowns appear here
        """
    )
    st.stop()

strategy_results = results_state["strategy_results"]
benchmark        = results_state["benchmark"]
initial_capital  = results_state["initial_capital"]

render_results(
    strategy_results=strategy_results,
    benchmark=benchmark,
    initial_capital=initial_capital,
)

st.divider()

all_strategies = {s.strategy_id: s for s in discover_strategies()}
active_strategies = [
    all_strategies[r.strategy_id]
    for r in strategy_results
    if r.strategy_id in all_strategies
]

if prices is not None:
    render_todays_allocation(
        strategy_results=strategy_results,
        strategies=active_strategies,
        prices=prices,
        universe=universe,
        initial_capital=initial_capital,
    )
