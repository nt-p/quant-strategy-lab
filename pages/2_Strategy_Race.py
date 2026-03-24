"""Page 1: Strategy Race — full-screen animated equity race chart."""

import streamlit as st

st.set_page_config(page_title="Strategy Race", page_icon="🏁", layout="wide")

from ui.equity_race import render_equity_race  # noqa: E402
from ui.theme import inject_css  # noqa: E402

inject_css()
st.markdown(
    "<h2 style='font-size:1.4rem;font-weight:700;letter-spacing:-0.3px;"
    "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;margin-bottom:0.1rem;'>"
    "🏁 Strategy Race</h2>",
    unsafe_allow_html=True,
)

results_state = st.session_state.get("backtest_results")

if results_state:
    strategy_results = results_state["strategy_results"]
    benchmark = results_state["benchmark"]
    initial_capital = results_state["initial_capital"]

    st.caption(
        "Animated race — strategies compete frame by frame through time. "
        "Use the **Play** button or drag the time slider to scrub."
    )
    render_equity_race(strategy_results, benchmark, initial_capital)
else:
    st.info("Run a backtest from the home page to see the equity race chart here.")
    st.markdown(
        """
### How to get here
1. Go to the **home page** (QuantScope in the sidebar)
2. Pick assets, set a date range, and click **Load Data**
3. Toggle some strategies and click **Run Backtest**
4. Come back here — the animated race will be waiting
        """
    )
