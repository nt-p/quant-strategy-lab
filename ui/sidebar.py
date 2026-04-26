"""Sidebar UI components.

Three distinct sidebar modes:

1. render_portfolio_builder_sidebar() — Portfolio Hub (app.py)
   Market, asset picker, per-ticker weights, date range, Run Portfolio button.

2. render_strategy_sidebar() — Strategy Lab (pages/1_Strategy_Lab.py)
   Read-only portfolio status, strategy toggles, capital, Run Backtest button.

3. render_portfolio_sidebar() — All other pages
   Read-only portfolio + backtest status.

4. require_portfolio() — Guard used at the top of pages that need portfolio data.
"""

from datetime import date

import pandas as pd
import streamlit as st

from data.universe import (
    AU_BENCHMARK_TICKER,
    AU_CURATED_UNIVERSE,
    BENCHMARK_TICKER,
    CURATED_UNIVERSE,
    load_ticker_search_options,
)
from strategies.base import StrategyCategory
from strategies.registry import discover_strategies

# ── Portfolio templates ────────────────────────────────────────────────────────
# Each value maps ticker → weight (%). Must sum to 100.
PORTFOLIO_TEMPLATES: dict[str, dict[str, float]] = {
    "Classic 60/40": {
        "SPY": 60.0, "AGG": 40.0,
    },
    "All Weather (Dalio)": {
        "SPY": 30.0, "TLT": 40.0, "IEF": 15.0, "GLD": 7.5, "DBC": 7.5,
    },
    "Permanent Portfolio (Browne)": {
        "SPY": 25.0, "TLT": 25.0, "GLD": 25.0, "SHY": 25.0,
    },
    "Global 60/40": {
        "SPY": 36.0, "EFA": 24.0, "TLT": 28.0, "GLD": 12.0,
    },
    "US Tech Tilt": {
        "QQQ": 40.0, "SPY": 30.0, "TLT": 15.0, "GLD": 15.0,
    },
    "Endowment Style": {
        "SPY": 35.0, "EFA": 15.0, "VNQ": 10.0, "GLD": 10.0, "TLT": 20.0, "HYG": 10.0,
    },
    "ASX Core": {
        "VAS.AX": 50.0, "VGS.AX": 30.0, "IAF.AX": 10.0, "VAP.AX": 10.0,
    },
    "Vanguard Diversified Growth (ASX)": {
        "VGS.AX": 52.0, "VAS.AX": 18.0, "VGE.AX": 10.0,
        "IAF.AX": 10.0, "VAP.AX": 5.0, "VISM.AX": 5.0,
    },
}

_CATEGORY_LABELS = {
    StrategyCategory.PASSIVE: "Passive / Traditional",
    StrategyCategory.QUANTITATIVE: "Quantitative / Active",
    StrategyCategory.ML: "ML / Advanced",
}

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

def _rebalance_weights(changed_ticker: str, all_tickers: list[str]) -> None:
    """on_change callback: proportionally redistribute weights when one input changes.

    Scales the other tickers so they sum to exactly (100 - curr), then
    distributes any 1-cent rounding residual to the largest of them.
    This guarantees total_w == 100.00 after every change.
    """
    curr = st.session_state[f"hub_weight_{changed_ticker}"]
    prev = st.session_state.get(f"_hub_weight_prev_{changed_ticker}", curr)
    if abs(curr - prev) < 1e-9:
        return

    others = [t for t in all_tickers if t != changed_ticker]
    if not others:
        st.session_state[f"_hub_weight_prev_{changed_ticker}"] = round(curr, 2)
        return

    target_others = max(0.0, 100.0 - curr)
    others_sum = sum(st.session_state.get(f"hub_weight_{t}", 0.0) for t in others)

    if others_sum > 1e-9:
        # Scale proportionally to the exact target
        raw = {
            t: max(0.0, st.session_state.get(f"hub_weight_{t}", 0.0) * target_others / others_sum)
            for t in others
        }
    else:
        share = target_others / len(others)
        raw = {t: share for t in others}

    # Round to 2 dp, then push any residual onto the largest ticker
    rounded: dict[str, float] = {t: round(w, 2) for t, w in raw.items()}
    residual = round(target_others - sum(rounded.values()), 2)
    if abs(residual) >= 0.005:
        largest = max(others, key=lambda t: rounded.get(t, 0.0))
        rounded[largest] = round(rounded[largest] + residual, 2)

    for t, w in rounded.items():
        st.session_state[f"hub_weight_{t}"] = w

    for t in all_tickers:
        st.session_state[f"_hub_weight_prev_{t}"] = round(
            st.session_state.get(f"hub_weight_{t}", 0.0), 2
        )


_BENCHMARK_LABELS: dict[str, str] = {
    BENCHMARK_TICKER: "S&P 500 (SPY)",
    AU_BENCHMARK_TICKER: "S&P/ASX 200 (STW.AX)",
}

# Session state keys
_PRICES_KEY = "prices"
_LOADED_CONFIG_KEY = "loaded_config"


def _brand_header() -> None:
    st.sidebar.markdown(
        """
        <div style='padding: 0.6rem 0 0.9rem 0; border-bottom: 1px solid #252836;
                    margin-bottom: 0.25rem;'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #4ecdc4;
                        letter-spacing: -0.3px; font-family: "DM Sans", sans-serif;'>
                QuantScope
            </div>
            <div style='font-size: 0.62rem; color: #636b78; letter-spacing: 2.5px;
                        font-weight: 600; margin-top: 4px; font-family: "DM Sans", sans-serif;'>
                PORTFOLIO &nbsp;·&nbsp; RESEARCH &nbsp;·&nbsp; TERMINAL
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Theme toggle (shared) ─────────────────────────────────────────────────────

def _render_theme_toggle() -> None:
    """Light / dark mode toggle fixed at the bottom of every sidebar."""
    st.sidebar.markdown("---")
    dark = st.session_state.get("dark_mode", True)
    label = "🌙 Dark mode" if dark else "☀️ Light mode"
    st.sidebar.toggle(label, value=dark, key="dark_mode")


# ── 1. Portfolio Hub sidebar ───────────────────────────────────────────────────

def render_portfolio_builder_sidebar() -> dict:
    """Render the Portfolio Hub sidebar.

    Returns
    -------
    dict with keys:
        tickers, weights, start_date, end_date, benchmark_ticker, run
    """
    _brand_header()

    # ── Market ───────────────────────────────────────────────────────────────
    st.sidebar.header("Market")

    MARKET_US   = "🇺🇸 US Market"
    MARKET_AU   = "🇦🇺 Australian Market"
    MARKET_BOTH = "🌐 Both Markets"

    market = st.sidebar.radio(
        "Select market",
        options=[MARKET_US, MARKET_AU, MARKET_BOTH],
        label_visibility="collapsed",
    )

    if market == MARKET_US:
        active_universe = CURATED_UNIVERSE
        benchmark_ticker: str = BENCHMARK_TICKER
    elif market == MARKET_AU:
        active_universe = AU_CURATED_UNIVERSE
        benchmark_ticker = AU_BENCHMARK_TICKER
    else:
        active_universe = {**CURATED_UNIVERSE, **AU_CURATED_UNIVERSE}
        benchmark_ticker = BENCHMARK_TICKER  # resolved below

    if market == MARKET_BOTH:
        st.sidebar.header("Benchmark")
        benchmark_ticker = st.sidebar.selectbox(
            "Benchmark index",
            options=[BENCHMARK_TICKER, AU_BENCHMARK_TICKER],
            format_func=lambda t: _BENCHMARK_LABELS.get(t, t),
        )
    else:
        st.sidebar.caption(f"Benchmark: {_BENCHMARK_LABELS.get(benchmark_ticker, benchmark_ticker)}")

    # ── Asset Picker ─────────────────────────────────────────────────────────
    st.sidebar.header("Holdings")

    # Template picker
    template_names = ["(custom)"] + list(PORTFOLIO_TEMPLATES.keys())
    template_choice: str = st.sidebar.selectbox(
        "Portfolio template",
        options=template_names,
        key="_hub_template_selection",
        help="Pre-fill holdings and weights from a known allocation strategy.",
    )

    # Apply template when selection changes — update multiselect + weights
    # before the multiselect widget is rendered so it picks up the new default.
    prev_template = st.session_state.get("_hub_template_prev", "(custom)")
    if template_choice != prev_template:
        if template_choice != "(custom)":
            tmpl = PORTFOLIO_TEMPLATES[template_choice]
            st.session_state["_hub_ticker_multiselect"] = list(tmpl.keys())
            for ticker, w in tmpl.items():
                st.session_state[f"hub_weight_{ticker}"] = w
                st.session_state[f"_hub_weight_prev_{ticker}"] = w
            st.session_state["_hub_tickers_prev"] = list(tmpl.keys())
        else:
            st.session_state["_hub_ticker_multiselect"] = []
        st.session_state["_hub_template_prev"] = template_choice

    # Determine search market filter
    search_market = (
        "US" if market == MARKET_US else "AU" if market == MARKET_AU else "BOTH"
    )
    search_tickers, label_map = load_ticker_search_options(search_market)

    # Ensure any pre-seeded tickers that aren't in the CSV still appear as options
    current_selection: list[str] = st.session_state.get("_hub_ticker_multiselect", [])
    extra_options = [t for t in current_selection if t not in label_map]
    all_options = search_tickers + extra_options

    selected_tickers: list[str] = st.sidebar.multiselect(
        "Search tickers",
        options=all_options,
        format_func=lambda t: label_map.get(t, t),
        key="_hub_ticker_multiselect",
        placeholder="Type to search by name, ticker, or sector…",
    )

    # Fallback text input for tickers not in the bundled CSV
    extra_input = st.sidebar.text_input(
        "Unlisted tickers",
        placeholder="e.g. BRK-B, 2330.TW, RMD.AX",
        help="Comma-separated tickers not found in the search list above.",
    )
    extra_tickers = [t.strip().upper() for t in extra_input.split(",") if t.strip()]

    tickers = list(dict.fromkeys(selected_tickers + extra_tickers))

    if not tickers:
        st.sidebar.warning("Select at least one asset.")

    # ── Weights ───────────────────────────────────────────────────────────────
    weights: dict[str, float] = {}
    if tickers:
        st.sidebar.markdown(
            "<p style='font-size:0.72rem;color:#636b78;font-weight:600;"
            "text-transform:uppercase;letter-spacing:1.2px;margin:0.6rem 0 0.3rem 0;"
            "font-family:\"DM Sans\",sans-serif;'>Weights (%)</p>",
            unsafe_allow_html=True,
        )
        equal_w = round(100.0 / len(tickers), 2)

        # Reset weights when ticker list changes
        prev_tickers = st.session_state.get("_hub_tickers_prev", [])
        if set(tickers) != set(prev_tickers):
            template_weights = (
                PORTFOLIO_TEMPLATES.get(template_choice, {})
                if template_choice != "(custom)"
                else {}
            )
            # Assign weights; give the last ticker any residual so sum == 100.0
            assigned: list[tuple[str, float]] = []
            for ticker in tickers:
                w = template_weights.get(ticker, equal_w)
                assigned.append((ticker, round(w, 2)))
            residual = round(100.0 - sum(w for _, w in assigned), 2)
            if abs(residual) >= 0.005 and assigned:
                t0, w0 = assigned[-1]
                assigned[-1] = (t0, round(w0 + residual, 2))
            for ticker, w in assigned:
                st.session_state[f"hub_weight_{ticker}"] = w
                st.session_state[f"_hub_weight_prev_{ticker}"] = w
            st.session_state["_hub_tickers_prev"] = list(tickers)

        # Initialise missing keys (first run)
        for ticker in tickers:
            if f"hub_weight_{ticker}" not in st.session_state:
                st.session_state[f"hub_weight_{ticker}"] = equal_w
            if f"_hub_weight_prev_{ticker}" not in st.session_state:
                st.session_state[f"_hub_weight_prev_{ticker}"] = st.session_state[f"hub_weight_{ticker}"]

        # Render number inputs — on_change handles redistribution
        total_w = 0.0
        for ticker in tickers:
            w = st.sidebar.number_input(
                ticker,
                min_value=0.0,
                max_value=100.0,
                step=0.01,
                format="%.2f",
                key=f"hub_weight_{ticker}",
                on_change=_rebalance_weights,
                args=(ticker, tickers),
            )
            weights[ticker] = w
            total_w += w

        # Weight sum indicator
        if abs(total_w - 100.0) < 0.01:
            st.sidebar.success(f"Weights sum to {total_w:.2f}%", icon="✅")
        else:
            remainder = 100.0 - total_w
            color = "#f6ad55" if abs(remainder) < 20 else "#f56565"
            direction = f"+{remainder:.2f}%" if remainder > 0 else f"{remainder:.2f}%"
            st.sidebar.markdown(
                f"<p style='font-size:0.78rem;color:{color};margin:0;"
                f"font-family:\"DM Sans\",sans-serif;'>"
                f"Weights sum to {total_w:.2f}% ({direction} to 100%)</p>",
                unsafe_allow_html=True,
            )

    # ── Date Range ────────────────────────────────────────────────────────────
    st.sidebar.header("Date Range")

    date_range = st.sidebar.slider(
        "Portfolio window",
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        value=(date(2015, 1, 1), date.today()),
        format="YYYY-MM-DD",
    )
    start_date = pd.Timestamp(date_range[0], tz="UTC")
    end_date   = pd.Timestamp(date_range[1], tz="UTC")

    # ── Run Button ────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    run = st.sidebar.button(
        "Run Portfolio",
        type="primary",
        disabled=not tickers,
        use_container_width=True,
        help="Fetch price data and compute portfolio analytics.",
    )

    _render_theme_toggle()

    return {
        "tickers": tickers,
        "weights": weights,
        "start_date": start_date,
        "end_date": end_date,
        "benchmark_ticker": benchmark_ticker,
        "run": run,
    }


# ── 2. Strategy Lab sidebar ────────────────────────────────────────────────────

def render_strategy_sidebar() -> dict:
    """Render the Strategy Lab sidebar.

    Shows read-only portfolio status, then strategy toggles + Run Backtest.

    Returns
    -------
    dict with keys:
        selected_strategy_ids, initial_capital, benchmark_ticker, run
    """
    _brand_header()

    # ── Portfolio Status (read-only) ──────────────────────────────────────────
    portfolio = st.session_state.get("portfolio")
    if portfolio:
        holdings: dict[str, float] = portfolio.get("holdings", {})
        start: pd.Timestamp = portfolio.get("start_date")
        end: pd.Timestamp   = portfolio.get("end_date")
        tickers = list(holdings.keys())
        ticker_str = "  ".join(tickers) if len(tickers) <= 6 else "  ".join(tickers[:6]) + f"  +{len(tickers)-6}"

        st.sidebar.markdown(
            "<p style='font-size:0.70rem;color:#636b78;font-weight:600;"
            "text-transform:uppercase;letter-spacing:1.2px;margin:0.4rem 0 0.2rem 0;"
            "font-family:\"DM Sans\",sans-serif;'>Active Portfolio</p>",
            unsafe_allow_html=True,
        )
        st.sidebar.success(f"✓ {len(tickers)} holding(s)")
        st.sidebar.markdown(
            f"<p style='font-size:0.72rem;color:#a0a8b8;margin:0 0 0.1rem 0;"
            f"font-family:\"DM Sans\",sans-serif;'>{ticker_str}</p>",
            unsafe_allow_html=True,
        )
        if start and end:
            st.sidebar.caption(f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
    else:
        st.sidebar.info("No portfolio loaded.")

    benchmark_ticker: str = (portfolio or {}).get("benchmark_ticker", BENCHMARK_TICKER)

    # ── Strategies ────────────────────────────────────────────────────────────
    st.sidebar.header("Strategies")

    portfolio_loaded = portfolio is not None
    all_strategies = discover_strategies()
    selected_strategy_ids: list[str] = []

    if not all_strategies:
        st.sidebar.info("No strategies found.")
    else:
        if not portfolio_loaded:
            st.sidebar.caption("Load a portfolio first to enable strategies.")

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
                    value=(category == StrategyCategory.PASSIVE and portfolio_loaded),
                    disabled=not portfolio_loaded,
                    help=strategy.description,
                    key=f"strat_{strategy.strategy_id}",
                )
                if checked:
                    selected_strategy_ids.append(strategy.strategy_id)

    # ── Capital ───────────────────────────────────────────────────────────────
    st.sidebar.header("Capital")
    initial_capital = st.sidebar.number_input(
        "Initial capital ($)",
        min_value=1_000,
        max_value=100_000_000,
        value=100_000,
        step=10_000,
        disabled=not portfolio_loaded,
    )

    # ── Run Backtest ──────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    run = st.sidebar.button(
        "Run Backtest",
        type="primary",
        disabled=not portfolio_loaded or not selected_strategy_ids,
        use_container_width=True,
        help=(
            "Load a portfolio in Portfolio Hub first."
            if not portfolio_loaded
            else "Run the walk-forward backtest."
        ),
    )

    _render_theme_toggle()

    return {
        "selected_strategy_ids": selected_strategy_ids,
        "initial_capital": float(initial_capital),
        "benchmark_ticker": benchmark_ticker,
        "run": run,
    }


# ── 3. Read-only sidebar (all other pages) ─────────────────────────────────────

def render_portfolio_sidebar() -> None:
    """Render a read-only status sidebar for non-lab pages."""
    _brand_header()

    st.sidebar.header("Portfolio")
    portfolio = st.session_state.get("portfolio")
    if portfolio:
        holdings: dict[str, float] = portfolio.get("holdings", {})
        tickers = list(holdings.keys())
        ticker_str = "  ".join(tickers) if len(tickers) <= 8 else "  ".join(tickers[:8]) + f"  +{len(tickers)-8}"
        start: pd.Timestamp = portfolio.get("start_date")
        end: pd.Timestamp   = portfolio.get("end_date")

        st.sidebar.success(f"✓ {len(tickers)} holding(s) loaded")
        st.sidebar.markdown(
            f"<p style='font-size:0.75rem;color:#a0a8b8;margin:0;"
            f"font-family:\"DM Sans\",sans-serif;'>{ticker_str}</p>",
            unsafe_allow_html=True,
        )
        if start and end:
            st.sidebar.caption(f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
    else:
        st.sidebar.info("No portfolio loaded yet.")

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
        if strategy_results:
            best = max(strategy_results, key=lambda r: r.metrics.get("total_return", float("-inf")))
            best_ret = best.metrics.get("total_return", float("nan"))
            sign = "+" if best_ret >= 0 else ""
            st.sidebar.markdown(
                f"<p style='font-size:0.82rem;color:#4ecdc4;margin:0;"
                f"font-family:\"JetBrains Mono\",monospace;font-weight:600;'>"
                f"{best.strategy_name}</p>"
                f"<p style='font-size:0.80rem;color:#68d391;margin:0;"
                f"font-family:\"JetBrains Mono\",monospace;'>{sign}{best_ret:.1%}</p>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("No backtest run yet.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Return to **Portfolio Hub** to change holdings or **Strategy Lab** to run a backtest.")

    _render_theme_toggle()


# ── 4. Guard ───────────────────────────────────────────────────────────────────

def require_portfolio() -> None:
    """Stop page rendering if no portfolio has been configured and run.

    Call at the top of any page that needs portfolio data.
    Displays an info message with a link back to Portfolio Hub, then st.stop().
    """
    if not st.session_state.get("portfolio"):
        st.info(
            "Configure and run a portfolio in the **Portfolio Hub** first.",
            icon="🏗",
        )
        st.markdown(
            """
### How to get started
1. Go to **Portfolio Hub** in the sidebar navigation
2. Pick a market, select assets, and set weights
3. Set a date range and click **Run Portfolio**
4. Come back here — your portfolio analytics will be ready
            """
        )
        st.stop()


# ── Legacy: kept for backward compatibility with controller.py ─────────────────

def render_sidebar() -> dict:
    """Full interactive sidebar used by the legacy Strategy Lab flow.

    Retained so existing imports in controller.py do not break.
    New code should call render_strategy_sidebar() instead.
    """
    return render_portfolio_builder_sidebar()
