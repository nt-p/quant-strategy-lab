"""Backtest results UI components.

Four panels rendered after "Run Backtest":
1. Equity curve chart — all strategies + benchmark
2. Metrics comparison table — colour-coded by rank
3. Drawdown chart — underwater equity curves
4. Today's Allocation — live weights for each strategy as of most recent data
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.backtest import BacktestResult
from strategies.base import StrategyBase

from .colors import BENCHMARK_COLOR, assign_colors
from .equity_race import render_equity_race


# ── Public renderers ──────────────────────────────────────────────────────────

def render_results(
    strategy_results: list[BacktestResult],
    benchmark: BacktestResult,
    initial_capital: float,
) -> None:
    """Render the full results panel: equity chart, metrics table, drawdown chart.

    Parameters
    ----------
    strategy_results : list[BacktestResult]
        One result per enabled strategy (excludes benchmark).
    benchmark : BacktestResult
        SPY buy-and-hold benchmark.
    initial_capital : float
        Starting capital (for y-axis label).
    """
    all_results = strategy_results + [benchmark]
    colors = assign_colors(all_results)

    st.subheader("Equity Curves")
    render_equity_chart(all_results, colors, initial_capital)

    st.subheader("Strategy Comparison")
    render_metrics_table(all_results, benchmark, colors)

    st.subheader("Drawdown")
    render_drawdown_chart(all_results, colors)

    st.divider()
    st.subheader("Equity Race")
    st.caption(
        "Animated race — strategies compete frame by frame through time. "
        "Use the **Play** button or drag the time slider to scrub."
    )
    render_equity_race(strategy_results, benchmark, initial_capital)


def render_equity_chart(
    results: list[BacktestResult],
    colors: dict[str, str],
    initial_capital: float,
) -> None:
    """Multi-line Plotly equity chart.

    Parameters
    ----------
    results : list[BacktestResult]
        All results including benchmark.
    colors : dict[str, str]
        {strategy_id: hex_color}
    initial_capital : float
        Used for the y-axis title.
    """
    # Align all series to a common date index (inner join)
    date_sets = [set(r.dates) for r in results]
    common_dates = sorted(date_sets[0].intersection(*date_sets[1:]))

    fig = go.Figure()

    for r in results:
        date_to_equity = dict(zip(r.dates, r.equity))
        y = [date_to_equity.get(d, float("nan")) for d in common_dates]

        is_benchmark = r.category == "benchmark"
        color = colors[r.strategy_id]

        fig.add_trace(
            go.Scatter(
                x=common_dates,
                y=y,
                mode="lines",
                name=r.strategy_name,
                line=dict(
                    color=color,
                    width=1.5 if not is_benchmark else 1.5,
                    dash="dot" if is_benchmark else "solid",
                ),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"{r.strategy_name}: $%{{y:,.0f}}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=f"Portfolio Value ($, start = ${initial_capital:,.0f})",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_metrics_table(
    results: list[BacktestResult],
    benchmark: BacktestResult,
    colors: dict[str, str],
) -> None:
    """Colour-coded metrics comparison table.

    Green = best in column, Red = worst.  Benchmark row is always last.

    Parameters
    ----------
    results : list[BacktestResult]
        All results (strategies + benchmark).
    benchmark : BacktestResult
        Used for column ordering (benchmark always last).
    colors : dict[str, str]
        {strategy_id: hex_color}  (unused here but kept for API consistency)
    """
    _METRIC_COLS = {
        "total_return": ("Total Return", "{:+.1%}", True),
        "cagr": ("CAGR", "{:+.1%}", True),
        "ann_vol": ("Ann. Vol", "{:.1%}", False),
        "sharpe": ("Sharpe", "{:.2f}", True),
        "sortino": ("Sortino", "{:.2f}", True),
        "max_drawdown": ("Max DD", "{:.1%}", True),   # higher (less negative) = better
        "max_drawdown_duration": ("DD Days", "{:.0f}", False),
        "calmar": ("Calmar", "{:.2f}", True),
        "win_rate": ("Win Rate", "{:.1%}", True),
        "avg_monthly_turnover": ("Avg Mo. TO", "{:.1%}", False),
        "excess_return": ("Excess Ret", "{:+.1%}", True),
        "information_ratio": ("Info Ratio", "{:.2f}", True),
    }

    rows = []
    for r in results:
        row: dict[str, object] = {"Strategy": r.strategy_name}
        for key, (label, fmt, _) in _METRIC_COLS.items():
            val = r.metrics.get(key, float("nan"))
            try:
                row[label] = float(val)
            except (TypeError, ValueError):
                row[label] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")
    col_labels = [label for _, (label, _, _) in _METRIC_COLS.items()]

    # Build format dict for styler
    fmt_map = {label: fmt for _, (label, fmt, _) in _METRIC_COLS.items()}

    # Gradient: green = better, red = worse
    # For max_drawdown and dd_days, lower is worse (green for max, red for min)
    higher_is_better_cols = [
        label for _, (label, _, higher_is_better) in _METRIC_COLS.items() if higher_is_better
    ]
    lower_is_better_cols = [
        label for _, (label, _, higher_is_better) in _METRIC_COLS.items() if not higher_is_better
    ]

    styler = df.style.format(fmt_map, na_rep="—")

    for col in higher_is_better_cols:
        if col in df.columns:
            styler = styler.background_gradient(
                subset=[col], cmap="RdYlGn", axis=0
            )

    for col in lower_is_better_cols:
        if col in df.columns:
            styler = styler.background_gradient(
                subset=[col], cmap="RdYlGn_r", axis=0
            )

    st.dataframe(styler, use_container_width=True)


def render_drawdown_chart(
    results: list[BacktestResult],
    colors: dict[str, str],
) -> None:
    """Underwater (drawdown) chart for all strategies.

    Parameters
    ----------
    results : list[BacktestResult]
        All results including benchmark.
    colors : dict[str, str]
        {strategy_id: hex_color}
    """
    # Common date index
    date_sets = [set(r.dates) for r in results]
    common_dates = sorted(date_sets[0].intersection(*date_sets[1:]))

    fig = go.Figure()

    for r in results:
        date_to_dd = dict(zip(r.dates, r.drawdown))
        y = [date_to_dd.get(d, 0.0) for d in common_dates]

        is_benchmark = r.category == "benchmark"
        color = colors[r.strategy_id]

        fig.add_trace(
            go.Scatter(
                x=common_dates,
                y=[v * 100 for v in y],   # express as %
                mode="lines",
                name=r.strategy_name,
                fill="tozeroy",
                fillcolor=_hex_with_alpha(color, 0.15),
                line=dict(
                    color=color,
                    width=1.0,
                    dash="dot" if is_benchmark else "solid",
                ),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"{r.strategy_name}: %{{y:.1f}}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=320,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_todays_allocation(
    strategy_results: list[BacktestResult],
    strategies: list[StrategyBase],
    prices: pd.DataFrame,
    universe: list[str],
    initial_capital: float,
) -> None:
    """Render the 'Today's Allocation' section below the backtest charts.

    For each enabled strategy, calls get_weights() with the most recent date
    in the price data, then shows:
    - A table (Ticker, Weight %, Dollar Amount) + donut chart per strategy
    - A combined grouped bar chart comparing all strategies side-by-side

    Parameters
    ----------
    strategy_results : list[BacktestResult]
        Results from the backtest run — used to match strategy display names and
        colors to the strategy objects.
    strategies : list[StrategyBase]
        Live strategy instances to call get_weights() on.
    prices : pd.DataFrame
        Full OHLCV history in long format.  The most recent date is used as
        current_date for get_weights().
    universe : list[str]
        Tickers available in the loaded dataset.
    initial_capital : float
        Used to compute per-ticker dollar amounts.
    """
    current_date: pd.Timestamp = prices["date"].max()
    strategy_map = {s.strategy_id: s for s in strategies}

    # Assign colors consistent with the equity chart
    colors = assign_colors(strategy_results)

    # ── Compute weights for every strategy ────────────────────────────────────
    allocation_data: list[tuple[BacktestResult, dict[str, float]]] = []

    for result in strategy_results:
        strategy = strategy_map.get(result.strategy_id)
        if strategy is None:
            continue
        try:
            raw_weights = strategy.get_weights(prices, current_date, universe)
            # Normalise to sum exactly to 1.0
            total = sum(raw_weights.values())
            if total > 1e-9:
                weights = {t: w / total for t, w in raw_weights.items() if w > 1e-9}
            else:
                weights = {}
        except Exception as exc:
            st.warning(f"{result.strategy_name}: could not compute today's weights — {exc}")
            continue

        if weights:
            allocation_data.append((result, weights))

    if not allocation_data:
        st.info("No allocation data available for the selected strategies.")
        return

    # ── Section header ─────────────────────────────────────────────────────────
    st.subheader("Today's Allocation")
    st.caption(
        f"What each strategy would hold today · as of **{current_date.strftime('%Y-%m-%d')}** "
        f"· based on ${initial_capital:,.0f} initial capital"
    )

    # ── Per-strategy: table + donut ────────────────────────────────────────────
    for result, weights in allocation_data:
        color = colors.get(result.strategy_id, "#888888")
        st.markdown(f"#### {result.strategy_name}")

        # Build table rows sorted by weight descending
        rows = [
            {
                "Ticker": ticker,
                "Weight": weight,
                "Dollar Amount": weight * initial_capital,
            }
            for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        ]
        df_weights = pd.DataFrame(rows)

        col_table, col_chart = st.columns([1, 1])

        with col_table:
            styled = (
                df_weights.style
                .format({"Weight": "{:.1%}", "Dollar Amount": "${:,.0f}"})
                .hide(axis="index")
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

        with col_chart:
            fig = go.Figure(
                go.Pie(
                    labels=df_weights["Ticker"].tolist(),
                    values=df_weights["Weight"].tolist(),
                    hole=0.45,
                    textinfo="label+percent",
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Weight: %{percent}<br>"
                        f"Value: $%{{value:.1%}}"
                        "<extra></extra>"
                    ),
                    marker=dict(
                        colors=_pie_colors(len(df_weights), color),
                        line=dict(color="#1a1a2e", width=1),
                    ),
                )
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Combined grouped bar chart ─────────────────────────────────────────────
    st.markdown("#### What each strategy would hold today")
    st.caption("Grouped by ticker — each bar is one strategy's recommended weight")

    # Collect all tickers across all strategies, sorted by avg weight desc
    all_tickers_weights: dict[str, list[float]] = {}
    for _result, weights in allocation_data:
        for ticker, w in weights.items():
            all_tickers_weights.setdefault(ticker, []).append(w)

    sorted_tickers = sorted(
        all_tickers_weights.keys(),
        key=lambda t: sum(all_tickers_weights[t]) / len(all_tickers_weights[t]),
        reverse=True,
    )

    fig_bar = go.Figure()
    for result, weights in allocation_data:
        color = colors.get(result.strategy_id, "#888888")
        y_vals = [weights.get(t, 0.0) * 100 for t in sorted_tickers]
        fig_bar.add_trace(
            go.Bar(
                name=result.strategy_name,
                x=sorted_tickers,
                y=y_vals,
                marker_color=color,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{result.strategy_name}: %{{y:.1f}}%"
                    "<extra></extra>"
                ),
            )
        )

    fig_bar.update_layout(
        barmode="group",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_with_alpha(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' + alpha to 'rgba(r,g,b,a)' string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _pie_colors(n: int, base_color: str) -> list[str]:
    """Generate n colours for a pie chart, shading from base_color toward white.

    Parameters
    ----------
    n : int
        Number of slices.
    base_color : str
        Hex color for the first (largest) slice.

    Returns
    -------
    list[str]
        List of n hex colors.
    """
    h = base_color.lstrip("#")
    br, bg, bb = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    result = []
    for i in range(n):
        # Blend toward #2a2a4a (dark background) as slices get smaller
        t = i / max(n - 1, 1)
        r = int(br + (42 - br) * t)
        g = int(bg + (42 - bg) * t)
        b = int(bb + (74 - bb) * t)
        result.append(f"#{r:02x}{g:02x}{b:02x}")
    return result
