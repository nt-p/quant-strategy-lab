"""Page 5: Portfolio Hub — synthesis, construction, and comparison.

Sections
--------
1. Executive Summary    — top-line metric cards across all strategies
2. Risk–Return Map      — CAGR vs Vol scatter, bubble = Sharpe ratio
3. Strategy Leaderboard — sortable ranked table with all metrics
4. Composite Builder    — blend strategies with sliders, see live equity curve
5. Pairwise Returns     — scatter plots of daily returns between strategy pairs
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Portfolio Hub", page_icon="🏗", layout="wide")

from engine.backtest import BacktestResult  # noqa: E402
from ui.colors import BENCHMARK_COLOR, assign_colors  # noqa: E402
from ui.sidebar import render_portfolio_sidebar  # noqa: E402
from ui.theme import AXIS_STYLE, BG, CHART_LAYOUT, FONT, GRID, PLOT_BG, inject_css  # noqa: E402

inject_css()
render_portfolio_sidebar()

st.markdown(
    "<h2 style='font-size:1.4rem;font-weight:700;letter-spacing:-0.3px;"
    "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;margin-bottom:0.1rem;'>"
    "🏗 Portfolio Hub</h2>",
    unsafe_allow_html=True,
)
st.caption(
    "Cross-strategy synthesis — risk–return landscape, composite portfolio construction, "
    "and pairwise return analysis."
)

# ── Guard ─────────────────────────────────────────────────────────────────────

results_state = st.session_state.get("backtest_results")

if not results_state:
    st.info("Run a backtest from the home page to unlock the Portfolio Hub.")
    st.markdown(
        """
### How to get started
1. Go to the **home page** (Quant Strategy Lab in the sidebar)
2. Pick assets and a date range, then click **Load Data**
3. Toggle strategies and click **Run Backtest**
4. Return here for cross-strategy synthesis and portfolio construction
        """
    )
    st.stop()

strategy_results: list[BacktestResult] = results_state["strategy_results"]
benchmark: BacktestResult = results_state["benchmark"]
initial_capital: float = results_state["initial_capital"]

all_results = strategy_results + [benchmark]
colors = assign_colors(all_results)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _section_header(title: str, caption: str | None = None) -> None:
    st.markdown(
        f"<h3 style='font-size:1.05rem;font-weight:600;margin:1.2rem 0 0.15rem 0;"
        f"color:#e8ecf0;letter-spacing:-0.2px;font-family:\"DM Sans\",sans-serif;'>"
        f"{title}</h3>",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


_CATEGORY_COLORS = {
    "passive": "#63b3ed",
    "quantitative": "#f6ad55",
    "ml": "#b794f4",
    "benchmark": BENCHMARK_COLOR,
}


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — Executive Summary
# ════════════════════════════════════════════════════════════════════════════

_section_header("Executive Summary")

_m = {r.strategy_id: r.metrics for r in strategy_results}

if strategy_results:
    best_sharpe_result = max(strategy_results, key=lambda r: r.metrics.get("sharpe", float("-inf")))
    best_cagr_result   = max(strategy_results, key=lambda r: r.metrics.get("cagr", float("-inf")))
    best_dd_result     = max(strategy_results, key=lambda r: r.metrics.get("max_drawdown", float("-inf")))
    best_wr_result     = max(strategy_results, key=lambda r: r.metrics.get("win_rate", float("-inf")))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = best_sharpe_result.metrics.get("sharpe", float("nan"))
        st.metric(
            "Best Sharpe Ratio",
            f"{v:.2f}" if math.isfinite(v) else "—",
            delta=best_sharpe_result.strategy_name,
        )
    with c2:
        v = best_cagr_result.metrics.get("cagr", float("nan"))
        st.metric(
            "Best CAGR",
            f"{v:+.1%}" if math.isfinite(v) else "—",
            delta=best_cagr_result.strategy_name,
        )
    with c3:
        v = best_dd_result.metrics.get("max_drawdown", float("nan"))
        st.metric(
            "Shallowest Max DD",
            f"{v:.1%}" if math.isfinite(v) else "—",
            delta=best_dd_result.strategy_name,
        )
    with c4:
        v = best_wr_result.metrics.get("win_rate", float("nan"))
        st.metric(
            "Highest Win Rate",
            f"{v:.1%}" if math.isfinite(v) else "—",
            delta=best_wr_result.strategy_name,
        )

    # Benchmark reference row
    b_sharpe = benchmark.metrics.get("sharpe", float("nan"))
    b_cagr   = benchmark.metrics.get("cagr", float("nan"))
    b_dd     = benchmark.metrics.get("max_drawdown", float("nan"))
    b_wr     = benchmark.metrics.get("win_rate", float("nan"))
    st.caption(
        f"Benchmark ({benchmark.strategy_name}): "
        f"Sharpe {b_sharpe:.2f}  ·  CAGR {b_cagr:+.1%}  ·  "
        f"Max DD {b_dd:.1%}  ·  Win Rate {b_wr:.1%}"
    )

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Risk–Return Map
# ════════════════════════════════════════════════════════════════════════════

_section_header(
    "Risk–Return Map",
    "Each bubble is one strategy. x = annualised volatility, y = CAGR. "
    "Bubble size = Sharpe ratio (larger = better risk-adjusted return). "
    "Dashed lines are iso-Sharpe curves.",
)

fig_rr = go.Figure()

# Iso-Sharpe reference lines (Sharpe = 0.5, 1.0, 1.5)
_vol_range = np.linspace(0.0, 0.45, 200)
for iso_sharpe, iso_color in [(0.5, "#252836"), (1.0, "#2e3347"), (1.5, "#3a4160")]:
    fig_rr.add_trace(
        go.Scatter(
            x=_vol_range,
            y=_vol_range * iso_sharpe,
            mode="lines",
            line=dict(color=iso_color, width=1, dash="dot"),
            name=f"Sharpe = {iso_sharpe}",
            showlegend=True,
            hoverinfo="skip",
        )
    )

for r in all_results:
    vol  = r.metrics.get("ann_vol", float("nan"))
    cagr = r.metrics.get("cagr", float("nan"))
    srp  = r.metrics.get("sharpe", float("nan"))

    if not (math.isfinite(vol) and math.isfinite(cagr)):
        continue

    bubble_size = max(abs(srp) * 18, 8) if math.isfinite(srp) else 8
    color = colors[r.strategy_id]
    is_benchmark = r.category == "benchmark"

    fig_rr.add_trace(
        go.Scatter(
            x=[vol],
            y=[cagr],
            mode="markers+text",
            name=r.strategy_name,
            text=[r.strategy_name],
            textposition="top center",
            textfont=dict(size=9, color=color, family="DM Sans, sans-serif"),
            marker=dict(
                size=bubble_size,
                color=color,
                opacity=0.85,
                symbol="circle-open" if is_benchmark else "circle",
                line=dict(color=color, width=2),
            ),
            hovertemplate=(
                f"<b>{r.strategy_name}</b><br>"
                f"CAGR: {cagr:+.1%}<br>"
                f"Vol: {vol:.1%}<br>"
                f"Sharpe: {srp:.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

fig_rr.update_layout(
    **CHART_LAYOUT,
    xaxis_title="Annualised Volatility",
    yaxis_title="CAGR",
    xaxis_tickformat=".0%",
    yaxis_tickformat="+.0%",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=10, color=FONT, family="DM Sans, sans-serif"),
        bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=460,
)
fig_rr.update_xaxes(**AXIS_STYLE, range=[0, 0.45])
fig_rr.update_yaxes(**AXIS_STYLE)

st.plotly_chart(fig_rr, use_container_width=True)
st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Strategy Leaderboard
# ════════════════════════════════════════════════════════════════════════════

_section_header(
    "Strategy Leaderboard",
    "All strategies and the benchmark ranked by Sharpe ratio. "
    "Green = best in column, red = worst.",
)

_LEADERBOARD_COLS: dict[str, tuple[str, str, bool]] = {
    "cagr":                 ("CAGR",         "{:+.1%}", True),
    "ann_vol":              ("Ann. Vol",      "{:.1%}",  False),
    "sharpe":               ("Sharpe",        "{:.2f}",  True),
    "sortino":              ("Sortino",       "{:.2f}",  True),
    "max_drawdown":         ("Max DD",        "{:.1%}",  True),
    "max_drawdown_duration":("DD Days",       "{:.0f}",  False),
    "calmar":               ("Calmar",        "{:.2f}",  True),
    "win_rate":             ("Win Rate",      "{:.1%}",  True),
    "excess_return":        ("Excess Ret",    "{:+.1%}", True),
    "information_ratio":    ("Info Ratio",    "{:.2f}",  True),
    "avg_monthly_turnover": ("Avg Mo. TO",    "{:.1%}",  False),
}

_sorted_results = sorted(all_results, key=lambda r: r.metrics.get("sharpe", float("-inf")), reverse=True)

lb_rows = []
for rank, r in enumerate(_sorted_results, start=1):
    row: dict[str, object] = {"#": rank, "Strategy": r.strategy_name, "Category": r.category.title()}
    for key, (label, _, _) in _LEADERBOARD_COLS.items():
        val = r.metrics.get(key, float("nan"))
        try:
            row[label] = float(val)
        except (TypeError, ValueError):
            row[label] = float("nan")
    lb_rows.append(row)

df_lb = pd.DataFrame(lb_rows).set_index("#")

fmt_map = {label: fmt for _, (label, fmt, _) in _LEADERBOARD_COLS.items()}
higher_cols = [label for _, (label, _, hi) in _LEADERBOARD_COLS.items() if hi]
lower_cols  = [label for _, (label, _, hi) in _LEADERBOARD_COLS.items() if not hi]

styler = df_lb.style.format(fmt_map, na_rep="—")
for col in higher_cols:
    if col in df_lb.columns:
        styler = styler.background_gradient(subset=[col], cmap="RdYlGn", axis=0)
for col in lower_cols:
    if col in df_lb.columns:
        styler = styler.background_gradient(subset=[col], cmap="RdYlGn_r", axis=0)

st.dataframe(styler, use_container_width=True)
st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 4 — Composite Portfolio Builder
# ════════════════════════════════════════════════════════════════════════════

_section_header(
    "Composite Portfolio Builder",
    "Blend strategies by adjusting weights below. The equity curve is computed "
    "by combining pre-backtested daily return series — no rebalancing costs modelled. "
    "Remaining weight (to 100%) is treated as cash.",
)

# Align all strategy results + benchmark to a common date index
_date_sets = [set(r.dates) for r in all_results]
_common_dates: list[pd.Timestamp] = sorted(_date_sets[0].intersection(*_date_sets[1:]))

# Build returns DataFrame on common dates
_returns_df = pd.DataFrame(
    {r.strategy_id: dict(zip(r.dates, r.returns)) for r in strategy_results},
    index=_common_dates,
).fillna(0.0)

# Weight sliders — laid out in columns of 3
n_strats = len(strategy_results)
cols_per_row = 3
slider_cols = st.columns(min(cols_per_row, n_strats))

raw_weights: dict[str, float] = {}
for i, r in enumerate(strategy_results):
    with slider_cols[i % cols_per_row]:
        w = st.slider(
            r.strategy_name,
            min_value=0,
            max_value=100,
            value=round(100 // n_strats),
            step=5,
            key=f"hub_weight_{r.strategy_id}",
            format="%d%%",
        )
        raw_weights[r.strategy_id] = w / 100.0

total_invested = sum(raw_weights.values())
cash_weight = max(0.0, 1.0 - total_invested)

# Normalise display
if total_invested > 1.0 + 1e-6:
    # Over-invested: normalise down
    normalised = {sid: w / total_invested for sid, w in raw_weights.items()}
    cash_weight = 0.0
    st.warning(f"Weights sum to {total_invested:.0%} — normalised to 100% for calculation.")
else:
    normalised = dict(raw_weights)

weight_summary = "  ·  ".join(
    f"{r.strategy_name}: {normalised[r.strategy_id]:.0%}"
    for r in strategy_results
)
if cash_weight > 1e-3:
    weight_summary += f"  ·  Cash: {cash_weight:.0%}"
st.caption(f"Portfolio: {weight_summary}")

# Compute blended returns on common dates
blended_returns = np.zeros(len(_common_dates))
for sid, w in normalised.items():
    if sid in _returns_df.columns and w > 1e-9:
        blended_returns += w * _returns_df[sid].values
# Cash earns 0%

# Reconstruct equity curve
blended_equity = [initial_capital]
for ret in blended_returns[1:]:
    blended_equity.append(blended_equity[-1] * (1.0 + ret))

# Blended metrics
from engine.metrics import compute_metrics as _compute_metrics  # noqa: E402
bench_rets_aligned = [
    dict(zip(benchmark.dates, benchmark.returns)).get(d, 0.0)
    for d in _common_dates
]
blended_metrics = _compute_metrics(blended_equity, [0.0] + list(blended_returns[1:]), bench_rets_aligned)

# Chart: blended vs benchmark
fig_blend = go.Figure()

# Benchmark
bench_eq_map = dict(zip(benchmark.dates, benchmark.equity))
bench_eq_aligned = [bench_eq_map.get(d, float("nan")) for d in _common_dates]

fig_blend.add_trace(
    go.Scatter(
        x=_common_dates,
        y=bench_eq_aligned,
        mode="lines",
        name=benchmark.strategy_name,
        line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>Benchmark: $%{y:,.0f}<extra></extra>",
    )
)

# Individual strategies (muted)
for r in strategy_results:
    w = normalised.get(r.strategy_id, 0.0)
    if w < 1e-3:
        continue
    eq_map = dict(zip(r.dates, r.equity))
    eq_aligned = [eq_map.get(d, float("nan")) for d in _common_dates]
    color = colors[r.strategy_id]
    fig_blend.add_trace(
        go.Scatter(
            x=_common_dates,
            y=eq_aligned,
            mode="lines",
            name=f"{r.strategy_name} ({w:.0%})",
            line=dict(color=color, width=1.0, dash="dot"),
            opacity=0.45,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{r.strategy_name}: $%{{y:,.0f}}<extra></extra>",
        )
    )

# Blended composite (prominent)
fig_blend.add_trace(
    go.Scatter(
        x=_common_dates,
        y=blended_equity,
        mode="lines",
        name="Composite Portfolio",
        line=dict(color="#4ecdc4", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Composite: $%{y:,.0f}<extra></extra>",
    )
)

fig_blend.update_layout(
    **CHART_LAYOUT,
    xaxis_title=None,
    yaxis_title=f"Portfolio Value (start = ${initial_capital:,.0f})",
    yaxis_tickprefix="$",
    yaxis_tickformat=",.0f",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
        bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=420,
)
fig_blend.update_xaxes(**AXIS_STYLE)
fig_blend.update_yaxes(**AXIS_STYLE)
st.plotly_chart(fig_blend, use_container_width=True)

# Blended metrics summary
bm = blended_metrics
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
with mc1:
    st.metric("Composite CAGR", f"{bm.get('cagr', float('nan')):+.1%}" if math.isfinite(bm.get('cagr', float('nan'))) else "—")
with mc2:
    st.metric("Composite Sharpe", f"{bm.get('sharpe', float('nan')):.2f}" if math.isfinite(bm.get('sharpe', float('nan'))) else "—")
with mc3:
    st.metric("Composite Max DD", f"{bm.get('max_drawdown', float('nan')):.1%}" if math.isfinite(bm.get('max_drawdown', float('nan'))) else "—")
with mc4:
    st.metric("Composite Sortino", f"{bm.get('sortino', float('nan')):.2f}" if math.isfinite(bm.get('sortino', float('nan'))) else "—")
with mc5:
    st.metric("Excess Return", f"{bm.get('excess_return', float('nan')):+.1%}" if math.isfinite(bm.get('excess_return', float('nan'))) else "—")

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 5 — Pairwise Return Scatter
# ════════════════════════════════════════════════════════════════════════════

_section_header(
    "Pairwise Return Scatter",
    "Daily returns of one strategy plotted against another. "
    "A tight diagonal = high correlation (low diversification benefit). "
    "A diffuse cloud = low correlation (good complement).",
)

if len(strategy_results) < 2:
    st.info("Run at least two strategies to see pairwise comparisons.")
else:
    # Build list of pairs — all combinations, capped at 15
    import itertools
    all_pairs = list(itertools.combinations(strategy_results, 2))
    if len(all_pairs) > 15:
        all_pairs = all_pairs[:15]
        st.caption(f"Showing first 15 of {len(list(itertools.combinations(strategy_results, 2)))} pairs.")

    tab_labels = [f"{a.strategy_name[:10]} × {b.strategy_name[:10]}" for a, b in all_pairs]
    tabs = st.tabs(tab_labels)

    for tab, (r_a, r_b) in zip(tabs, all_pairs):
        with tab:
            # Align to common dates between these two (and skip day-0 zeros)
            common_ab = sorted(set(r_a.dates) & set(r_b.dates))
            if len(common_ab) < 30:
                st.info("Not enough overlapping dates for this pair.")
                continue

            ret_a_map = dict(zip(r_a.dates, r_a.returns))
            ret_b_map = dict(zip(r_b.dates, r_b.returns))
            xs = [ret_a_map[d] * 100 for d in common_ab[1:]]
            ys = [ret_b_map[d] * 100 for d in common_ab[1:]]

            # Pearson correlation
            corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")

            color_a = colors[r_a.strategy_id]

            fig_pair = go.Figure()
            fig_pair.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(
                        color=color_a,
                        size=3,
                        opacity=0.45,
                    ),
                    hovertemplate=(
                        f"{r_a.strategy_name}: %{{x:.2f}}%<br>"
                        f"{r_b.strategy_name}: %{{y:.2f}}%"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

            # OLS trend line
            if len(xs) > 2:
                m, b_int = np.polyfit(xs, ys, 1)
                x_line = [min(xs), max(xs)]
                y_line = [m * x + b_int for x in x_line]
                fig_pair.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="#f56565", width=1.5, dash="dash"),
                        name="OLS fit",
                        hoverinfo="skip",
                    )
                )

            fig_pair.add_hline(y=0, line=dict(color=GRID, width=0.8))
            fig_pair.add_vline(x=0, line=dict(color=GRID, width=0.8))

            corr_str = f"ρ = {corr:.3f}" if math.isfinite(corr) else "ρ = —"
            fig_pair.update_layout(
                **CHART_LAYOUT,
                xaxis_title=f"{r_a.strategy_name} daily return (%)",
                yaxis_title=f"{r_b.strategy_name} daily return (%)",
                xaxis_ticksuffix="%",
                yaxis_ticksuffix="%",
                title=dict(
                    text=f"{r_a.strategy_name} × {r_b.strategy_name}   <span style='font-size:12px;color:#94a3b8;'>{corr_str}</span>",
                    font=dict(size=13, color="#e8ecf0", family="DM Sans, sans-serif"),
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                height=400,
                showlegend=False,
            )
            fig_pair.update_xaxes(**AXIS_STYLE)
            fig_pair.update_yaxes(**AXIS_STYLE)

            st.plotly_chart(fig_pair, use_container_width=True)
