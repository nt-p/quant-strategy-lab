"""Portfolio Hub — entry point for QuantScope.

The Monday morning view. Configure holdings in the sidebar, click Run Portfolio,
and see your portfolio analytics: returns, risk, drawdown, attribution.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Portfolio Hub",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="expanded",
)

from engine.metrics import compute_metrics  # noqa: E402
from ui.controller import handle_portfolio_run  # noqa: E402
from ui.theme import AXIS_STYLE, BG, CHART_LAYOUT, FONT, GRID, PLOT_BG, inject_css  # noqa: E402

inject_css()

# ── Sidebar: portfolio builder + Run button ────────────────────────────────────
if handle_portfolio_run():
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='padding: 0.5rem 0 0.15rem 0;'>
        <h1 style='font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;
                   background: linear-gradient(90deg, #4ecdc4 0%, #81a4e8 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-family: "DM Sans", sans-serif;'>
            Portfolio Hub
        </h1>
        <p style='color: #636b78; font-size: 0.70rem; margin: 0.25rem 0 0 0;
                  letter-spacing: 2.5px; font-weight: 600;
                  font-family: "DM Sans", sans-serif;'>
            HOLDINGS &nbsp;·&nbsp; PERFORMANCE &nbsp;·&nbsp; RISK
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Guard: show prompt if no portfolio loaded ─────────────────────────────────
portfolio = st.session_state.get("portfolio")
portfolio_returns: pd.Series | None = st.session_state.get("portfolio_returns")
benchmark_returns: pd.Series | None = st.session_state.get("benchmark_returns")

if portfolio is None or portfolio_returns is None:
    st.info("Configure your holdings in the sidebar and click **Run Portfolio** to get started.", icon="🏗")
    st.markdown(
        """
### How it works
1. **Pick a market** — US, Australian, or both
2. **Select assets** — use a curated group or type any yfinance ticker
3. **Set weights** — default is equal weight; adjust as needed
4. **Set a date range** and click **Run Portfolio**
5. Your portfolio analytics appear here
        """
    )
    st.stop()

# ── Extract portfolio data ─────────────────────────────────────────────────────
holdings: dict[str, float]  = portfolio["holdings"]
start_date: pd.Timestamp    = portfolio["start_date"]
end_date: pd.Timestamp      = portfolio["end_date"]
benchmark_ticker: str       = portfolio.get("benchmark_ticker", "SPY")

returns_wide: pd.DataFrame | None = st.session_state.get("returns")

# Align benchmark to portfolio index
bench_aligned = benchmark_returns.reindex(portfolio_returns.index).fillna(0.0)

# Build equity curves (starting at 1.0)
port_equity  = (1.0 + portfolio_returns).cumprod()
bench_equity = (1.0 + bench_aligned).cumprod()

# ── Compute metrics ────────────────────────────────────────────────────────────
port_equity_list   = [1.0] + port_equity.tolist()
port_returns_list  = [0.0] + portfolio_returns.tolist()
bench_returns_list = [0.0] + bench_aligned.tolist()

port_metrics  = compute_metrics(port_equity_list, port_returns_list, bench_returns_list)
bench_metrics = compute_metrics(
    [1.0] + bench_equity.tolist(),
    [0.0] + bench_aligned.tolist(),
    [0.0] + bench_aligned.tolist(),
)

# ── Helpers ────────────────────────────────────────────────────────────────────

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


TEAL     = "#4ecdc4"
BENCH_C  = "#94a3b8"


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — KPI Cards
# ════════════════════════════════════════════════════════════════════════════

def _fmt(v: float, fmt: str) -> str:
    return fmt.format(v) if math.isfinite(v) else "—"


c1, c2, c3, c4 = st.columns(4)

with c1:
    pv = port_metrics.get("cagr", float("nan"))
    bv = bench_metrics.get("cagr", float("nan"))
    delta = pv - bv if math.isfinite(pv) and math.isfinite(bv) else None
    st.metric(
        "Ann. Return",
        _fmt(pv, "{:+.1%}"),
        delta=_fmt(delta, "{:+.1%}") + " vs benchmark" if delta is not None else None,
        delta_color="normal",
    )

with c2:
    pv = port_metrics.get("ann_vol", float("nan"))
    bv = bench_metrics.get("ann_vol", float("nan"))
    delta = pv - bv if math.isfinite(pv) and math.isfinite(bv) else None
    st.metric(
        "Ann. Volatility",
        _fmt(pv, "{:.1%}"),
        delta=_fmt(delta, "{:+.1%}") + " vs benchmark" if delta is not None else None,
        delta_color="inverse",  # lower vol is better
    )

with c3:
    pv = port_metrics.get("sharpe", float("nan"))
    bv = bench_metrics.get("sharpe", float("nan"))
    delta = pv - bv if math.isfinite(pv) and math.isfinite(bv) else None
    st.metric(
        "Sharpe Ratio",
        _fmt(pv, "{:.2f}"),
        delta=_fmt(delta, "{:+.2f}") + " vs benchmark" if delta is not None else None,
        delta_color="normal",
    )

with c4:
    pv = port_metrics.get("max_drawdown", float("nan"))
    bv = bench_metrics.get("max_drawdown", float("nan"))
    delta = pv - bv if math.isfinite(pv) and math.isfinite(bv) else None
    st.metric(
        "Max Drawdown",
        _fmt(pv, "{:.1%}"),
        delta=_fmt(delta, "{:+.1%}") + " vs benchmark" if delta is not None else None,
        delta_color="normal",   # less negative = green = better
    )

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Cumulative Return
# ════════════════════════════════════════════════════════════════════════════

_section_header("Cumulative Return", "Portfolio vs benchmark, rebased to 1.0 at start.")

log_scale = st.toggle("Log scale", value=False, key="log_scale_toggle")

fig_cum = go.Figure()

fig_cum.add_trace(go.Scatter(
    x=port_equity.index,
    y=port_equity.values,
    mode="lines",
    name="Portfolio",
    line=dict(color=TEAL, width=2.5),
    hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.3f}<extra></extra>",
))
fig_cum.add_trace(go.Scatter(
    x=bench_equity.index,
    y=bench_equity.values,
    mode="lines",
    name=benchmark_ticker,
    line=dict(color=BENCH_C, width=1.5, dash="dot"),
    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{benchmark_ticker}: %{{y:.3f}}<extra></extra>",
))

fig_cum.update_layout(
    **CHART_LAYOUT,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title=None,
    yaxis_title="Growth of $1",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
        bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=400,
)
fig_cum.update_xaxes(**AXIS_STYLE)
fig_cum.update_yaxes(**AXIS_STYLE)
st.plotly_chart(fig_cum, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Drawdown
# ════════════════════════════════════════════════════════════════════════════

_section_header("Drawdown", "Peak-to-trough loss at each point in time.")


def _drawdown_series(eq: pd.Series) -> pd.Series:
    running_max = eq.cummax()
    return (eq - running_max) / running_max


port_dd  = _drawdown_series(port_equity)
bench_dd = _drawdown_series(bench_equity)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=port_dd.index, y=(port_dd * 100).values,
    mode="lines", name="Portfolio",
    fill="tozeroy", fillcolor=_hex_alpha(TEAL, 0.12),
    line=dict(color=TEAL, width=1.5),
    hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.1f}%<extra></extra>",
))
fig_dd.add_trace(go.Scatter(
    x=bench_dd.index, y=(bench_dd * 100).values,
    mode="lines", name=benchmark_ticker,
    fill="tozeroy", fillcolor=_hex_alpha(BENCH_C, 0.07),
    line=dict(color=BENCH_C, width=1.0, dash="dot"),
    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{benchmark_ticker}: %{{y:.1f}}%<extra></extra>",
))

fig_dd.update_layout(
    **CHART_LAYOUT,
    xaxis_title=None,
    yaxis_title="Drawdown (%)",
    yaxis_ticksuffix="%",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
        bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=300,
)
fig_dd.update_xaxes(**AXIS_STYLE)
fig_dd.update_yaxes(**AXIS_STYLE)
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 4 — Rolling 12-Month Return
# ════════════════════════════════════════════════════════════════════════════

_section_header("Rolling 12-Month Return", "Annual return computed over a trailing 252-day window.")

window_days = 252
port_roll  = portfolio_returns.rolling(window_days).apply(lambda r: (1 + r).prod() - 1, raw=True)
bench_roll = bench_aligned.rolling(window_days).apply(lambda r: (1 + r).prod() - 1, raw=True)
port_roll  = port_roll.dropna()
bench_roll = bench_roll.reindex(port_roll.index).fillna(0.0)

bar_colors = [TEAL if v >= 0 else "#f56565" for v in port_roll.values]

fig_roll = go.Figure()
fig_roll.add_trace(go.Bar(
    x=port_roll.index,
    y=(port_roll * 100).values,
    name="Portfolio",
    marker_color=bar_colors,
    opacity=0.85,
    hovertemplate="%{x|%Y-%m-%d}<br>Portfolio 12M: %{y:.1f}%<extra></extra>",
))
fig_roll.add_trace(go.Scatter(
    x=bench_roll.index,
    y=(bench_roll * 100).values,
    mode="lines",
    name=benchmark_ticker,
    line=dict(color=BENCH_C, width=1.2, dash="dot"),
    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{benchmark_ticker} 12M: %{{y:.1f}}%<extra></extra>",
))
fig_roll.add_hline(y=0, line=dict(color=GRID, width=0.8))

fig_roll.update_layout(
    **CHART_LAYOUT,
    xaxis_title=None,
    yaxis_title="12M Return (%)",
    yaxis_ticksuffix="%",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
        bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=320,
)
fig_roll.update_xaxes(**AXIS_STYLE)
fig_roll.update_yaxes(**AXIS_STYLE)
st.plotly_chart(fig_roll, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 5 — Three-panel: Allocation | Monthly Heatmap | Risk Decomposition
# ════════════════════════════════════════════════════════════════════════════

col_donut, col_heat, col_risk = st.columns([1, 1.6, 1])

# ── Allocation donut ──────────────────────────────────────────────────────────
with col_donut:
    _section_header("Allocation")

    tickers_list = list(holdings.keys())
    weights_list = [holdings[t] * 100 for t in tickers_list]

    _TEAL_PALETTE = [
        "#4ecdc4", "#81a4e8", "#f6ad55", "#b794f4",
        "#68d391", "#f56565", "#63b3ed", "#e2b862",
        "#a0aec0", "#76e4f7",
    ]
    pie_colors = [_TEAL_PALETTE[i % len(_TEAL_PALETTE)] for i in range(len(tickers_list))]

    fig_donut = go.Figure(go.Pie(
        labels=tickers_list,
        values=weights_list,
        hole=0.48,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        marker=dict(colors=pie_colors, line=dict(color=BG, width=2)),
    ))
    fig_donut.update_layout(
        **CHART_LAYOUT,
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ── Monthly returns heatmap ───────────────────────────────────────────────────
with col_heat:
    _section_header("Monthly Returns")

    monthly = (
        portfolio_returns
        .resample("ME")
        .apply(lambda r: (1 + r).prod() - 1)
    )
    monthly.index = monthly.index.to_period("M")

    years  = sorted(monthly.index.year.unique(), reverse=True)
    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    z = []
    text = []
    for yr in years:
        row_z, row_t = [], []
        for mo in months:
            period = pd.Period(year=yr, month=mo, freq="M")
            val = monthly.get(period, float("nan"))
            row_z.append(val * 100 if not math.isnan(val) else float("nan"))
            row_t.append(f"{val*100:+.1f}%" if not math.isnan(val) else "")
        z.append(row_z)
        text.append(row_t)

    fig_heat = go.Figure(go.Heatmap(
        z=z,
        x=month_labels,
        y=[str(yr) for yr in years],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=9, color="#e8ecf0", family="JetBrains Mono, monospace"),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(
            title="%",
            tickfont=dict(color=FONT, size=9),
            titlefont=dict(color=FONT, size=10),
            outlinewidth=0,
            bgcolor=PLOT_BG,
            thickness=10,
        ),
        hovertemplate="%{y} %{x}<br>Return: %{text}<extra></extra>",
    ))
    fig_heat.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(tickfont=dict(size=9, color=FONT), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=9, color=FONT), showgrid=False, zeroline=False),
        margin=dict(l=0, r=30, t=10, b=0),
        height=280,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Risk decomposition pie ────────────────────────────────────────────────────
with col_risk:
    _section_header("Risk Contribution")

    if returns_wide is not None and len(holdings) > 1:
        avail = [t for t in holdings if t in returns_wide.columns]
        w_vec = np.array([holdings[t] for t in avail])
        ret_sub = returns_wide[avail].fillna(0.0)
        cov = ret_sub.cov().values * 252  # annualised

        port_var = float(w_vec @ cov @ w_vec)
        if port_var > 0:
            rc = w_vec * (cov @ w_vec) / port_var  # marginal risk contribution
            rc_labels = avail
            rc_values = (rc * 100).tolist()

            fig_risk = go.Figure(go.Pie(
                labels=rc_labels,
                values=rc_values,
                hole=0.48,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Risk contribution: %{percent}<extra></extra>",
                marker=dict(colors=pie_colors[:len(avail)], line=dict(color=BG, width=2)),
            ))
            fig_risk.update_layout(
                **CHART_LAYOUT,
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            st.caption(
                "Risk contribution = wᵢ·(Σw)ᵢ / w′Σw. "
                "Equal-weight ≠ equal-risk."
            )
        else:
            st.info("Insufficient data for risk decomposition.")
    else:
        st.info("Risk decomposition requires at least two holdings.")

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 6 — Full Metrics Table
# ════════════════════════════════════════════════════════════════════════════

_section_header("Performance Summary", "Portfolio vs benchmark — all key statistics.")

_METRIC_ROWS = [
    ("Ann. Return (CAGR)",    "cagr",             "{:+.2%}"),
    ("Ann. Volatility",       "ann_vol",           "{:.2%}"),
    ("Sharpe Ratio",          "sharpe",            "{:.3f}"),
    ("Sortino Ratio",         "sortino",           "{:.3f}"),
    ("Calmar Ratio",          "calmar",            "{:.3f}"),
    ("Max Drawdown",          "max_drawdown",      "{:.2%}"),
    ("Win Rate",              "win_rate",          "{:.1%}"),
    ("Excess Return vs Bmk",  "excess_return",     "{:+.2%}"),
    ("Information Ratio",     "information_ratio", "{:.3f}"),
]

rows = []
for label, key, fmt in _METRIC_ROWS:
    pv = port_metrics.get(key, float("nan"))
    bv = bench_metrics.get(key, float("nan"))
    rows.append({
        "Metric":    label,
        "Portfolio": fmt.format(pv) if math.isfinite(pv) else "—",
        "Benchmark": fmt.format(bv) if math.isfinite(bv) else "—",
    })

df_summary = pd.DataFrame(rows).set_index("Metric")
st.dataframe(df_summary, use_container_width=True)
