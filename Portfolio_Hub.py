"""Portfolio Hub — the Monday morning view.

Configure holdings in the sidebar, click Run Portfolio,
and see your portfolio analytics: returns, risk, drawdown, attribution.
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

# Build equity curves (starting at 1.0) — used for drawdown and metrics
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
# Section 2 — Asset Information Cards
# ════════════════════════════════════════════════════════════════════════════

def _get_asset_info(ticker: str) -> dict:
    """Fetch yfinance Ticker.info for *ticker*, cached in session_state.

    Returns the raw info dict (or empty dict on any error).
    """
    cache: dict = st.session_state.setdefault("_asset_info_cache", {})
    if ticker not in cache:
        try:
            import yfinance as yf
            cache[ticker] = yf.Ticker(ticker).info or {}
        except Exception:
            cache[ticker] = {}
    return cache[ticker]


def _fmt_market_cap(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "—"
    v = float(v)
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"


def _truncate_sentences(text: str | None, max_sentences: int = 2) -> str:
    if not text:
        return "—"
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences])


def _asset_type_label(info: dict) -> str:
    qt = (info.get("quoteType") or "").upper()
    category = (info.get("category") or "").lower()
    if qt == "ETF":
        if "bond" in category or "fixed income" in category or "treasury" in category:
            return "Bond ETF"
        if "real estate" in category or "reit" in category:
            return "REIT ETF"
        return "ETF"
    if qt == "EQUITY":
        return "Stock"
    if qt == "MUTUALFUND":
        return "Mutual Fund"
    if qt == "CURRENCY":
        return "FX"
    if qt == "FUTURE":
        return "Futures"
    return qt if qt else "—"


def _ytd_and_1y_returns(ticker: str, prices: pd.DataFrame | None) -> tuple[float | None, float | None]:
    """Compute YTD and 1Y returns from the already-fetched prices DataFrame."""
    if prices is None or ticker not in prices.columns:
        return None, None
    s = prices[ticker].dropna()
    if len(s) < 2:
        return None, None
    latest = s.iloc[-1]
    latest_date = s.index[-1]

    # YTD: first trading day of the current year
    ytd_return: float | None = None
    year_start = s[s.index.year == latest_date.year]
    if len(year_start) > 0:
        ytd_return = float(latest / year_start.iloc[0] - 1)

    # 1Y: 252 trading days back
    one_year_return: float | None = None
    if len(s) >= 253:
        one_year_return = float(latest / s.iloc[-253] - 1)

    return ytd_return, one_year_return


def _return_html(label: str, value: float | None) -> str:
    if value is None or not math.isfinite(value):
        txt = "—"
        color = "#a0a8b8"
    else:
        sign = "+" if value >= 0 else ""
        txt = f"{sign}{value:.1%}"
        color = "#68d391" if value >= 0 else "#f56565"
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-top:4px;'>"
        f"<span style='font-size:0.70rem;color:#636b78;text-transform:uppercase;"
        f"letter-spacing:0.8px;font-weight:600;'>{label}</span>"
        f"<span style='font-size:0.80rem;font-family:\"JetBrains Mono\",monospace;"
        f"font-weight:600;color:{color};'>{txt}</span>"
        f"</div>"
    )


def _field_row_html(label: str, value: str) -> str:
    safe = value if value else "—"
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"margin-top:5px;gap:8px;'>"
        f"<span style='font-size:0.68rem;color:#636b78;text-transform:uppercase;"
        f"letter-spacing:0.8px;font-weight:600;white-space:nowrap;flex-shrink:0;'>{label}</span>"
        f"<span style='font-size:0.76rem;color:#c4cad6;text-align:right;"
        f"font-family:\"DM Sans\",sans-serif;'>{safe}</span>"
        f"</div>"
    )


def _render_asset_cards(
    holdings: dict[str, float],
    prices: pd.DataFrame | None,
) -> None:
    _section_header("Holdings", "Asset details and returns for each position.")

    tickers_ordered = list(holdings.keys())
    n = len(tickers_ordered)
    cols_per_row = min(3, n) if n > 0 else 1

    # Chunk tickers into rows
    rows_of_tickers: list[list[str]] = []
    for i in range(0, n, cols_per_row):
        rows_of_tickers.append(tickers_ordered[i : i + cols_per_row])

    for row_tickers in rows_of_tickers:
        cols = st.columns(cols_per_row)
        for col, ticker in zip(cols, row_tickers):
            weight = holdings[ticker]
            info = _get_asset_info(ticker)
            asset_type = _asset_type_label(info)
            ytd, one_y = _ytd_and_1y_returns(ticker, prices)

            # Common fields
            long_name: str = info.get("longName") or info.get("shortName") or ticker

            # Build type-specific body HTML
            is_etf = asset_type in ("ETF", "Bond ETF", "REIT ETF", "Mutual Fund")
            body_parts: list[str] = []

            if is_etf:
                fund_family = info.get("fundFamily") or info.get("issuer") or None
                if fund_family:
                    body_parts.append(_field_row_html("Manager", fund_family))

                expense = info.get("annualReportExpenseRatio") or info.get("totalExpenseRatio") or None
                if expense is not None:
                    try:
                        body_parts.append(_field_row_html("Expense Ratio", f"{float(expense):.2%}"))
                    except (TypeError, ValueError):
                        pass

                # Top 5 holdings
                raw_holdings = info.get("holdings") or []
                if raw_holdings:
                    holding_lines = []
                    for h in raw_holdings[:5]:
                        h_name = h.get("holdingName") or h.get("symbol") or "—"
                        h_pct = h.get("holdingPercent")
                        if h_pct is not None:
                            try:
                                h_line = f"{h_name} ({float(h_pct):.1%})"
                            except (TypeError, ValueError):
                                h_line = h_name
                        else:
                            h_line = h_name
                        holding_lines.append(h_line)
                    holdings_str = "<br>".join(holding_lines)
                    body_parts.append(
                        f"<div style='margin-top:6px;'>"
                        f"<div style='font-size:0.68rem;color:#636b78;text-transform:uppercase;"
                        f"letter-spacing:0.8px;font-weight:600;margin-bottom:3px;'>Top Holdings</div>"
                        f"<div style='font-size:0.73rem;color:#c4cad6;line-height:1.5;"
                        f"font-family:\"DM Sans\",sans-serif;'>{holdings_str}</div>"
                        f"</div>"
                    )

            else:
                # Stock
                sector = info.get("sector") or None
                if sector:
                    body_parts.append(_field_row_html("Sector", sector))

                mkt_cap = info.get("marketCap")
                if mkt_cap:
                    body_parts.append(_field_row_html("Market Cap", _fmt_market_cap(mkt_cap)))

                pe = info.get("trailingPE")
                if pe is not None:
                    try:
                        body_parts.append(_field_row_html("Trailing P/E", f"{float(pe):.1f}×"))
                    except (TypeError, ValueError):
                        pass

                hi52 = info.get("fiftyTwoWeekHigh")
                lo52 = info.get("fiftyTwoWeekLow")
                if hi52 and lo52:
                    try:
                        body_parts.append(_field_row_html("52W Range", f"${float(lo52):.2f} – ${float(hi52):.2f}"))
                    except (TypeError, ValueError):
                        pass

            # Description (both types)
            summary = _truncate_sentences(info.get("longBusinessSummary"), 2)
            if summary and summary != "—":
                body_parts.append(
                    f"<div style='margin-top:8px;font-size:0.72rem;color:#8892a0;"
                    f"line-height:1.5;font-family:\"DM Sans\",sans-serif;border-top:1px solid #252836;"
                    f"padding-top:7px;'>{summary}</div>"
                )

            body_html = "\n".join(body_parts) if body_parts else ""

            # Assemble full card
            card_html = f"""
<div style='background:#1a1e2a;border:1px solid #252836;border-radius:10px;
            padding:14px 16px 12px 16px;height:100%;font-family:"DM Sans",sans-serif;'>
  <!-- Header row: ticker + weight -->
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
    <span style='font-size:1.05rem;font-weight:700;color:#f0f2f5;
                 font-family:"JetBrains Mono",monospace;letter-spacing:-0.3px;'>{ticker}</span>
    <span style='background:#1e2d3d;color:#4ecdc4;padding:2px 8px;border-radius:4px;
                 font-size:0.70rem;font-weight:700;font-family:"JetBrains Mono",monospace;
                 letter-spacing:0.5px;'>{weight:.1%}</span>
  </div>
  <!-- Name + type -->
  <div style='font-size:0.78rem;color:#a0a8b8;margin-bottom:2px;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
       title='{long_name}'>{long_name}</div>
  <div style='display:inline-block;background:#1e2d1e;color:#68d391;padding:1px 6px;
              border-radius:3px;font-size:0.63rem;font-weight:700;letter-spacing:0.8px;
              text-transform:uppercase;margin-bottom:8px;'>{asset_type}</div>
  <!-- Type-specific fields -->
  {body_html}
  <!-- Returns row -->
  <div style='margin-top:10px;padding-top:8px;border-top:1px solid #252836;'>
    {_return_html("YTD", ytd)}
    {_return_html("1Y", one_y)}
  </div>
</div>
"""
            with col:
                st.markdown(card_html, unsafe_allow_html=True)


_render_asset_cards(holdings, returns_wide)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Cumulative Return
# ════════════════════════════════════════════════════════════════════════════

_section_header("Portfolio Value")

# ── Investment controls ────────────────────────────────────────────────────
ic1, ic2, ic3 = st.columns([1, 1, 1])
with ic1:
    initial_investment = st.number_input(
        "Initial investment ($)",
        min_value=0,
        max_value=100_000_000,
        value=10_000,
        step=1_000,
        format="%d",
        key="hub_initial_investment",
    )
with ic2:
    dca_enabled = st.toggle("Periodic contributions", value=False, key="hub_dca_enabled")

if dca_enabled:
    dc1, dc2, dc3 = st.columns([1, 1, 1])
    with dc1:
        dca_amount = st.number_input(
            "Contribution ($)",
            min_value=0,
            max_value=1_000_000,
            value=500,
            step=100,
            format="%d",
            key="hub_dca_amount",
        )
    with dc2:
        dca_freq = st.selectbox(
            "Frequency",
            options=["Monthly", "Weekly", "Quarterly", "Annually"],
            index=0,
            key="hub_dca_freq",
        )
    with dc3:
        show_invested = st.toggle("Show total invested", value=True, key="hub_show_invested")
else:
    dca_amount = 0
    dca_freq = "Monthly"
    show_invested = False


def _contribution_dates(returns: pd.Series, freq: str) -> set:
    """First trading day in each period."""
    idx = returns.index
    period_map = {"Monthly": "M", "Weekly": "W", "Quarterly": "Q", "Annually": "Y"}
    periods = idx.to_period(period_map[freq])
    seen: set = set()
    dates: set = set()
    for date, period in zip(idx, periods):
        if period not in seen:
            seen.add(period)
            dates.add(date)
    return dates


def _dollar_equity(
    returns: pd.Series,
    initial: float,
    contrib: float,
    contrib_dates: set,
) -> pd.Series:
    """Simulate a dollar portfolio with optional periodic contributions."""
    n = len(returns)
    values = np.empty(n)
    balance = float(initial)
    for i in range(n):
        if i > 0 and returns.index[i] in contrib_dates:
            balance += contrib
        balance *= 1.0 + returns.iloc[i]
        values[i] = balance
    return pd.Series(values, index=returns.index)


contrib_dates: set = (
    _contribution_dates(portfolio_returns, dca_freq) if dca_enabled and dca_amount > 0 else set()
)

port_dollar  = _dollar_equity(portfolio_returns, initial_investment, dca_amount if dca_enabled else 0, contrib_dates)
bench_dollar = _dollar_equity(bench_aligned,     initial_investment, dca_amount if dca_enabled else 0, contrib_dates)

# Cumulative invested (step function for "break-even" reference)
if dca_enabled and show_invested and dca_amount > 0:
    invested_vals = np.empty(len(portfolio_returns))
    running = float(initial_investment)
    for i, date in enumerate(portfolio_returns.index):
        if i > 0 and date in contrib_dates:
            running += dca_amount
        invested_vals[i] = running
    total_invested_series = pd.Series(invested_vals, index=portfolio_returns.index)
else:
    total_invested_series = None

log_scale = st.toggle("Log scale", value=False, key="log_scale_toggle")

fig_cum = go.Figure()

if total_invested_series is not None:
    fig_cum.add_trace(go.Scatter(
        x=total_invested_series.index,
        y=total_invested_series.values,
        mode="lines",
        name="Total Invested",
        line=dict(color="#636b78", width=1.2, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Invested: $%{y:,.0f}<extra></extra>",
    ))

fig_cum.add_trace(go.Scatter(
    x=bench_dollar.index,
    y=bench_dollar.values,
    mode="lines",
    name=benchmark_ticker,
    line=dict(color=BENCH_C, width=1.5, dash="dot"),
    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{benchmark_ticker}: $%{{y:,.0f}}<extra></extra>",
))
fig_cum.add_trace(go.Scatter(
    x=port_dollar.index,
    y=port_dollar.values,
    mode="lines",
    name="Portfolio",
    line=dict(color=TEAL, width=2.5),
    hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: $%{y:,.0f}<extra></extra>",
))

_freq_label = {"Monthly": "month", "Weekly": "week", "Quarterly": "quarter", "Annually": "year"}
_dca_caption = (
    f"${initial_investment:,.0f} initial investment"
    + (
        f" + ${dca_amount:,.0f} per {_freq_label.get(dca_freq, dca_freq.lower())} contributions"
        if dca_enabled and dca_amount > 0 else ""
    )
)

fig_cum.update_layout(
    **CHART_LAYOUT,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title=None,
    yaxis_title="Portfolio Value ($)",
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
fig_cum.update_xaxes(**AXIS_STYLE)
fig_cum.update_yaxes(**AXIS_STYLE)
st.plotly_chart(fig_cum, use_container_width=True)
st.caption(_dca_caption)

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
            title=dict(text="%", font=dict(color=FONT, size=10)),
            tickfont=dict(color=FONT, size=9),
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
