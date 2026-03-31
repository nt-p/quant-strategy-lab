"""Page 4: Risk Analytics — distributional, drawdown, and stress risk for the active portfolio."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as sp_stats
import streamlit as st

st.set_page_config(page_title="Risk Analytics", page_icon="📊", layout="wide")

from modules.risk import (  # noqa: E402
    apply_stress_scenarios,
    compute_cvar_from_series,
    compute_drawdown_from_series,
    compute_parametric_var,
    compute_rolling_var,
    compute_sensitivity_impact,
    compute_var_from_series,
    compute_worst_drawdown_periods,
)
from ui.sidebar import render_portfolio_sidebar, require_portfolio  # noqa: E402
from ui.theme import AXIS_STYLE, CHART_LAYOUT, FONT, GRID, inject_css  # noqa: E402

inject_css()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' + alpha to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


TEAL  = "#4ecdc4"
GREY  = "#636b78"
GREEN = "#68d391"
RED   = "#f56565"


# ── Sidebar + header ───────────────────────────────────────────────────────────
render_portfolio_sidebar()

st.markdown(
    """
    <div style='padding: 0.5rem 0 0.15rem 0;'>
        <h1 style='font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;
                   background: linear-gradient(90deg, #4ecdc4 0%, #81a4e8 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-family: "DM Sans", sans-serif;'>
            Risk Analytics
        </h1>
        <p style='color: #636b78; font-size: 0.70rem; margin: 0.25rem 0 0 0;
                  letter-spacing: 2.5px; font-weight: 600;
                  font-family: "DM Sans", sans-serif;'>
            DISTRIBUTION &nbsp;·&nbsp; DRAWDOWN &nbsp;·&nbsp; STRESS TESTING
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ── Guard ─────────────────────────────────────────────────────────────────────
require_portfolio()

portfolio_returns: pd.Series | None = st.session_state.get("portfolio_returns")
if portfolio_returns is None:
    st.info(
        "Run the portfolio in **Portfolio Hub** first — return data not yet computed.",
        icon="📊",
    )
    st.stop()

portfolio        = st.session_state["portfolio"]
benchmark_returns: pd.Series | None = st.session_state.get("benchmark_returns")
holdings: dict[str, float]          = portfolio["holdings"]
benchmark_ticker: str               = portfolio.get("benchmark_ticker", "SPY")

bench_aligned: pd.Series = (
    benchmark_returns.reindex(portfolio_returns.index).fillna(0.0)
    if benchmark_returns is not None
    else pd.Series(0.0, index=portfolio_returns.index)
)


# ── Cached stress scenarios (avoid re-fetching on every interaction) ──────────
@st.cache_data(show_spinner="Fetching historical scenario data…", ttl=3_600)
def _load_scenarios(
    holdings_tuple: tuple[tuple[str, float], ...],
    benchmark: str,
) -> pd.DataFrame:
    return apply_stress_scenarios(dict(holdings_tuple), benchmark)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dist, tab_dd, tab_stress = st.tabs(
    ["Return Distribution & VaR", "Drawdown Analysis", "Stress Testing"]
)


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Return Distribution & VaR
# ════════════════════════════════════════════════════════════════════════════
with tab_dist:
    st.caption(
        "Histogram of daily portfolio returns with a fitted normal distribution overlaid. "
        "Fat tails (excess kurtosis > 0) indicate more crash risk than the normal model implies."
    )

    rets_pct = portfolio_returns.dropna() * 100.0  # convert to %

    mu_d   = float(rets_pct.mean())
    sig_d  = float(rets_pct.std(ddof=1))
    skew_d = float(sp_stats.skew(rets_pct.values, bias=False))
    kurt_d = float(sp_stats.kurtosis(rets_pct.values, fisher=True, bias=False))

    # ── Histogram + fitted normal ─────────────────────────────────────────
    x_fit = np.linspace(float(rets_pct.min()), float(rets_pct.max()), 400)
    y_fit = sp_stats.norm.pdf(x_fit, mu_d, sig_d)

    var_95_val = compute_var_from_series(portfolio_returns, 0.95) * 100.0
    var_99_val = compute_var_from_series(portfolio_returns, 0.99) * 100.0

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Histogram(
            x=rets_pct,
            name="Daily Returns",
            histnorm="probability density",
            nbinsx=80,
            marker_color=_rgba(TEAL, 0.50),
            marker_line=dict(color=_rgba(TEAL, 0.15), width=0.3),
            hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.5f}<extra></extra>",
        )
    )
    fig_hist.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name="Fitted Normal",
            line=dict(color="#f6ad55", width=2.0),
            hoverinfo="skip",
        )
    )

    if not math.isnan(var_95_val):
        fig_hist.add_vline(
            x=-var_95_val,
            line=dict(color=RED, width=1.5, dash="dash"),
            annotation_text="VaR 95%",
            annotation_font=dict(color=RED, size=9),
            annotation_position="top left",
        )
    if not math.isnan(var_99_val):
        fig_hist.add_vline(
            x=-var_99_val,
            line=dict(color="#fc8181", width=1.2, dash="dot"),
            annotation_text="VaR 99%",
            annotation_font=dict(color="#fc8181", size=9),
            annotation_position="bottom left",
        )

    # Stats annotation box
    stat_text = (
        f"μ = {mu_d:+.3f}%   σ = {sig_d:.3f}%<br>"
        f"Skewness = {skew_d:+.3f}   Excess Kurtosis = {kurt_d:+.3f}"
    )
    fig_hist.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.97,
        text=stat_text,
        showarrow=False,
        align="right",
        font=dict(size=10, color=FONT, family="JetBrains Mono, monospace"),
        bgcolor="rgba(22,27,39,0.88)",
        bordercolor=GRID,
        borderwidth=1,
        borderpad=7,
    )

    fig_hist.update_layout(
        **CHART_LAYOUT,
        barmode="overlay",
        xaxis_title="Daily Return (%)",
        yaxis_title="Probability Density",
        xaxis_ticksuffix="%",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
            bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
    )
    fig_hist.update_xaxes(**AXIS_STYLE)
    fig_hist.update_yaxes(**AXIS_STYLE)

    st.plotly_chart(fig_hist, use_container_width=True)

    # ── VaR / CVaR table ──────────────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:0.6rem 0 0.2rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Value-at-Risk Summary</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "VaR: maximum loss not exceeded on (1 − confidence) fraction of trading days. "
        "CVaR / Expected Shortfall: average loss on days that breach the VaR threshold. "
        "Parametric VaR assumes normally distributed returns."
    )

    var_95h = compute_var_from_series(portfolio_returns, 0.95)
    var_99h = compute_var_from_series(portfolio_returns, 0.99)
    var_95p = compute_parametric_var(portfolio_returns, 0.95)
    var_99p = compute_parametric_var(portfolio_returns, 0.99)
    cvar_95 = compute_cvar_from_series(portfolio_returns, 0.95)
    cvar_99 = compute_cvar_from_series(portfolio_returns, 0.99)

    var_df = pd.DataFrame(
        {
            "95% Confidence": [-var_95h, -var_95p, -cvar_95],
            "99% Confidence": [-var_99h, -var_99p, -cvar_99],
        },
        index=pd.Index(
            ["Historical VaR", "Parametric VaR", "CVaR (Expected Shortfall)"],
            name="Metric",
        ),
    )

    var_styler = (
        var_df.style
        .format("{:.2%}", na_rep="—")
        .background_gradient(cmap="RdYlGn", axis=None)
    )
    st.dataframe(var_styler, use_container_width=True)

    # ── Rolling 252-day VaR chart ─────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:1.2rem 0 0.2rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>"
        "Rolling 252-Day Historical VaR (95%)</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Rolling one-year window. Spikes indicate periods of elevated tail risk.")

    rolling_var = (
        compute_rolling_var(portfolio_returns, window=252, confidence=0.95) * 100.0
    ).dropna()

    if len(rolling_var) > 0:
        fig_rvar = go.Figure()
        fig_rvar.add_trace(
            go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values,
                mode="lines",
                name="Rolling VaR 95%",
                line=dict(color=TEAL, width=1.5),
                fill="tozeroy",
                fillcolor=_rgba(TEAL, 0.08),
                hovertemplate="%{x|%Y-%m-%d}<br>VaR 95%: %{y:.2f}%<extra></extra>",
            )
        )
        fig_rvar.update_layout(
            **CHART_LAYOUT,
            xaxis_title=None,
            yaxis_title="Daily VaR (%)",
            yaxis_ticksuffix="%",
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            showlegend=False,
        )
        fig_rvar.update_xaxes(**AXIS_STYLE)
        fig_rvar.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_rvar, use_container_width=True)
    else:
        st.caption("Insufficient history for rolling VaR (requires at least 63 trading days).")


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Drawdown Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_dd:
    st.caption(
        "Underwater equity curves show the depth of loss from the portfolio's peak value at every point "
        "in time. Portfolio in teal, benchmark in grey."
    )

    port_dd  = compute_drawdown_from_series(portfolio_returns) * 100.0
    bench_dd = compute_drawdown_from_series(bench_aligned) * 100.0

    # ── Underwater chart ──────────────────────────────────────────────────
    fig_dd = go.Figure()

    fig_dd.add_trace(
        go.Scatter(
            x=bench_dd.index,
            y=bench_dd.values,
            mode="lines",
            name=benchmark_ticker,
            line=dict(color=GREY, width=1.2, dash="dot"),
            fill="tozeroy",
            fillcolor=_rgba(GREY, 0.08),
            hovertemplate="%{x|%Y-%m-%d}<br>" + benchmark_ticker + ": %{y:.2f}%<extra></extra>",
        )
    )
    fig_dd.add_trace(
        go.Scatter(
            x=port_dd.index,
            y=port_dd.values,
            mode="lines",
            name="Portfolio",
            line=dict(color=TEAL, width=1.8),
            fill="tozeroy",
            fillcolor=_rgba(TEAL, 0.12),
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.2f}%<extra></extra>",
        )
    )

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
        height=360,
    )
    fig_dd.update_xaxes(**AXIS_STYLE)
    fig_dd.update_yaxes(**AXIS_STYLE)

    st.plotly_chart(fig_dd, use_container_width=True)

    # ── 5 worst drawdown periods ──────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:0.6rem 0 0.2rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>5 Worst Drawdown Periods</h3>",
        unsafe_allow_html=True,
    )

    worst_dd = compute_worst_drawdown_periods(portfolio_returns, n=5)

    if worst_dd.empty:
        st.info("No significant drawdown periods detected in this date range.")
    else:
        display_dd = worst_dd.copy()
        display_dd["Depth"] = display_dd["Depth"].map(
            lambda x: f"{x:.1%}" if not math.isnan(float(x)) else "—"
        )
        for col in ["Peak Date", "Trough Date"]:
            display_dd[col] = pd.to_datetime(display_dd[col]).dt.strftime("%Y-%m-%d")
        display_dd["Recovery Date"] = display_dd["Recovery Date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "Ongoing"
        )
        display_dd["Duration (days)"] = display_dd["Duration (days)"].apply(
            lambda x: f"{int(x):,d}" if pd.notna(x) else "—"
        )
        display_dd.index = range(1, len(display_dd) + 1)
        st.dataframe(display_dd, use_container_width=True)

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:1.2rem 0 0.3rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Drawdown Summary</h3>",
        unsafe_allow_html=True,
    )

    max_dd_val = float(port_dd.min())
    uw_mask    = port_dd < -0.01
    avg_dd_val = float(port_dd[uw_mask].mean()) if uw_mask.any() else 0.0
    pct_uw     = float(uw_mask.mean()) * 100.0
    avg_rec    = (
        float(worst_dd["Duration (days)"].mean())
        if not worst_dd.empty and worst_dd["Duration (days)"].notna().any()
        else float("nan")
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Max Drawdown", f"{max_dd_val:.1f}%")
    with c2:
        st.metric("Avg Drawdown (when underwater)", f"{avg_dd_val:.1f}%")
    with c3:
        st.metric(
            "Avg Recovery (days)",
            f"{avg_rec:.0f}" if not math.isnan(avg_rec) else "—",
        )
    with c4:
        st.metric("% Time Underwater", f"{pct_uw:.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Stress Testing
# ════════════════════════════════════════════════════════════════════════════
with tab_stress:

    # ── Historical scenarios ──────────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:0 0 0.2rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Historical Scenario Analysis</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Current portfolio weights applied to each holding's actual return during the scenario period. "
        "Portfolio weights are rescaled proportionally for any tickers without price history in that window."
    )

    holdings_tuple = tuple(sorted(holdings.items()))
    scenario_df    = _load_scenarios(holdings_tuple, benchmark_ticker)

    def _fmt_pct(val: object) -> str:
        if isinstance(val, float) and not math.isnan(val):
            return f"{val:+.1%}"
        return "—"

    def _color_relative(val: object) -> str:
        if isinstance(val, float) and not math.isnan(val):
            return f"color: {GREEN}" if val >= 0 else f"color: {RED}"
        return ""

    sc_styler = (
        scenario_df.style
        .format({"Portfolio": _fmt_pct, "Benchmark": _fmt_pct, "Relative": _fmt_pct})
        .applymap(_color_relative, subset=["Relative"])
    )
    st.dataframe(sc_styler, use_container_width=True, hide_index=True)

    st.markdown(
        "<p style='font-size:0.75rem;color:#636b78;margin:0.4rem 0 0 0;"
        "font-family:\"DM Sans\",sans-serif;'>"
        "⚠ These are linear approximations based on historical period returns applied to current weights. "
        "Actual outcomes will differ due to composition changes, non-linearities, and market microstructure."
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Sensitivity sliders ───────────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:0 0 0.2rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Sensitivity Analysis</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Estimated portfolio impact using asset-class duration and beta proxies. "
        "Tickers not in the lookup table are treated as equity (beta = 1)."
    )

    col_eq, col_yd, col_sp = st.columns(3)
    with col_eq:
        equity_shock_pct = st.slider(
            "Equity market shock (%)",
            min_value=-30,
            max_value=0,
            value=-20,
            step=5,
            format="%d%%",
        )
    with col_yd:
        yield_shock_bps = st.slider(
            "Interest rate shock (bps, +ve = rising)",
            min_value=0,
            max_value=300,
            value=100,
            step=25,
        )
    with col_sp:
        spread_shock_bps = st.slider(
            "Credit spread widening (bps)",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
        )

    # holdings are stored as decimal fractions (0-1); the function expects 0-100 scale
    holdings_pct = {t: w * 100.0 for t, w in holdings.items()}
    sensitivity  = compute_sensitivity_impact(
        holdings_pct,
        equity_shock=equity_shock_pct / 100.0,
        yield_shock_bps=float(yield_shock_bps),
        spread_shock_bps=float(spread_shock_bps),
    )
    total_impact = float(sensitivity["total"])
    breakdown    = sensitivity["breakdown"]
    classified   = sensitivity["classified_as"]

    # ── Total impact card ─────────────────────────────────────────────────
    impact_color = GREEN if total_impact >= 0 else RED
    sign_label   = "gain" if total_impact >= 0 else "loss"
    st.markdown(
        f"<div style='background:#1a1e2a;border:1px solid #252836;border-radius:8px;"
        f"padding:1rem 1.2rem;margin:0.8rem 0;display:inline-block;min-width:220px;'>"
        f"<p style='font-size:0.70rem;color:#636b78;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:1.2px;margin:0 0 0.25rem 0;"
        f"font-family:\"DM Sans\",sans-serif;'>Estimated Portfolio Impact</p>"
        f"<p style='font-size:2.0rem;font-weight:700;color:{impact_color};"
        f"font-family:\"JetBrains Mono\",monospace;margin:0;line-height:1.1;'>"
        f"{total_impact:+.2%}"
        f"</p>"
        f"<p style='font-size:0.72rem;color:#636b78;margin:0.25rem 0 0 0;"
        f"font-family:\"DM Sans\",sans-serif;'>Estimated {sign_label} · linear approximation only</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Per-ticker breakdown (collapsible) ────────────────────────────────
    if breakdown:
        rows = [
            {
                "Ticker": t,
                "Weight": holdings.get(t, 0.0),  # already 0-1 decimal
                "Asset Class": classified.get(t, "Equity"),
                "Estimated Impact": v,
            }
            for t, v in sorted(breakdown.items(), key=lambda x: x[1])
        ]
        bd_df = pd.DataFrame(rows)
        bd_styler = (
            bd_df.style
            .format({"Weight": "{:.1%}", "Estimated Impact": "{:+.2%}"})
            .applymap(
                lambda v: f"color: {GREEN}" if isinstance(v, float) and v >= 0
                else f"color: {RED}",
                subset=["Estimated Impact"],
            )
        )
        with st.expander("Per-holding breakdown", expanded=False):
            st.dataframe(bd_styler, use_container_width=True, hide_index=True)

    st.markdown(
        "<p style='font-size:0.75rem;color:#636b78;margin:0.6rem 0 0 0;"
        "font-family:\"DM Sans\",sans-serif;'>"
        "⚠ These are linear approximations based on historical period returns applied to current weights. "
        "Actual outcomes will differ due to non-linearities, correlation changes, and portfolio dynamics "
        "not captured here."
        "</p>",
        unsafe_allow_html=True,
    )
