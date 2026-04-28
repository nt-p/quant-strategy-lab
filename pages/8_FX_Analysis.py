"""Page 8: FX Analysis — G10 carry, momentum, and PPP value.

Independent page — no portfolio required.

Tabs:
  1. Carry     — Lustig, Roussanov & Verdelhan (2011), RFS
  2. Momentum  — Moskowitz, Ooi & Pedersen (2012), JFE (TSMOM applied to FX)
  3. Value     — PPP deviation (Rogoff, 1996)
  4. Combined  — Equal-weight blend of all three

Data: yfinance (spot) + FRED (interest rates).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FX Analysis", page_icon="💱", layout="wide")

# Load .env before any module imports that need FRED key
try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from modules.fx import (  # noqa: E402
    G10_PAIRS,
    backtest_metrics,
    compute_carry_signals,
    compute_momentum_signals,
    compute_monthly_returns,
    compute_ppp_signals,
    fetch_fx_prices,
    fetch_fx_rates,
    run_carry_backtest,
    run_combined_backtest,
    run_momentum_backtest,
    run_value_backtest,
    vix_series,
)
from modules.macro import get_fred_api_key  # noqa: E402
from ui.sidebar import render_portfolio_sidebar  # noqa: E402
from ui.theme import AXIS_STYLE, CHART_LAYOUT, FONT, GRID, inject_css  # noqa: E402

inject_css()

# ── Colour palette ───────────────────────────────────────────────────────────
TEAL   = "#4ecdc4"
CORAL  = "#f56565"
AMBER  = "#f6ad55"
PURPLE = "#9f7aea"
BLUE   = "#63b3ed"
GREEN  = "#68d391"
GREY   = "#636b78"
GREY_L = "#a0a8b8"

BG      = "#0f1116"
PLOT_BG = "#161b27"

CCY_COLORS = {
    "AUD": TEAL,
    "GBP": BLUE,
    "EUR": AMBER,
    "JPY": CORAL,
    "CAD": GREEN,
    "CHF": PURPLE,
    "NZD": "#81e6d9",
    "NOK": "#d4a853",
    "SEK": "#b794f4",
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
render_portfolio_sidebar()

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='padding:0.5rem 0 0.15rem 0;'>
        <h1 style='font-size:1.9rem;font-weight:700;margin:0;letter-spacing:-0.5px;
                   background:linear-gradient(90deg,#4ecdc4 0%,#63b3ed 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-family:"DM Sans",sans-serif;'>
            FX Analysis
        </h1>
        <p style='color:#636b78;font-size:0.70rem;margin:0.25rem 0 0 0;
                  letter-spacing:2.5px;font-weight:600;
                  font-family:"DM Sans",sans-serif;'>
            G10 CARRY &nbsp;·&nbsp; MOMENTUM &nbsp;·&nbsp; PPP VALUE
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<hr style='margin:0.5rem 0 1.2rem 0;'>", unsafe_allow_html=True)

# ── Date range selector ───────────────────────────────────────────────────────
col_date1, col_date2, _ = st.columns([1, 1, 4])
with col_date1:
    start_date = st.selectbox(
        "History",
        ["2000-01-01", "2005-01-01", "2010-01-01", "2015-01-01"],
        index=0,
        label_visibility="visible",
    )
with col_date2:
    st.write("")  # spacer

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices(start: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fetch_fx_prices(start=start)


@st.cache_data(ttl=7200, show_spinner=False)
def _load_rates(start: str) -> pd.DataFrame:
    return fetch_fx_rates(start=start)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_vix(start: str) -> pd.Series:
    return vix_series(start=start)


with st.spinner("Loading G10 spot prices…"):
    prices = _load_prices(start_date)

if prices.empty:
    st.error("Could not load FX price data from yfinance. Please check your internet connection.")
    st.stop()

fred_key = get_fred_api_key()
rates_available = fred_key is not None

if rates_available:
    with st.spinner("Loading FRED interest rates…"):
        rates = _load_rates(start_date)
else:
    rates = pd.DataFrame()

monthly_rets = compute_monthly_returns(prices)
available_ccys = [c for c in prices.columns if c in G10_PAIRS]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_carry, tab_mom, tab_ppp, tab_combined = st.tabs(
    ["Carry", "Momentum (TSMOM)", "Value (PPP)", "Combined"]
)


# ── Helper: cumulative equity curve ──────────────────────────────────────────
def _equity_curve(rets: pd.Series, label: str, color: str) -> go.Scatter:
    cum = (1 + rets).cumprod()
    return go.Scatter(
        x=cum.index,
        y=cum.values,
        name=label,
        line=dict(color=color, width=2),
    )


def _metrics_table(metrics_dict: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in metrics_dict.items():
        if not m:
            continue
        rows.append({
            "Strategy":   name,
            "Ann. Return": f"{m['ann_return']*100:.1f}%",
            "Ann. Vol":    f"{m['ann_vol']*100:.1f}%",
            "Sharpe":      f"{m['sharpe']:.2f}",
            "Max DD":      f"{m['max_dd']*100:.1f}%",
            "Hit Rate":    f"{m['hit_rate']*100:.0f}%",
        })
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: CARRY
# ════════════════════════════════════════════════════════════════════════════
with tab_carry:
    st.markdown("#### Carry Trade — Long High-Yield, Short Low-Yield Currencies")
    st.caption(
        "Lustig, H., Roussanov, N. & Verdelhan, A. (2011). Common Risk Factors in "
        "Currency Markets. *Review of Financial Studies*, 24(11), 3731–3777."
    )

    if not rates_available:
        st.warning(
            "FRED_API_KEY not set — interest rate differentials unavailable. "
            "Add your key to `.env` to enable carry signals. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    elif rates.empty:
        st.warning("FRED rate data could not be loaded. Check your API key.")
    else:
        # Current carry snapshot ─────────────────────────────────────────────
        month_ends = monthly_rets.index
        carry = compute_carry_signals(rates, month_ends)

        if not carry.empty:
            latest_carry = carry.iloc[-1].dropna().sort_values(ascending=False)
            usd_rate = rates["USD"].reindex(carry.index, method="ffill").iloc[-1] if "USD" in rates.columns else np.nan

            st.markdown(f"**USD rate (Fed Funds): {usd_rate:.2f}%** — carry differentials below")

            # Bar chart: carry differentials
            colors_bar = [TEAL if v >= 0 else CORAL for v in latest_carry.values]
            fig_carry_bar = go.Figure(
                go.Bar(
                    x=[G10_PAIRS[c]["name"] if c in G10_PAIRS else c for c in latest_carry.index],
                    y=latest_carry.values,
                    marker_color=colors_bar,
                    text=[f"{v:+.2f}%" for v in latest_carry.values],
                    textposition="outside",
                )
            )
            fig_carry_bar.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Current Carry Differential vs USD (% p.a.)", font=dict(size=13, color=FONT)),
                yaxis_title="Rate Differential (pp)",
                height=300,
                showlegend=False,
                margin=dict(t=50, b=40, l=50, r=20),
            )
            fig_carry_bar.update_xaxes(**AXIS_STYLE)
            fig_carry_bar.update_yaxes(**{**AXIS_STYLE, "zeroline": True, "zerolinecolor": GRID, "zerolinewidth": 1})
            st.plotly_chart(fig_carry_bar, use_container_width=True)

            # Signal table
            col_sig, col_space = st.columns([2, 2])
            with col_sig:
                n_sorted = latest_carry.sort_values(ascending=False)
                longs  = n_sorted.index[:3].tolist()
                shorts = n_sorted.index[-3:].tolist()
                signal_df = pd.DataFrame({
                    "Currency": [G10_PAIRS.get(c, {}).get("name", c) for c in n_sorted.index],
                    "Carry (pp)": [f"{v:+.2f}%" for v in n_sorted.values],
                    "Signal": ["🟢 Long" if c in longs else "🔴 Short" if c in shorts else "— Neutral"
                               for c in n_sorted.index],
                })
                st.dataframe(signal_df, use_container_width=True, hide_index=True)

        # Backtest equity curve ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### Carry Strategy Backtest")

        carry_rets = run_carry_backtest(monthly_rets, carry)

        if not carry_rets.dropna().empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(_equity_curve(carry_rets.dropna(), "Carry (Long–Short)", TEAL))
            fig_eq.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Cumulative Return — Carry Strategy (Long Top 3, Short Bottom 3)", font=dict(size=13, color=FONT)),
                yaxis_title="Growth of $1",
                height=320,
                margin=dict(t=50, b=40, l=60, r=20),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT, size=11)),
            )
            fig_eq.update_xaxes(**AXIS_STYLE)
            fig_eq.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_eq, use_container_width=True)

            # Carry vs VIX ───────────────────────────────────────────────────
            vix = _load_vix(start_date)
            if not vix.empty:
                st.markdown("##### Carry Return vs VIX — Carry Crash Dynamic")
                st.caption(
                    "High-carry currencies collapse when risk appetite drops sharply. "
                    "The negative correlation with VIX spikes is the signature of carry crash "
                    "(Brunnermeier, Nagel & Pedersen, 2009)."
                )
                common_idx = carry_rets.dropna().index.intersection(vix.index)
                if len(common_idx) > 0:
                    fig_vix = go.Figure()
                    fig_vix.add_trace(go.Scatter(
                        x=common_idx,
                        y=carry_rets.loc[common_idx].cumsum() * 100,
                        name="Carry cumsum (%)",
                        line=dict(color=TEAL, width=2),
                        yaxis="y1",
                    ))
                    fig_vix.add_trace(go.Scatter(
                        x=vix.loc[common_idx].index,
                        y=vix.loc[common_idx].values,
                        name="VIX",
                        line=dict(color=CORAL, width=1.5, dash="dot"),
                        yaxis="y2",
                        opacity=0.8,
                    ))
                    fig_vix.update_layout(
                        **CHART_LAYOUT,
                        height=300,
                        margin=dict(t=40, b=40, l=60, r=60),
                        yaxis=dict(title="Carry (cum. %)", **AXIS_STYLE),
                        yaxis2=dict(
                            title="VIX",
                            overlaying="y",
                            side="right",
                            **AXIS_STYLE,
                            showgrid=False,
                        ),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT, size=11)),
                    )
                    fig_vix.update_xaxes(**AXIS_STYLE)
                    st.plotly_chart(fig_vix, use_container_width=True)

            # Metrics
            m = backtest_metrics(carry_rets)
            if m:
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Ann. Return", f"{m['ann_return']*100:.1f}%")
                mc2.metric("Ann. Vol",    f"{m['ann_vol']*100:.1f}%")
                mc3.metric("Sharpe",      f"{m['sharpe']:.2f}")
                mc4.metric("Max Drawdown", f"{m['max_dd']*100:.1f}%")
                mc5.metric("Hit Rate",     f"{m['hit_rate']*100:.0f}%")

    with st.expander("DOL and HML carry factors"):
        st.markdown(
            """
            Lustig et al. (2011) decompose carry returns into two risk factors:

            - **DOL (Dollar factor)** — average return across all G10 currencies vs USD.
              Captures broad USD risk; highly correlated with global equity beta.
            - **HML (High-minus-Low carry)** — return on the carry trade itself
              (long high-yield, short low-yield). Compensates for *crash risk* —
              the tendency of carry to unwind sharply in crises.

            A positive Sharpe on HML does not imply free money: it reflects a premium
            for bearing the risk of sudden, large drawdowns during global deleveraging events
            (2008, 2015 CNY revaluation, 2020 COVID). The carry trade is metaphorically
            "picking up nickels in front of a steamroller."
            """
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: MOMENTUM (TSMOM)
# ════════════════════════════════════════════════════════════════════════════
with tab_mom:
    st.markdown("#### Time-Series Momentum (TSMOM) — FX")
    st.caption(
        "Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). Time Series Momentum. "
        "*Journal of Financial Economics*, 104(2), 228–250."
    )
    st.markdown(
        "**Signal:** for each currency, go long if the prior 12-month return is positive, "
        "short if negative. Scale position size by target volatility / realised volatility "
        "so each pair contributes equal risk. Target vol = 10% p.a."
    )

    signals, _ = compute_momentum_signals(prices)
    mom_rets = run_momentum_backtest(prices)

    # Current signals ────────────────────────────────────────────────────────
    if not signals.empty:
        latest_sig = signals.iloc[-1].dropna().sort_values(ascending=False)
        latest_rets_12m = (
            prices.resample("ME").last()
            .pct_change(13)
            .shift(1)
            .iloc[-1]
            .dropna()
        )

        sig_df = pd.DataFrame({
            "Currency": [G10_PAIRS.get(c, {}).get("name", c) for c in latest_sig.index],
            "12m Return": [
                f"{latest_rets_12m.get(c, np.nan)*100:.1f}%"
                if not np.isnan(latest_rets_12m.get(c, np.nan)) else "—"
                for c in latest_sig.index
            ],
            "Position Size": [f"{v:+.2f}×" for v in latest_sig.values],
            "Signal": ["🟢 Long" if v > 0 else "🔴 Short" for v in latest_sig.values],
        })

        col_m1, col_m2 = st.columns([2, 3])
        with col_m1:
            st.markdown("**Current TSMOM Signals**")
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        with col_m2:
            # Position size bar
            bar_vals = latest_sig.values
            bar_cols = [TEAL if v >= 0 else CORAL for v in bar_vals]
            bar_names = [G10_PAIRS.get(c, {}).get("name", c) for c in latest_sig.index]
            fig_sig = go.Figure(go.Bar(
                x=bar_names,
                y=bar_vals,
                marker_color=bar_cols,
                text=[f"{v:+.2f}×" for v in bar_vals],
                textposition="outside",
            ))
            fig_sig.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Position Sizes (vol-scaled)", font=dict(size=13, color=FONT)),
                yaxis_title="Position (×)",
                height=280,
                showlegend=False,
                margin=dict(t=50, b=40, l=50, r=20),
            )
            fig_sig.update_xaxes(**AXIS_STYLE)
            fig_sig.update_yaxes(**{**AXIS_STYLE, "zeroline": True, "zerolinecolor": GRID, "zerolinewidth": 1})
            st.plotly_chart(fig_sig, use_container_width=True)

    # Backtest ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### TSMOM Backtest — All G10 FX")

    if not mom_rets.empty:
        fig_mom = go.Figure()
        fig_mom.add_trace(_equity_curve(mom_rets, "TSMOM FX", TEAL))
        fig_mom.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Cumulative Return — FX Time-Series Momentum", font=dict(size=13, color=FONT)),
            yaxis_title="Growth of $1",
            height=320,
            margin=dict(t=50, b=40, l=60, r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT, size=11)),
        )
        fig_mom.update_xaxes(**AXIS_STYLE)
        fig_mom.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_mom, use_container_width=True)

        m = backtest_metrics(mom_rets)
        if m:
            mm1, mm2, mm3, mm4, mm5 = st.columns(5)
            mm1.metric("Ann. Return",  f"{m['ann_return']*100:.1f}%")
            mm2.metric("Ann. Vol",     f"{m['ann_vol']*100:.1f}%")
            mm3.metric("Sharpe",       f"{m['sharpe']:.2f}")
            mm4.metric("Max Drawdown", f"{m['max_dd']*100:.1f}%")
            mm5.metric("Hit Rate",     f"{m['hit_rate']*100:.0f}%")

    # Per-currency contribution ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Per-Currency Return Contribution")
    sig_lag, monthly_rets_ccy = compute_momentum_signals(prices)
    ccy_contrib = (sig_lag.shift(1) * monthly_rets_ccy).mean() * 12  # annualised

    ccy_contrib = ccy_contrib.dropna().sort_values(ascending=False)
    if not ccy_contrib.empty:
        fig_contrib = go.Figure(go.Bar(
            x=[G10_PAIRS.get(c, {}).get("name", c) for c in ccy_contrib.index],
            y=ccy_contrib.values * 100,
            marker_color=[TEAL if v >= 0 else CORAL for v in ccy_contrib.values],
            text=[f"{v*100:.2f}%" for v in ccy_contrib.values],
            textposition="outside",
        ))
        fig_contrib.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Average Annual Contribution per Currency Pair", font=dict(size=13, color=FONT)),
            yaxis_title="Contribution (% p.a.)",
            height=280,
            showlegend=False,
            margin=dict(t=50, b=40, l=50, r=20),
        )
        fig_contrib.update_xaxes(**AXIS_STYLE)
        fig_contrib.update_yaxes(**{**AXIS_STYLE, "zeroline": True, "zerolinecolor": GRID, "zerolinewidth": 1})
        st.plotly_chart(fig_contrib, use_container_width=True)

    with st.expander("How vol-scaling works"):
        st.markdown(
            r"""
            The raw signal is $\text{sign}(r_{t-12,t-1})$ — just +1 or −1.
            Vol-scaling converts this into a *risk-adjusted* position:

            $$w_t = \text{sign}(r_{t-12,t-1}) \times \frac{\sigma^*}{\hat{\sigma}_t}$$

            where $\sigma^* = 10\%$ is the target volatility and $\hat{\sigma}_t$ is the
            60-day realised daily volatility (annualised). Position size is capped at 2×
            to prevent excessive leverage in low-volatility regimes.

            This means the strategy always targets roughly 10% annualised vol regardless
            of which pair or regime — making Sharpe ratios comparable across pairs and time.
            """
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: VALUE (PPP)
# ════════════════════════════════════════════════════════════════════════════
with tab_ppp:
    st.markdown("#### PPP Value — Purchasing Power Parity Deviation")
    st.caption(
        "Rogoff, K. (1996). The Purchasing Power Parity Puzzle. "
        "*Journal of Economic Literature*, 34(2), 647–668."
    )
    st.info(
        "PPP fair values are OECD-based estimates embedded at module level "
        "(AUD≈0.74, GBP≈1.38, EUR≈1.22, JPY≈0.0092, CAD≈0.82, CHF≈0.89, NZD≈0.67). "
        "These are long-run anchors, not near-term forecasts — PPP deviations can "
        "persist for years before mean-reverting.",
        icon="ℹ️",
    )

    ppp_dev = compute_ppp_signals(prices)

    if not ppp_dev.empty:
        # Current PPP snapshot ───────────────────────────────────────────────
        latest_ppp = ppp_dev.iloc[-1].dropna().sort_values()
        latest_spot = prices.iloc[-1].dropna()

        ppp_df = pd.DataFrame({
            "Currency": [G10_PAIRS.get(c, {}).get("name", c) for c in latest_ppp.index],
            "Spot (FX/USD)": [
                f"{latest_spot.get(c, np.nan):.4f}" if c in latest_spot else "—"
                for c in latest_ppp.index
            ],
            "PPP Fair Value": [
                f"{G10_PAIRS.get(c, {}).get('ppp', np.nan):.4f}" if c in G10_PAIRS else "—"
                for c in latest_ppp.index
            ],
            "Deviation": [f"{v*100:+.1f}%" for v in latest_ppp.values],
            "Signal": [
                "🟢 Undervalued (buy)" if v < -0.10 else
                "🔴 Overvalued (sell)" if v > 0.10 else
                "— Near fair value"
                for v in latest_ppp.values
            ],
        })

        col_p1, col_p2 = st.columns([2, 3])
        with col_p1:
            st.markdown("**Current PPP Deviation**")
            st.dataframe(ppp_df, use_container_width=True, hide_index=True)
        with col_p2:
            bar_col = [TEAL if v < 0 else CORAL for v in latest_ppp.values]
            fig_ppp = go.Figure(go.Bar(
                x=[G10_PAIRS.get(c, {}).get("name", c) for c in latest_ppp.index],
                y=latest_ppp.values * 100,
                marker_color=bar_col,
                text=[f"{v*100:+.1f}%" for v in latest_ppp.values],
                textposition="outside",
            ))
            fig_ppp.add_hline(y=0, line=dict(color=GREY_L, width=1, dash="dot"))
            fig_ppp.add_hrect(y0=-10, y1=10, fillcolor="rgba(100,100,100,0.07)", line_width=0)
            fig_ppp.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Spot vs PPP Fair Value — % Deviation", font=dict(size=13, color=FONT)),
                yaxis_title="Deviation from PPP (%)",
                height=300,
                showlegend=False,
                margin=dict(t=50, b=40, l=50, r=20),
                annotations=[dict(
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    text="±10% band", font=dict(color=GREY, size=10),
                    showarrow=False, opacity=0.6,
                )],
            )
            fig_ppp.update_xaxes(**AXIS_STYLE)
            fig_ppp.update_yaxes(**{**AXIS_STYLE, "zeroline": True, "zerolinecolor": GRID, "zerolinewidth": 1})
            st.plotly_chart(fig_ppp, use_container_width=True)

        # Historical deviation chart ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### Historical PPP Deviation Over Time")
        ccy_to_plot = st.multiselect(
            "Select currencies",
            options=ppp_dev.columns.tolist(),
            default=ppp_dev.columns.tolist()[:4],
            format_func=lambda c: G10_PAIRS.get(c, {}).get("name", c),
        )
        if ccy_to_plot:
            fig_hist = go.Figure()
            for ccy in ccy_to_plot:
                if ccy not in ppp_dev.columns:
                    continue
                fig_hist.add_trace(go.Scatter(
                    x=ppp_dev.index,
                    y=ppp_dev[ccy] * 100,
                    name=G10_PAIRS.get(ccy, {}).get("name", ccy),
                    line=dict(color=CCY_COLORS.get(ccy, GREY_L), width=1.8),
                ))
            fig_hist.add_hline(y=0, line=dict(color=GREY_L, width=1, dash="dot"))
            fig_hist.update_layout(
                **CHART_LAYOUT,
                title=dict(text="PPP Deviation History (% above/below fair value)", font=dict(size=13, color=FONT)),
                yaxis_title="Deviation (%)",
                height=360,
                margin=dict(t=50, b=40, l=60, r=20),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT, size=11)),
            )
            fig_hist.update_xaxes(**AXIS_STYLE)
            fig_hist.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Value backtest ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### PPP Value Strategy Backtest")
        val_rets = run_value_backtest(prices)
        if not val_rets.empty:
            fig_val = go.Figure()
            fig_val.add_trace(_equity_curve(val_rets, "PPP Value (Long–Short)", AMBER))
            fig_val.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Cumulative Return — PPP Value Strategy", font=dict(size=13, color=FONT)),
                yaxis_title="Growth of $1",
                height=300,
                margin=dict(t=50, b=40, l=60, r=20),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT, size=11)),
            )
            fig_val.update_xaxes(**AXIS_STYLE)
            fig_val.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_val, use_container_width=True)

            m = backtest_metrics(val_rets)
            if m:
                mv1, mv2, mv3, mv4, mv5 = st.columns(5)
                mv1.metric("Ann. Return",  f"{m['ann_return']*100:.1f}%")
                mv2.metric("Ann. Vol",     f"{m['ann_vol']*100:.1f}%")
                mv3.metric("Sharpe",       f"{m['sharpe']:.2f}")
                mv4.metric("Max Drawdown", f"{m['max_dd']*100:.1f}%")
                mv5.metric("Hit Rate",     f"{m['hit_rate']*100:.0f}%")

        st.caption(
            "PPP is a very long-run anchor. Deviations can persist 5–10 years. "
            "The strategy is most useful as a contrarian signal at extremes (>30% deviation), "
            "not as a high-frequency trading signal."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: COMBINED
# ════════════════════════════════════════════════════════════════════════════
with tab_combined:
    st.markdown("#### Combined FX Strategy — Equal-Weight Blend")
    st.markdown(
        "An equal-weight average of carry, momentum, and PPP value returns. "
        "Blending reduces drawdowns by combining strategies with low mutual correlation — "
        "carry suffers in crises, momentum thrives in trending regimes, "
        "PPP value is a slow mean-reversion anchor."
    )

    mom_rets_c  = run_momentum_backtest(prices)
    val_rets_c  = run_value_backtest(prices)

    if rates_available and not rates.empty:
        carry_c = compute_carry_signals(rates, monthly_rets.index)
        carry_rets_c = run_carry_backtest(monthly_rets, carry_c)
    else:
        carry_rets_c = pd.Series(dtype=float, name="carry")

    combined_rets = run_combined_backtest(carry_rets_c, mom_rets_c, val_rets_c)

    if not combined_rets.empty:
        # Equity curves ──────────────────────────────────────────────────────
        fig_all = go.Figure()

        for rets, label, color in [
            (carry_rets_c,  "Carry",    TEAL),
            (mom_rets_c,    "Momentum", BLUE),
            (val_rets_c,    "Value",    AMBER),
            (combined_rets, "Combined", GREEN),
        ]:
            if not rets.empty and not rets.dropna().empty:
                fig_all.add_trace(_equity_curve(rets.dropna(), label, color))

        fig_all.update_layout(
            **CHART_LAYOUT,
            title=dict(
                text="Cumulative Return — Carry / Momentum / Value / Combined",
                font=dict(size=13, color=FONT),
            ),
            yaxis_title="Growth of $1",
            height=380,
            margin=dict(t=50, b=40, l=60, r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT, size=11)),
        )
        fig_all.update_xaxes(**AXIS_STYLE)
        fig_all.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_all, use_container_width=True)

        # Metrics comparison table ───────────────────────────────────────────
        st.markdown("##### Performance Comparison")
        metrics_dict: dict[str, dict] = {}
        for rets, label in [
            (carry_rets_c,  "Carry"),
            (mom_rets_c,    "Momentum (TSMOM)"),
            (val_rets_c,    "Value (PPP)"),
            (combined_rets, "Combined"),
        ]:
            if not rets.empty:
                metrics_dict[label] = backtest_metrics(rets.dropna())

        tbl = _metrics_table(metrics_dict)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True)

        # Correlation matrix ─────────────────────────────────────────────────
        st.markdown("##### Return Correlation Matrix")

        corr_df = pd.concat([
            s.rename(n) for s, n in [
                (carry_rets_c,  "Carry"),
                (mom_rets_c,    "Momentum"),
                (val_rets_c,    "Value"),
            ] if not s.empty
        ], axis=1).dropna()

        if not corr_df.empty and corr_df.shape[1] > 1:
            corr = corr_df.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=[[0, CORAL], [0.5, PLOT_BG], [1, TEAL]],
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=14, color="#f0f2f5"),
                showscale=True,
            ))
            fig_corr.update_layout(
                **CHART_LAYOUT,
                title=dict(text="Pairwise Return Correlations (monthly)", font=dict(size=13, color=FONT)),
                height=280,
                margin=dict(t=50, b=40, l=80, r=60),
            )
            fig_corr.update_xaxes(tickfont=dict(color=FONT, size=11), showgrid=False)
            fig_corr.update_yaxes(tickfont=dict(color=FONT, size=11), showgrid=False)
            st.plotly_chart(fig_corr, use_container_width=True)

            st.caption(
                "Low correlation between carry and momentum is the key diversification benefit. "
                "Carry earns a risk premium in calm markets; momentum captures trending regimes. "
                "Blending both reduces drawdown without proportionally reducing return."
            )
    else:
        st.warning(
            "Combined strategy requires at least momentum and value data to compute. "
            "Carry will be included automatically once FRED_API_KEY is configured."
        )

    with st.expander("Academic references"):
        st.markdown(
            """
            **Carry**
            Lustig, H., Roussanov, N. & Verdelhan, A. (2011). Common Risk Factors in Currency Markets.
            *Review of Financial Studies*, 24(11), 3731–3777. DOI: 10.1093/rfs/hhq138

            **Momentum**
            Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). Time Series Momentum.
            *Journal of Financial Economics*, 104(2), 228–250. DOI: 10.1016/j.jfineco.2011.11.003

            **Value (PPP)**
            Rogoff, K. (1996). The Purchasing Power Parity Puzzle.
            *Journal of Economic Literature*, 34(2), 647–668.

            **Carry crash risk**
            Brunnermeier, M.K., Nagel, S. & Pedersen, L.H. (2009). Carry Trades and Currency Crashes.
            *NBER Macroeconomics Annual*, 23, 313–347. DOI: 10.1086/648701
            """
        )
