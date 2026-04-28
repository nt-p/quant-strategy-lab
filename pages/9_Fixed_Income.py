"""Page 9: Fixed Income — Yield curve, duration/convexity, credit spreads.

Independent page — no portfolio required.

Tabs:
  1. Yield Curve          — interactive curve with 1Y comparison, roll-down
  2. Duration & Convexity — bond calculator with price shock table
  3. Credit Spreads       — HY/IG OAS z-scores, Cochrane-Piazzesi factor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fixed Income", page_icon="📈", layout="wide")

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from modules.fixed_income import (  # noqa: E402
    TREASURY_TICKERS,
    TENOR_YEARS,
    bond_price,
    cochrane_piazzesi_factor,
    convexity,
    credit_zscore_table,
    curve_one_year_ago,
    fetch_credit_data,
    fetch_yield_curve,
    latest_curve,
    macaulay_duration,
    modified_duration,
    price_change_table,
    price_yield_curve_data,
    roll_down,
    spread_10y2y,
    zscore_series,
)
from modules.macro import get_fred_api_key  # noqa: E402
from ui.sidebar import render_portfolio_sidebar  # noqa: E402
from ui.theme import AXIS_STYLE, CHART_LAYOUT, FONT, GRID, apply_theme_to_plotly_figure, inject_css  # noqa: E402

inject_css()

# ── Colour palette ───────────────────────────────────────────────────────────
TEAL   = "#4ecdc4"
CORAL  = "#f56565"
AMBER  = "#f6ad55"
GREY   = "#636b78"
BLUE   = "#63b3ed"
PURPLE = "#9f7aea"
GREEN  = "#68d391"
BG     = "#0f1116"

# ── Sidebar ──────────────────────────────────────────────────────────────────
render_portfolio_sidebar()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Fixed Income")
st.caption(
    "Treasury yield curve dynamics, bond duration and convexity, "
    "and credit spread regime signals."
)

# ── FRED key check ───────────────────────────────────────────────────────────
api_key = get_fred_api_key()
if not api_key:
    st.error(
        "FRED API key not found. Add `FRED_API_KEY=your_key` to your `.env` file. "
        "Free key: https://fred.stlouisfed.org/docs/api/api_key.html"
    )
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Yield Curve", "Duration & Convexity", "Credit Spreads"])


# ═══════════════════════════════════════════════════════════════════════════ #
#  TAB 1 — YIELD CURVE
# ═══════════════════════════════════════════════════════════════════════════ #
with tab1:
    with st.spinner("Fetching Treasury yields…"):
        try:
            yc_df = st.cache_data(fetch_yield_curve, ttl=3600)(start="2000-01-01")
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if yc_df.empty:
        st.warning("No yield curve data available.")
        st.stop()

    tickers_ordered = list(TREASURY_TICKERS.keys())
    labels_ordered  = list(TREASURY_TICKERS.values())
    tenors_ordered  = [TENOR_YEARS[t] for t in tickers_ordered]

    latest  = latest_curve(yc_df)
    ago1y   = curve_one_year_ago(yc_df)
    roll    = roll_down(yc_df, horizon_months=6)
    s10y2y  = spread_10y2y(yc_df).dropna()

    # ── KPI row ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    spread_now   = latest.get("DGS10", np.nan) - latest.get("DGS2", np.nan)
    spread_prev  = ago1y.get("DGS10", np.nan) - ago1y.get("DGS2", np.nan)
    k1.metric("2Y Yield",     f"{latest.get('DGS2',  np.nan):.2f}%",
              delta=f"{latest.get('DGS2', np.nan) - ago1y.get('DGS2', np.nan):+.2f}% vs 1Y ago")
    k2.metric("10Y Yield",    f"{latest.get('DGS10', np.nan):.2f}%",
              delta=f"{latest.get('DGS10', np.nan) - ago1y.get('DGS10', np.nan):+.2f}% vs 1Y ago")
    k3.metric("30Y Yield",    f"{latest.get('DGS30', np.nan):.2f}%",
              delta=f"{latest.get('DGS30', np.nan) - ago1y.get('DGS30', np.nan):+.2f}% vs 1Y ago")
    k4.metric("10Y-2Y Spread", f"{spread_now:+.2f}%",
              delta=f"{spread_now - spread_prev:+.2f}% vs 1Y ago",
              delta_color="off")

    st.markdown("---")

    # ── Yield curve snapshot ─────────────────────────────────────────────────
    curve_vals_now  = [latest.get(t, np.nan) for t in tickers_ordered]
    curve_vals_ago  = [ago1y.get(t, np.nan) for t in labels_ordered]
    # ago1y uses the raw FRED column names
    curve_vals_ago  = [ago1y.get(t, np.nan) for t in tickers_ordered]

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=labels_ordered, y=curve_vals_now,
        mode="lines+markers",
        name="Current",
        line=dict(color=TEAL, width=2.5),
        marker=dict(size=7),
    ))
    fig_curve.add_trace(go.Scatter(
        x=labels_ordered, y=curve_vals_ago,
        mode="lines+markers",
        name="1 Year Ago",
        line=dict(color=GREY, width=1.5, dash="dot"),
        marker=dict(size=5),
    ))
    fig_curve.update_layout(
        **CHART_LAYOUT,
        title="US Treasury Yield Curve",
        xaxis_title="Maturity",
        yaxis_title="Yield (%)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT)),
        height=380,
    )
    fig_curve.update_xaxes(**AXIS_STYLE)
    fig_curve.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_curve, use_container_width=True)

    # ── 10Y-2Y spread history ────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig_spread = go.Figure()

        # NBER recession shading
        try:
            rec_df = st.cache_data(
                lambda: __import__("modules.macro", fromlist=["fetch_series"]).fetch_series(
                    ["USREC"], start="2000-01-01"
                ),
                ttl=86400,
            )()
            if "USREC" in rec_df.columns:
                rec = rec_df["USREC"].reindex(s10y2y.index, method="ffill").fillna(0)
                in_rec = False
                for date, val in rec.items():
                    if val == 1 and not in_rec:
                        rec_start = date
                        in_rec = True
                    elif val == 0 and in_rec:
                        fig_spread.add_vrect(
                            x0=str(rec_start), x1=str(date),
                            fillcolor="rgba(216,90,48,0.12)", line_width=0,
                        )
                        in_rec = False
        except Exception:
            pass

        fig_spread.add_trace(go.Scatter(
            x=s10y2y.index, y=s10y2y.values,
            mode="lines",
            name="10Y-2Y Spread",
            line=dict(color=TEAL, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(78,205,196,0.08)",
        ))
        fig_spread.add_hline(y=0, line=dict(color=CORAL, width=1, dash="dot"))
        fig_spread.update_layout(
            **CHART_LAYOUT,
            title="10Y-2Y Yield Spread — Inversion as Recession Signal",
            xaxis_title="",
            yaxis_title="Spread (%)",
            height=320,
            showlegend=False,
        )
        fig_spread.update_xaxes(**AXIS_STYLE)
        fig_spread.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_spread, use_container_width=True)

    with col_r:
        st.markdown("**Roll-Down Estimates (6m horizon)**")
        st.caption(
            "Approximate return from riding the yield curve. "
            "A bond 'rolls down' to a lower-yielding maturity as time passes, "
            "generating a capital gain if the curve is upward-sloping."
        )
        roll_df = pd.DataFrame([
            {"Maturity": label, "Est. Roll-Down (%)": v}
            for label, v in roll.items()
            if not np.isnan(v)
        ])
        if not roll_df.empty:
            st.dataframe(
                roll_df.style.format({"Est. Roll-Down (%)": "{:+.3f}%"})
                    .background_gradient(
                        subset=["Est. Roll-Down (%)"],
                        cmap="RdYlGn",
                        vmin=-0.5, vmax=0.5,
                    ),
                use_container_width=True,
                height=310,
            )

    st.caption(
        "Source: FRED constant-maturity Treasury yields. "
        "Roll-down assumes an unchanged curve shape over the horizon. "
        "NBER recession shading for context."
    )


# ═══════════════════════════════════════════════════════════════════════════ #
#  TAB 2 — DURATION & CONVEXITY
# ═══════════════════════════════════════════════════════════════════════════ #
with tab2:
    st.markdown("#### Bond Duration & Convexity Calculator")
    st.caption(
        "Duration measures the sensitivity of a bond's price to yield changes — "
        "a first-order (linear) approximation. Convexity captures the curvature: "
        "the second derivative of price with respect to yield."
    )

    col_inputs, col_metrics = st.columns([1, 1])

    with col_inputs:
        face_value   = st.number_input("Face Value ($)", value=1000.0, step=100.0,
                                        min_value=100.0, max_value=1_000_000.0)
        coupon_pct   = st.slider("Annual Coupon Rate (%)", 0.0, 15.0, 5.0, 0.25)
        ytm_pct      = st.slider("Yield-to-Maturity (%)", 0.1, 20.0, 5.0, 0.1)
        maturity_yr  = st.slider("Years to Maturity", 0.5, 30.0, 10.0, 0.5)
        freq         = st.selectbox("Coupon Frequency", [1, 2, 4], index=1,
                                    format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}[x])

    coupon_rate = coupon_pct / 100
    ytm         = ytm_pct / 100

    px      = bond_price(face_value, coupon_rate, ytm, maturity_yr, freq)
    mac_dur = macaulay_duration(face_value, coupon_rate, ytm, maturity_yr, freq)
    mod_dur = modified_duration(face_value, coupon_rate, ytm, maturity_yr, freq)
    cx      = convexity(face_value, coupon_rate, ytm, maturity_yr, freq)
    dv01    = px * mod_dur / 10_000  # DV01 = $ change per 1bp

    with col_metrics:
        st.markdown("**Key Analytics**")
        m1, m2 = st.columns(2)
        m1.metric("Clean Price",          f"${px:,.2f}",
                  delta=f"{'Premium' if px > face_value else 'Discount' if px < face_value else 'Par'}")
        m2.metric("Macaulay Duration",    f"{mac_dur:.3f} yrs")
        m3, m4 = st.columns(2)
        m3.metric("Modified Duration",    f"{mod_dur:.3f}")
        m4.metric("Convexity",            f"{cx:.4f}")
        m5, m6 = st.columns(2)
        m5.metric("DV01",                 f"${dv01:.4f}",
                  help="Dollar value of 1 basis-point move in yield")
        m6.metric("Coupon ÷ YTM",
                  "At Par" if abs(coupon_rate - ytm) < 1e-4 else
                  f"{'> YTM → Premium' if coupon_rate > ytm else '< YTM → Discount'}",
                  delta=None)

        with st.expander("Formula reference"):
            st.latex(
                r"D_{mac} = \frac{\sum_{t=1}^{n} t \cdot PV(CF_t)}{P}, \quad"
                r"D_{mod} = \frac{D_{mac}}{1 + y/m}"
            )
            st.latex(
                r"\frac{\Delta P}{P} \approx -D_{mod} \cdot \Delta y"
                r"+ \frac{1}{2} \cdot C \cdot (\Delta y)^2"
            )
            st.caption(
                "Where C = convexity, y = YTM, m = coupon frequency per year."
            )

    st.markdown("---")
    st.markdown("**Price Sensitivity to Yield Shocks**")

    shock_df = price_change_table(face_value, coupon_rate, ytm, maturity_yr, freq)
    styled = shock_df.style.format({
        "New YTM (%)":         "{:.3f}",
        "New Price ($)":       "${:,.4f}",
        "$ Change":            "{:+,.4f}",
        "% Change":            "{:+.3f}%",
        "Duration Est. (%)":   "{:+.3f}%",
        "Convexity Adj. (%)":  "{:+.4f}%",
    }).background_gradient(
        subset=["% Change"],
        cmap="RdYlGn_r",
    )
    st.dataframe(styled, use_container_width=True, height=340)

    st.markdown("---")
    st.markdown("**Price-Yield Relationship**")

    ytms_pct, prices = price_yield_curve_data(
        face_value, coupon_rate, maturity_yr, ytm, freq
    )

    fig_py = go.Figure()
    fig_py.add_trace(go.Scatter(
        x=ytms_pct, y=prices,
        mode="lines",
        name="Price-Yield Curve",
        line=dict(color=TEAL, width=2.5),
    ))
    # Tangent (duration approximation)
    tangent_ys   = np.array([ytm_pct - 2, ytm_pct + 2])
    tangent_px   = np.array([
        px * (1 - mod_dur * (y / 100 - ytm)) for y in tangent_ys
    ])
    fig_py.add_trace(go.Scatter(
        x=tangent_ys, y=tangent_px,
        mode="lines",
        name="Duration Tangent",
        line=dict(color=AMBER, width=1.5, dash="dot"),
    ))
    fig_py.add_trace(go.Scatter(
        x=[ytm_pct], y=[px],
        mode="markers",
        name="Current",
        marker=dict(color=CORAL, size=10, symbol="circle"),
    ))
    fig_py.update_layout(
        **CHART_LAYOUT,
        title=f"Price vs Yield — {maturity_yr:.1f}Y Bond (Coupon {coupon_pct:.2f}%)",
        xaxis_title="Yield (%)",
        yaxis_title="Price ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT)),
        height=380,
    )
    fig_py.update_xaxes(**AXIS_STYLE)
    fig_py.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_py, use_container_width=True)

    st.caption(
        "The curved line shows the true (convex) price-yield relationship. "
        "The dotted tangent is the duration linear approximation. "
        "The gap between the two is the convexity benefit — "
        "bonds gain more than duration predicts when yields fall, "
        "and lose less than duration predicts when yields rise."
    )


# ═══════════════════════════════════════════════════════════════════════════ #
#  TAB 3 — CREDIT SPREADS
# ═══════════════════════════════════════════════════════════════════════════ #
with tab3:
    with st.spinner("Fetching credit spread data…"):
        try:
            credit_df = st.cache_data(fetch_credit_data, ttl=3600)(start="2000-01-01")
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if credit_df.empty:
        st.warning("No credit spread data available.")
        st.stop()

    # ── Current spread z-score table ────────────────────────────────────────
    st.markdown("#### Credit Spread Regime")
    st.caption(
        "OAS — Option-Adjusted Spread — measures the yield premium of corporate bonds "
        "over Treasuries after stripping out any embedded options. "
        "High z-scores → spreads are wide relative to history → potential value; "
        "low z-scores → spreads tight → historically low compensation for credit risk."
    )

    zscore_tbl = credit_zscore_table(credit_df)

    def _signal_color(signal: str) -> str:
        if "Wide" in signal:
            return "color: #68d391"
        if "Tight" in signal:
            return "color: #f56565"
        return "color: #f6ad55"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(
            zscore_tbl.style.applymap(
                _signal_color, subset=["Signal"]
            ).format({
                "Current":      "{:.1f}",
                "1Y Average":   "{:.1f}",
                "Z-Score (1Y)": "{:+.2f}",
            }),
            use_container_width=True,
            height=120,
        )

    with c2:
        if "BAMLH0A0HYM2" in credit_df.columns:
            hy_z = zscore_series(credit_df["BAMLH0A0HYM2"].dropna(), window=252)
            ig_z = zscore_series(credit_df["BAMLC0A0CM"].dropna(), window=252) \
                   if "BAMLC0A0CM" in credit_df.columns else None
            fig_z = go.Figure()
            fig_z.add_hline(y=1.5, line=dict(color=GREEN, width=1, dash="dot"),
                             annotation_text="Wide (buy signal)", annotation_font_color=GREEN)
            fig_z.add_hline(y=-1.5, line=dict(color=CORAL, width=1, dash="dot"),
                             annotation_text="Tight (caution)", annotation_font_color=CORAL)
            fig_z.add_trace(go.Scatter(
                x=hy_z.index, y=hy_z.values,
                mode="lines", name="HY OAS Z-Score",
                line=dict(color=AMBER, width=1.8),
            ))
            if ig_z is not None:
                fig_z.add_trace(go.Scatter(
                    x=ig_z.index, y=ig_z.values,
                    mode="lines", name="IG OAS Z-Score",
                    line=dict(color=BLUE, width=1.5),
                ))
            fig_z.update_layout(
                **CHART_LAYOUT,
                title="Credit Spread Z-Scores (1Y Rolling)",
                xaxis_title="", yaxis_title="Z-Score",
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT)),
                height=260,
            )
            fig_z.update_xaxes(**AXIS_STYLE)
            fig_z.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("---")

    # ── Spread history charts ────────────────────────────────────────────────
    st.markdown("#### Spread History")
    ch_l, ch_r = st.columns(2)

    for col, sid, label, colour in [
        (ch_l, "BAMLH0A0HYM2", "HY OAS Spread (bp)", AMBER),
        (ch_r, "BAMLC0A0CM",   "IG OAS Spread (bp)", BLUE),
    ]:
        if sid not in credit_df.columns:
            continue
        s = credit_df[sid].dropna()
        fig_s = go.Figure()

        # NBER recession shading
        if "USREC" in credit_df.columns:
            rec = credit_df["USREC"].reindex(s.index, method="ffill").fillna(0)
            in_rec = False
            for date, val in rec.items():
                if val == 1 and not in_rec:
                    rec_start = date
                    in_rec = True
                elif val == 0 and in_rec:
                    fig_s.add_vrect(
                        x0=str(rec_start), x1=str(date),
                        fillcolor="rgba(216,90,48,0.12)", line_width=0,
                    )
                    in_rec = False

        fig_s.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=label,
            line=dict(color=colour, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(colour[1:3], 16)},{int(colour[3:5], 16)},{int(colour[5:7], 16)},0.08)",
        ))
        fig_s.add_hline(
            y=float(s.mean()),
            line=dict(color=GREY, width=1, dash="dot"),
            annotation_text=f"Long-run avg {s.mean():.0f}bp",
            annotation_font_color=GREY,
        )
        fig_s.update_layout(
            **CHART_LAYOUT,
            title=label,
            xaxis_title="", yaxis_title="Basis Points",
            showlegend=False,
            height=280,
        )
        fig_s.update_xaxes(**AXIS_STYLE)
        fig_s.update_yaxes(**AXIS_STYLE)
        col.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # ── Cochrane-Piazzesi factor ─────────────────────────────────────────────
    st.markdown("#### Cochrane-Piazzesi Bond Return Factor")
    st.caption(
        "The CP factor (Cochrane & Piazzesi, 2005, AER) predicts excess bond returns "
        "using a tent-shaped linear combination of forward rates. "
        "A high CP value historically precedes high excess returns across the yield curve."
    )

    try:
        yc_for_cp = st.cache_data(fetch_yield_curve, ttl=3600)(start="2000-01-01")
        cp = cochrane_piazzesi_factor(yc_for_cp).dropna()

        if not cp.empty:
            cp_z = zscore_series(cp, window=252)
            current_cp = cp.iloc[-1]
            current_cp_z = cp_z.iloc[-1] if len(cp_z) else np.nan
            signal = (
                "Attractive — above-average excess bond returns historically predicted"
                if current_cp_z > 0.5 else
                "Unattractive — below-average excess bond returns historically predicted"
                if current_cp_z < -0.5 else
                "Neutral"
            )

            kp1, kp2, kp3 = st.columns(3)
            kp1.metric("CP Factor (current)", f"{current_cp:.3f}")
            kp2.metric("CP Z-Score (1Y)",     f"{current_cp_z:+.2f}")
            kp3.metric("Signal", signal)

            fig_cp = go.Figure()
            fig_cp.add_hline(y=0, line=dict(color=GREY, width=1))
            fig_cp.add_trace(go.Scatter(
                x=cp.index, y=cp.values,
                mode="lines", name="CP Factor",
                line=dict(color=PURPLE, width=1.8),
            ))
            fig_cp.update_layout(
                **CHART_LAYOUT,
                title="Cochrane-Piazzesi Factor — Bond Excess Return Predictor",
                xaxis_title="", yaxis_title="CP Factor",
                showlegend=False,
                height=300,
            )
            fig_cp.update_xaxes(**AXIS_STYLE)
            fig_cp.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_cp, use_container_width=True)

            with st.expander("Methodology"):
                st.markdown(
                    """
                    **Cochrane & Piazzesi (2005)** showed that a single factor — a
                    tent-shaped combination of forward rates — explains a large fraction
                    of the time-series variation in excess bond returns across maturities.

                    The factor is estimated here as an approximation from FRED
                    constant-maturity yields:

                    $$CP_t \\approx -2.14\\,y_1 + 0.81\\,y_2 + 1.48\\,y_5 - 0.99\\,y_{10}
                    + 0.54\\,\\bar{y}_{2-5}$$

                    where $y_n$ is the $n$-year constant-maturity yield.
                    The original paper uses Fama-Bliss zero-coupon data — this is
                    a replication approximation, not a direct reproduction.

                    **Reference:** Cochrane, J.H. & Piazzesi, M. (2005). Bond Risk Premia.
                    *American Economic Review*, 95(1), 138–160.
                    DOI: [10.1257/0002828053828581](https://doi.org/10.1257/0002828053828581)
                    """
                )
    except Exception:
        st.info("Cochrane-Piazzesi factor requires yield curve data.")

    st.caption(
        "Source: FRED (BAMLH0A0HYM2, BAMLC0A0CM, DGS series). "
        "NBER recession shading for context. "
        "Z-scores computed on a 252-day trailing window."
    )
