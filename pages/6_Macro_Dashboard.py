"""Page 6: Macro Dashboard — independent page, no portfolio required.

Sections:
  1. Economic Snapshot  — 6 FRED metric tiles
  2. Regime Quadrant    — Ilmanen (2011) growth/inflation quadrant
  3. Yield Curve        — current + 1Y-ago, 10Y-2Y history
  4. Macro Stress Index — composite VIX + HY OAS + curve inversion gauge
  5. HMM Regime         — Hamilton (1989) 2-state classifier
  6. Recommendation     — regime-conditional allocation guidance

Data: FRED via fredapi (FRED_API_KEY required).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Macro Dashboard", page_icon="🌐", layout="wide")

try:
    from dotenv import load_dotenv  # type: ignore[import]
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # .env handled by shell or Streamlit secrets

from modules.macro import (  # noqa: E402
    compute_regime_coordinates,
    compute_snapshot_metrics,
    compute_stress_index,
    fetch_series,
    fetch_yield_curve,
    fit_hmm_regime,
    get_fred_api_key,
    get_regime_recommendation,
    regime_label,
    stress_level,
    FRED_SERIES,
    SNAPSHOT_SERIES,
)
from ui.sidebar import render_portfolio_sidebar  # noqa: E402
from ui.theme import AXIS_STYLE, CHART_LAYOUT, FONT, GRID, inject_css  # noqa: E402

inject_css()

# ── Colour palette ──────────────────────────────────────────────────────────
TEAL   = "#4ecdc4"
BLUE   = "#81a4e8"
GREEN  = "#68d391"
AMBER  = "#f6ad55"
RED    = "#f56565"
GREY   = "#636b78"
GREY_L = "#a0a8b8"

BG     = "#0f1116"
PLOT_BG = "#161b27"

# ── Sidebar ─────────────────────────────────────────────────────────────────
render_portfolio_sidebar()

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='padding: 0.5rem 0 0.15rem 0;'>
        <h1 style='font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;
                   background: linear-gradient(90deg, #4ecdc4 0%, #81a4e8 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-family: "DM Sans", sans-serif;'>
            Macro Dashboard
        </h1>
        <p style='color: #636b78; font-size: 0.70rem; margin: 0.25rem 0 0 0;
                  letter-spacing: 2.5px; font-weight: 600;
                  font-family: "DM Sans", sans-serif;'>
            ECONOMIC REGIME &nbsp;·&nbsp; YIELD CURVE &nbsp;·&nbsp; STRESS INDEX
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── API key check ─────────────────────────────────────────────────────────────
if not get_fred_api_key():
    st.error(
        "**FRED_API_KEY not configured.**  \n"
        "Add `FRED_API_KEY=your_key` to your `.env` file.  \n"
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
        icon="🔑",
    )
    st.stop()


# ── Data loading ─────────────────────────────────────────────────────────────

ALL_SERIES = list(set(
    SNAPSHOT_SERIES
    + ["CPILFESL", "INDPRO", "USREC", "T10Y2Y", "BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS",
       "T10YIE", "RECPROUSM156N"]
))

YIELD_MATURITIES = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS20", "DGS30"]
MATURITY_LABELS  = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]
MATURITY_YEARS   = [1/12, 3/12, 6/12, 1, 2, 5, 10, 20, 30]

START_DATE = "2000-01-01"


@st.cache_data(ttl=3600, show_spinner=False)
def _load_macro_data() -> pd.DataFrame:
    return fetch_series(ALL_SERIES, start=START_DATE)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_yield_curve() -> pd.DataFrame:
    raw = fetch_series(YIELD_MATURITIES, start=START_DATE)
    if raw.empty:
        return raw
    rename = dict(zip(YIELD_MATURITIES, MATURITY_LABELS))
    cols_present = {k: v for k, v in rename.items() if k in raw.columns}
    raw.rename(columns=cols_present, inplace=True)
    return raw


with st.spinner("Loading FRED data…"):
    try:
        macro_df   = _load_macro_data()
        yield_df   = _load_yield_curve()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to fetch FRED data: {exc}", icon="⚠")
        st.stop()

if macro_df.empty:
    st.error("No FRED data returned. Check your API key and network connection.", icon="⚠")
    st.stop()


# ── Helper: NBER recession shading ──────────────────────────────────────────

def _recession_shapes(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add NBER recession shading to a Plotly figure."""
    if "USREC" not in df.columns:
        return
    rec = df["USREC"].ffill().fillna(0)
    in_rec = False
    start_rec: pd.Timestamp | None = None
    for date, val in rec.items():
        if val == 1 and not in_rec:
            in_rec = True
            start_rec = date
        elif val == 0 and in_rec:
            in_rec = False
            fig.add_vrect(
                x0=start_rec, x1=date,
                fillcolor="rgba(255,255,255,0.04)",
                layer="below", line_width=0,
            )
    if in_rec and start_rec is not None:
        fig.add_vrect(
            x0=start_rec, x1=rec.index[-1],
            fillcolor="rgba(255,255,255,0.04)",
            layer="below", line_width=0,
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Economic Snapshot
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h2 style='margin-bottom:0.6rem;'>Economic Snapshot</h2>",
    unsafe_allow_html=True,
)

snapshot = compute_snapshot_metrics(macro_df)

_TILE_CONFIGS = {
    "INDPRO":   {"fmt": ".1f",  "scale": 1},
    "CPIAUCSL": {"fmt": ".2%",  "scale": 1},
    "UNRATE":   {"fmt": ".1f",  "scale": 1},
    "DFF":      {"fmt": ".2f",  "scale": 1},
    "T10Y2Y":   {"fmt": ".2f",  "scale": 1},
    "VIXCLS":   {"fmt": ".1f",  "scale": 1},
}


def _tile_html(
    name: str,
    value: float,
    mom: float,
    zscore: float,
    trend: str,
    unit: str,
    fmt: str,
    invert_good: bool,
) -> str:
    """Render a metric tile as HTML."""
    if np.isnan(value):
        val_str = "N/A"
    elif fmt == ".2%":
        val_str = f"{value:.2%}"
    elif fmt == ".2f":
        val_str = f"{value:.2f}"
    else:
        val_str = f"{value:.1f}"

    # MoM badge
    if np.isnan(mom):
        mom_html = ""
    else:
        mom_pct = mom * 100
        good = (mom_pct > 0) != invert_good   # invert_good: lower = better
        mom_color = GREEN if good else RED
        sign = "+" if mom_pct >= 0 else ""
        mom_html = (
            f"<span style='font-size:0.70rem;color:{mom_color};"
            f"font-family:\"JetBrains Mono\",monospace;'>{sign}{mom_pct:.2f}% MoM</span>"
        )

    # Z-score badge
    if abs(zscore) > 1.5:
        z_color = RED if (zscore > 0) != invert_good else GREEN
        z_html = (
            f"<span style='font-size:0.68rem;color:{z_color};"
            f"font-family:\"JetBrains Mono\",monospace;'>z={zscore:+.1f}</span>"
        )
    else:
        z_html = (
            f"<span style='font-size:0.68rem;color:{GREY_L};"
            f"font-family:\"JetBrains Mono\",monospace;'>z={zscore:+.1f}</span>"
        )

    trend_color = GREEN if trend == "↑" else RED if trend == "↓" else GREY_L

    return f"""
    <div style='background:#1a1e2a;border:1px solid #252836;border-radius:10px;
                padding:1rem 1.1rem;height:100%;'>
        <div style='font-size:0.68rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:1.4px;color:{GREY};font-family:"DM Sans",sans-serif;
                    margin-bottom:0.4rem;'>{name}</div>
        <div style='display:flex;align-items:baseline;gap:0.4rem;margin-bottom:0.25rem;'>
            <span style='font-size:1.55rem;font-weight:600;color:#f0f2f5;
                         font-family:"JetBrains Mono",monospace;'>{val_str}</span>
            <span style='font-size:0.72rem;color:{GREY_L};'>{unit}</span>
            <span style='font-size:1.1rem;color:{trend_color};'>{trend}</span>
        </div>
        <div style='display:flex;gap:0.6rem;flex-wrap:wrap;'>{mom_html}{z_html}</div>
    </div>
    """


cols = st.columns(6)
for i, metric in enumerate(snapshot[:6]):
    sid = metric["series_id"]
    cfg = _TILE_CONFIGS.get(sid, {"fmt": ".2f", "scale": 1})
    with cols[i]:
        st.markdown(
            _tile_html(
                name=metric["name"],
                value=metric["latest_value"],
                mom=metric["mom_change"],
                zscore=metric["zscore_1y"],
                trend=metric["trend"],
                unit=metric["unit"],
                fmt=cfg["fmt"],
                invert_good=metric["invert_good"],
            ),
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Regime Quadrant + Stress Index (side by side)
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h2 style='margin-bottom:0.6rem;'>Macro Regime</h2>",
    unsafe_allow_html=True,
)
st.caption(
    "Framework: Ilmanen (2011) *Expected Returns*. Bridgewater All Weather. "
    "X = IP 6-month momentum z-score (growth). Y = Core CPI 6-month momentum z-score (inflation)."
)

col_quad, col_stress = st.columns([3, 2], gap="large")

# ── Regime Quadrant ──────────────────────────────────────────────────────────
with col_quad:
    if "INDPRO" in macro_df.columns and "CPILFESL" in macro_df.columns:
        regime_df = compute_regime_coordinates(macro_df["INDPRO"], macro_df["CPILFESL"])

        if not regime_df.empty:
            # Limit history to 10 years for the path
            cutoff = regime_df["date"].max() - pd.DateOffset(years=10)
            hist   = regime_df[regime_df["date"] >= cutoff].copy()
            curr   = regime_df.iloc[-1]

            current_regime = regime_label(float(curr["x"]), float(curr["y"]))

            fig_q = go.Figure()

            # Quadrant shading
            _QUAD_COLORS = {
                "Goldilocks":  "rgba(104,211,145,0.07)",
                "Reflation":   "rgba(246,173,85,0.07)",
                "Stagflation": "rgba(245,101,101,0.07)",
                "Deflation":   "rgba(129,164,232,0.07)",
            }
            _QUAD_LABEL_POS = {
                "Goldilocks":  ( 1.8, -1.8),
                "Reflation":   ( 1.8,  1.8),
                "Stagflation": (-1.8,  1.8),
                "Deflation":   (-1.8, -1.8),
            }
            _QUAD_TEXT_COLOR = {
                "Goldilocks":  "#68d391",
                "Reflation":   "#f6ad55",
                "Stagflation": "#f56565",
                "Deflation":   "#81a4e8",
            }
            for quad, (qx, qy) in _QUAD_LABEL_POS.items():
                fig_q.add_annotation(
                    x=qx, y=qy, text=quad,
                    font=dict(size=10, color=_QUAD_TEXT_COLOR[quad], family="DM Sans"),
                    showarrow=False, opacity=0.7,
                )

            # Quadrant shaded regions
            for quad, col in _QUAD_COLORS.items():
                lx = 0 if quad in ("Goldilocks", "Reflation") else -4
                hx = 4 if quad in ("Goldilocks", "Reflation") else 0
                ly = 0 if quad in ("Reflation", "Stagflation") else -4
                hy = 4 if quad in ("Reflation", "Stagflation") else 0
                fig_q.add_shape(
                    type="rect", x0=lx, x1=hx, y0=ly, y1=hy,
                    fillcolor=col, line_width=0, layer="below",
                )

            # Crosshair lines
            fig_q.add_hline(y=0, line_color=GREY, line_width=0.8, opacity=0.5)
            fig_q.add_vline(x=0, line_color=GREY, line_width=0.8, opacity=0.5)

            # Historical 10Y path
            fig_q.add_trace(go.Scatter(
                x=hist["x"], y=hist["y"],
                mode="lines",
                line=dict(color="rgba(99,107,120,0.5)", width=1),
                name="10Y path",
                showlegend=False,
                hovertemplate=(
                    "<b>%{customdata|%b %Y}</b><br>"
                    "Growth: %{x:.2f}σ<br>Inflation: %{y:.2f}σ<extra></extra>"
                ),
                customdata=hist["date"],
            ))

            # Current dot
            fig_q.add_trace(go.Scatter(
                x=[curr["x"]], y=[curr["y"]],
                mode="markers+text",
                marker=dict(color=TEAL, size=14, symbol="circle",
                            line=dict(color="#0f1116", width=2)),
                text=["NOW"],
                textposition="top center",
                textfont=dict(size=9, color=TEAL, family="DM Sans"),
                name="Current",
                showlegend=False,
                hovertemplate=(
                    f"<b>Current ({curr['date'].strftime('%b %Y')})</b><br>"
                    f"Regime: {current_regime}<br>"
                    f"Growth: {curr['x']:.2f}σ<br>Inflation: {curr['y']:.2f}σ<extra></extra>"
                ),
            ))

            axis_range = 3.5
            fig_q.update_layout(
                **CHART_LAYOUT,
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(
                    **AXIS_STYLE,
                    range=[-axis_range, axis_range],
                    title=dict(text="Growth (IP 6m momentum z-score)", font=dict(size=10, color=GREY_L)),
                ),
                yaxis=dict(
                    **AXIS_STYLE,
                    range=[-axis_range, axis_range],
                    title=dict(text="Inflation (Core CPI 6m momentum z-score)", font=dict(size=10, color=GREY_L)),
                ),
            )

            st.plotly_chart(fig_q, use_container_width=True, config={"displayModeBar": False})

            # Current regime badge
            rec_info = get_regime_recommendation(current_regime)
            badge_color = rec_info["color"]
            st.markdown(
                f"<div style='background:#1a1e2a;border:1px solid #252836;border-radius:8px;"
                f"padding:0.7rem 1rem;margin-top:-0.5rem;'>"
                f"<span style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:1.4px;color:{GREY};font-family:\"DM Sans\",sans-serif;'>Current Regime</span>"
                f"<div style='font-size:1.05rem;font-weight:700;color:{badge_color};"
                f"font-family:\"DM Sans\",sans-serif;margin-top:0.15rem;'>{current_regime}</div>"
                f"<div style='font-size:0.78rem;color:{GREY_L};margin-top:0.3rem;"
                f"font-family:\"DM Sans\",sans-serif;line-height:1.5;'>{rec_info['summary']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Insufficient data for regime quadrant.")
    else:
        st.info("INDPRO or CPILFESL data unavailable.")

# ── Macro Stress Index ────────────────────────────────────────────────────────
with col_stress:
    st.markdown(
        "<h3 style='margin-bottom:0.3rem;'>Macro Stress Index</h3>",
        unsafe_allow_html=True,
    )
    st.caption("z(VIX) + z(HY OAS) + z(−10Y2Y) — composite. Higher = more stress.")

    can_stress = all(c in macro_df.columns for c in ["VIXCLS", "BAMLH0A0HYM2", "T10Y2Y"])
    if can_stress:
        stress_series = compute_stress_index(
            macro_df["VIXCLS"].dropna(),
            macro_df["BAMLH0A0HYM2"].dropna(),
            macro_df["T10Y2Y"].dropna(),
        )
        if not stress_series.empty:
            current_stress = float(stress_series.iloc[-1])
            stress_lbl, stress_color = stress_level(current_stress)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(current_stress, 2),
                domain={"x": [0, 1], "y": [0, 1]},
                number={"font": {"color": stress_color, "size": 36, "family": "JetBrains Mono"}},
                gauge={
                    "axis": {
                        "range": [-3, 4],
                        "tickwidth": 1,
                        "tickcolor": GREY,
                        "tickfont": {"size": 9, "color": GREY_L},
                    },
                    "bar": {"color": stress_color, "thickness": 0.22},
                    "bgcolor": PLOT_BG,
                    "borderwidth": 1,
                    "bordercolor": GRID,
                    "steps": [
                        {"range": [-3,  -0.5], "color": "rgba(104,211,145,0.15)"},
                        {"range": [-0.5, 0.5], "color": "rgba(246,173,85,0.15)"},
                        {"range": [0.5,  1.5], "color": "rgba(237,137,54,0.15)"},
                        {"range": [1.5,  2.5], "color": "rgba(245,101,101,0.15)"},
                        {"range": [2.5,  4.0], "color": "rgba(155,44,44,0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": stress_color, "width": 3},
                        "thickness": 0.8,
                        "value": current_stress,
                    },
                },
            ))
            fig_g.update_layout(
                **CHART_LAYOUT,
                height=230,
                margin=dict(l=10, r=10, t=20, b=5),
            )
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"<div style='text-align:center;font-size:1.3rem;font-weight:700;"
                f"color:{stress_color};font-family:\"DM Sans\",sans-serif;"
                f"margin-top:-1rem;'>{stress_lbl}</div>",
                unsafe_allow_html=True,
            )

            # Rolling stress chart
            st.markdown(
                "<div style='margin-top:1.2rem;font-size:0.72rem;font-weight:600;"
                "text-transform:uppercase;letter-spacing:1.3px;color:#636b78;"
                "font-family:\"DM Sans\",sans-serif;margin-bottom:0.3rem;'>"
                "Stress Index — 5Y History</div>",
                unsafe_allow_html=True,
            )
            cutoff_5y = stress_series.index.max() - pd.DateOffset(years=5)
            stress_5y = stress_series[stress_series.index >= cutoff_5y]

            fig_s = go.Figure()

            # Colour by stress level
            colors_hist = [stress_level(v)[1] for v in stress_5y]
            fig_s.add_trace(go.Scatter(
                x=stress_5y.index,
                y=stress_5y.values,
                mode="lines",
                line=dict(color=TEAL, width=1.5),
                fill="tozeroy",
                fillcolor="rgba(78,205,196,0.08)",
                name="Stress",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Stress: %{y:.2f}<extra></extra>",
            ))
            fig_s.add_hline(y=1.5, line_color=RED, line_dash="dot", line_width=1, opacity=0.6,
                            annotation_text="High", annotation_position="top right",
                            annotation_font=dict(size=9, color=RED))
            fig_s.add_hline(y=2.5, line_color="#9b2c2c", line_dash="dot", line_width=1, opacity=0.6,
                            annotation_text="Crisis", annotation_position="top right",
                            annotation_font=dict(size=9, color="#9b2c2c"))

            _recession_shapes(fig_s, macro_df[macro_df.index >= cutoff_5y])

            fig_s.update_layout(
                **CHART_LAYOUT,
                height=160,
                margin=dict(l=0, r=10, t=10, b=5),
                showlegend=False,
                xaxis=dict(**AXIS_STYLE),
                yaxis=dict(**AXIS_STYLE, title=None),
            )
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Insufficient data for stress index.")
    else:
        st.info("VIX, HY OAS, or T10Y2Y data unavailable.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Yield Curve
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    "<h2 style='margin-bottom:0.2rem;'>Yield Curve</h2>",
    unsafe_allow_html=True,
)

tab_curve, tab_spread = st.tabs(["Current Shape", "10Y–2Y Spread History"])

with tab_curve:
    if not yield_df.empty:
        present_labels = [l for l in MATURITY_LABELS if l in yield_df.columns]
        present_years  = [MATURITY_YEARS[MATURITY_LABELS.index(l)] for l in present_labels]

        # Current curve
        today_row = yield_df[present_labels].dropna(how="all").iloc[-1]
        today_date = today_row.name

        # 1Y ago
        one_year_ago = today_date - pd.DateOffset(years=1)
        past_row_idx = yield_df.index.get_indexer([one_year_ago], method="nearest")[0]
        past_row = yield_df[present_labels].iloc[past_row_idx]

        fig_yc = go.Figure()

        fig_yc.add_trace(go.Scatter(
            x=present_years,
            y=past_row.values,
            mode="lines+markers",
            name=f"1Y ago ({past_row.name.strftime('%b %Y')})",
            line=dict(color=GREY, width=1.5, dash="dot"),
            marker=dict(color=GREY, size=5),
            hovertemplate="%{y:.2f}%<extra>1Y Ago</extra>",
        ))

        fig_yc.add_trace(go.Scatter(
            x=present_years,
            y=today_row.values,
            mode="lines+markers",
            name=f"Current ({today_date.strftime('%b %Y')})",
            line=dict(color=TEAL, width=2.5),
            marker=dict(color=TEAL, size=7),
            hovertemplate="%{y:.2f}%<extra>Current</extra>",
        ))

        # Add maturity labels to x-axis
        fig_yc.update_layout(
            **CHART_LAYOUT,
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(
                **AXIS_STYLE,
                tickvals=present_years,
                ticktext=present_labels,
                title=dict(text="Maturity", font=dict(size=10, color=GREY_L)),
            ),
            yaxis=dict(
                **AXIS_STYLE,
                title=dict(text="Yield (%)", font=dict(size=10, color=GREY_L)),
            ),
            legend=dict(
                orientation="h", y=1.06, x=0,
                font=dict(size=10, color=GREY_L),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_yc, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Yield curve data unavailable.")

with tab_spread:
    if "T10Y2Y" in macro_df.columns:
        spread = macro_df["T10Y2Y"].dropna()

        fig_sp = go.Figure()

        # Fill areas: positive = normal, negative = inverted
        pos = spread.clip(lower=0)
        neg = spread.clip(upper=0)

        fig_sp.add_trace(go.Scatter(
            x=spread.index, y=pos.values,
            fill="tozeroy", mode="none",
            fillcolor="rgba(104,211,145,0.15)", name="Normal",
        ))
        fig_sp.add_trace(go.Scatter(
            x=spread.index, y=neg.values,
            fill="tozeroy", mode="none",
            fillcolor="rgba(245,101,101,0.2)", name="Inverted",
        ))
        fig_sp.add_trace(go.Scatter(
            x=spread.index, y=spread.values,
            mode="lines", name="10Y–2Y",
            line=dict(color=TEAL, width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Spread: %{y:.2f}pp<extra></extra>",
        ))
        fig_sp.add_hline(y=0, line_color=GREY, line_width=1, opacity=0.6)

        _recession_shapes(fig_sp, macro_df)

        fig_sp.update_layout(
            **CHART_LAYOUT,
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(**AXIS_STYLE),
            yaxis=dict(
                **AXIS_STYLE,
                title=dict(text="Spread (pp)", font=dict(size=10, color=GREY_L)),
            ),
            legend=dict(
                orientation="h", y=1.06, x=0,
                font=dict(size=10, color=GREY_L),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.caption("Grey bands = NBER recessions. Red fill = yield curve inversion.")
        st.plotly_chart(fig_sp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("T10Y2Y data unavailable.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HMM Regime Classifier
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    "<h2 style='margin-bottom:0.2rem;'>HMM Regime Classifier</h2>",
    unsafe_allow_html=True,
)
st.caption(
    "Hamilton (1989) 2-state Gaussian HMM on monthly [IP change, Core CPI change, VIX, HY OAS]. "
    "States labelled Risk-On / Risk-Off by mean VIX in each state."
)

hmm_ok = all(c in macro_df.columns for c in ["INDPRO", "CPILFESL", "VIXCLS", "BAMLH0A0HYM2"])

if hmm_ok:
    with st.spinner("Fitting HMM…"):
        try:
            states, probs, risk_off_idx = fit_hmm_regime(
                macro_df["INDPRO"],
                macro_df["CPILFESL"],
                macro_df["VIXCLS"],
                macro_df["BAMLH0A0HYM2"],
            )

            risk_on_idx   = 1 - risk_off_idx
            state_labels  = {risk_off_idx: "Risk-Off", risk_on_idx: "Risk-On"}
            state_colors  = {risk_off_idx: RED, risk_on_idx: GREEN}

            # Probability of risk-off state
            prob_risk_off = probs[f"state_{risk_off_idx}"]
            current_prob  = float(prob_risk_off.iloc[-1])
            current_state = "Risk-Off" if current_prob >= 0.5 else "Risk-On"
            state_color   = RED if current_state == "Risk-Off" else GREEN

            col_hmm1, col_hmm2 = st.columns([3, 1], gap="large")

            with col_hmm1:
                # Regime probability time-series
                fig_hmm = go.Figure()

                fig_hmm.add_trace(go.Scatter(
                    x=prob_risk_off.index,
                    y=prob_risk_off.values,
                    mode="lines",
                    line=dict(color=RED, width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(245,101,101,0.12)",
                    name="P(Risk-Off)",
                    hovertemplate="<b>%{x|%b %Y}</b><br>P(Risk-Off): %{y:.2f}<extra></extra>",
                ))
                fig_hmm.add_hline(y=0.5, line_color=GREY, line_dash="dot", line_width=1,
                                  opacity=0.7, annotation_text="50%",
                                  annotation_font=dict(size=9, color=GREY_L))

                _recession_shapes(fig_hmm, macro_df.resample("ME").last())

                fig_hmm.update_layout(
                    **CHART_LAYOUT,
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(**AXIS_STYLE),
                    yaxis=dict(
                        **AXIS_STYLE,
                        range=[0, 1],
                        title=dict(text="P(Risk-Off)", font=dict(size=10, color=GREY_L)),
                        tickformat=".0%",
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_hmm, use_container_width=True, config={"displayModeBar": False})
                st.caption("Grey bands = NBER recessions.")

            with col_hmm2:
                st.markdown(
                    f"<div style='background:#1a1e2a;border:1px solid #252836;border-radius:10px;"
                    f"padding:1.2rem 1.1rem;text-align:center;margin-top:1rem;'>"
                    f"<div style='font-size:0.68rem;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:1.4px;color:{GREY};font-family:\"DM Sans\",sans-serif;"
                    f"margin-bottom:0.5rem;'>Current State</div>"
                    f"<div style='font-size:1.5rem;font-weight:700;color:{state_color};"
                    f"font-family:\"DM Sans\",sans-serif;'>{current_state}</div>"
                    f"<div style='font-size:0.90rem;color:{GREY_L};font-family:\"JetBrains Mono\",monospace;"
                    f"margin-top:0.4rem;'>P = {current_prob:.1%}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        except Exception as exc:  # noqa: BLE001
            st.warning(f"HMM fitting failed: {exc}", icon="⚠")
else:
    st.info("Required FRED series unavailable for HMM fitting.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Regime Recommendation
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    "<h2 style='margin-bottom:0.6rem;'>Allocation Recommendation</h2>",
    unsafe_allow_html=True,
)

# Determine current regime for recommendation
_rec_regime = "Goldilocks"
if "INDPRO" in macro_df.columns and "CPILFESL" in macro_df.columns:
    try:
        _rc = compute_regime_coordinates(macro_df["INDPRO"], macro_df["CPILFESL"])
        if not _rc.empty:
            _last = _rc.iloc[-1]
            _rec_regime = regime_label(float(_last["x"]), float(_last["y"]))
    except Exception:  # noqa: BLE001
        pass

rec = get_regime_recommendation(_rec_regime)

col_rec, col_tbl = st.columns([2, 1], gap="large")

with col_rec:
    st.markdown(
        f"<div style='background:#1a1e2a;border:1px solid #252836;border-left:3px solid {rec['color']};"
        f"border-radius:8px;padding:1.1rem 1.3rem;'>"
        f"<div style='font-size:0.68rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:1.4px;color:{GREY};font-family:\"DM Sans\",sans-serif;"
        f"margin-bottom:0.4rem;'>Regime-Conditional Guidance</div>"
        f"<div style='font-size:1.1rem;font-weight:700;color:{rec['color']};"
        f"font-family:\"DM Sans\",sans-serif;margin-bottom:0.5rem;'>{rec['label']}</div>"
        f"<div style='font-size:0.84rem;color:{GREY_L};font-family:\"DM Sans\",sans-serif;"
        f"line-height:1.6;'>{rec['summary']}</div>"
        f"<div style='margin-top:0.8rem;font-size:0.68rem;color:{GREY};"
        f"font-family:\"DM Sans\",sans-serif;'>"
        f"Source: Ilmanen (2011), <em>Expected Returns</em>, Wiley. "
        f"Bridgewater All Weather framework. "
        f"This is regime-conditional historical guidance, not a return forecast.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_tbl:
    suggested = rec["suggested"]
    _stance_color = {
        "+Overweight": GREEN,
        "Neutral":     GREY_L,
        "Underweight": RED,
    }
    rows_html = ""
    for asset, stance in suggested.items():
        color = _stance_color.get(stance, GREY_L)
        rows_html += (
            f"<tr>"
            f"<td style='padding:0.35rem 0.6rem;color:#c4cad6;font-size:0.80rem;"
            f"font-family:\"DM Sans\",sans-serif;'>{asset}</td>"
            f"<td style='padding:0.35rem 0.6rem;text-align:right;'>"
            f"<span style='font-size:0.76rem;font-weight:700;color:{color};"
            f"font-family:\"JetBrains Mono\",monospace;'>{stance}</span></td>"
            f"</tr>"
        )
    st.markdown(
        f"<div style='background:#1a1e2a;border:1px solid #252836;border-radius:8px;overflow:hidden;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>"
        f"<th style='padding:0.5rem 0.6rem;font-size:0.65rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:1.2px;color:{GREY};"
        f"font-family:\"DM Sans\",sans-serif;text-align:left;border-bottom:1px solid #252836;'>Asset Class</th>"
        f"<th style='padding:0.5rem 0.6rem;font-size:0.65rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:1.2px;color:{GREY};"
        f"font-family:\"DM Sans\",sans-serif;text-align:right;border-bottom:1px solid #252836;'>Stance</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='margin-top:0.5rem;font-size:0.70rem;color:#636b78;"
    "font-family:\"DM Sans\",sans-serif;'>"
    "⚠ Guidance based on historical regime averages. Not investment advice. "
    "Regimes can transition rapidly. Always conduct independent analysis.</div>",
    unsafe_allow_html=True,
)
