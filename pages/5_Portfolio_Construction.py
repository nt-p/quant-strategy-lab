"""Page 5: Portfolio Construction — Efficient Frontier, Risk Parity, Black-Litterman."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Construction", page_icon="⚙️", layout="wide")

from pypfopt import BlackLittermanModel, EfficientFrontier  # noqa: E402
from pypfopt import expected_returns as pf_expected_returns  # noqa: E402
from pypfopt import risk_models as pf_risk_models  # noqa: E402
from ui.sidebar import render_portfolio_sidebar  # noqa: E402
from ui.theme import AXIS_STYLE, CHART_LAYOUT, FONT, GRID, inject_css  # noqa: E402

inject_css()

# ── Palette ────────────────────────────────────────────────────────────────────
TEAL   = "#4ecdc4"
GREY   = "#636b78"
GREEN  = "#68d391"
RED    = "#f56565"
PURPLE = "#b794f4"
ORANGE = "#f6ad55"
BLUE   = "#63b3ed"
RF_RATE = 0.04  # risk-free rate proxy

_DONUT_PALETTE = [
    TEAL, ORANGE, PURPLE, GREEN, BLUE, RED,
    "#f687b3", "#76e4f7", "#9ae6b4", "#fbd38d",
    "#fc8181", "#b2f5ea", "#c3dafe", "#e9d8fd",
    "#feebc8", "#bee3f8",
]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _subheader(text: str) -> None:
    st.markdown(
        f"<h3 style='font-size:1.0rem;font-weight:600;margin:0.8rem 0 0.25rem 0;"
        f"color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>{text}</h3>",
        unsafe_allow_html=True,
    )


# ── Sidebar + header ───────────────────────────────────────────────────────────
render_portfolio_sidebar()

st.markdown(
    """
    <div style='padding: 0.5rem 0 0.15rem 0;'>
        <h1 style='font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;
                   background: linear-gradient(90deg, #4ecdc4 0%, #81a4e8 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-family: "DM Sans", sans-serif;'>
            Portfolio Construction
        </h1>
        <p style='color: #636b78; font-size: 0.70rem; margin: 0.25rem 0 0 0;
                  letter-spacing: 2.5px; font-weight: 600;
                  font-family: "DM Sans", sans-serif;'>
            EFFICIENT FRONTIER &nbsp;·&nbsp; RISK PARITY &nbsp;·&nbsp; BLACK-LITTERMAN
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ── Guard ──────────────────────────────────────────────────────────────────────
if not st.session_state.get("portfolio"):
    st.info(
        "Configure and run a portfolio in the **Portfolio Hub** first.",
        icon="🏗",
    )
    st.stop()

portfolio: dict    = st.session_state["portfolio"]
holdings: dict[str, float] = portfolio["holdings"]
port_start: pd.Timestamp | None = portfolio.get("start_date")
port_end: pd.Timestamp | None   = portfolio.get("end_date")


# ── Universe editor ────────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-size:0.70rem;color:#636b78;font-weight:600;"
    "text-transform:uppercase;letter-spacing:1.8px;margin:0.2rem 0 0.3rem 0;"
    "font-family:\"DM Sans\",sans-serif;'>Asset Universe</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Pre-populated from your active portfolio. Add or remove tickers freely — "
    "changes here do not affect your active portfolio."
)

portfolio_tickers = list(holdings.keys())
universe: list[str] = st.multiselect(
    "Tickers in optimisation universe",
    options=portfolio_tickers,
    default=portfolio_tickers,
    label_visibility="collapsed",
)

extra_raw = st.text_input(
    "Add extra tickers (comma-separated)",
    placeholder="e.g. NVDA, VXUS, BND",
    key="pc_extra",
)
if extra_raw.strip():
    for t in (tok.strip().upper() for tok in extra_raw.split(",") if tok.strip()):
        if t not in universe:
            universe.append(t)

col_s, col_e, col_btn = st.columns([1, 1, 1])
with col_s:
    sel_start = st.date_input(
        "Start date",
        value=port_start.date() if port_start else date(2015, 1, 1),
        max_value=date.today(),
        key="pc_start",
    )
with col_e:
    sel_end = st.date_input(
        "End date",
        value=port_end.date() if port_end else date.today(),
        max_value=date.today(),
        key="pc_end",
    )
with col_btn:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    run_opt = st.button("Compute Optimisation", type="primary", use_container_width=True)

if run_opt:
    if len(universe) < 2:
        st.error("Select at least 2 tickers.")
        st.stop()
    st.session_state["pc_tickers"]   = list(universe)
    st.session_state["pc_start_str"] = str(sel_start)
    st.session_state["pc_end_str"]   = str(sel_end)
    st.session_state["pc_ready"]     = True

if not st.session_state.get("pc_ready"):
    st.info(
        "Configure the universe and date range above, then click **Compute Optimisation**.",
        icon="ℹ️",
    )
    st.stop()


# ── Price data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching price data…", ttl=3_600)
def _load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # lazy import
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.ffill().dropna(how="all")


tickers_key = tuple(sorted(st.session_state["pc_tickers"]))
prices_raw  = _load_prices(
    tickers_key,
    st.session_state["pc_start_str"],
    st.session_state["pc_end_str"],
)

if prices_raw.empty:
    st.error("Could not fetch price data for the selected tickers and date range.")
    st.stop()

# Require at least 252 observations per ticker; then drop all rows with any NaN
prices_raw  = prices_raw.dropna(thresh=252, axis=1)
prices_clean: pd.DataFrame = prices_raw.dropna()

available: list[str] = list(prices_clean.columns)

if len(available) < 2:
    st.error("Fewer than 2 tickers have sufficient price history. Extend the date range or change tickers.")
    st.stop()

dropped = set(tickers_key) - set(available)
if dropped:
    st.warning(f"Dropped tickers with insufficient history: {', '.join(sorted(dropped))}")

st.caption(
    f"Optimising over **{len(available)} assets** · "
    f"{len(prices_clean):,} trading days · "
    f"{st.session_state['pc_start_str']} → {st.session_state['pc_end_str']}"
)


# ── Shared computations ────────────────────────────────────────────────────────

mu_pf: pd.Series     = pf_expected_returns.mean_historical_return(prices_clean)
S_pf: pd.DataFrame   = pf_risk_models.sample_cov(prices_clean)
mu_arr: np.ndarray   = mu_pf.values.astype(float)
cov_arr: np.ndarray  = S_pf.values.astype(float)
n: int               = len(available)
vols_ann: np.ndarray = np.sqrt(np.diag(cov_arr))

# Current portfolio weight vector (rescaled to available tickers, equal-weight fallback)
raw_curr: np.ndarray = np.array([holdings.get(t, 0.0) for t in available], dtype=float)
curr_w: np.ndarray   = raw_curr / raw_curr.sum() if raw_curr.sum() > 1e-9 else np.ones(n) / n


# ── Pure-computation helpers (no Streamlit) ────────────────────────────────────

def _port_stats(w: np.ndarray) -> tuple[float, float, float]:
    """(ann_return, ann_vol, sharpe) for weight vector w."""
    ret = float(mu_arr @ w)
    vol = float(np.sqrt(w @ cov_arr @ w))
    sr  = (ret - RF_RATE) / vol if vol > 1e-9 else 0.0
    return ret, vol, sr


def _risk_contribs(w: np.ndarray) -> np.ndarray:
    """Absolute risk contribution per asset (sums to portfolio vol)."""
    pv  = float(np.sqrt(w @ cov_arr @ w))
    mrc = cov_arr @ w
    return w * mrc / (pv + 1e-12)


def _erc_weights() -> np.ndarray:
    """Equal Risk Contribution via SLSQP."""
    def _obj(w: np.ndarray) -> float:
        rc     = _risk_contribs(w)
        target = rc.mean()
        return float(np.sum((rc - target) ** 2))

    res = minimize(
        _obj,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(1e-4, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    w = np.maximum(res.x, 0.0)
    return w / w.sum()


def _max_div_weights() -> np.ndarray:
    """Maximum Diversification portfolio via SLSQP."""
    def _neg_dr(w: np.ndarray) -> float:
        return -float(w @ vols_ann) / (float(np.sqrt(w @ cov_arr @ w)) + 1e-12)

    res = minimize(
        _neg_dr,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(1e-4, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    w = np.maximum(res.x, 0.0)
    return w / w.sum()


def _donut_fig(weights: np.ndarray, labels: list[str], title: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels,
        values=weights,
        hole=0.55,
        marker=dict(
            colors=_DONUT_PALETTE[:len(labels)],
            line=dict(color="#161b27", width=2),
        ),
        textinfo="percent",
        textfont=dict(size=10, color=FONT, family="DM Sans, sans-serif"),
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        showlegend=False,
        margin=dict(l=0, r=0, t=28, b=0),
        height=210,
        title=dict(
            text=title,
            font=dict(size=10, color=FONT, family="DM Sans, sans-serif"),
            x=0.5,
        ),
        annotations=[dict(
            text=f"<b>{len(labels)}<br>assets</b>",
            x=0.5, y=0.5,
            font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
            showarrow=False,
        )],
    )
    return fig


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_ef, tab_rp, tab_bl = st.tabs(["Efficient Frontier", "Risk Parity", "Black-Litterman"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — EFFICIENT FRONTIER
# ════════════════════════════════════════════════════════════════════════════════

with tab_ef:
    st.caption(
        "5,000 randomly sampled portfolios coloured by Sharpe ratio. "
        "The efficient frontier traces the minimum-variance portfolio at each return level. "
        "★ marks Maximum Sharpe, Minimum Variance, and Current Portfolio. "
        "All estimates from historical mean returns and sample covariance — not a forecast."
    )

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    rng    = np.random.default_rng(42)
    mc_w   = rng.dirichlet(np.ones(n), size=5_000)
    mc_ret = mc_w @ mu_arr
    mc_vol = np.sqrt(np.einsum("ij,jk,ik->i", mc_w, cov_arr, mc_w))
    mc_sr  = (mc_ret - RF_RATE) / mc_vol

    # ── PyPortfolioOpt: Max Sharpe & Min Variance ─────────────────────────────
    ms_w: np.ndarray | None = None
    mv_w: np.ndarray | None = None
    ms_ret, ms_vol, ms_sr   = 0.0, 0.0, 0.0
    mv_ret, mv_vol, mv_sr   = 0.0, 0.0, 0.0

    try:
        ef_ms = EfficientFrontier(mu_pf, S_pf, weight_bounds=(0, 1))
        ef_ms.max_sharpe(risk_free_rate=RF_RATE)
        ms_d = ef_ms.clean_weights()
        ms_w = np.array([ms_d.get(t, 0.0) for t in available])
        ms_ret, ms_vol, ms_sr = _port_stats(ms_w)
    except Exception:
        pass

    try:
        ef_mv = EfficientFrontier(mu_pf, S_pf, weight_bounds=(0, 1))
        ef_mv.min_volatility()
        mv_d = ef_mv.clean_weights()
        mv_w = np.array([mv_d.get(t, 0.0) for t in available])
        mv_ret, mv_vol, mv_sr = _port_stats(mv_w)
    except Exception:
        pass

    curr_ret, curr_vol, curr_sr = _port_stats(curr_w)

    # ── Efficient frontier curve ──────────────────────────────────────────────
    ef_curve_v: list[float] = []
    ef_curve_r: list[float] = []
    mu_lo = float(mu_arr.min()) * 1.05
    mu_hi = float(mu_arr.max()) * 0.95
    if mu_lo < mu_hi:
        for tr in np.linspace(mu_lo, mu_hi, 40):
            try:
                ef_t = EfficientFrontier(mu_pf, S_pf, weight_bounds=(0, 1))
                ef_t.efficient_return(target_return=float(tr))
                r, v, _ = ef_t.portfolio_performance(verbose=False, risk_free_rate=RF_RATE)
                ef_curve_v.append(v)
                ef_curve_r.append(r)
            except Exception:
                pass

    # ── Build figure ──────────────────────────────────────────────────────────
    fig_ef = go.Figure()

    # Trace 0: Monte Carlo scatter
    fig_ef.add_trace(go.Scatter(
        x=mc_vol,
        y=mc_ret,
        mode="markers",
        name="Random portfolios",
        marker=dict(
            color=mc_sr,
            colorscale="Viridis",
            size=4,
            opacity=0.55,
            colorbar=dict(
                title=dict(text="Sharpe", font=dict(size=10, color=FONT)),
                tickfont=dict(size=9, color=FONT),
                thickness=12,
                len=0.7,
                borderwidth=0,
            ),
            line=dict(width=0),
        ),
        hovertemplate=(
            "Vol: %{x:.2%}<br>Return: %{y:.2%}<br>"
            "Sharpe: %{marker.color:.2f}<extra></extra>"
        ),
    ))

    # Trace 1: efficient frontier curve
    if ef_curve_v:
        fig_ef.add_trace(go.Scatter(
            x=ef_curve_v,
            y=ef_curve_r,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color=TEAL, width=2.5),
            hoverinfo="skip",
        ))

    # Trace 2: starred points
    star_x, star_y, star_text, star_col = [], [], [], []
    if ms_w is not None:
        star_x.append(ms_vol); star_y.append(ms_ret)
        star_text.append("Max Sharpe"); star_col.append(GREEN)
    if mv_w is not None:
        star_x.append(mv_vol); star_y.append(mv_ret)
        star_text.append("Min Variance"); star_col.append(BLUE)
    star_x.append(curr_vol); star_y.append(curr_ret)
    star_text.append("Current"); star_col.append(ORANGE)

    fig_ef.add_trace(go.Scatter(
        x=star_x,
        y=star_y,
        mode="markers+text",
        name="Key portfolios",
        marker=dict(
            symbol="star",
            size=18,
            color=star_col,
            line=dict(color="#161b27", width=1),
        ),
        text=star_text,
        textposition="top center",
        textfont=dict(size=10, color=FONT, family="DM Sans, sans-serif"),
        hovertemplate="%{text}<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    fig_ef.update_layout(
        **CHART_LAYOUT,
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
            bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    fig_ef.update_xaxes(**AXIS_STYLE)
    fig_ef.update_yaxes(**AXIS_STYLE)

    ef_event = st.plotly_chart(
        fig_ef,
        use_container_width=True,
        key="ef_scatter",
        on_select="rerun",
        selection_mode="points",
    )

    # ── Click → selected portfolio weights ────────────────────────────────────
    _sel_w: np.ndarray | None = None
    _sel_label = ""

    if ef_event and hasattr(ef_event, "selection") and ef_event.selection.points:
        pt         = ef_event.selection.points[0]
        curve_num  = pt.get("curve_number", -1)
        pt_idx     = pt.get("point_index", 0)

        if curve_num == 0 and pt_idx < len(mc_w):
            _sel_w     = mc_w[pt_idx]
            _sel_label = f"Random portfolio #{pt_idx + 1}"
        elif curve_num == 2 and pt_idx < len(star_text):
            _map = {"Max Sharpe": ms_w, "Min Variance": mv_w, "Current": curr_w}
            _sel_label = star_text[pt_idx]
            _sel_w     = _map.get(_sel_label)

    if _sel_w is not None:
        _r, _v, _s = _port_stats(_sel_w)
        st.markdown(
            f"<p style='font-size:0.83rem;color:{TEAL};font-weight:600;"
            f"font-family:\"DM Sans\",sans-serif;margin:0.5rem 0 0.2rem 0;'>"
            f"Selected: {_sel_label} &nbsp;·&nbsp; "
            f"Return {_r:.2%} &nbsp;·&nbsp; Vol {_v:.2%} &nbsp;·&nbsp; Sharpe {_s:.2f}"
            f"</p>",
            unsafe_allow_html=True,
        )
        _wdf = (
            pd.DataFrame({"Ticker": available, "Weight": _sel_w})
            .sort_values("Weight", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            _wdf.style.format({"Weight": "{:.2%}"}),
            use_container_width=False,
            hide_index=True,
        )

    st.divider()

    # ── Starred portfolio metrics comparison ──────────────────────────────────
    _subheader("Starred Portfolio Comparison")
    st.caption(
        "Annualised estimates from historical mean returns and sample covariance. "
        "Sharpe uses a 4% risk-free rate. Not a forecast of future performance."
    )

    starred: dict[str, tuple[np.ndarray, float, float, float]] = {}
    if ms_w is not None:
        starred["Maximum Sharpe ★"] = (ms_w, ms_ret, ms_vol, ms_sr)
    if mv_w is not None:
        starred["Minimum Variance ★"] = (mv_w, mv_ret, mv_vol, mv_sr)
    starred["Current Portfolio ★"] = (curr_w, curr_ret, curr_vol, curr_sr)

    star_cols = st.columns(len(starred))
    for col, (label, (w_s, r_s, v_s, s_s)) in zip(star_cols, starred.items()):
        w_series = pd.Series(w_s, index=available)
        positive = w_series[w_series > 1e-3]
        with col:
            st.markdown(
                f"<p style='font-size:0.80rem;font-weight:700;color:{TEAL};"
                f"font-family:\"DM Sans\",sans-serif;margin:0 0 0.5rem 0;'>{label}</p>",
                unsafe_allow_html=True,
            )
            st.metric("Ann. Return (est.)", f"{r_s:.2%}")
            st.metric("Ann. Vol (est.)",    f"{v_s:.2%}")
            st.metric("Sharpe (est.)",      f"{s_s:.2f}")
            st.metric("Max Weight",         f"{w_series.max():.2%}")
            st.metric(
                "Min Weight",
                f"{positive.min():.2%}" if not positive.empty else "—",
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK PARITY
# ════════════════════════════════════════════════════════════════════════════════

with tab_rp:
    st.caption(
        "Equal allocation by weight is rarely equal allocation by risk. "
        "Three risk-parity methods redistribute risk more evenly across holdings."
    )

    # ── Compute variants ──────────────────────────────────────────────────────
    iv     = 1.0 / (vols_ann + 1e-12)
    naive_w = iv / iv.sum()
    erc_w   = _erc_weights()
    maxd_w  = _max_div_weights()

    # Risk contributions (normalised to sum to 100%)
    def _rc_pct(w: np.ndarray) -> np.ndarray:
        rc = _risk_contribs(w)
        return rc / (rc.sum() + 1e-12)

    naive_rc_pct = _rc_pct(naive_w)
    erc_rc_pct   = _rc_pct(erc_w)
    maxd_rc_pct  = _rc_pct(maxd_w)
    curr_rc_pct  = _rc_pct(curr_w)

    # ── Key insight card ──────────────────────────────────────────────────────
    top2_idx   = np.argsort(curr_rc_pct)[-2:][::-1]
    top2_ticks = [available[i] for i in top2_idx]
    top2_share = float(curr_rc_pct[top2_idx].sum())

    # Concentration in current vs ERC
    curr_hhi = float(np.sum(curr_rc_pct ** 2))  # Herfindahl–Hirschman
    erc_hhi  = float(np.sum(erc_rc_pct ** 2))
    conc_reduction = (curr_hhi - erc_hhi) / (curr_hhi + 1e-12)

    st.markdown(
        f"<div style='background:#1a1e2a;border:1px solid #252836;border-radius:8px;"
        f"padding:0.8rem 1.1rem;margin:0.4rem 0 1rem 0;'>"
        f"<p style='font-size:0.85rem;color:#e8ecf0;margin:0;font-family:\"DM Sans\",sans-serif;'>"
        f"<span style='color:{ORANGE};font-weight:700;'>Key insight — risk ≠ weight:</span> "
        f"In your current portfolio, <b>{top2_ticks[0]}</b> and <b>{top2_ticks[1]}</b> "
        f"account for <span style='color:{RED};font-weight:700;'>{top2_share:.0%}</span> of total portfolio risk "
        f"despite a smaller weight allocation. "
        f"Switching to ERC reduces the risk concentration index by "
        f"<span style='color:{GREEN};font-weight:700;'>{conc_reduction:.0%}</span>."
        f"</p></div>",
        unsafe_allow_html=True,
    )

    # ── Three-column layout ───────────────────────────────────────────────────
    variants = [
        (
            "1 · Naive Inverse-Vol",
            "w<sub>i</sub> ∝ 1/σ<sub>i</sub> · Simple, no covariance required.",
            naive_w, naive_rc_pct,
        ),
        (
            "2 · Equal Risk Contribution (ERC)",
            "Solve so RC<sub>i</sub> = w<sub>i</sub>(Σw)<sub>i</sub>/(w′Σw) is equal for all i.",
            erc_w, erc_rc_pct,
        ),
        (
            "3 · Maximum Diversification",
            "Maximise (w′σ) / √(w′Σw) — the diversification ratio.",
            maxd_w, maxd_rc_pct,
        ),
    ]

    rp_cols = st.columns(3)
    for col, (label, desc, w_arr, rc_arr) in zip(rp_cols, variants):
        with col:
            st.markdown(
                f"<p style='font-size:0.84rem;font-weight:700;color:{TEAL};"
                f"font-family:\"DM Sans\",sans-serif;margin:0 0 0.1rem 0;'>{label}</p>"
                f"<p style='font-size:0.72rem;color:{GREY};"
                f"font-family:\"DM Sans\",sans-serif;margin:0 0 0.6rem 0;'>{desc}</p>",
                unsafe_allow_html=True,
            )

            # Donut
            st.plotly_chart(
                _donut_fig(w_arr, available, ""),
                use_container_width=True,
            )

            # Combined weights + risk-contrib table
            wdf = pd.DataFrame({
                "Ticker":       available,
                "Weight":       w_arr,
                "Risk Contrib": rc_arr,
            }).sort_values("Weight", ascending=False).reset_index(drop=True)

            st.dataframe(
                wdf.style
                    .format({"Weight": "{:.2%}", "Risk Contrib": "{:.2%}"})
                    .background_gradient(cmap="YlOrRd", subset=["Risk Contrib"], axis=0),
                use_container_width=True,
                hide_index=True,
            )

            # Risk contribution horizontal bar
            rc_sorted = sorted(zip(available, rc_arr), key=lambda x: x[1])
            rc_ticks  = [x[0] for x in rc_sorted]
            rc_vals   = [x[1] * 100 for x in rc_sorted]

            fig_rc = go.Figure(go.Bar(
                x=rc_vals,
                y=rc_ticks,
                orientation="h",
                marker=dict(color=TEAL, opacity=0.82),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_rc.update_layout(
                **CHART_LAYOUT,
                xaxis_title="Risk Contribution (%)",
                xaxis_ticksuffix="%",
                margin=dict(l=0, r=0, t=8, b=0),
                height=max(160, n * 28),
                showlegend=False,
            )
            fig_rc.update_xaxes(**AXIS_STYLE)
            fig_rc.update_yaxes(**{**AXIS_STYLE, "tickfont": dict(color=FONT, size=9)})
            st.plotly_chart(fig_rc, use_container_width=True)

    # ── Summary comparison table ──────────────────────────────────────────────
    st.divider()
    _subheader("Risk Parity Summary")

    def _summary_row(name: str, w: np.ndarray, rc: np.ndarray) -> dict:
        r, v, s = _port_stats(w)
        return {
            "Method":             name,
            "Ann. Return (est.)": f"{r:.2%}",
            "Ann. Vol (est.)":    f"{v:.2%}",
            "Sharpe (est.)":      f"{s:.2f}",
            "Max Weight":         f"{w.max():.2%}",
            "Max Risk Contrib":   f"{rc.max():.2%}",
            "HHI (risk conc.)":   f"{np.sum(rc ** 2):.4f}",
        }

    summary_df = pd.DataFrame([
        _summary_row("Current Portfolio",       curr_w,  curr_rc_pct),
        _summary_row("Naive Inverse-Vol",        naive_w, naive_rc_pct),
        _summary_row("Equal Risk Contribution",  erc_w,   erc_rc_pct),
        _summary_row("Maximum Diversification",  maxd_w,  maxd_rc_pct),
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.caption(
        "HHI = Σ RC²ᵢ (Herfindahl–Hirschman Index of risk concentration). "
        "Lower is more diversified. Minimum possible = 1/n."
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — BLACK-LITTERMAN
# ════════════════════════════════════════════════════════════════════════════════

with tab_bl:
    st.caption(
        "Black-Litterman starts from the market-implied equilibrium (reverse CAPM), "
        "then blends in your forward-looking views using Bayesian updating. "
        "The posterior expected returns feed a mean-variance optimisation. "
        "Reference: Black, F. & Litterman, R. (1992). "
        "Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28–43."
    )

    # ── Step 1: equilibrium returns ───────────────────────────────────────────
    _subheader("Step 1 — Market-Implied Equilibrium Returns (Prior)")
    st.caption(
        "Π = δ · Σ · w_mkt via reverse optimisation. "
        "Market weights approximated by equal weight (no live market-cap data). "
        "Risk aversion δ = 2.5 (Idzorek, 2005 calibration)."
    )

    delta   = 2.5
    mkt_w   = np.ones(n) / n
    pi_arr  = delta * cov_arr @ mkt_w
    pi_s    = pd.Series(pi_arr, index=available)

    # Bar chart of equilibrium returns
    pi_sorted = pi_s.sort_values(ascending=False)
    fig_pi = go.Figure(go.Bar(
        x=list(pi_sorted.index),
        y=list(pi_sorted.values),
        marker_color=[_rgba(BLUE, 0.85)] * len(pi_sorted),
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ))
    fig_pi.update_layout(
        **CHART_LAYOUT,
        yaxis_title="Implied Equilibrium Return",
        yaxis_tickformat=".1%",
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        showlegend=False,
    )
    fig_pi.update_xaxes(**AXIS_STYLE)
    fig_pi.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_pi, use_container_width=True)

    st.divider()

    # ── Step 2: user views ────────────────────────────────────────────────────
    _subheader("Step 2 — Your Views (Absolute Expected Annual Returns)")
    st.caption(
        "Enter your expected annual return for each asset you have a view on. "
        "Leave at 0.0% to exclude that asset from the views. "
        "Confidence (0.1 = very uncertain, 1.0 = very certain) controls how much "
        "the posterior shifts toward your view vs the equilibrium."
    )

    view_values: dict[str, float]      = {}
    view_confidences: dict[str, float] = {}

    n_view_cols = min(n, 4)
    view_grid   = st.columns(n_view_cols)
    for i, ticker in enumerate(available):
        with view_grid[i % n_view_cols]:
            vv = st.number_input(
                f"{ticker} (%/yr)",
                min_value=-50.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                format="%.1f",
                key=f"bl_view_{ticker}",
            )
            vc = st.slider(
                f"Confidence",
                min_value=0.10,
                max_value=1.0,
                value=0.50,
                step=0.05,
                key=f"bl_conf_{ticker}",
                label_visibility="collapsed",
            )
            if abs(vv) > 0.005:
                view_values[ticker]      = vv / 100.0
                view_confidences[ticker] = vc

    st.divider()

    # ── Step 3: posterior ─────────────────────────────────────────────────────
    _subheader("Step 3 — Posterior Returns (Prior ⊕ Views)")

    mu_bl_s: pd.Series
    S_bl_df: pd.DataFrame
    bl_computed = False

    if view_values:
        try:
            view_conf_list = [view_confidences[k] for k in view_values]
            bl_model = BlackLittermanModel(
                S_pf,
                pi=pi_s,
                absolute_views=view_values,
                omega="idzorek",
                view_confidences=view_conf_list,
            )
            mu_bl_s    = bl_model.bl_returns()
            S_bl_df    = bl_model.bl_cov()
            bl_computed = True
        except Exception as exc:
            st.warning(f"Black-Litterman blending failed: {exc}")
            mu_bl_s = pi_s.copy()
            S_bl_df = S_pf.copy()
    else:
        st.info(
            "No views entered — showing equilibrium returns as posterior. "
            "Enter expected returns above to see the Bayesian update.",
            icon="ℹ️",
        )
        mu_bl_s = pi_s.copy()
        S_bl_df = S_pf.copy()

    # Prior vs posterior grouped bar
    fig_bl = go.Figure()
    fig_bl.add_trace(go.Bar(
        x=list(pi_s.index),
        y=list(pi_s.values),
        name="Equilibrium (Prior)",
        marker_color=_rgba(GREY, 0.80),
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ))
    fig_bl.add_trace(go.Bar(
        x=list(mu_bl_s.index),
        y=list(mu_bl_s.values),
        name="Posterior (BL)",
        marker_color=_rgba(TEAL, 0.90),
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ))
    fig_bl.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        yaxis_title="Expected Annual Return",
        yaxis_tickformat=".1%",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
            bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=320,
    )
    fig_bl.update_xaxes(**AXIS_STYLE)
    fig_bl.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig_bl, use_container_width=True)

    st.divider()

    # ── Step 4: optimal portfolio from posterior ──────────────────────────────
    _subheader("Step 4 — Optimal Portfolio from Posterior")
    st.caption(
        "Maximum Sharpe portfolio optimised using the Black-Litterman posterior returns "
        "and posterior covariance. Long-only, weights sum to 1."
    )

    bl_w:  np.ndarray | None = None
    bl_ret, bl_vol, bl_sr    = 0.0, 0.0, 0.0

    try:
        ef_bl = EfficientFrontier(mu_bl_s, S_bl_df, weight_bounds=(0, 1))
        ef_bl.max_sharpe(risk_free_rate=RF_RATE)
        bl_d  = ef_bl.clean_weights()
        bl_w  = np.array([bl_d.get(t, 0.0) for t in available])
        bl_ret, bl_vol, bl_sr = _port_stats(bl_w)
    except Exception as exc:
        st.warning(f"Optimisation failed: {exc}")

    if bl_w is not None:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Est. Annual Return", f"{bl_ret:.2%}")
        with c2: st.metric("Est. Annual Vol",    f"{bl_vol:.2%}")
        with c3: st.metric("Sharpe (est.)",      f"{bl_sr:.2f}")

        bl_wdf = (
            pd.DataFrame({"Ticker": available, "Weight": bl_w})
            .sort_values("Weight", ascending=False)
            .reset_index(drop=True)
        )
        active_bl = bl_wdf[bl_wdf["Weight"] > 1e-3].reset_index(drop=True)

        col_tbl, col_chart = st.columns([1, 1])
        with col_tbl:
            st.dataframe(
                active_bl.style.format({"Weight": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
        with col_chart:
            st.plotly_chart(
                _donut_fig(
                    active_bl["Weight"].values,
                    active_bl["Ticker"].tolist(),
                    "BL Optimal",
                ),
                use_container_width=True,
            )

        st.divider()

        if st.button(
            "⬆ Adopt as Active Portfolio",
            type="primary",
            key="bl_adopt",
            help="Overwrites the active portfolio holdings in session state. "
                 "Return to Portfolio Hub and click Run Portfolio to recompute analytics.",
        ):
            new_h = {t: float(w) for t, w in zip(available, bl_w) if w > 1e-3}
            total_h = sum(new_h.values())
            st.session_state["portfolio"]["holdings"] = {t: w / total_h for t, w in new_h.items()}
            st.success(
                "Active portfolio updated. Return to **Portfolio Hub** and click "
                "**Run Portfolio** to recompute all analytics.",
                icon="✅",
            )

    st.markdown(
        "<p style='font-size:0.75rem;color:#636b78;margin:0.8rem 0 0 0;"
        "font-family:\"DM Sans\",sans-serif;'>"
        "⚠ Estimates are highly sensitive to view inputs, risk aversion δ, and the historical "
        "covariance window. Market weights are approximated by equal weight. "
        "Posterior shifts are Bayesian-weighted — large confidence on imprecise views can "
        "dominate the equilibrium prior. This is not investment advice."
        "</p>",
        unsafe_allow_html=True,
    )
