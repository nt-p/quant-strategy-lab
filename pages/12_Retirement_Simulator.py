"""Page 12: Retirement Outcome Simulator.

Standalone — does not require an active portfolio.
Monte Carlo simulation of retirement wealth paths.
Quantifies probability of reaching a user-defined income goal and the
long-run cost of fees.

References
----------
Bengen, W.P. (1994). Determining withdrawal rates using historical data.
    Journal of Financial Planning, 7(4), 171–180.
Cooley, P.L., Hubbard, C.M. & Walz, D.T. (1998). Retirement savings:
    Choosing a withdrawal rate that is sustainable. AAII Journal, 20(2), 16–21.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Retirement Simulator", page_icon="🏖", layout="wide")

from ui.theme import apply_theme, apply_theme_to_plotly_figure  # noqa: E402

apply_theme()

# ── Return assumptions ─────────────────────────────────────────────────────────

PORTFOLIO_ASSUMPTIONS: dict[str, dict[str, float]] = {
    "Conservative (30/70)": {"mean": 0.055, "vol": 0.075},
    "Balanced (60/40)":     {"mean": 0.075, "vol": 0.110},
    "Growth (80/20)":       {"mean": 0.090, "vol": 0.145},
    "All Equities (100/0)": {"mean": 0.100, "vol": 0.170},
}

_SHORT_LABELS: dict[str, str] = {
    "Conservative (30/70)": "Conservative",
    "Balanced (60/40)":     "Balanced",
    "Growth (80/20)":       "Growth",
    "All Equities (100/0)": "All Equities",
}

_FEE_SENSITIVITY_COLS = [0.07, 0.20, 0.50, 1.00]


# ── Simulation ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_simulation(
    current_value: float,
    monthly_contribution: float,
    years_to_retirement: int,
    annual_fee: float,
    mean_return: float,
    vol_return: float,
    n_simulations: int,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a vectorised Monte Carlo retirement simulation.

    Parameters
    ----------
    current_value : float
        Starting portfolio value in dollars.
    monthly_contribution : float
        Fixed monthly contribution in dollars.
    years_to_retirement : int
        Number of years until retirement.
    annual_fee : float
        Annual fee as a decimal (e.g. 0.002 for 0.20%).
    mean_return : float
        Expected gross annual return as a decimal.
    vol_return : float
        Annual return volatility as a decimal.
    n_simulations : int
        Number of Monte Carlo paths.
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    terminal_values : np.ndarray, shape (n_simulations,)
        Portfolio value at retirement for each path.
    path_percentiles : np.ndarray, shape (years_to_retirement, 5)
        P10, P25, P50, P75, P90 at each annual time step.
    """
    rng = np.random.default_rng(random_seed)
    log_mean = np.log(1 + mean_return - annual_fee) - 0.5 * vol_return**2
    annual_returns = rng.normal(log_mean, vol_return, size=(n_simulations, years_to_retirement))
    gross_factors = np.exp(annual_returns)
    annual_contribution = monthly_contribution * 12

    paths = np.zeros((n_simulations, years_to_retirement))
    portfolio = np.full(n_simulations, current_value, dtype=float)
    for t in range(years_to_retirement):
        portfolio = (portfolio + annual_contribution) * gross_factors[:, t]
        paths[:, t] = portfolio

    terminal_values = paths[:, -1]
    path_percentiles = np.percentile(paths, [10, 25, 50, 75, 90], axis=0).T
    return terminal_values, path_percentiles


# ── Sidebar: inputs ────────────────────────────────────────────────────────────

st.sidebar.markdown(
    """
    <div style='padding:0.6rem 0 0.9rem 0;border-bottom:1px solid #252836;
                margin-bottom:0.5rem;'>
        <div style='font-size:1.05rem;font-weight:700;color:#4ecdc4;
                    letter-spacing:-0.3px;font-family:"DM Sans",sans-serif;'>
            QuantScope
        </div>
        <div style='font-size:0.62rem;color:#636b78;letter-spacing:2.5px;
                    font-weight:600;margin-top:4px;font-family:"DM Sans",sans-serif;'>
            PORTFOLIO &nbsp;·&nbsp; RESEARCH &nbsp;·&nbsp; TERMINAL
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Your Profile")

current_age: int = st.sidebar.slider("Current age", min_value=18, max_value=70, value=30)
retirement_age: int = st.sidebar.slider(
    "Retirement age",
    min_value=current_age + 1,
    max_value=80,
    value=min(65, max(current_age + 1, 65)),
)

st.sidebar.header("Portfolio")

current_value: float = float(
    st.sidebar.number_input(
        "Current savings ($)",
        min_value=0,
        max_value=10_000_000,
        value=50_000,
        step=1_000,
        format="%d",
    )
)
monthly_contribution: float = float(
    st.sidebar.number_input(
        "Monthly contribution ($)",
        min_value=0,
        max_value=20_000,
        value=1_000,
        step=100,
        format="%d",
    )
)
portfolio_type: str = st.sidebar.selectbox(
    "Asset allocation",
    options=list(PORTFOLIO_ASSUMPTIONS.keys()),
    index=1,
)
annual_fee_pct: float = st.sidebar.slider(
    "Annual fee (%)",
    min_value=0.00,
    max_value=2.00,
    value=0.20,
    step=0.01,
    format="%.2f",
)

st.sidebar.header("Goal")

target_annual_income: float = float(
    st.sidebar.number_input(
        "Target retirement income ($/yr)",
        min_value=0,
        max_value=500_000,
        value=60_000,
        step=1_000,
        format="%d",
    )
)
safe_withdrawal_rate: float = st.sidebar.number_input(
    "Safe withdrawal rate (%)",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.1,
    format="%.1f",
)

st.sidebar.header("Simulation")

n_simulations: int = st.sidebar.selectbox(
    "Number of simulations",
    options=[500, 1_000, 5_000],
    index=1,
)

st.sidebar.markdown("---")
_dark = st.session_state.get("dark_mode", True)
st.sidebar.toggle("🌙 Dark mode" if _dark else "☀️ Light mode", value=_dark, key="dark_mode")

# ── Derived values ─────────────────────────────────────────────────────────────

years_to_retirement: int = retirement_age - current_age
annual_fee: float = annual_fee_pct / 100.0
required_nest_egg: float = target_annual_income / (safe_withdrawal_rate / 100.0)

assumptions = PORTFOLIO_ASSUMPTIONS[portfolio_type]
terminal_values, path_percentiles = run_simulation(
    current_value=current_value,
    monthly_contribution=monthly_contribution,
    years_to_retirement=years_to_retirement,
    annual_fee=annual_fee,
    mean_return=assumptions["mean"],
    vol_return=assumptions["vol"],
    n_simulations=n_simulations,
)

prob_success: float = float((terminal_values >= required_nest_egg).mean() * 100)
median_wealth: float = float(np.median(terminal_values))
p10_wealth: float = float(np.percentile(terminal_values, 10))
p90_wealth: float = float(np.percentile(terminal_values, 90))

current_year = datetime.date.today().year

# ── Page header ───────────────────────────────────────────────────────────────

st.title("Retirement Outcome Simulator")
st.caption(
    "How likely are you to reach your retirement goal? "
    "Adjust your inputs to explore how contributions, asset allocation, "
    "and fees affect your long-term outcomes."
)

st.info(
    f"To generate **${target_annual_income:,.0f}/year** at a "
    f"**{safe_withdrawal_rate:.1f}%** withdrawal rate, "
    f"you need **${required_nest_egg:,.0f}** at retirement.",
    icon="🎯",
)

# ── Row 1: KPI cards ──────────────────────────────────────────────────────────

if prob_success >= 70:
    prob_color = "#68d391"
    prob_label = "On track"
elif prob_success >= 40:
    prob_color = "#f6ad55"
    prob_label = "Borderline"
else:
    prob_color = "#f56565"
    prob_label = "Off track"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Probability of success", f"{prob_success:.0f}%")
    st.markdown(
        f"<p style='font-size:0.75rem;color:{prob_color};font-weight:600;"
        f"margin:-0.6rem 0 0 0;font-family:\"DM Sans\",sans-serif;'>"
        f"● {prob_label}</p>",
        unsafe_allow_html=True,
    )
with k2:
    st.metric("Median at retirement", f"${median_wealth:,.0f}")
with k3:
    st.metric("P10 outcome (bad luck)", f"${p10_wealth:,.0f}")
with k4:
    st.metric("P90 outcome (good luck)", f"${p90_wealth:,.0f}")

st.markdown("---")

# ── Row 2: Fan chart ──────────────────────────────────────────────────────────

years_axis = list(range(current_year + 1, current_year + years_to_retirement + 1))

p10 = path_percentiles[:, 0]
p25 = path_percentiles[:, 1]
p50 = path_percentiles[:, 2]
p75 = path_percentiles[:, 3]
p90 = path_percentiles[:, 4]


def _band(
    x: list[int],
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    fillcolor: str,
    name: str = "",
) -> go.Scatter:
    """Closed polygon band between two percentile lines."""
    return go.Scatter(
        x=x + x[::-1],
        y=list(y_upper) + list(y_lower[::-1]),
        fill="toself",
        fillcolor=fillcolor,
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=bool(name),
        name=name,
        hoverinfo="skip",
    )


fig_fan = go.Figure()

# Bands (bottom to top: red, teal, green)
fig_fan.add_trace(_band(years_axis, p10, p25, "rgba(216,90,48,0.15)", "P10–P25"))
fig_fan.add_trace(_band(years_axis, p25, p75, "rgba(29,158,117,0.20)", "P25–P75"))
fig_fan.add_trace(_band(years_axis, p75, p90, "rgba(29,158,117,0.12)", "P75–P90"))

# Percentile lines
fig_fan.add_trace(go.Scatter(
    x=years_axis, y=p10,
    line=dict(color="rgba(216,90,48,0.7)", width=1),
    showlegend=True, name="P10",
    hovertemplate="P10: $%{y:,.0f}<extra></extra>",
))
fig_fan.add_trace(go.Scatter(
    x=years_axis, y=p90,
    line=dict(color="rgba(29,158,117,0.7)", width=1),
    showlegend=True, name="P90",
    hovertemplate="P90: $%{y:,.0f}<extra></extra>",
))
fig_fan.add_trace(go.Scatter(
    x=years_axis, y=p50,
    line=dict(color="#4ecdc4", width=2.5),
    showlegend=True, name="Median",
    hovertemplate="Median: $%{y:,.0f}<extra></extra>",
))

# Goal line
fig_fan.add_hline(
    y=required_nest_egg,
    line_dash="dash",
    line_color="#f6ad55",
    line_width=1.5,
    annotation_text=f"Goal: ${required_nest_egg:,.0f}",
    annotation_position="top right",
    annotation_font_color="#f6ad55",
)

fig_fan.update_layout(
    title=f"Portfolio value over {years_to_retirement} years ({n_simulations:,} simulations)",
    xaxis_title="Year",
    yaxis_title="Portfolio value ($)",
    yaxis_tickformat="$,.0f",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=420,
)
apply_theme_to_plotly_figure(fig_fan)
st.plotly_chart(fig_fan, use_container_width=True)

# ── Row 3: Histogram | Fee impact ─────────────────────────────────────────────

col_hist, col_fee = st.columns(2)

# ── Left: outcome distribution histogram ──────────────────────────────────────
with col_hist:
    counts, bin_edges = np.histogram(terminal_values, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_colors = [
        "rgba(216,90,48,0.75)" if c < required_nest_egg else "rgba(29,158,117,0.75)"
        for c in bin_centers
    ]

    fig_hist = go.Figure(go.Bar(
        x=bin_centers,
        y=counts,
        marker_color=bar_colors,
        hovertemplate="$%{x:,.0f}: %{y} paths<extra></extra>",
    ))
    fig_hist.add_vline(
        x=required_nest_egg,
        line_dash="dash",
        line_color="#f6ad55",
        line_width=1.5,
        annotation_text=f"{prob_success:.0f}% reached the goal",
        annotation_position="top right",
        annotation_font_color="#f6ad55",
    )
    fig_hist.update_layout(
        title="Distribution of retirement outcomes",
        xaxis_title="Portfolio value at retirement ($)",
        yaxis_title="Number of simulations",
        xaxis_tickformat="$,.0f",
        showlegend=False,
        height=360,
    )
    apply_theme_to_plotly_figure(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Right: fee impact bar chart ───────────────────────────────────────────────
with col_fee:
    fee_levels_pct = [0.07, 0.20, 0.50, 1.00, 1.50]
    fee_medians: list[float] = []
    for fl in fee_levels_pct:
        tv_fee, _ = run_simulation(
            current_value=current_value,
            monthly_contribution=monthly_contribution,
            years_to_retirement=years_to_retirement,
            annual_fee=fl / 100.0,
            mean_return=assumptions["mean"],
            vol_return=assumptions["vol"],
            n_simulations=n_simulations,
        )
        fee_medians.append(float(np.median(tv_fee)))

    baseline_median = fee_medians[0]  # 0.07% = index ETF baseline

    bar_colors_fee = [
        "#4ecdc4" if abs(fl - annual_fee_pct) < 0.005 else "#3a4155"
        for fl in fee_levels_pct
    ]
    annotations_fee = []
    for i, (fl, fm) in enumerate(zip(fee_levels_pct, fee_medians)):
        loss = fm - baseline_median
        label = "baseline" if i == 0 else f"−${abs(loss):,.0f} vs index ETF"
        annotations_fee.append(dict(
            x=f"{fl:.2f}%",
            y=fm,
            text=label,
            showarrow=False,
            yanchor="bottom",
            font=dict(size=9, color="#a0a8b8"),
            yshift=4,
        ))

    fig_fee = go.Figure(go.Bar(
        x=[f"{fl:.2f}%" for fl in fee_levels_pct],
        y=fee_medians,
        marker_color=bar_colors_fee,
        hovertemplate="%{x} fee: $%{y:,.0f} median<extra></extra>",
    ))
    fig_fee.update_layout(
        title=f"Cost of fees over {years_to_retirement} years",
        xaxis_title="Annual fee",
        yaxis_title="Median portfolio at retirement ($)",
        yaxis_tickformat="$,.0f",
        annotations=annotations_fee,
        showlegend=False,
        height=360,
    )
    apply_theme_to_plotly_figure(fig_fee)
    st.plotly_chart(fig_fee, use_container_width=True)

# ── Row 4: Sensitivity table ──────────────────────────────────────────────────

st.subheader("Probability of success: allocation × fee")
st.caption(
    "% of simulations reaching the retirement goal. "
    "Green ≥ 70% · Amber 40–70% · Red < 40%. "
    "Bold = your current selection."
)

sens_rows: dict[str, list[float]] = {}
for pt, params in PORTFOLIO_ASSUMPTIONS.items():
    row: list[float] = []
    for fl in _FEE_SENSITIVITY_COLS:
        tv_s, _ = run_simulation(
            current_value=current_value,
            monthly_contribution=monthly_contribution,
            years_to_retirement=years_to_retirement,
            annual_fee=fl / 100.0,
            mean_return=params["mean"],
            vol_return=params["vol"],
            n_simulations=n_simulations,
        )
        row.append(float((tv_s >= required_nest_egg).mean() * 100))
    sens_rows[_SHORT_LABELS[pt]] = row

sens_df = pd.DataFrame(
    sens_rows,
    index=[f"{fl:.2f}%" for fl in _FEE_SENSITIVITY_COLS],
).T  # rows = portfolio type, columns = fee level

# Round to integers for display
sens_display = sens_df.round(0).astype(int)


def _cell_style(val: float) -> str:
    if val >= 70:
        return "background-color: rgba(72,187,120,0.22); color: #68d391"
    elif val >= 40:
        return "background-color: rgba(246,173,85,0.20); color: #f6ad55"
    return "background-color: rgba(245,101,101,0.22); color: #f56565"


def _format_cell(val: int, row_label: str, col_label: str) -> str:
    is_current = (
        row_label == _SHORT_LABELS[portfolio_type]
        and col_label == f"{annual_fee_pct:.2f}%"
    )
    return f"**{val}%**" if is_current else f"{val}%"


styled = sens_display.style.applymap(_cell_style)
st.dataframe(styled, use_container_width=True)

# ── Caveats ───────────────────────────────────────────────────────────────────

with st.expander("Methodology and assumptions"):
    st.markdown("""
- Returns are simulated using a log-normal model calibrated to long-run
  historical asset class returns. Past returns do not guarantee future results.
- Inflation is not explicitly modelled. All figures are in today's dollars
  (real returns assumed).
- The 4% safe withdrawal rate is based on Bengen (1994) and the Trinity Study
  (Cooley et al., 1998). It may not be appropriate for all investors or
  market conditions.
- This tool is for educational and illustrative purposes only.
  It is not financial advice.

**References**

Bengen, W.P. (1994). Determining withdrawal rates using historical data.
*Journal of Financial Planning*, 7(4), 171–180.

Cooley, P.L., Hubbard, C.M. & Walz, D.T. (1998). Retirement savings:
Choosing a withdrawal rate that is sustainable. *AAII Journal*, 20(2), 16–21.
    """)
