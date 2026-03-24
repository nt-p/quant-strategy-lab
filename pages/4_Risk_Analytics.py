"""Page 4: Risk Analytics — distributional and factor risk analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Risk Analytics", page_icon="📐", layout="wide")

from engine.backtest import BacktestResult  # noqa: E402
from modules.risk import (  # noqa: E402
    compute_beta_alpha,
    compute_correlation_matrix,
    compute_cvar,
    compute_drawdown_stats,
    compute_r_squared,
    compute_tail_stats,
    compute_var,
)
from ui.colors import assign_colors  # noqa: E402
from ui.theme import AXIS_STYLE, BG, CHART_LAYOUT, FONT, GRID, PLOT_BG, inject_css  # noqa: E402

inject_css()


# ── Local helper ──────────────────────────────────────────────────────────────

def _hex_with_alpha(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' + alpha float to an 'rgba(r,g,b,a)' string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


st.markdown(
    "<h2 style='font-size:1.4rem;font-weight:700;letter-spacing:-0.3px;"
    "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;margin-bottom:0.1rem;'>"
    "📐 Risk Analytics</h2>",
    unsafe_allow_html=True,
)

# ── Guard: require a completed backtest ───────────────────────────────────────

results_state = st.session_state.get("backtest_results")

if not results_state:
    st.info("Run a backtest from the home page to unlock risk analytics.")
    st.markdown(
        """
### How to get started
1. Go to the **home page** (QuantScope in the sidebar)
2. Pick assets, set a date range, and click **Load Data**
3. Toggle strategies and click **Run Backtest**
4. Return here for distributional and factor risk analysis
        """
    )
    st.stop()

strategy_results: list[BacktestResult] = results_state["strategy_results"]
benchmark: BacktestResult = results_state["benchmark"]

all_results = strategy_results + [benchmark]
colors = assign_colors(all_results)

# ── Strategy selector ─────────────────────────────────────────────────────────

st.caption(
    "Select which strategies to include in the analysis. "
    "The benchmark (SPY) is always available for comparison."
)

name_to_result: dict[str, BacktestResult] = {r.strategy_name: r for r in all_results}
all_names = [r.strategy_name for r in strategy_results]   # exclude benchmark from default
benchmark_name = benchmark.strategy_name

selected_names: list[str] = st.multiselect(
    "Strategies",
    options=all_names,
    default=all_names,
    label_visibility="collapsed",
)

include_benchmark = st.checkbox(
    f"Include benchmark ({benchmark_name})",
    value=True,
)

if not selected_names:
    st.warning("Select at least one strategy to display.")
    st.stop()

selected_results = [name_to_result[n] for n in selected_names]
if include_benchmark:
    selected_results = selected_results + [benchmark]

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_dist, tab_dd, tab_corr, tab_factor = st.tabs(
    ["Return Distribution", "Drawdown Analysis", "Correlation Matrix", "Factor Exposure"]
)


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Return Distribution
# ════════════════════════════════════════════════════════════════════════════

with tab_dist:
    st.caption(
        "Histogram of daily returns with VaR/CVaR overlays at the 95% confidence level. "
        "Skewness < 0 indicates a left tail (more crash risk than boom upside)."
    )

    # ── Histogram ──────────────────────────────────────────────────────────
    fig_hist = go.Figure()

    for r in selected_results:
        rets = np.array(r.returns[1:], dtype=float) * 100   # convert to %
        is_benchmark = r.category == "benchmark"
        color = colors[r.strategy_id]

        fig_hist.add_trace(
            go.Histogram(
                x=rets,
                name=r.strategy_name,
                nbinsx=80,
                opacity=0.55 if not is_benchmark else 0.35,
                marker_color=color,
                hovertemplate=(
                    f"{r.strategy_name}<br>"
                    "Return: %{x:.2f}%<br>"
                    "Count: %{y}"
                    "<extra></extra>"
                ),
            )
        )

    # VaR / CVaR lines — draw for each selected strategy
    for r in selected_results:
        color = colors[r.strategy_id]
        var_val = compute_var(r.returns) * 100
        cvar_val = compute_cvar(r.returns) * 100

        if not math.isnan(var_val):
            fig_hist.add_vline(
                x=-var_val,
                line=dict(color=color, width=1.2, dash="dash"),
                annotation_text=f"VaR {r.strategy_name[:6]}",
                annotation_font=dict(color=color, size=9),
                annotation_position="top right",
            )
        if not math.isnan(cvar_val):
            fig_hist.add_vline(
                x=-cvar_val,
                line=dict(color=color, width=1.0, dash="dot"),
                annotation_text=f"CVaR {r.strategy_name[:6]}",
                annotation_font=dict(color=color, size=9),
                annotation_position="bottom right",
            )

    fig_hist.update_layout(
        **CHART_LAYOUT,
        barmode="overlay",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
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

    # ── Per-strategy stat cards ────────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:0.5rem 0 0.25rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Distribution Statistics</h3>",
        unsafe_allow_html=True,
    )

    rows = []
    for r in selected_results:
        ts = compute_tail_stats(r.returns)
        var_val = compute_var(r.returns)
        cvar_val = compute_cvar(r.returns)
        rows.append(
            {
                "Strategy": r.strategy_name,
                "Skewness": ts["skewness"],
                "Excess Kurtosis": ts["excess_kurtosis"],
                "Tail Ratio": ts["tail_ratio"],
                "VaR 95%": var_val,
                "CVaR 95%": cvar_val,
                "Best Day": ts["best_day"],
                "Worst Day": ts["worst_day"],
                "% Positive Days": ts["pct_positive"],
            }
        )

    df_dist = pd.DataFrame(rows).set_index("Strategy")

    fmt = {
        "Skewness": "{:.3f}",
        "Excess Kurtosis": "{:.3f}",
        "Tail Ratio": "{:.2f}",
        "VaR 95%": "{:.2%}",
        "CVaR 95%": "{:.2%}",
        "Best Day": "{:+.2%}",
        "Worst Day": "{:+.2%}",
        "% Positive Days": "{:.1%}",
    }

    # Highlight: lower VaR/CVaR is better; higher skewness (less negative) is better
    styler = (
        df_dist.style
        .format(fmt, na_rep="—")
        .background_gradient(subset=["VaR 95%", "CVaR 95%"], cmap="RdYlGn_r", axis=0)
        .background_gradient(subset=["Skewness"], cmap="RdYlGn", axis=0)
        .background_gradient(subset=["Best Day", "Worst Day", "% Positive Days"], cmap="RdYlGn", axis=0)
    )

    st.dataframe(styler, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Drawdown Analysis
# ════════════════════════════════════════════════════════════════════════════

with tab_dd:
    st.caption(
        "Underwater equity curves showing peak-to-trough loss at each point in time, "
        "with per-strategy drawdown statistics below."
    )

    # ── Underwater chart ───────────────────────────────────────────────────
    date_sets = [set(r.dates) for r in selected_results]
    common_dates = sorted(date_sets[0].intersection(*date_sets[1:]))

    fig_dd = go.Figure()

    for r in selected_results:
        date_to_dd = dict(zip(r.dates, r.drawdown))
        y = [date_to_dd.get(d, 0.0) * 100 for d in common_dates]
        is_benchmark = r.category == "benchmark"
        color = colors[r.strategy_id]

        fig_dd.add_trace(
            go.Scatter(
                x=common_dates,
                y=y,
                mode="lines",
                name=r.strategy_name,
                fill="tozeroy",
                fillcolor=_hex_with_alpha(color, 0.12),
                line=dict(
                    color=color,
                    width=1.2,
                    dash="dot" if is_benchmark else "solid",
                ),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"{r.strategy_name}: %{{y:.2f}}%"
                    "<extra></extra>"
                ),
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
        height=380,
    )
    fig_dd.update_xaxes(**AXIS_STYLE)
    fig_dd.update_yaxes(**AXIS_STYLE)

    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Drawdown stats table ───────────────────────────────────────────────
    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:0.5rem 0 0.25rem 0;"
        "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Drawdown Statistics</h3>",
        unsafe_allow_html=True,
    )

    dd_rows = []
    for r in selected_results:
        ds = compute_drawdown_stats(r.drawdown)
        max_dd_dur = r.metrics.get("max_drawdown_duration", float("nan"))
        dd_rows.append(
            {
                "Strategy": r.strategy_name,
                "Max Drawdown": ds["max_drawdown"],
                "Max DD Duration (days)": max_dd_dur,
                "Avg Drawdown": ds["avg_drawdown"],
                "% Time Underwater": ds["pct_time_underwater"],
                "DD Episodes": ds["n_drawdown_periods"],
            }
        )

    df_dd = pd.DataFrame(dd_rows).set_index("Strategy")

    dd_fmt = {
        "Max Drawdown": "{:.1%}",
        "Max DD Duration (days)": "{:.0f}",
        "Avg Drawdown": "{:.1%}",
        "% Time Underwater": "{:.1%}",
        "DD Episodes": "{:.0f}",
    }

    # Higher (less negative) max drawdown = better
    dd_styler = (
        df_dd.style
        .format(dd_fmt, na_rep="—")
        .background_gradient(subset=["Max Drawdown"], cmap="RdYlGn", axis=0)
        .background_gradient(
            subset=["Max DD Duration (days)", "% Time Underwater", "DD Episodes"],
            cmap="RdYlGn_r",
            axis=0,
        )
    )

    st.dataframe(dd_styler, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Correlation Matrix
# ════════════════════════════════════════════════════════════════════════════

with tab_corr:
    st.caption(
        "Pearson correlation of daily returns across strategies. "
        "Low or negative correlations indicate diversification benefit when combining strategies."
    )

    corr_df = compute_correlation_matrix(selected_results)

    if corr_df.empty:
        st.info("Need at least two strategies with overlapping dates to compute a correlation matrix.")
    else:
        names = corr_df.columns.tolist()
        z = corr_df.values
        text = [[f"{z[i][j]:.2f}" for j in range(len(names))] for i in range(len(names))]

        fig_corr = go.Figure(
            go.Heatmap(
                z=z,
                x=names,
                y=names,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=10, color="#e8ecf0", family="JetBrains Mono, monospace"),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="ρ",
                    tickfont=dict(color=FONT, size=10),
                    titlefont=dict(color=FONT, size=11),
                    outlinewidth=0,
                    bgcolor=PLOT_BG,
                ),
                hovertemplate=(
                    "%{y} × %{x}<br>"
                    "Correlation: %{text}"
                    "<extra></extra>"
                ),
            )
        )

        n = len(names)
        fig_corr.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(
                tickfont=dict(size=10, color=FONT),
                tickangle=-35,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                tickfont=dict(size=10, color=FONT),
                showgrid=False,
                zeroline=False,
                autorange="reversed",
            ),
            margin=dict(l=10, r=10, t=30, b=80),
            height=max(340, 60 * n + 100),
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        # Summary: lowest and highest pairwise correlations (excluding self)
        if n > 2:
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((names[i], names[j], float(z[i][j])))

            pairs.sort(key=lambda x: x[2])

            col_low, col_high = st.columns(2)
            with col_low:
                st.markdown(
                    "<p style='font-size:0.78rem;color:#636b78;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:1px;'>Lowest Correlation</p>",
                    unsafe_allow_html=True,
                )
                for a, b_name, rho in pairs[:3]:
                    st.markdown(
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.82rem;"
                        f"color:#4ecdc4;'>{rho:+.3f}</span>"
                        f"<span style='font-size:0.80rem;color:#c4cad6;'> — {a} × {b_name}</span>",
                        unsafe_allow_html=True,
                    )
            with col_high:
                st.markdown(
                    "<p style='font-size:0.78rem;color:#636b78;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:1px;'>Highest Correlation</p>",
                    unsafe_allow_html=True,
                )
                for a, b_name, rho in pairs[-3:][::-1]:
                    st.markdown(
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.82rem;"
                        f"color:#f56565;'>{rho:+.3f}</span>"
                        f"<span style='font-size:0.80rem;color:#c4cad6;'> — {a} × {b_name}</span>",
                        unsafe_allow_html=True,
                    )


# ════════════════════════════════════════════════════════════════════════════
# Tab 4 — Factor Exposure
# ════════════════════════════════════════════════════════════════════════════

with tab_factor:
    st.caption(
        f"Market beta and Jensen's alpha vs the {benchmark_name} benchmark. "
        "Beta > 1 amplifies market moves; beta < 1 reduces them. "
        "Alpha is annualised excess return unexplained by market exposure."
    )

    # Exclude the benchmark from factor exposure (beta vs itself = 1, alpha = 0)
    factor_results = [r for r in selected_results if r.category != "benchmark"]

    if not factor_results:
        st.info("Select at least one non-benchmark strategy to display factor exposure.")
    else:
        strat_names = [r.strategy_name for r in factor_results]
        betas, alphas, r_squareds = [], [], []

        for r in factor_results:
            beta, alpha = compute_beta_alpha(r.returns, benchmark.returns)
            r2 = compute_r_squared(r.returns, benchmark.returns)
            betas.append(beta)
            alphas.append(alpha)
            r_squareds.append(r2)

        strat_colors = [colors[r.strategy_id] for r in factor_results]

        # ── Beta bar chart ─────────────────────────────────────────────────
        col_beta, col_alpha = st.columns(2)

        with col_beta:
            st.markdown(
                "<h3 style='font-size:0.95rem;font-weight:600;margin:0 0 0.25rem 0;"
                "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Market Beta</h3>",
                unsafe_allow_html=True,
            )
            fig_beta = go.Figure(
                go.Bar(
                    x=strat_names,
                    y=betas,
                    marker_color=strat_colors,
                    text=[f"{b:.2f}" if not math.isnan(b) else "—" for b in betas],
                    textposition="outside",
                    textfont=dict(size=10, color=FONT, family="JetBrains Mono, monospace"),
                    hovertemplate="<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>",
                )
            )
            fig_beta.add_hline(
                y=1.0,
                line=dict(color="#636b78", width=1, dash="dot"),
                annotation_text="β = 1",
                annotation_font=dict(color="#636b78", size=9),
            )
            fig_beta.update_layout(
                **CHART_LAYOUT,
                xaxis_title=None,
                yaxis_title="Beta (β)",
                xaxis_tickangle=-30,
                margin=dict(l=0, r=0, t=20, b=60),
                height=340,
                showlegend=False,
            )
            fig_beta.update_xaxes(**AXIS_STYLE)
            fig_beta.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_beta, use_container_width=True)

        with col_alpha:
            st.markdown(
                "<h3 style='font-size:0.95rem;font-weight:600;margin:0 0 0.25rem 0;"
                "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Annualised Alpha</h3>",
                unsafe_allow_html=True,
            )
            alpha_colors = [
                "#68d391" if (not math.isnan(a) and a >= 0) else "#f56565"
                for a in alphas
            ]
            fig_alpha = go.Figure(
                go.Bar(
                    x=strat_names,
                    y=[a * 100 for a in alphas],   # express as %
                    marker_color=alpha_colors,
                    text=[
                        f"{a*100:+.2f}%" if not math.isnan(a) else "—"
                        for a in alphas
                    ],
                    textposition="outside",
                    textfont=dict(size=10, color=FONT, family="JetBrains Mono, monospace"),
                    hovertemplate="<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>",
                )
            )
            fig_alpha.add_hline(
                y=0,
                line=dict(color="#636b78", width=1, dash="dot"),
            )
            fig_alpha.update_layout(
                **CHART_LAYOUT,
                xaxis_title=None,
                yaxis_title="Annualised Alpha (%)",
                yaxis_ticksuffix="%",
                xaxis_tickangle=-30,
                margin=dict(l=0, r=0, t=20, b=60),
                height=340,
                showlegend=False,
            )
            fig_alpha.update_xaxes(**AXIS_STYLE)
            fig_alpha.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_alpha, use_container_width=True)

        # ── Factor stats table ─────────────────────────────────────────────
        st.markdown(
            "<h3 style='font-size:1.0rem;font-weight:600;margin:0.5rem 0 0.25rem 0;"
            "color:#e8ecf0;font-family:\"DM Sans\",sans-serif;'>Factor Statistics</h3>",
            unsafe_allow_html=True,
        )

        factor_rows = []
        for r, beta, alpha, r2 in zip(factor_results, betas, alphas, r_squareds):
            factor_rows.append(
                {
                    "Strategy": r.strategy_name,
                    "Beta (β)": beta,
                    "Ann. Alpha (α)": alpha,
                    "R²": r2,
                    "Treynor Ratio": (
                        r.metrics.get("cagr", float("nan")) / beta
                        if not math.isnan(beta) and abs(beta) > 1e-6
                        else float("nan")
                    ),
                }
            )

        df_factor = pd.DataFrame(factor_rows).set_index("Strategy")

        factor_fmt = {
            "Beta (β)": "{:.3f}",
            "Ann. Alpha (α)": "{:+.2%}",
            "R²": "{:.3f}",
            "Treynor Ratio": "{:.3f}",
        }

        factor_styler = (
            df_factor.style
            .format(factor_fmt, na_rep="—")
            .background_gradient(subset=["Ann. Alpha (α)", "Treynor Ratio"], cmap="RdYlGn", axis=0)
            .background_gradient(subset=["R²"], cmap="RdYlGn_r", axis=0)
        )

        st.dataframe(factor_styler, use_container_width=True)

        st.caption(
            "**Treynor Ratio** = CAGR / Beta — return per unit of market risk. "
            "**R²** = fraction of return variance explained by the benchmark; "
            "low R² indicates a strategy's returns are driven by factors other than the market."
        )
