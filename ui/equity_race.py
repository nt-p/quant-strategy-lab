"""Animated Equity Race chart — Phase 4.

Three animated subplots in one Plotly figure driven by a single time slider:
  Row 1 — Equity curves      (the race)
  Row 2 — Drawdown           (underwater plot)
  Row 3 — Rolling 60-d Sharpe

A live leaderboard (strategies ranked by current total return) is embedded
as a text annotation that updates with every animation frame.

~N_FRAMES evenly-spaced sample points are used as both the frame markers
and the data points rendered inside each frame.  This keeps the generated
JSON small: the final frame has at most N_FRAMES data points per series,
rather than the full ~2 500-day daily series.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from engine.backtest import BacktestResult
from .colors import assign_colors

# ── Theme ──────────────────────────────────────────────────────────────────────
_BG      = "#0f1116"   # page background (matches config.toml)
_PLOT_BG = "#161b27"   # chart plot area
_GRID    = "#252836"   # subtle gridlines
_FONT    = "#c4cad6"   # axis labels and tick text

_N_FRAMES    = 250     # target number of animation frames
_FRAME_MS    = 40      # ms per frame during playback  (~25 fps)


# ── Public API ─────────────────────────────────────────────────────────────────

def render_equity_race(
    strategy_results: list[BacktestResult],
    benchmark: BacktestResult,
    initial_capital: float,
) -> None:
    """Render the animated Equity Race section.

    Builds a single Plotly figure with three animated subplots (equity,
    drawdown, rolling Sharpe) plus a live leaderboard annotation.  Renders
    at full container width via ``st.plotly_chart``.

    Parameters
    ----------
    strategy_results : list[BacktestResult]
        One result per enabled strategy (excludes benchmark).
    benchmark : BacktestResult
        SPY buy-and-hold benchmark result.
    initial_capital : float
        Starting portfolio value — used to compute % returns for the
        leaderboard.
    """
    all_results: list[BacktestResult] = strategy_results + [benchmark]
    colors = assign_colors(all_results)

    # ── Common date index ──────────────────────────────────────────────────────
    date_sets = [set(r.dates) for r in all_results]
    common: list[pd.Timestamp] = sorted(date_sets[0].intersection(*date_sets[1:]))
    if len(common) < 10:
        st.warning("Not enough overlapping dates across strategies to build the race animation.")
        return

    # ── Sample ~N_FRAMES evenly-spaced dates ──────────────────────────────────
    n = len(common)
    step = max(1, n // _N_FRAMES)
    idxs: list[int] = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    sampled: list[pd.Timestamp] = [common[i] for i in idxs]
    n_frames = len(sampled)

    # ── Pre-compute values at sampled dates only ───────────────────────────────
    eq_v: dict[str, list[float]] = {}
    dd_v: dict[str, list[float]] = {}
    rs_v: dict[str, list[float]] = {}

    for r in all_results:
        e_lkp = dict(zip(r.dates, r.equity))
        d_lkp = dict(zip(r.dates, [v * 100.0 for v in r.drawdown]))
        s_lkp = dict(zip(r.dates, r.rolling_sharpe))
        eq_v[r.strategy_id] = [e_lkp.get(d, float("nan")) for d in sampled]
        dd_v[r.strategy_id] = [d_lkp.get(d, float("nan")) for d in sampled]
        rs_v[r.strategy_id] = [s_lkp.get(d, float("nan")) for d in sampled]

    # ── Stable y-axis ranges (pre-computed from full data) ────────────────────
    def _finite(vals: list[float]) -> list[float]:
        return [v for v in vals if math.isfinite(v)]

    all_eq = _finite([v for sid in eq_v for v in eq_v[sid]])
    all_dd = _finite([v for sid in dd_v for v in dd_v[sid]])
    all_rs = _finite([v for sid in rs_v for v in rs_v[sid]])

    eq_range = [min(all_eq) * 0.95, max(all_eq) * 1.05] if all_eq else None
    dd_range = [min(all_dd) * 1.1, 2.0] if all_dd else None
    rs_range = [min(all_rs) - 0.5, max(all_rs) + 0.5] if all_rs else None

    # ── Figure skeleton ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.23, 0.22],
        vertical_spacing=0.04,
    )

    # ── Initial traces (day-0 data — overwritten by frames) ───────────────────
    for r in all_results:
        is_b = r.category == "benchmark"
        c = colors[r.strategy_id]
        common_kw = dict(mode="lines", legendgroup=r.strategy_id)

        # Row 1 — Equity
        fig.add_trace(
            go.Scatter(
                x=sampled[:1],
                y=eq_v[r.strategy_id][:1],
                name=r.strategy_name,
                showlegend=True,
                line=dict(
                    color=c,
                    width=2.0 if not is_b else 1.5,
                    dash="dash" if is_b else "solid",
                ),
                hovertemplate=(
                    f"<b>{r.strategy_name}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: $%{y:,.0f}<extra></extra>"
                ),
                **common_kw,
            ),
            row=1, col=1,
        )

        # Row 2 — Drawdown
        fig.add_trace(
            go.Scatter(
                x=sampled[:1],
                y=dd_v[r.strategy_id][:1],
                name=r.strategy_name,
                showlegend=False,
                fill="tozeroy",
                fillcolor=_hex_alpha(c, 0.12),
                line=dict(color=c, width=1.0, dash="dash" if is_b else "solid"),
                hovertemplate=(
                    f"<b>{r.strategy_name}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Drawdown: %{y:.2f}%<extra></extra>"
                ),
                **common_kw,
            ),
            row=2, col=1,
        )

        # Row 3 — Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=sampled[:1],
                y=rs_v[r.strategy_id][:1],
                name=r.strategy_name,
                showlegend=False,
                line=dict(color=c, width=1.0, dash="dash" if is_b else "solid"),
                hovertemplate=(
                    f"<b>{r.strategy_name}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Rolling Sharpe: %{y:.2f}<extra></extra>"
                ),
                **common_kw,
            ),
            row=3, col=1,
        )

    # ── Animation frames ───────────────────────────────────────────────────────
    frames: list[go.Frame] = []
    for k in range(n_frames):
        fd = sampled[k]
        xs = sampled[: k + 1]

        frame_data = []
        for r in all_results:
            frame_data.append(go.Scatter(x=xs, y=eq_v[r.strategy_id][: k + 1]))
            frame_data.append(go.Scatter(x=xs, y=dd_v[r.strategy_id][: k + 1]))
            frame_data.append(go.Scatter(x=xs, y=rs_v[r.strategy_id][: k + 1]))

        lb_text = _leaderboard_html(all_results, eq_v, k, initial_capital)

        frames.append(
            go.Frame(
                data=frame_data,
                name=fd.strftime("%Y-%m-%d"),
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0.99,
                            y=0.99,
                            xref="paper",
                            yref="paper",
                            xanchor="right",
                            yanchor="top",
                            text=lb_text,
                            showarrow=False,
                            font=dict(size=11, color=_FONT, family="JetBrains Mono, monospace"),
                            bgcolor="rgba(22,27,39,0.92)",
                            bordercolor=_GRID,
                            borderwidth=1,
                            borderpad=8,
                            align="left",
                        )
                    ]
                ),
            )
        )

    fig.frames = frames

    # ── Initial leaderboard annotation (frame 0) ───────────────────────────────
    fig.update_layout(
        annotations=[
            dict(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="top",
                text=_leaderboard_html(all_results, eq_v, 0, initial_capital),
                showarrow=False,
                font=dict(size=11, color=_FONT, family="JetBrains Mono, monospace"),
                bgcolor="rgba(22,27,39,0.92)",
                bordercolor=_GRID,
                borderwidth=1,
                borderpad=8,
                align="left",
            )
        ]
    )

    # ── Play / Pause + time slider ─────────────────────────────────────────────
    slider_steps = [
        dict(
            args=[
                [f.name],
                dict(
                    frame=dict(duration=_FRAME_MS, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                ),
            ],
            label=f.name,
            method="animate",
        )
        for f in frames
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.0,
                y=0.0,
                xanchor="right",
                yanchor="top",
                pad=dict(r=10, t=87),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=_FRAME_MS, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                font=dict(size=12, color=_FONT, family="DM Sans, sans-serif"),
                bgcolor="#1e2333",
                bordercolor=_GRID,
                borderwidth=1,
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.05,
                y=0.0,
                len=0.94,
                xanchor="left",
                yanchor="top",
                pad=dict(b=10, t=60),
                currentvalue=dict(
                    font=dict(size=12, color=_FONT, family="JetBrains Mono, monospace"),
                    prefix="Date: ",
                    visible=True,
                    xanchor="center",
                ),
                transition=dict(duration=0),
                bgcolor=_PLOT_BG,
                bordercolor=_GRID,
                tickcolor=_GRID,
                font=dict(color=_FONT, size=9, family="DM Sans, sans-serif"),
            )
        ],
    )

    # ── X-axis: fixed range so the line draws left-to-right ───────────────────
    fig.update_xaxes(
        range=[sampled[0], sampled[-1]],
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=0.5,
        zeroline=False,
        showline=False,
        tickfont=dict(color=_FONT, size=10),
        showspikes=True,
        spikecolor=_GRID,
        spikethickness=1,
        spikedash="dot",
        spikesnap="cursor",
        spikemode="across",
    )

    # ── Y-axes ─────────────────────────────────────────────────────────────────
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor=_GRID,
        zerolinewidth=1,
        tickfont=dict(color=_FONT, size=10),
    )

    eq_y_kw: dict = dict(
        title_text="Portfolio Value ($)",
        tickprefix="$",
        tickformat=",.0f",
        title_font=dict(color=_FONT, size=11),
    )
    if eq_range:
        eq_y_kw["range"] = eq_range
    fig.update_yaxes(**eq_y_kw, row=1, col=1)

    dd_y_kw: dict = dict(
        title_text="Drawdown (%)",
        ticksuffix="%",
        title_font=dict(color=_FONT, size=11),
    )
    if dd_range:
        dd_y_kw["range"] = dd_range
    fig.update_yaxes(**dd_y_kw, row=2, col=1)

    rs_y_kw: dict = dict(
        title_text="Rolling Sharpe",
        title_font=dict(color=_FONT, size=11),
    )
    if rs_range:
        rs_y_kw["range"] = rs_range
    fig.update_yaxes(**rs_y_kw, row=3, col=1)

    # ── Global layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_FONT, family="DM Sans, sans-serif", size=11),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=11, color=_FONT, family="DM Sans, sans-serif"),
            bgcolor="rgba(22,27,39,0.85)",
            bordercolor=_GRID,
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=60, b=120),
        height=760,
    )

    st.plotly_chart(fig, use_container_width=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _leaderboard_html(
    all_results: list[BacktestResult],
    eq_v: dict[str, list[float]],
    frame_idx: int,
    initial_capital: float,
) -> str:
    """Build the HTML leaderboard annotation text for a given frame.

    Parameters
    ----------
    all_results : list[BacktestResult]
        All results (strategies + benchmark) in display order.
    eq_v : dict[str, list[float]]
        Pre-sampled equity values keyed by strategy_id.
    frame_idx : int
        Current animation frame index into eq_v lists.
    initial_capital : float
        Starting portfolio value for % return computation.

    Returns
    -------
    str
        HTML string for use in a Plotly annotation ``text`` field.
    """
    rows: list[tuple[str, float]] = []
    for r in all_results:
        eq = eq_v[r.strategy_id][frame_idx]
        if math.isfinite(eq):
            ret_pct = (eq / initial_capital - 1.0) * 100.0
            label = r.strategy_name + (" (B)" if r.category == "benchmark" else "")
            rows.append((label, ret_pct))

    rows.sort(key=lambda x: x[1], reverse=True)

    lines = ["<b>Leaderboard</b>"]
    for i, (name, ret) in enumerate(rows):
        short = name if len(name) <= 22 else name[:21] + "\u2026"
        sign = "+" if ret >= 0 else ""
        lines.append(f"{i + 1:>2}. {short:<22} {sign}{ret:.1f}%")

    return "<br>".join(lines)


def _hex_alpha(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' + alpha float to an 'rgba(r,g,b,a)' CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
