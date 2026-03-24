"""Distinct color assignments for strategies and the benchmark.

Colors are chosen to be visually separable on both dark and light
Streamlit themes.  The benchmark always uses a dashed grey line.
"""

from __future__ import annotations

from engine.backtest import BacktestResult

# Premium fintech palette — harmonious, clearly distinguishable on dark bg
_PALETTE = [
    "#4ecdc4",  # muted teal (primary accent)
    "#f56565",  # soft coral
    "#f6ad55",  # warm amber
    "#9f7aea",  # soft purple
    "#63b3ed",  # sky blue
    "#68d391",  # sage green
    "#fc8181",  # rose
    "#76e4f7",  # light cyan
    "#d4a853",  # warm gold
    "#b794f4",  # lavender
    "#81e6d9",  # mint
    "#c6a96e",  # sand
]

BENCHMARK_COLOR = "#94a3b8"  # slate grey


def assign_colors(results: list[BacktestResult]) -> dict[str, str]:
    """Return a {strategy_id: hex_color} mapping for the given results.

    The benchmark always receives BENCHMARK_COLOR.  Strategy colors are
    drawn from _PALETTE in result order (wrapping if > 12 strategies).

    Parameters
    ----------
    results : list[BacktestResult]
        All results to color, benchmark included.

    Returns
    -------
    dict[str, str]
        {strategy_id: "#rrggbb"}
    """
    colors: dict[str, str] = {}
    palette_idx = 0
    for r in results:
        if r.category == "benchmark":
            colors[r.strategy_id] = BENCHMARK_COLOR
        else:
            colors[r.strategy_id] = _PALETTE[palette_idx % len(_PALETTE)]
            palette_idx += 1
    return colors
