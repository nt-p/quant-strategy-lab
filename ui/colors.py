"""Distinct color assignments for strategies and the benchmark.

Colors are chosen to be visually separable on both dark and light
Streamlit themes.  The benchmark always uses a dashed grey line.
"""

from __future__ import annotations

from engine.backtest import BacktestResult

# Ordered palette — vivid, high-contrast, colourblind-friendly where possible
_PALETTE = [
    "#00b4d8",  # sky blue
    "#e63946",  # red
    "#2dc653",  # green
    "#f4a261",  # orange
    "#c77dff",  # purple
    "#f1c40f",  # yellow
    "#ff6b9d",  # pink
    "#06d6a0",  # teal
    "#ff9f1c",  # amber
    "#a8dadc",  # pale cyan
    "#e9c46a",  # sand
    "#264653",  # dark teal
]

BENCHMARK_COLOR = "#adb5bd"  # grey


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
