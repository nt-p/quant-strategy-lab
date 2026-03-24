"""Data preview UI component.

Renders after a successful "Load Data":
- Summary table: ticker, coverage dates, trading days, latest close, total return
- Normalised price chart: all selected assets rebased to 100 on the first day
  of the selected window, so the user can visually verify data quality.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.universe import get_ticker_currency
from ui.theme import AXIS_STYLE, CHART_LAYOUT, FONT, GRID


def render_data_preview(
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fundamentals: dict[str, dict] | None = None,
) -> None:
    """Display a data quality panel for the loaded price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Full OHLCV data in long format (all history, all tickers).
    start_date : pd.Timestamp
        Start of the user-selected backtest window (for display).
    end_date : pd.Timestamp
        End of the user-selected backtest window (for display).
    fundamentals : dict[str, dict] or None
        Optional fundamental data keyed by ticker.  When provided, a
        fundamentals summary table is rendered below the price summary.
    """
    # Slice to the selected window for display purposes
    window = prices[(prices["date"] >= start_date) & (prices["date"] <= end_date)].copy()

    if window.empty:
        st.warning("No data in the selected date range. Adjust the date slider.")
        return

    st.markdown(
        "<h3 style='font-size:1.0rem;font-weight:600;margin:0.25rem 0 0.15rem 0;"
        "color:#e8ecf0;letter-spacing:-0.2px;font-family:\"DM Sans\",sans-serif;'>"
        "Data Preview</h3>",
        unsafe_allow_html=True,
    )

    _render_summary_table(window)
    st.markdown("")
    if fundamentals:
        _render_fundamentals_table(fundamentals)
        st.markdown("")
    _render_normalised_chart(window)


def _render_summary_table(window: pd.DataFrame) -> None:
    """Render per-ticker summary statistics as a styled dataframe."""
    rows = []
    for ticker, grp in window.groupby("ticker"):
        grp = grp.sort_values("date")
        first_close = grp["close"].iloc[0]
        last_close = grp["close"].iloc[-1]
        total_return = (last_close / first_close - 1.0) * 100.0

        rows.append(
            {
                "Ticker": ticker,
                "Currency": get_ticker_currency(ticker),
                "From": grp["date"].iloc[0].date(),
                "To": grp["date"].iloc[-1].date(),
                "Trading Days": len(grp),
                "Latest Close": last_close,
                "Total Return": total_return,
            }
        )

    summary = pd.DataFrame(rows).set_index("Ticker")

    # Format display columns
    fmt = {
        "Latest Close": "{:.2f}",
        "Total Return": "{:+.1f}%",
    }

    st.dataframe(
        summary.style.format(fmt).background_gradient(
            subset=["Total Return"], cmap="RdYlGn", vmin=-50, vmax=200
        ),
        use_container_width=True,
    )


def _render_fundamentals_table(fundamentals: dict[str, dict]) -> None:
    """Render a per-ticker fundamentals summary table.

    Parameters
    ----------
    fundamentals : dict[str, dict]
        Keyed by ticker.  Each value is a dict with keys: market_cap,
        trailing_pe, price_to_book, dividend_yield, revenue_growth,
        earnings_growth.  Any value may be None if unavailable.
    """
    rows = []
    for ticker, data in sorted(fundamentals.items()):
        mkt_cap = data.get("market_cap")
        rows.append(
            {
                "Ticker": ticker,
                "Market Cap ($B)": (mkt_cap / 1e9) if mkt_cap is not None else None,
                "P/E (trailing)": data.get("trailing_pe"),
                "P/B": data.get("price_to_book"),
                "Div Yield": data.get("dividend_yield"),
                "Rev Growth": data.get("revenue_growth"),
                "EPS Growth": data.get("earnings_growth"),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows).set_index("Ticker")

    fmt: dict[str, str] = {}
    for col in ["Market Cap ($B)", "P/E (trailing)", "P/B"]:
        if col in df.columns:
            fmt[col] = "{:.1f}"
    for col in ["Div Yield", "Rev Growth", "EPS Growth"]:
        if col in df.columns:
            fmt[col] = "{:.1%}"

    has_au = any(ticker.upper().endswith(".AX") for ticker in fundamentals)
    note = " | AUD-denominated for .AX tickers" if has_au else ""
    st.caption(f"Fundamentals (current snapshot, cached 7 days){note}")
    st.dataframe(
        df.style.format(fmt, na_rep="—"),
        use_container_width=True,
    )


def _render_normalised_chart(window: pd.DataFrame) -> None:
    """Render a Plotly line chart of normalised close prices (rebased to 100)."""
    pivot = window.pivot_table(values="close", index="date", columns="ticker")
    pivot = pivot.sort_index()

    # Rebase: first non-NaN value for each ticker → 100
    rebased = pivot.div(pivot.bfill().iloc[0]) * 100.0

    fig = go.Figure()
    for ticker in rebased.columns:
        series = rebased[ticker].dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=ticker,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Normalised Close Prices (rebased to 100)",
            font=dict(size=13, color="#e8ecf0", family="DM Sans, sans-serif"),
        ),
        xaxis_title=None,
        yaxis_title="Value (start = 100)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=FONT, family="DM Sans, sans-serif"),
            bgcolor="rgba(22,27,39,0.85)", bordercolor=GRID, borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=420,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)

    st.plotly_chart(fig, use_container_width=True)
