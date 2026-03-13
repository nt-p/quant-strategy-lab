"""Data preview UI component.

Renders after a successful "Load Data":
- Summary table: ticker, coverage dates, trading days, latest close, total return
- Normalised price chart: all selected assets rebased to 100 on the first day
  of the selected window, so the user can visually verify data quality.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_data_preview(
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
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
    """
    # Slice to the selected window for display purposes
    window = prices[(prices["date"] >= start_date) & (prices["date"] <= end_date)].copy()

    if window.empty:
        st.warning("No data in the selected date range. Adjust the date slider.")
        return

    st.subheader("Data Preview")

    _render_summary_table(window)
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
        title="Normalised Close Prices (rebased to 100)",
        xaxis_title=None,
        yaxis_title="Value (start = 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)
