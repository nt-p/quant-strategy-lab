"""Asset universe definitions for QuantScope.

CURATED_UNIVERSE provides grouped shortcuts in the sidebar.
Users can also type any valid yfinance ticker manually.
"""

import pandas as pd

CURATED_UNIVERSE: dict[str, list[str]] = {
    "US Equity ETFs": ["SPY", "QQQ", "IWM", "DIA", "VOO"],
    "Sector ETFs": [
        "XLF", "XLK", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    ],
    "Fixed Income": ["TLT", "IEF", "SHY", "HYG", "LQD", "BND", "AGG"],
    "Alternatives": ["GLD", "SLV", "DBC", "VNQ", "BITO"],
    "International": ["EFA", "EEM", "VWO", "MCHI", "EWJ", "EWG", "EWU"],
    "Mega Cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "BRK-B", "JPM", "V",
    ],
}

# Flat sorted list of all curated tickers (no duplicates)
ALL_CURATED_TICKERS: list[str] = sorted(
    {t for tickers in CURATED_UNIVERSE.values() for t in tickers}
)

# Date range constants
DATA_START = "2000-01-01"
BENCHMARK_TICKER = "SPY"


def get_end_date() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return pd.Timestamp.today().strftime("%Y-%m-%d")
