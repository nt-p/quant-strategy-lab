"""Asset universe definitions for QuantScope.

CURATED_UNIVERSE provides grouped shortcuts in the sidebar for US assets.
AU_CURATED_UNIVERSE provides grouped shortcuts for Australian (ASX) assets.
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

AU_CURATED_UNIVERSE: dict[str, list[str]] = {
    "ASX Broad Market": ["STW.AX", "IOZ.AX", "VAS.AX", "A200.AX"],
    "ASX Sector ETFs": ["VHY.AX", "MVW.AX", "QUAL.AX", "OZR.AX", "OZF.AX"],
    "ASX Blue Chips": [
        "BHP.AX", "CBA.AX", "CSL.AX", "WBC.AX", "NAB.AX",
        "ANZ.AX", "WES.AX", "MQG.AX", "FMG.AX", "RIO.AX",
        "WDS.AX", "TLS.AX", "WOW.AX", "ALL.AX", "GMG.AX",
    ],
    "Australian Fixed Income": ["IAF.AX", "BOND.AX", "GOVT.AX", "PLUS.AX"],
    "Australian REITs & Alternatives": ["VAP.AX", "MVA.AX", "GOLD.AX", "QAU.AX"],
    "International from ASX": ["IVV.AX", "IOO.AX", "VGS.AX", "VISM.AX", "VGE.AX", "ASIA.AX"],
}

# Flat sorted list of all curated tickers (no duplicates)
ALL_CURATED_TICKERS: list[str] = sorted(
    {t for tickers in CURATED_UNIVERSE.values() for t in tickers}
)
AU_ALL_CURATED_TICKERS: list[str] = sorted(
    {t for tickers in AU_CURATED_UNIVERSE.values() for t in tickers}
)

# Date range constants
DATA_START = "2000-01-01"
BENCHMARK_TICKER = "SPY"
AU_BENCHMARK_TICKER = "STW.AX"


def get_end_date() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def is_au_ticker(ticker: str) -> bool:
    """Return True if the ticker is an ASX-listed security (ends with .AX)."""
    return ticker.upper().endswith(".AX")


def get_ticker_currency(ticker: str) -> str:
    """Return the display currency for a ticker (AUD for .AX, USD otherwise)."""
    return "AUD" if is_au_ticker(ticker) else "USD"
