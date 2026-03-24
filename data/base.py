"""DataSource abstract base class.

V1 provider: yfinance. V2: swap to Databento/OpenBB via config.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract base for all data providers.

    Implementations must return OHLCV data in long format,
    asset metadata via get_asset_info(), and fundamental data
    via fetch_fundamentals().
    """

    @abstractmethod
    def fetch_ohlcv(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV data for the given tickers and date range.

        Parameters
        ----------
        tickers : list[str]
            List of ticker symbols (yfinance format).
        start : str
            Start date in YYYY-MM-DD format.
        end : str
            End date in YYYY-MM-DD format (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: [date, ticker, open, high, low, close, volume].
            Sorted by (ticker, date). No NaN in close. Dates are UTC.
        """
        ...

    @abstractmethod
    def fetch_fundamentals(
        self,
        tickers: list[str],
        date: pd.Timestamp | None = None,
    ) -> dict[str, dict]:
        """Fetch fundamental data for the given tickers as of *date*.

        For providers that only expose current snapshots (e.g. yfinance),
        the *date* parameter is accepted but not used — the most-recent
        available values are returned.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to query.
        date : pd.Timestamp or None
            As-of date.  Ignored by the yfinance provider.

        Returns
        -------
        dict[str, dict]
            ``{ticker: {field: value}}`` where each inner dict contains at
            minimum the following keys (``None`` when data unavailable):

            - ``market_cap``      float | None   — market capitalisation ($)
            - ``trailing_pe``     float | None   — trailing price-to-earnings
            - ``price_to_book``   float | None   — price-to-book ratio
            - ``dividend_yield``  float | None   — annual dividend yield (0–1)
            - ``revenue_growth``  float | None   — YoY revenue growth rate (0–1)
            - ``earnings_growth`` float | None   — YoY earnings growth rate (0–1)
        """
        ...

    @abstractmethod
    def get_asset_info(self, ticker: str) -> dict:
        """Fetch metadata for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        dict
            Keys: ticker, name, category, description.
        """
        ...
