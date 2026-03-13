"""DataSource abstract base class.

V1 provider: yfinance. V2: swap to Databento/OpenBB via config.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract base for all data providers.

    Implementations must return OHLCV data in long format and
    asset metadata via get_asset_info().
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
