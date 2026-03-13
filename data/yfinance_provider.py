"""YFinance data provider with parquet caching.

Fetches OHLCV data from yfinance and caches results as parquet files.
Cache is considered stale if max date < today - 1 day.
"""

import warnings

import pandas as pd
import yfinance as yf

from .base import DataSource
from .cache import load_cache, save_cache


class YFinanceProvider(DataSource):
    """Data provider backed by yfinance with local parquet cache."""

    def fetch_ohlcv(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV data, using parquet cache where available.

        Parameters
        ----------
        tickers : list[str]
            List of ticker symbols.
        start : str
            Start date YYYY-MM-DD.
        end : str
            End date YYYY-MM-DD (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: [date, ticker, open, high, low, close, volume].
            Sorted by (ticker, date). No NaN in close.
        """
        frames: list[pd.DataFrame] = []

        for ticker in tickers:
            cached = load_cache(ticker, start, end)
            if cached is not None:
                frames.append(cached)
                continue

            df = self._fetch_single(ticker, start, end)
            if df is not None and not df.empty:
                save_cache(df, ticker, start, end)
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
        return result

    def _fetch_single(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        """Download data for one ticker from yfinance.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        start, end : str
            Date range.

        Returns
        -------
        pd.DataFrame or None
            Long-format OHLCV, or None if fetch fails.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
        except Exception:
            return None

        if raw is None or raw.empty:
            return None

        # Flatten MultiIndex columns if present (happens with single ticker too in newer yfinance)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"adj close": "close"}) if "adj close" in raw.columns else raw

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(raw.columns)):
            return None

        raw = raw[["open", "high", "low", "close", "volume"]].copy()
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw.dropna(subset=["close"])

        df = raw.reset_index().rename(columns={"Date": "date", "Datetime": "date", "index": "date"})
        # Ensure the date column is named correctly regardless of index name
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"], utc=True)

        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    def get_asset_info(self, ticker: str) -> dict:
        """Fetch metadata for a ticker from yfinance.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        dict
            Keys: ticker, name, category, description.
        """
        try:
            info = yf.Ticker(ticker).info
            return {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName") or ticker,
                "category": info.get("quoteType", "Unknown"),
                "description": info.get("longBusinessSummary", ""),
            }
        except Exception:
            return {"ticker": ticker, "name": ticker, "category": "Unknown", "description": ""}
