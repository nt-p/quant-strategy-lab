"""Parquet cache with staleness checking.

OHLCV cache key: {ticker}_{start}_{end}.parquet  — stale after 1 day.
Fundamentals cache key: fundamentals_{ticker}.parquet — stale after 7 days.
"""

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"

_FUNDAMENTALS_STALE_DAYS = 7


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe_ticker = ticker.replace("/", "_").replace(".", "_")
    return CACHE_DIR / f"{safe_ticker}_{start}_{end}.parquet"


def _fundamentals_cache_path(ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "_").replace(".", "_")
    return CACHE_DIR / f"fundamentals_{safe_ticker}.parquet"


def load_cache(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Load cached OHLCV data if it exists and is fresh.

    Parameters
    ----------
    ticker, start, end : str
        Cache key components.

    Returns
    -------
    pd.DataFrame or None
        Cached DataFrame if fresh, otherwise None.
    """
    path = _cache_path(ticker, start, end)
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None

    # Stale if max cached date < today - 1 day
    today = pd.Timestamp.today(tz="UTC").normalize()
    max_cached = pd.Timestamp(df["date"].max()).tz_convert("UTC").normalize()
    if max_cached < today - pd.Timedelta(days=1):
        return None

    return df


def save_cache(df: pd.DataFrame, ticker: str, start: str, end: str) -> None:
    """Persist OHLCV DataFrame to parquet cache.

    Parameters
    ----------
    df : pd.DataFrame
        Data to cache.
    ticker, start, end : str
        Cache key components.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, start, end)
    df.to_parquet(path, index=False)


def load_fundamentals_cache(ticker: str) -> dict | None:
    """Load cached fundamentals for a ticker if the cache is fresh (< 7 days old).

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    dict or None
        Fundamentals dict if cached and fresh, otherwise None.
    """
    path = _fundamentals_cache_path(ticker)
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if df.empty or "fetched_at" not in df.columns:
        return None

    fetched_at = pd.Timestamp(df["fetched_at"].iloc[0])
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.tz_localize("UTC")
    today = pd.Timestamp.today(tz="UTC").normalize()
    if today - fetched_at.normalize() > pd.Timedelta(days=_FUNDAMENTALS_STALE_DAYS):
        return None

    row = df.drop(columns=["fetched_at"]).iloc[0].to_dict()
    return row


def save_fundamentals_cache(ticker: str, fundamentals: dict) -> None:
    """Persist fundamentals dict to parquet cache.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    fundamentals : dict
        Fundamental data to cache.  A ``fetched_at`` timestamp is added automatically.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    row = {**fundamentals, "fetched_at": pd.Timestamp.now(tz="UTC")}
    df = pd.DataFrame([row])
    path = _fundamentals_cache_path(ticker)
    df.to_parquet(path, index=False)
