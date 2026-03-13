"""Parquet cache with staleness checking.

Cache key: {ticker}_{start}_{end}.parquet
Stale if cached max date < today - 1 day.
"""

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe_ticker = ticker.replace("/", "_").replace(".", "_")
    return CACHE_DIR / f"{safe_ticker}_{start}_{end}.parquet"


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
