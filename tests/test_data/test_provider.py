"""Tests for YFinanceProvider and cache layer."""

import pandas as pd
import pytest

from data.yfinance_provider import YFinanceProvider
from data.cache import load_cache, save_cache


def test_fetch_spy_returns_dataframe():
    """Smoke test: fetch a small slice of SPY data."""
    provider = YFinanceProvider()
    df = provider.fetch_ohlcv(["SPY"], start="2023-01-01", end="2023-01-31")

    assert not df.empty, "Expected non-empty DataFrame for SPY Jan 2023"
    assert set(df.columns) >= {"date", "ticker", "open", "high", "low", "close", "volume"}
    assert (df["ticker"] == "SPY").all()
    assert df["close"].notna().all()


def test_cache_round_trip(tmp_path, monkeypatch):
    """Cache save → load returns the same DataFrame."""
    import data.cache as cache_module

    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)

    # Use today's date so the staleness check passes
    today = pd.Timestamp.today(tz="UTC").normalize()
    yesterday = today - pd.Timedelta(days=1)
    df = pd.DataFrame(
        {
            "date": [yesterday, today],
            "ticker": ["SPY", "SPY"],
            "open": [380.0, 381.0],
            "high": [385.0, 386.0],
            "low": [378.0, 379.0],
            "close": [383.0, 384.0],
            "volume": [1_000_000, 1_100_000],
        }
    )

    save_cache(df, "SPY", "2000-01-01", "today")
    loaded = load_cache("SPY", "2000-01-01", "today")

    assert loaded is not None
    assert len(loaded) == len(df)
