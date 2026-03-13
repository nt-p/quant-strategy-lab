"""Shared test fixtures for QuantScope."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Long-format OHLCV DataFrame with two tickers over 500 trading days."""
    rng = np.random.default_rng(42)
    tickers = ["AAAA", "BBBB"]
    dates = pd.bdate_range("2020-01-01", periods=500, tz="UTC")

    rows = []
    for ticker in tickers:
        close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, len(dates)))
        for i, d in enumerate(dates):
            c = close[i]
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": c * (1 + rng.uniform(-0.005, 0.005)),
                    "high": c * (1 + rng.uniform(0, 0.01)),
                    "low": c * (1 - rng.uniform(0, 0.01)),
                    "close": c,
                    "volume": rng.integers(1_000_000, 10_000_000),
                }
            )

    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
