"""
STRATEGY TEMPLATE — copy this file to start a new strategy.
============================================================

Instructions
------------
1. Copy this file:
       cp strategies/_example.py strategies/my_strategy.py

2. Fill in the five metadata class attributes (strategy_id, name, …).

3. Implement get_weights().  That is the ONLY required method.

4. The strategy appears in the dashboard automatically — no other
   changes needed anywhere in the codebase.

Contract for get_weights()
--------------------------
- `prices` contains OHLCV data in LONG format for ALL tickers in the
  universe, from the dataset start date up to and INCLUDING
  `current_date`.  It NEVER contains future data.
- Return a dict {ticker: weight}.  Weights should sum to ~1.0.
  Tickers omitted from the dict are treated as zero weight.
- Use only data where prices["date"] <= current_date.  The engine
  enforces this too, but it is your responsibility to avoid lookahead.

Available helpers (inherited from StrategyBase)
-----------------------------------------------
  self._pivot_close(prices)            → DataFrame (date × ticker) of close prices
  self._equal_weight(tickers)          → {ticker: 1/N} for each ticker
  self._zscore_cross_sectional(df)     → row-wise z-score of a pivot table

Rebalance frequency
-------------------
Set `rebalance_frequency` to one of:
  RebalanceFrequency.NEVER      — weights set once on day 1, never changed
  RebalanceFrequency.WEEKLY     — every 5 trading days
  RebalanceFrequency.MONTHLY    — first trading day of each month  (default)
  RebalanceFrequency.QUARTERLY  — first trading day of each quarter
"""

import pandas as pd

from .base import RebalanceFrequency, StrategyBase, StrategyCategory


class MomentumTopN(StrategyBase):
    """Simple top-N momentum strategy.

    At each rebalance date, rank assets by trailing lookback return and
    allocate equal weight to the top N assets.
    """

    strategy_id = "momentum_top_n"
    name = "Top-N Momentum"
    category = StrategyCategory.QUANTITATIVE
    description = "Equal-weight the top N assets ranked by trailing momentum."
    long_description = (
        "This strategy computes each asset's trailing return over a fixed "
        "lookback window and ranks the universe cross-sectionally. At each "
        "rebalance date, it holds the top N assets with equal weights."
    )

    rebalance_frequency = RebalanceFrequency.MONTHLY

    lookback_days = 60
    top_n = 5

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
    ) -> dict[str, float]:
        # Step 1: build close-price matrix
        close = self._pivot_close(prices)

        # Step 2: keep only tickers in current universe
        close = close[[c for c in close.columns if c in universe]]

        if close.empty:
            return {}

        # Step 3: make sure current_date is in the index
        if current_date not in close.index:
            return {}

        # Step 4: find the row position of the rebalance date
        current_idx = close.index.get_loc(current_date)

        # Need at least lookback_days of history
        if current_idx < self.lookback_days:
            return {}

        # Step 5: get current and past prices
        current_prices = close.iloc[current_idx]
        past_prices = close.iloc[current_idx - self.lookback_days]

        # Step 6: compute trailing momentum
        momentum = (current_prices / past_prices) - 1.0

        # Step 7: drop assets with missing data
        momentum = momentum.dropna()

        if momentum.empty:
            return {}

        # Step 8: rank and choose top N
        selected = momentum.sort_values(ascending=False).head(self.top_n).index.tolist()

        if not selected:
            return {}

        # Step 9: equal weight the winners
        return self._equal_weight(selected)