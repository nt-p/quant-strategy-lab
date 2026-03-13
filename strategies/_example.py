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


class ExampleEqualWeight(StrategyBase):
    """Trivial equal-weight strategy — used only as a template reference.

    Assigns identical weight to every asset in the universe on every
    rebalance date.  There is no signal, no ranking, and no look-back
    period.  This is the simplest possible valid strategy and is useful
    for verifying that the backtest pipeline is wired up correctly.
    """

    # ── Metadata ──────────────────────────────────────────────────
    strategy_id = "example_equal_weight"
    name = "Example: Equal Weight"
    category = StrategyCategory.PASSIVE
    description = "Template strategy — equal weight across all assets. Copy to create your own."
    long_description = (
        "This strategy assigns a weight of 1/N to each of the N assets in the "
        "universe at every rebalance date.  It is the baseline passive allocation "
        "and exists solely as a template to demonstrate the correct structure for "
        "a QuantScope strategy file."
    )

    # ── Rebalance cadence ─────────────────────────────────────────
    # Monthly is the default; override if your strategy requires a different cadence.
    rebalance_frequency = RebalanceFrequency.MONTHLY

    # ── Core logic ────────────────────────────────────────────────

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
    ) -> dict[str, float]:
        """Return equal weight for every asset in the universe.

        Parameters
        ----------
        prices : pd.DataFrame
            Long-format OHLCV data up to and including current_date.
        current_date : pd.Timestamp
            The rebalance date. Only use data <= this date.
        universe : list[str]
            All tickers available at this rebalance date.

        Returns
        -------
        dict[str, float]
            {ticker: 1/N} for each ticker in the universe.
        """
        # ── Step 1: filter to assets that actually have price data ──
        # Some assets may have missing data early in the history.
        close = self._pivot_close(prices)  # DataFrame: index=date, columns=ticker
        available = [t for t in universe if t in close.columns and close[t].notna().any()]

        if not available:
            return {}

        # ── Step 2: assign equal weight ────────────────────────────
        # _equal_weight() is a helper from StrategyBase.
        return self._equal_weight(available)

    # ── ML hook example (not used here) ───────────────────────────
    # Uncomment and implement these for ML strategies:
    #
    # def requires_training(self) -> bool:
    #     return True
    #
    # def train(self, prices: pd.DataFrame, train_end: pd.Timestamp) -> None:
    #     # Fit your model here using prices up to train_end.
    #     # Store fitted model as self._model = ...
    #     pass
