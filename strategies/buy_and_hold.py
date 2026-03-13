"""Buy & Hold strategy — the simplest possible passive allocation.

Equal-weight the selected universe on day 1 and never rebalance.
Portfolio drifts with market returns indefinitely.

This is the canonical baseline against which all active strategies compete.
When the universe is a single asset (e.g. SPY), the equity curve is
mathematically identical to the benchmark, which is used as the Phase 3
integration smoke-test.

Reference
---------
Sharpe, W. (1991). The Arithmetic of Active Management. *Financial Analysts
Journal*, 47(1), 7–9. — explains why buy-and-hold is the passive benchmark
that the average active strategy must underperform before costs.
"""

from __future__ import annotations

import pandas as pd

from .base import RebalanceFrequency, StrategyBase, StrategyCategory


class BuyAndHold(StrategyBase):
    """Equal-weight buy-and-hold across the selected universe.

    Weights are set once on day 1 and never changed.  Portfolio value
    drifts with daily market returns — no transaction costs incurred
    after the initial allocation.

    With a single-asset universe of SPY, this strategy reproduces the
    benchmark equity curve exactly (to floating-point precision).
    """

    strategy_id = "buy_and_hold"
    name = "Buy & Hold"
    category = StrategyCategory.PASSIVE
    description = "Equal-weight the universe on day 1 and never rebalance."
    long_description = (
        "Buy & Hold allocates 1/N of the portfolio to each of the N selected "
        "assets at inception and then does nothing.  Weights drift as prices "
        "move, reflecting the true passive experience of an investor who "
        "makes a one-time purchase and ignores daily fluctuations.\n\n"
        "This is the simplest possible strategy and serves as the fundamental "
        "baseline: an investor who cannot outperform buy-and-hold (net of costs) "
        "should prefer a passive index fund.\n\n"
        "When the universe contains only SPY, the equity curve is identical to "
        "the S&P 500 benchmark shown in all charts."
    )

    rebalance_frequency = RebalanceFrequency.NEVER

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
    ) -> dict[str, float]:
        """Return equal weights across all assets that have price data.

        Parameters
        ----------
        prices : pd.DataFrame
            Long-format OHLCV data up to and including current_date.
        current_date : pd.Timestamp
            The rebalance date (only called once on day 1 for this strategy).
        universe : list[str]
            All tickers available at this rebalance date.

        Returns
        -------
        dict[str, float]
            {ticker: 1/N} for each ticker that has at least one close price.
        """
        close = self._pivot_close(prices)
        available = [t for t in universe if t in close.columns and close[t].notna().any()]
        return self._equal_weight(available)
