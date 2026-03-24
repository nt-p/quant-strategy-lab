import numpy as np
import pandas as pd

from .base import RebalanceFrequency, StrategyBase, StrategyCategory


class TimeSeriesMomentumVolScaled(StrategyBase):
    """Time-series momentum with skip-month signal and inverse-vol scaling."""

    strategy_id = "ts_mom_skip_vol"
    name = "TS Momentum (Skip-Month, Vol-Scaled)"
    category = StrategyCategory.QUANTITATIVE
    description = (
        "Long assets with positive skip-month momentum, weighted by inverse volatility."
    )
    long_description = (
        "This strategy applies a time-series momentum rule to each asset individually. "
        "The momentum signal is measured using a skip-month return, comparing the price "
        "from 1 month ago to the price from 12 months ago, which avoids the most recent "
        "month to reduce short-term reversal effects. Assets with positive momentum are "
        "included in the portfolio, and weights are assigned inversely proportional to "
        "recent realized volatility so that lower-volatility assets receive larger weights."
    )

    rebalance_frequency = RebalanceFrequency.MONTHLY

    # Approximate trading-day conventions
    lookback_long = 252   # ~12 months
    skip_recent = 21      # ~1 month
    vol_lookback = 63     # ~3 months

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
    ) -> dict[str, float]:
        close = self._pivot_close(prices)

        # Keep only current-universe tickers
        available = [t for t in universe if t in close.columns]
        if not available or current_date not in close.index:
            return {}

        close = close[available].sort_index()

        current_idx = close.index.get_loc(current_date)

        # Need enough data for skip-month momentum and volatility estimation
        min_required = max(self.lookback_long, self.vol_lookback) + self.skip_recent
        if current_idx < min_required:
            return {}

        # t-21 and t-252 for skip-month signal
        price_1m_ago = close.iloc[current_idx - self.skip_recent]
        price_12m_ago = close.iloc[current_idx - self.lookback_long]

        signal = (price_1m_ago / price_12m_ago) - 1.0

        # Use recent daily returns up to current_date for vol estimate
        returns = close.pct_change()
        vol_window = returns.iloc[current_idx - self.vol_lookback + 1 : current_idx + 1]
        vol = vol_window.std()

        # Keep assets with valid positive signal and valid positive vol
        eligible = signal[(signal > 0)].index.tolist()
        if not eligible:
            return {}

        signal = signal.loc[eligible]
        vol = vol.loc[eligible]

        valid = vol[(vol > 0) & vol.notna()].index.tolist()
        if not valid:
            return {}

        inv_vol = 1.0 / vol.loc[valid]
        weights = inv_vol / inv_vol.sum()

        return {ticker: float(w) for ticker, w in weights.items()}