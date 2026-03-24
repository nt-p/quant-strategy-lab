import numpy as np
import pandas as pd

from .base import RebalanceFrequency, StrategyBase, StrategyCategory


class ResidualMomentumStrategy(StrategyBase):
    """Cross-sectional residual momentum strategy.

    For each asset, regress its daily returns on a market return proxy over a
    rolling lookback window. Compute cumulative residual return and hold the
    top-N assets with the strongest positive residual momentum.
    """

    strategy_id = "residual_momentum"
    name = "Residual Momentum"
    category = StrategyCategory.QUANTITATIVE
    description = "Ranks assets by momentum after removing market beta."
    long_description = (
        "This strategy estimates each asset's market beta using rolling daily returns, "
        "computes residual returns relative to a market proxy, and ranks assets by "
        "cumulative residual return over the lookback window. It then allocates equal "
        "weight to the top-ranked assets."
    )

    rebalance_frequency = RebalanceFrequency.MONTHLY

    lookback_days = 126
    top_n = 5
    min_obs = 60

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
    ) -> dict[str, float]:
        close = self._pivot_close(prices).sort_index()

        available = [t for t in universe if t in close.columns]
        if not available or current_date not in close.index:
            return {}

        close = close[available]

        current_idx = close.index.get_loc(current_date)
        if current_idx < self.lookback_days:
            return {}

        # Rolling window ending at current_date
        window = close.iloc[current_idx - self.lookback_days : current_idx + 1]

        # Daily returns
        rets = window.pct_change().dropna(how="all")
        if rets.empty:
            return {}

        # Keep only assets with enough observations
        valid_cols = [c for c in rets.columns if rets[c].count() >= self.min_obs]
        if not valid_cols:
            return {}

        rets = rets[valid_cols]

        # Simple market proxy: equal-weight average return across valid assets
        market_ret = rets.mean(axis=1)

        scores: dict[str, float] = {}

        for ticker in valid_cols:
            asset_ret = rets[ticker].dropna()

            # Align with market series
            aligned = pd.concat(
                [asset_ret.rename("asset"), market_ret.rename("market")],
                axis=1,
                join="inner",
            ).dropna()

            if len(aligned) < self.min_obs:
                continue

            x = aligned["market"].values
            y = aligned["asset"].values

            var_m = np.var(x)
            if var_m <= 1e-12:
                continue

            beta = np.cov(y, x, ddof=1)[0, 1] / var_m
            alpha = y.mean() - beta * x.mean()

            residual = y - (alpha + beta * x)

            # Sum of residual returns as residual momentum score
            scores[ticker] = float(np.sum(residual))

        if not scores:
            return {}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        # Optional: keep only positive residual momentum names
        selected = [ticker for ticker, score in ranked if score > 0][: self.top_n]

        if not selected:
            return {}

        return self._equal_weight(selected)