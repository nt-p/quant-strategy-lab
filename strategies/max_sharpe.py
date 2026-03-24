import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import RebalanceFrequency, StrategyBase, StrategyCategory


class MaximumSharpeStrategy(StrategyBase):
    """Long-only maximum Sharpe ratio portfolio."""

    strategy_id = "maximum_sharpe"
    name = "Maximum Sharpe"
    category = StrategyCategory.QUANTITATIVE
    description = "Long-only portfolio that maximizes estimated Sharpe ratio."
    long_description = (
        "This strategy estimates expected returns and covariance from a rolling "
        "historical window, then solves for the long-only fully-invested "
        "portfolio that maximizes the estimated Sharpe ratio."
    )

    rebalance_frequency = RebalanceFrequency.MONTHLY
    lookback_days = 126
    risk_free_rate = 0.0

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
    ) -> dict[str, float]:

        close = self._pivot_close(prices)

        # keep only current universe
        cols = [c for c in close.columns if c in universe]
        close = close[cols]

        if close.empty or current_date not in close.index:
            return {}

        current_idx = close.index.get_loc(current_date)
        if current_idx < self.lookback_days:
            return {}

        window = close.iloc[current_idx - self.lookback_days: current_idx + 1]
        rets = window.pct_change().dropna(axis=0, how="any").dropna(axis=1, how="any")

        if rets.empty or rets.shape[1] == 0:
            return {}

        tickers = rets.columns.tolist()
        mu = rets.mean().values
        cov = rets.cov().values

        n = len(tickers)

        # small ridge regularization for numerical stability
        cov = cov + 1e-6 * np.eye(n)

        def neg_sharpe(w):
            port_ret = np.dot(w, mu) - self.risk_free_rate / 252.0
            port_vol = np.sqrt(np.dot(w, cov @ w))
            if port_vol <= 1e-12:
                return 1e6
            return -port_ret / port_vol

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe,
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )

        if not result.success:
            return self._equal_weight(tickers)

        weights = result.x
        weights = np.maximum(weights, 0.0)

        total = weights.sum()
        if total <= 1e-12:
            return self._equal_weight(tickers)

        weights = weights / total
        return {ticker: float(w) for ticker, w in zip(tickers, weights)}