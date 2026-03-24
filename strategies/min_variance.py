# strategies/min_variance.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from .base import StrategyBase, StrategyCategory


class MinVarianceStrategy(StrategyBase):
    strategy_id = "min_variance"
    name = "Minimum Variance"
    category = StrategyCategory.QUANTITATIVE
    description = "Minimise portfolio volatility using covariance optimisation"
    long_description = """
    Finds the portfolio weights that minimise total portfolio variance:
    
        min  w'Σw
        s.t. Σwᵢ = 1, wᵢ ≥ 0
    
    Uses Ledoit-Wolf shrinkage on the covariance matrix for numerical
    stability. Falls back to equal weight if optimisation fails or
    insufficient data is available.
    """

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
        fundamentals: dict | None = None,
    ) -> dict[str, float]:

        # 1. Pivot to get close prices: index=date, columns=tickers
        #    Then compute daily simple returns and drop NaN rows
        close = self._pivot_close(prices)
        returns = close.pct_change().dropna(how="all")
        # 2. Check if you have enough data (at least 60 days).
        #    If not, fall back to equal weight.
        if len(returns) < 60:
            return {ticker: 1 / len(universe) for ticker in universe if ticker in close.columns}
        # 3. Take the last 252 days of returns (or fewer if unavailable).
        #    Drop any columns (tickers) that have zero variance — these
        #    break the covariance estimate.
        returns = returns.iloc[-252:]
        returns = returns.loc[:, returns.var() > 0]
        # 4. Estimate the covariance matrix using Ledoit-Wolf shrinkage:
        #    lw = LedoitWolf().fit(returns_array)
        #    cov = lw.covariance_
        #    This gives you a well-conditioned covariance matrix.
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_

        # 5. Define the objective function:
        #    portfolio_variance(w) = w @ cov @ w
        def portfolio_variance(weights: np.ndarray) -> float:
            return weights.T @ cov @ weights
        # 6. Set up the optimiser:
        #    - constraint: weights sum to 1
        #    - bounds: each weight between 0 and max_weight
        #      (use max_weight = 0.4 to prevent over-concentration)
        #    - initial guess: equal weight
        num_assets = cov.shape[0]
        initial_weights = np.array([1 / num_assets] * num_assets)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        max_weight = min(0.4, 1.0 / num_assets + 0.2)  # adaptive cap
        bounds = [(0, max_weight) for _ in range(num_assets)]
        # 7. Run scipy.optimize.minimize with method='SLSQP'
        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )
        # 8. Check result.success. If the optimiser failed, log a
        #    warning and return equal weights.
        if not result.success:
            return {ticker: 1 / len(universe) for ticker in universe if ticker in close.columns}
        # 9. Clean up: set any tiny negative weights (numerical noise
        #    like -1e-15) to 0, then renormalise so weights sum to 1.
  
        weights = np.clip(result.x, 0, None)
        weights = weights / weights.sum()  # renormalise after clipping
        # 10. Build and return the {ticker: weight} dict.
        #     Include tickers that were dropped (zero variance) with
        #     weight 0.
        optimised_tickers = returns.columns.tolist()
        weight_dict = {}
        for ticker in universe:
            if ticker in optimised_tickers:
                weight_dict[ticker] = weights[optimised_tickers.index(ticker)]
            else:
                weight_dict[ticker] = 0.0
        return weight_dict
