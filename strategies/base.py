"""StrategyBase ABC — all strategies inherit from this."""

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class StrategyCategory(str, Enum):
    PASSIVE = "passive"
    QUANTITATIVE = "quantitative"
    ML = "ml"


class RebalanceFrequency(str, Enum):
    """How often the engine calls get_weights() during a backtest."""

    NEVER = "never"        # weights set once on day 1 and never changed
    WEEKLY = "weekly"      # every 5 trading days
    MONTHLY = "monthly"    # first trading day of each calendar month
    QUARTERLY = "quarterly"  # first trading day of each calendar quarter


class StrategyBase(ABC):
    """Base class for ALL strategies.

    Drop a new file in strategies/ that subclasses this and it
    automatically appears in the dashboard — no other wiring needed.

    Required
    --------
    - Override the five class-attribute metadata fields below.
    - Implement get_weights().

    Optional
    --------
    - Override rebalance_frequency (default: MONTHLY).
    - Override requires_training() + train() for ML strategies.
    """

    # ── Metadata (override as class attributes in each strategy) ──
    strategy_id: str = ""               # unique slug: "momentum_12m"
    name: str = ""                       # display name: "12-Month Momentum"
    category: StrategyCategory = StrategyCategory.QUANTITATIVE
    description: str = ""               # one-liner shown on the toggle card
    long_description: str = ""          # methodology detail for the deep-dive page

    # ── Rebalance cadence (override if not monthly) ───────────────
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY

    # ── Core interface ────────────────────────────────────────────

    @abstractmethod
    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
        fundamentals: dict[str, dict] | None = None,
    ) -> dict[str, float]:
        """Compute portfolio weights for the given rebalance date.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data in long format with columns:
            [date, ticker, open, high, low, close, volume].
            Contains ALL data from dataset start up to and INCLUDING
            current_date. NEVER contains future data.
        current_date : pd.Timestamp
            The rebalance date. Only use data <= this date.
        universe : list[str]
            Tickers available in the current universe.
        fundamentals : dict[str, dict] or None
            Optional fundamental data keyed by ticker.  Each value is a dict
            with keys: market_cap, trailing_pe, price_to_book, dividend_yield,
            revenue_growth, earnings_growth (any may be None if unavailable).
            Existing strategies that do not declare this parameter continue to
            work — the engine detects the signature and omits it when absent.

        Returns
        -------
        dict[str, float]
            {ticker: weight}. Weights should sum to ~1.0.
            Omitted tickers are assumed weight 0.
        """
        ...

    # ── Optional hooks for ML strategies ──────────────────────────

    def requires_training(self) -> bool:
        """Return True if this strategy needs train() called before backtesting."""
        return False

    def train(self, prices: pd.DataFrame, train_end: pd.Timestamp) -> None:
        """Train the model using data up to train_end.

        Called once by the engine before the walk-forward loop begins,
        only when requires_training() returns True.

        Parameters
        ----------
        prices : pd.DataFrame
            Full historical OHLCV data up to train_end in long format.
        train_end : pd.Timestamp
            Training cutoff. Do not use data after this date.
        """
        pass

    # ── Helpers available to all strategies ───────────────────────

    def _pivot_close(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Convert long-format prices to a close-price pivot table.

        Parameters
        ----------
        prices : pd.DataFrame
            Long-format OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            index=date, columns=ticker, values=close.
        """
        return prices.pivot_table(values="close", index="date", columns="ticker")

    def _zscore_cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score each row cross-sectionally (subtract row mean, divide by row std).

        Parameters
        ----------
        df : pd.DataFrame
            Pivot table (index=date, columns=tickers).

        Returns
        -------
        pd.DataFrame
            Same shape, each row has mean≈0, std≈1.
        """
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    def _equal_weight(self, tickers: list[str]) -> dict[str, float]:
        """Return a uniform weight allocation across the given tickers.

        Parameters
        ----------
        tickers : list[str]
            Tickers to include. Empty list → empty dict.

        Returns
        -------
        dict[str, float]
            Each ticker mapped to 1 / len(tickers).
        """
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
