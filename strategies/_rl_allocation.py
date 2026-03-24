import numpy as np
import pandas as pd

from .base import RebalanceFrequency, StrategyBase, StrategyCategory


class RLAllocationStrategy(StrategyBase):
    """Q-learning allocation between a risky asset and cash."""

    strategy_id = "rl_allocation"
    name = "RL Allocation"
    category = StrategyCategory.QUANTITATIVE
    description = "Uses Q-learning to allocate between a risky asset and cash."
    long_description = (
        "This strategy uses tabular Q-learning to learn a dynamic allocation "
        "policy between a risky asset and cash using recent momentum, "
        "volatility, and trend features."
    )

    rebalance_frequency = RebalanceFrequency.MONTHLY

    actions = [0.0, 0.25, 0.5, 0.75, 1.0]

    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    n_epochs = 20
    turnover_penalty = 0.001

    lookback_mom_short = 20
    lookback_mom_long = 60
    lookback_vol = 20
    lookback_ma = 20

    def __init__(self):
        super().__init__()
        self.q_table: dict[tuple, float] = {}
        self.primary_ticker: str | None = None

    def requires_training(self) -> bool:
        return True

    def _build_features(self, close: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=close.index)
        df["price"] = close
        df["ret_1"] = close.pct_change()
        df["mom_20"] = close / close.shift(self.lookback_mom_short) - 1.0
        df["mom_60"] = close / close.shift(self.lookback_mom_long) - 1.0
        df["vol_20"] = df["ret_1"].rolling(self.lookback_vol).std()
        df["ma_20"] = close.rolling(self.lookback_ma).mean()
        df["ma_gap_20"] = close / df["ma_20"] - 1.0
        return df

    def _discretize_value(self, x: float, bins: list[float]) -> int:
        return int(np.digitize(x, bins))

    def _state_from_row(self, row: pd.Series) -> tuple[int, int, int, int] | None:
        vals = [row["mom_20"], row["mom_60"], row["vol_20"], row["ma_gap_20"]]
        if any(pd.isna(v) for v in vals):
            return None

        s1 = self._discretize_value(row["mom_20"], [-0.05, 0.05])
        s2 = self._discretize_value(row["mom_60"], [-0.10, 0.10])
        s3 = self._discretize_value(row["vol_20"], [0.01, 0.03])
        s4 = self._discretize_value(row["ma_gap_20"], [-0.02, 0.02])

        return (s1, s2, s3, s4)

    def _get_q(self, state, action_idx: int) -> float:
        return self.q_table.get((state, action_idx), 0.0)

    def _choose_action(self, state, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.actions))

        q_vals = [self._get_q(state, a) for a in range(len(self.actions))]
        return int(np.argmax(q_vals))

    def train(self, prices: pd.DataFrame, start_date: pd.Timestamp) -> None:
        if prices.empty:
            return

        tickers = sorted(prices["ticker"].unique())
        if not tickers:
            return

        self.primary_ticker = tickers[0]

        asset_prices = prices[prices["ticker"] == self.primary_ticker].copy()
        asset_prices = asset_prices.sort_values("date")

        close = asset_prices.set_index("date")["close"].dropna()
        if close.empty:
            return

        features = self._build_features(close)
        features["next_ret"] = close.pct_change().shift(-1)
        features = features.dropna().copy()

        if features.empty:
            return

        for _ in range(self.n_epochs):
            prev_action_idx = 0

            for i in range(len(features) - 1):
                row = features.iloc[i]
                next_row = features.iloc[i + 1]

                state = self._state_from_row(row)
                next_state = self._state_from_row(next_row)

                if state is None or next_state is None:
                    continue

                action_idx = self._choose_action(state, self.epsilon)
                action = self.actions[action_idx]
                prev_action = self.actions[prev_action_idx]

                reward = action * float(row["next_ret"])
                reward -= self.turnover_penalty * abs(action - prev_action)

                old_q = self._get_q(state, action_idx)
                next_q_max = max(
                    self._get_q(next_state, a) for a in range(len(self.actions))
                )

                self.q_table[(state, action_idx)] = old_q + self.alpha * (
                    reward + self.gamma * next_q_max - old_q
                )

                prev_action_idx = action_idx

    def get_weights(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        universe: list[str],
    ) -> dict[str, float]:
        if self.primary_ticker is None or self.primary_ticker not in universe:
            return {}

        asset_prices = prices[prices["ticker"] == self.primary_ticker].copy()
        asset_prices = asset_prices.sort_values("date")

        close = asset_prices.set_index("date")["close"].dropna()
        features = self._build_features(close)

        if current_date not in features.index:
            return {}

        row = features.loc[current_date]
        state = self._state_from_row(row)

        if state is None:
            return {}

        q_vals = [self._get_q(state, a) for a in range(len(self.actions))]
        best_action_idx = int(np.argmax(q_vals))
        allocation = self.actions[best_action_idx]

        return {self.primary_ticker: float(allocation)}