"""Gymnasium Portfolio Environment for RL training.

Reward: Differential Sharpe Ratio (Moody & Saffell, IEEE TNN 2001)
        D_t = (σ̂·R_t − 0.5·μ̂·δ) / σ̂^1.5  where δ = R_t − μ̂
        minus λ·‖w_t − w_{t-1}‖₁ transaction cost penalty

State : flattened (lookback × n_assets × n_features) features + current weights
Action: (n_assets,) logits → softmax → portfolio weights

Paper references
----------------
Moody, J. & Saffell, M. (2001). Learning to Trade via Direct Reinforcement.
    IEEE Transactions on Neural Networks, 12(4), 875–889.
Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE: bool = True
except ImportError:
    GYM_AVAILABLE = False

LOOKBACK: int = 60
LAMBDA_COST: float = 0.001


if GYM_AVAILABLE:

    class PortfolioEnv(gym.Env):
        """Portfolio allocation environment for PPO training.

        Parameters
        ----------
        returns : np.ndarray
            Shape (T, n_assets).  Daily log returns per asset.
        features : np.ndarray
            Shape (T, n_assets, n_features).  Normalised ML features per asset per day.
        lookback : int
            Number of historical days included in the observation window.
        lambda_cost : float
            Transaction cost penalty coefficient (λ in the reward formula).
        """

        metadata: dict = {"render_modes": []}

        def __init__(
            self,
            returns: np.ndarray,
            features: np.ndarray,
            lookback: int = LOOKBACK,
            lambda_cost: float = LAMBDA_COST,
        ) -> None:
            super().__init__()

            self.returns = returns.astype(np.float32)
            self.features = features.astype(np.float32)
            self.lookback = lookback
            self.lambda_cost = lambda_cost

            T, n_assets, n_features = features.shape
            self.T = T
            self.n_assets = n_assets
            self.n_features = n_features

            # Observation: flattened lookback window + current weights
            obs_dim: int = lookback * n_assets * n_features + n_assets
            self.observation_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(obs_dim,),
                dtype=np.float32,
            )

            # Action: logits for each asset (softmax applied inside step)
            self.action_space = spaces.Box(
                low=-3.0,
                high=3.0,
                shape=(n_assets,),
                dtype=np.float32,
            )

            # Exponentially weighted moments for differential Sharpe reward
            self._mu_hat: float = 0.0
            self._sigma_hat: float = 1.0
            self._eta: float = 0.01  # smoothing coefficient

            # Internal state (initialised in reset)
            self.t: int = lookback
            self.weights: np.ndarray = np.ones(n_assets, dtype=np.float32) / n_assets

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[np.ndarray, dict]:
            """Reset environment to the beginning of the time series.

            Returns
            -------
            obs : np.ndarray
                Initial observation.
            info : dict
                Empty info dict (Gymnasium convention).
            """
            super().reset(seed=seed)

            self.t = self.lookback
            self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
            self._mu_hat = 0.0
            self._sigma_hat = 1.0

            return self._obs(), {}

        def step(
            self,
            action: np.ndarray,
        ) -> tuple[np.ndarray, float, bool, bool, dict]:
            """Execute one portfolio rebalance step.

            Parameters
            ----------
            action : np.ndarray
                Raw logits of shape (n_assets,).  Softmax converts to weights.

            Returns
            -------
            obs : np.ndarray
            reward : float
                Differential Sharpe Ratio minus transaction cost penalty.
            terminated : bool
            truncated : bool
            info : dict
                Contains ``weights``, ``portfolio_return``, and ``turnover``.
            """
            # Softmax → target weights on the probability simplex
            a = np.asarray(action, dtype=np.float32)
            exp_a = np.exp(a - a.max())
            new_w = exp_a / exp_a.sum()

            # Turnover (one-way L1 distance)
            turnover = float(np.abs(new_w - self.weights).sum())
            cost = self.lambda_cost * turnover

            # Portfolio return net of transaction cost
            r_t = float((self.returns[self.t] * new_w).sum()) - cost

            # Update exponentially weighted moments
            delta = r_t - self._mu_hat
            self._mu_hat += self._eta * delta
            self._sigma_hat += self._eta * (delta ** 2 - self._sigma_hat)

            # Differential Sharpe Ratio reward (Moody & Saffell 2001, eq. 8)
            if self._sigma_hat > 1e-8:
                reward = float(
                    (self._sigma_hat * r_t - 0.5 * self._mu_hat * delta)
                    / (self._sigma_hat ** 1.5)
                )
            else:
                reward = 0.0

            self.weights = new_w
            self.t += 1

            terminated = bool(self.t >= self.T - 1)

            return (
                self._obs(),
                reward,
                terminated,
                False,
                {
                    "weights": new_w.copy(),
                    "portfolio_return": r_t,
                    "turnover": turnover,
                },
            )

        def render(self) -> None:
            """No-op render (not used in training)."""
            pass

        def _obs(self) -> np.ndarray:
            """Construct the current observation vector.

            Returns
            -------
            np.ndarray
                Shape (lookback * n_assets * n_features + n_assets,).
            """
            win = self.features[self.t - self.lookback : self.t]
            return np.concatenate(
                [win.reshape(-1), self.weights],
                dtype=np.float32,
            )

else:
    # Stub so that imports don't fail when gymnasium is unavailable
    class PortfolioEnv:  # type: ignore[no-redef]
        """Placeholder when gymnasium is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "gymnasium is required for PortfolioEnv.  "
                "Install it with: pip install gymnasium"
            )
