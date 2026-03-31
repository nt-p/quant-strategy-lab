"""Deep learning models for return prediction: LSTM and Temporal Fusion Transformer.

Implements walk-forward training with strict no-lookahead bias.  All sequences
are built from data up to and including the current date only.

Models
------
LSTMModel     — two-layer LSTM binary direction classifier (P(return > 0))
SimpleTFT     — lightweight TFT with Variable Selection Networks, LSTM encoder,
                multi-head self-attention, and P10/P50/P90 quantile heads

References
----------
Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
    Neural Computation, 9(8), 1735–1780.
Fischer, T. & Krauss, C. (2018). Deep learning with long short-term memory
    networks for financial market predictions.
    European Journal of Operational Research, 270(2), 654–669.
Lim, B., Arık, S.Ö., Loeff, N. & Pfister, T. (2021). Temporal Fusion
    Transformers for Interpretable Multi-horizon Time Series Forecasting.
    International Journal of Forecasting, 37(4), 1748–1764.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modules.ml_signals import (
    FEATURE_COLS,
    FEATURE_META,
    FORECAST_HORIZON,
    MIN_TRAIN_DAYS,
    RETRAIN_EVERY_DAYS,
    compute_features_for_ticker,
)

# ── Torch availability guard ──────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE: bool = True
except ImportError:
    TORCH_AVAILABLE = False

SEQUENCE_LEN: int = 60   # lookback window in trading days
N_FEATURES: int = len(FEATURE_COLS)

_TAUS: list[float] = [0.1, 0.5, 0.9]   # quantile levels for TFT


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DeepLearningResult:
    """Output of a walk-forward deep-learning evaluation.

    Attributes
    ----------
    model_name : str
    predictions : pd.Series
        P(return > 0) per out-of-sample sample.  Index = integer position.
    actuals : pd.Series
        Binary direction labels (0/1).  Same index as predictions.
    oos_accuracy : float
        Out-of-sample direction accuracy.
    hit_rate : float
        Accuracy on high-confidence predictions (P ≥ 0.65).
    p10, p50, p90 : pd.Series or None
        TFT quantile return predictions (not probabilities).
    vsn_importance : np.ndarray or None
        Shape (n_features,).  Mean VSN gate weights across test window.
    attn_map : np.ndarray or None
        Shape (seq_len, seq_len).  Mean self-attention map over test window.
    training_losses : list[float]
        BCE / pinball loss per epoch for the last training window.
    final_model : object or None
        Fitted PyTorch model (last training window).
    final_scaler : StandardScaler or None
        Feature scaler fitted on last training window.
    """

    model_name: str
    predictions: pd.Series
    actuals: pd.Series
    oos_accuracy: float
    hit_rate: float
    p10: pd.Series | None = None
    p50: pd.Series | None = None
    p90: pd.Series | None = None
    vsn_importance: np.ndarray | None = None
    attn_map: np.ndarray | None = None
    training_losses: list[float] = field(default_factory=list)
    final_model: object | None = None
    final_scaler: StandardScaler | None = None


# ── Model definitions (only defined when torch is available) ──────────────────

if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        """Two-layer LSTM binary classifier.

        Architecture
        ------------
        LSTM(n_features, 64)  → dropout(0.2)
        LSTM(64, 32)           → dropout(0.2)
        Linear(32, 1)          → Sigmoid → P(return > 0)
        """

        def __init__(self, n_features: int = N_FEATURES) -> None:
            super().__init__()
            self.lstm1 = nn.LSTM(
                input_size=n_features,
                hidden_size=64,
                batch_first=True,
            )
            self.drop1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(
                input_size=64,
                hidden_size=32,
                batch_first=True,
            )
            self.drop2 = nn.Dropout(0.2)
            self.fc = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Shape (batch, seq_len, n_features).

            Returns
            -------
            torch.Tensor
                Shape (batch,) — P(return > 0).
            """
            out1, _ = self.lstm1(x)
            out1 = self.drop1(out1)
            out2, _ = self.lstm2(out1)
            out2 = self.drop2(out2)
            # Take only the last time step
            last = out2[:, -1, :]
            logit = self.fc(last)
            return self.sigmoid(logit).squeeze(-1)

    class SimpleTFT(nn.Module):
        """Lightweight Temporal Fusion Transformer for return distribution forecasting.

        Architecture
        ------------
        Variable Selection Network  — per-time-step softmax gates
        Input projection            — Linear(n_features, 32)
        LSTM encoder                — LSTM(32, 32, batch_first=True)
        Manual scaled dot-product attention (avoids nn.MultiheadAttention fused
        kernel which can segfault on macOS/MPS with certain input patterns)
        Layer norm + residual
        Three quantile heads        — Linear(32,16) → ELU → Linear(16,1) for P10/P50/P90

        Returns both predictions and interpretability artefacts:
        vsn_weights  (B, T, F)  — Variable Selection Network gates
        attn_weights (B, T, T)  — self-attention map
        """

        def __init__(
            self,
            n_features: int = N_FEATURES,
            hidden: int = 32,
        ) -> None:
            super().__init__()
            self.n_features = n_features
            self.hidden = hidden
            self._scale = hidden ** -0.5

            # Variable Selection Network: learns which features matter per time step
            self.vsn_fc = nn.Linear(n_features, n_features)

            # Input projection
            self.input_proj = nn.Linear(n_features, hidden)

            # LSTM encoder
            self.encoder = nn.LSTM(hidden, hidden, batch_first=True)

            # Manual attention projections (Q, K, V) — single head, stable on all platforms
            self.attn_q = nn.Linear(hidden, hidden, bias=False)
            self.attn_k = nn.Linear(hidden, hidden, bias=False)
            self.attn_v = nn.Linear(hidden, hidden, bias=False)

            self.layer_norm = nn.LayerNorm(hidden)

            # Three quantile output heads: P10, P50, P90
            def _head() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(hidden, 16),
                    nn.ELU(),
                    nn.Linear(16, 1),
                )

            self.head_p10 = _head()
            self.head_p50 = _head()
            self.head_p90 = _head()

        def forward(
            self,
            x: torch.Tensor,
            return_weights: bool = False,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            "torch.Tensor | None",
            "torch.Tensor | None",
        ]:
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Shape (batch, seq_len, n_features).
            return_weights : bool
                When True also returns VSN gates and attention map for interpretability.

            Returns
            -------
            p10, p50, p90 : torch.Tensor  shape (batch,)
            vsn_weights   : torch.Tensor or None  shape (batch, seq_len, n_features)
            attn_weights  : torch.Tensor or None  shape (batch, seq_len, seq_len)
            """
            # Variable Selection Network — softmax over features per time step
            vsn_logits = self.vsn_fc(x)                        # (B, T, F)
            vsn_weights = torch.softmax(vsn_logits, dim=-1)    # (B, T, F)
            x_selected = x * vsn_weights                       # (B, T, F)

            # Input projection
            x_proj = self.input_proj(x_selected)               # (B, T, hidden)

            # LSTM encoder
            enc_out, _ = self.encoder(x_proj)                  # (B, T, hidden)

            # Manual scaled dot-product attention — single head, pure Python/torch ops,
            # no fused kernel, stable on macOS and all CPU backends.
            Q = self.attn_q(enc_out)                           # (B, T, hidden)
            K = self.attn_k(enc_out)                           # (B, T, hidden)
            V = self.attn_v(enc_out)                           # (B, T, hidden)
            scores = torch.bmm(Q, K.transpose(1, 2)) * self._scale  # (B, T, T)
            attn_w = torch.softmax(scores, dim=-1)             # (B, T, T)
            attn_out = torch.bmm(attn_w, V)                    # (B, T, hidden)

            # Residual + layer norm
            out = self.layer_norm(enc_out + attn_out)          # (B, T, hidden)

            # Use the last time step for prediction
            last = out[:, -1, :]                               # (B, hidden)

            p10 = self.head_p10(last).squeeze(-1)
            p50 = self.head_p50(last).squeeze(-1)
            p90 = self.head_p90(last).squeeze(-1)

            if return_weights:
                return p10, p50, p90, vsn_weights, attn_w
            else:
                return p10, p50, p90, None, None


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball (quantile) loss for a single quantile level tau."""
    diff = target - pred
    return torch.where(diff >= 0, tau * diff, (tau - 1.0) * diff).mean()


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_sequence_dataset(
    prices: pd.DataFrame,
    tickers: list[str],
    macro_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Build a 3-D sequence dataset for LSTM / TFT training.

    Each sample is a sliding window of SEQUENCE_LEN trading days.
    The label is computed on data strictly after the window end (no lookahead).

    Parameters
    ----------
    prices : pd.DataFrame
        Long-format OHLCV with columns [date, ticker, close, volume].
    tickers : list[str]
    macro_df : pd.DataFrame or None

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, SEQUENCE_LEN, N_FEATURES).
    y_reg : np.ndarray
        Shape (n_samples,).  Forward FORECAST_HORIZON-day return (float).
    y_cls : np.ndarray
        Shape (n_samples,).  Binary direction label (0 or 1).
    sample_tickers : list[str]
        Ticker label for each sample row.
    dates : pd.DatetimeIndex
        Date of the *last day* in each sequence window.
    """
    X_list: list[np.ndarray] = []
    y_reg_list: list[float] = []
    y_cls_list: list[int] = []
    ticker_list: list[str] = []
    date_list: list[pd.Timestamp] = []

    for ticker in tickers:
        subset = prices[prices["ticker"] == ticker][["date", "close", "volume"]].copy()
        min_rows = SEQUENCE_LEN + FORECAST_HORIZON + MIN_TRAIN_DAYS
        if subset.empty or len(subset) < min_rows:
            continue

        feat_df = compute_features_for_ticker(subset, macro_df)
        feat_df = feat_df.dropna(
            subset=FEATURE_COLS,
            thresh=len(FEATURE_COLS) // 2,
        ).fillna(feat_df.median(numeric_only=True))

        close_idx = subset.set_index("date")["close"].sort_index()
        fwd_ret = close_idx.pct_change(FORECAST_HORIZON).shift(-FORECAST_HORIZON)
        fwd_ret = fwd_ret.reindex(feat_df.index)

        # Keep only rows that have both features and a forward label
        valid_mask = fwd_ret.notna()
        feat_valid = feat_df.loc[valid_mask]
        fwd_valid = fwd_ret.loc[valid_mask]

        if len(feat_valid) < SEQUENCE_LEN + 10:
            continue

        feat_arr = feat_valid[FEATURE_COLS].values.astype(np.float32)
        fwd_arr = fwd_valid.values.astype(np.float32)
        date_arr = feat_valid.index

        # Sliding window
        for i in range(SEQUENCE_LEN, len(feat_arr)):
            window = feat_arr[i - SEQUENCE_LEN : i]   # (SEQUENCE_LEN, N_FEATURES)
            X_list.append(window)
            y_reg_list.append(float(fwd_arr[i - 1]))
            y_cls_list.append(int(fwd_arr[i - 1] > 0.0))
            ticker_list.append(ticker)
            date_list.append(date_arr[i - 1])

    if not X_list:
        empty = np.empty((0, SEQUENCE_LEN, N_FEATURES), dtype=np.float32)
        return (
            empty,
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            [],
            pd.DatetimeIndex([]),
        )

    X = np.stack(X_list, axis=0)
    y_reg = np.array(y_reg_list, dtype=np.float32)
    y_cls = np.array(y_cls_list, dtype=np.int64)
    dates_idx = pd.DatetimeIndex(date_list)

    return X, y_reg, y_cls, ticker_list, dates_idx


# ── Walk-forward LSTM ─────────────────────────────────────────────────────────

def run_walk_forward_lstm(
    X: np.ndarray,
    y_cls: np.ndarray,
    dates: pd.DatetimeIndex,
    max_epochs: int = 10,
    max_windows: int = 8,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> "DeepLearningResult":
    """Walk-forward binary direction classification with LSTM.

    Training is identical to XGBoost walk-forward: expanding window starting
    at 60% of the timeline, retraining every RETRAIN_EVERY_DAYS observations.
    Features are normalised per training window using StandardScaler.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, SEQUENCE_LEN, N_FEATURES).
    y_cls : np.ndarray
        Shape (n_samples,) binary direction labels.
    dates : pd.DatetimeIndex
        Date of the last day in each sequence.  Same length as X.
    max_epochs : int
        Maximum epochs per walk-forward window.  Default 10.
    max_windows : int
        Maximum number of walk-forward retraining windows.  Caps runtime on
        long date ranges.  Default 8.
    progress_callback : callable or None
        Optional ``(window_idx, total_windows) -> None`` called after each
        window completes.  Pass ``None`` to disable.

    Returns
    -------
    DeepLearningResult
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for LSTM training. "
            "Install it with: pip install torch>=2.1.0"
        )

    from sklearn.metrics import accuracy_score  # noqa: PLC0415

    n = len(X)
    if n < 50:
        raise ValueError(
            f"Insufficient samples for walk-forward LSTM training ({n} < 50). "
            "Extend the date range or add more tickers."
        )

    unique_dates = pd.Series(dates).sort_values().unique()
    n_dates = len(unique_dates)

    # Determine train/test split index (60% train start)
    if n_dates >= MIN_TRAIN_DAYS:
        train_cutoff_idx = int(n_dates * 0.6)
    else:
        train_cutoff_idx = max(int(n * 0.6), 30)

    # Map each sample to its date rank
    date_to_rank = {d: i for i, d in enumerate(unique_dates)}
    sample_ranks = np.array([date_to_rank.get(d, 0) for d in dates])

    # Collect OOS predictions
    all_preds: list[float] = []
    all_actuals: list[int] = []
    last_losses: list[float] = []
    final_model: LSTMModel | None = None
    final_scaler: StandardScaler | None = None

    # Compute total windows so caller can show progress
    _step = max(1, RETRAIN_EVERY_DAYS)
    _total_windows = min(
        max_windows,
        max(1, int(np.ceil((n_dates - train_cutoff_idx) / _step))),
    )

    step = _step
    cutoff = train_cutoff_idx
    window_idx = 0

    while cutoff < n_dates and window_idx < max_windows:
        train_mask = sample_ranks < cutoff
        test_end = min(cutoff + step, n_dates)
        test_mask = (sample_ranks >= cutoff) & (sample_ranks < test_end)

        if not train_mask.any() or not test_mask.any():
            cutoff = test_end
            continue

        X_train_raw = X[train_mask]
        y_train = y_cls[train_mask]
        X_test_raw = X[test_mask]
        y_test = y_cls[test_mask]

        # Normalise per-feature across training window
        scaler = StandardScaler()
        n_train, seq, nf = X_train_raw.shape
        X_train_flat = X_train_raw.reshape(-1, nf)
        X_test_flat = X_test_raw.reshape(-1, nf)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, seq, nf)
        X_test_scaled = scaler.transform(X_test_flat).reshape(-1, seq, nf)
        X_train_scaled = np.clip(X_train_scaled, -10.0, 10.0).astype(np.float32)
        X_test_scaled = np.clip(X_test_scaled, -10.0, 10.0).astype(np.float32)

        # 80/20 val split within training data
        val_split = min(int(len(X_train_scaled) * 0.8), len(X_train_scaled) - 1)
        X_tr = torch.tensor(X_train_scaled[:val_split])
        y_tr = torch.tensor(y_train[:val_split], dtype=torch.float32)
        X_val = torch.tensor(X_train_scaled[val_split:])
        y_val = torch.tensor(y_train[val_split:], dtype=torch.float32)

        # Build model and train
        model = LSTMModel(n_features=nf)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        tr_dataset = TensorDataset(X_tr, y_tr)
        tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True)

        best_val_loss = float("inf")
        patience = 7
        patience_counter = 0
        epoch_losses: list[float] = []

        model.train()
        for epoch in range(max_epochs):
            ep_loss = 0.0
            for xb, yb in tr_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            epoch_losses.append(ep_loss / max(len(tr_loader), 1))

            # Validation loss for early stopping
            if len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = loss_fn(val_pred, y_val).item()
                model.train()
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        last_losses = epoch_losses
        final_model = model
        final_scaler = scaler
        window_idx += 1
        if progress_callback is not None:
            progress_callback(window_idx, _total_windows)

        # Predict on test set
        model.eval()
        X_test_t = torch.tensor(X_test_scaled)
        with torch.no_grad():
            probs = model(X_test_t).numpy()

        all_preds.extend(probs.tolist())
        all_actuals.extend(y_test.tolist())

        cutoff = test_end

    if not all_preds:
        # Fallback: single 80/20 split
        split = int(n * 0.8)
        scaler = StandardScaler()
        n_tr, seq, nf = X[:split].shape
        X_tr_flat = X[:split].reshape(-1, nf)
        X_te_flat = X[split:].reshape(-1, nf)
        X_tr_s = scaler.fit_transform(X_tr_flat).reshape(n_tr, seq, nf).astype(np.float32)
        X_te_s = scaler.transform(X_te_flat).reshape(-1, seq, nf).astype(np.float32)

        model = LSTMModel(n_features=nf)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        X_t = torch.tensor(np.clip(X_tr_s, -10.0, 10.0))
        y_t = torch.tensor(y_cls[:split], dtype=torch.float32)
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(np.clip(X_te_s, -10.0, 10.0))).numpy()
        all_preds = probs.tolist()
        all_actuals = y_cls[split:].tolist()
        last_losses = []
        final_model = model
        final_scaler = scaler

    preds_arr = np.array(all_preds, dtype=np.float32)
    actuals_arr = np.array(all_actuals, dtype=np.int64)
    pred_labels = (preds_arr >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score  # noqa: PLC0415

    oos_accuracy = float(accuracy_score(actuals_arr, pred_labels))
    high_conf = preds_arr >= 0.65
    hit_rate = (
        float(accuracy_score(actuals_arr[high_conf], pred_labels[high_conf]))
        if high_conf.any()
        else oos_accuracy
    )

    return DeepLearningResult(
        model_name="LSTM",
        predictions=pd.Series(preds_arr),
        actuals=pd.Series(actuals_arr),
        oos_accuracy=oos_accuracy,
        hit_rate=hit_rate,
        training_losses=last_losses,
        final_model=final_model,
        final_scaler=final_scaler,
    )


# ── Walk-forward TFT ──────────────────────────────────────────────────────────

_TFT_INFER_BATCH = 256   # max samples per inference pass to cap peak memory


def run_walk_forward_tft(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    dates: pd.DatetimeIndex,
    max_epochs: int = 10,
    max_windows: int = 4,
    max_samples: int = 8_000,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> "DeepLearningResult":
    """Walk-forward quantile forecasting with the SimpleTFT.

    Trains with pinball loss at tau = 0.1, 0.5, 0.9.  Direction signal derived
    from sign(P50) — positive P50 → long signal.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, SEQUENCE_LEN, N_FEATURES).
    y_reg : np.ndarray
        Shape (n_samples,) forward returns (float).
    y_cls : np.ndarray
        Shape (n_samples,) binary direction labels (for accuracy reporting).
    dates : pd.DatetimeIndex
        Date index aligned to X.
    max_epochs : int
        Maximum epochs per walk-forward window.  Default 10.
    max_windows : int
        Maximum number of walk-forward retraining windows.  Default 8.
    progress_callback : callable or None
        Optional ``(window_idx, total_windows) -> None`` called after each window.

    Returns
    -------
    DeepLearningResult with p10, p50, p90, vsn_importance, attn_map populated.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TFT training. "
            "Install it with: pip install torch>=2.1.0"
        )

    from sklearn.metrics import accuracy_score  # noqa: PLC0415

    # Single-threaded avoids macOS thread-pool instability on Apple Silicon
    torch.set_num_threads(1)

    n = len(X)
    if n < 50:
        raise ValueError(
            f"Insufficient samples for walk-forward TFT training ({n} < 50)."
        )

    # Cap samples to prevent OOM on large portfolios — keep the most recent data.
    # Sort by date first so the cap keeps the most recent samples across all tickers.
    if n > max_samples:
        sort_idx = np.argsort(dates)
        X = X[sort_idx][-max_samples:]
        y_reg = y_reg[sort_idx][-max_samples:]
        y_cls = y_cls[sort_idx][-max_samples:]
        dates = dates[sort_idx][-max_samples:]
        n = max_samples

    # Sanitise regression targets — forward returns can be ±inf on degenerate tickers
    y_reg = np.nan_to_num(y_reg, nan=0.0, posinf=1.0, neginf=-1.0).clip(-1.0, 1.0)

    unique_dates = pd.Series(dates).sort_values().unique()
    n_dates = len(unique_dates)

    train_cutoff_idx = (
        int(n_dates * 0.6) if n_dates >= MIN_TRAIN_DAYS else max(int(n * 0.6), 30)
    )

    date_to_rank = {d: i for i, d in enumerate(unique_dates)}
    sample_ranks = np.array([date_to_rank.get(d, 0) for d in dates])

    all_p10: list[float] = []
    all_p50: list[float] = []
    all_p90: list[float] = []
    all_actuals: list[int] = []
    last_losses: list[float] = []
    final_model: SimpleTFT | None = None
    final_scaler: StandardScaler | None = None
    last_vsn: np.ndarray | None = None
    last_attn: np.ndarray | None = None

    _step = max(1, RETRAIN_EVERY_DAYS)
    _total_windows = min(
        max_windows,
        max(1, int(np.ceil((n_dates - train_cutoff_idx) / _step))),
    )

    step = _step
    cutoff = train_cutoff_idx
    window_idx = 0

    while cutoff < n_dates and window_idx < max_windows:
        train_mask = sample_ranks < cutoff
        test_end = min(cutoff + step, n_dates)
        test_mask = (sample_ranks >= cutoff) & (sample_ranks < test_end)

        if not train_mask.any() or not test_mask.any():
            cutoff = test_end
            continue

        X_train_raw = X[train_mask]
        y_tr_reg = y_reg[train_mask]
        X_test_raw = X[test_mask]
        y_te_cls = y_cls[test_mask]

        # Normalise
        scaler = StandardScaler()
        n_tr_samples, seq, nf = X_train_raw.shape
        X_tr_scaled = (
            scaler.fit_transform(X_train_raw.reshape(-1, nf))
            .reshape(n_tr_samples, seq, nf)
            .astype(np.float32)
        )
        X_te_scaled = (
            scaler.transform(X_test_raw.reshape(-1, nf))
            .reshape(-1, seq, nf)
            .astype(np.float32)
        )
        # Clip then sanitise: NaN/inf in feature columns (e.g. RSI on short series)
        # would propagate into LSTM hidden state and crash native kernels.
        X_tr_scaled = np.nan_to_num(
            np.clip(X_tr_scaled, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0
        )
        X_te_scaled = np.nan_to_num(
            np.clip(X_te_scaled, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0
        )

        # 80/20 val split
        val_split = min(int(len(X_tr_scaled) * 0.8), len(X_tr_scaled) - 1)

        X_t = torch.tensor(X_tr_scaled[:val_split])
        y_t = torch.tensor(y_tr_reg[:val_split], dtype=torch.float32)
        X_v = torch.tensor(X_tr_scaled[val_split:])
        y_v = torch.tensor(y_tr_reg[val_split:], dtype=torch.float32)

        model = SimpleTFT(n_features=nf)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        tr_dataset = TensorDataset(X_t, y_t)
        tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 7
        epoch_losses: list[float] = []

        is_last_window = (window_idx + 1 >= max_windows) or (test_end >= n_dates)

        model.train()
        for epoch in range(max_epochs):
            ep_loss = 0.0
            for xb, yb in tr_loader:
                optimizer.zero_grad()
                # return_weights=False → fast-path attention (no weight tensor computed)
                p10_pred, p50_pred, p90_pred, _, _ = model(xb, return_weights=False)
                loss = (
                    _pinball_loss(p10_pred, yb, 0.1)
                    + _pinball_loss(p50_pred, yb, 0.5)
                    + _pinball_loss(p90_pred, yb, 0.9)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            epoch_losses.append(ep_loss / max(len(tr_loader), 1))

            if len(X_v) > 0:
                model.eval()
                with torch.no_grad():
                    p10v, p50v, p90v, _, _ = model(X_v, return_weights=False)
                    val_loss = (
                        _pinball_loss(p10v, y_v, 0.1)
                        + _pinball_loss(p50v, y_v, 0.5)
                        + _pinball_loss(p90v, y_v, 0.9)
                    ).item()
                model.train()
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        last_losses = epoch_losses
        final_model = model
        final_scaler = scaler
        window_idx += 1
        if progress_callback is not None:
            progress_callback(window_idx, _total_windows)

        # Inference on test window — batched to cap peak memory allocation
        # Only request interpretability weights on the last window
        model.eval()
        p10_chunks: list[np.ndarray] = []
        p50_chunks: list[np.ndarray] = []
        p90_chunks: list[np.ndarray] = []
        vsn_acc: list[np.ndarray] = []
        attn_acc: list[np.ndarray] = []

        n_te = len(X_te_scaled)
        for b_start in range(0, n_te, _TFT_INFER_BATCH):
            b_end = min(b_start + _TFT_INFER_BATCH, n_te)
            X_b = torch.tensor(X_te_scaled[b_start:b_end])
            with torch.no_grad():
                p10_b, p50_b, p90_b, vsn_b, attn_b = model(
                    X_b, return_weights=is_last_window
                )
            p10_chunks.append(p10_b.numpy())
            p50_chunks.append(p50_b.numpy())
            p90_chunks.append(p90_b.numpy())
            if is_last_window and vsn_b is not None and attn_b is not None:
                vsn_acc.append(vsn_b.numpy())
                attn_acc.append(attn_b.numpy())

        all_p10.extend(np.concatenate(p10_chunks).tolist())
        all_p50.extend(np.concatenate(p50_chunks).tolist())
        all_p90.extend(np.concatenate(p90_chunks).tolist())
        all_actuals.extend(y_te_cls.tolist())

        # Keep interpretability artefacts from the last window
        if is_last_window and vsn_acc:
            vsn_cat = np.concatenate(vsn_acc, axis=0)    # (n_te, seq, features)
            attn_cat = np.concatenate(attn_acc, axis=0)  # (n_te, seq, seq)
            last_vsn = vsn_cat.mean(axis=(0, 1))          # (n_features,)
            last_attn = attn_cat.mean(axis=0)             # (seq, seq)

        cutoff = test_end

    if not all_p50:
        # Fallback single split
        split = int(n * 0.8)
        scaler = StandardScaler()
        n_sp, seq, nf = X[:split].shape
        X_tr_s = (
            scaler.fit_transform(X[:split].reshape(-1, nf))
            .reshape(n_sp, seq, nf)
            .astype(np.float32)
        )
        X_te_s = (
            scaler.transform(X[split:].reshape(-1, nf))
            .reshape(-1, seq, nf)
            .astype(np.float32)
        )
        X_tr_clean = torch.tensor(
            np.nan_to_num(np.clip(X_tr_s, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)
        )
        y_tr_clean = torch.tensor(y_reg[:split], dtype=torch.float32)
        model = SimpleTFT(n_features=nf)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Use DataLoader — never forward the full training set at once (OOM risk)
        fb_tr_dataset = TensorDataset(X_tr_clean, y_tr_clean)
        fb_tr_loader = DataLoader(fb_tr_dataset, batch_size=64, shuffle=True)
        model.train()
        for _ in range(10):
            for xb, yb in fb_tr_loader:
                optimizer.zero_grad()
                p10p, p50p, p90p, _, _ = model(xb, return_weights=False)
                loss = (
                    _pinball_loss(p10p, yb, 0.1)
                    + _pinball_loss(p50p, yb, 0.5)
                    + _pinball_loss(p90p, yb, 0.9)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        X_te_clipped = np.nan_to_num(
            np.clip(X_te_s, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0
        )
        fb_p10_chunks: list[np.ndarray] = []
        fb_p50_chunks: list[np.ndarray] = []
        fb_p90_chunks: list[np.ndarray] = []
        fb_vsn_acc: list[np.ndarray] = []
        fb_attn_acc: list[np.ndarray] = []
        n_fb_te = len(X_te_clipped)
        for b_start in range(0, n_fb_te, _TFT_INFER_BATCH):
            b_end = min(b_start + _TFT_INFER_BATCH, n_fb_te)
            X_b = torch.tensor(X_te_clipped[b_start:b_end])
            with torch.no_grad():
                p10_b, p50_b, p90_b, vsn_b, attn_b = model(X_b, return_weights=True)
            fb_p10_chunks.append(p10_b.numpy())
            fb_p50_chunks.append(p50_b.numpy())
            fb_p90_chunks.append(p90_b.numpy())
            if vsn_b is not None and attn_b is not None:
                fb_vsn_acc.append(vsn_b.numpy())
                fb_attn_acc.append(attn_b.numpy())
        all_p10 = np.concatenate(fb_p10_chunks).tolist()
        all_p50 = np.concatenate(fb_p50_chunks).tolist()
        all_p90 = np.concatenate(fb_p90_chunks).tolist()
        all_actuals = y_cls[split:].tolist()
        if fb_vsn_acc:
            last_vsn = np.concatenate(fb_vsn_acc, axis=0).mean(axis=(0, 1))
            last_attn = np.concatenate(fb_attn_acc, axis=0).mean(axis=0)
        last_losses = []
        final_model = model
        final_scaler = scaler

    p50_arr = np.array(all_p50, dtype=np.float32)
    actuals_arr = np.array(all_actuals, dtype=np.int64)

    # Direction accuracy from sign(P50)
    pred_direction = (p50_arr > 0.0).astype(int)
    from sklearn.metrics import accuracy_score  # noqa: PLC0415

    oos_accuracy = float(accuracy_score(actuals_arr, pred_direction))

    # Convert p50 to probability via sigmoid for hit-rate calculation
    p50_prob = 1.0 / (1.0 + np.exp(-p50_arr * 10.0))  # scaled sigmoid
    high_conf = p50_prob >= 0.65
    hit_rate = (
        float(accuracy_score(actuals_arr[high_conf], pred_direction[high_conf]))
        if high_conf.any()
        else oos_accuracy
    )

    return DeepLearningResult(
        model_name="TFT",
        predictions=pd.Series(p50_prob),
        actuals=pd.Series(actuals_arr),
        oos_accuracy=oos_accuracy,
        hit_rate=hit_rate,
        p10=pd.Series(np.array(all_p10, dtype=np.float32)),
        p50=pd.Series(p50_arr),
        p90=pd.Series(np.array(all_p90, dtype=np.float32)),
        vsn_importance=last_vsn,
        attn_map=last_attn,
        training_losses=last_losses,
        final_model=final_model,
        final_scaler=final_scaler,
    )


# ── RL feature builder ────────────────────────────────────────────────────────

def build_rl_features(
    prices: pd.DataFrame,
    tickers: list[str],
    macro_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build returns and feature arrays for the RL environment.

    All assets are aligned to a common date index (inner join).
    Features are normalised per-day across assets using a rolling StandardScaler
    to avoid lookahead bias.

    Parameters
    ----------
    prices : pd.DataFrame
        Long-format OHLCV with columns [date, ticker, close, volume].
    tickers : list[str]
    macro_df : pd.DataFrame or None

    Returns
    -------
    returns_arr : np.ndarray
        Shape (T, n_assets).  Daily returns per asset.
    features_arr : np.ndarray
        Shape (T, n_assets, n_features).  Normalised feature per asset per day.
    dates : pd.DatetimeIndex
        T common trading days.
    """
    feat_by_ticker: dict[str, pd.DataFrame] = {}
    ret_by_ticker: dict[str, pd.Series] = {}

    for ticker in tickers:
        subset = prices[prices["ticker"] == ticker][["date", "close", "volume"]].copy()
        if subset.empty or len(subset) < MIN_TRAIN_DAYS + 50:
            continue

        feat_df = compute_features_for_ticker(subset, macro_df)
        feat_df = (
            feat_df.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) // 2)
            .fillna(feat_df.median(numeric_only=True))
        )

        close_idx = subset.set_index("date")["close"].sort_index().astype(float)
        daily_ret = close_idx.pct_change()

        # Align returns to feat_df index
        ret_aligned = daily_ret.reindex(feat_df.index).fillna(0.0)

        feat_by_ticker[ticker] = feat_df[FEATURE_COLS]
        ret_by_ticker[ticker] = ret_aligned

    if not feat_by_ticker:
        empty_feat = np.empty((0, len(tickers), N_FEATURES), dtype=np.float32)
        empty_ret = np.empty((0, len(tickers)), dtype=np.float32)
        return empty_ret, empty_feat, pd.DatetimeIndex([])

    # Inner join on dates across all tickers
    common_dates: pd.DatetimeIndex = feat_by_ticker[
        list(feat_by_ticker.keys())[0]
    ].index
    for t in feat_by_ticker:
        common_dates = common_dates.intersection(feat_by_ticker[t].index)

    common_dates = common_dates.sort_values()

    valid_tickers = list(feat_by_ticker.keys())
    n_t = len(valid_tickers)
    n_f = N_FEATURES
    T = len(common_dates)

    if T < 10:
        empty_feat = np.empty((0, n_t, n_f), dtype=np.float32)
        empty_ret = np.empty((0, n_t), dtype=np.float32)
        return empty_ret, empty_feat, pd.DatetimeIndex([])

    returns_arr = np.zeros((T, n_t), dtype=np.float32)
    features_arr = np.zeros((T, n_t, n_f), dtype=np.float32)

    for j, ticker in enumerate(valid_tickers):
        ret_s = ret_by_ticker[ticker].reindex(common_dates).fillna(0.0)
        feat_s = feat_by_ticker[ticker].reindex(common_dates).fillna(0.0)
        returns_arr[:, j] = ret_s.values.astype(np.float32)
        features_arr[:, j, :] = feat_s.values.astype(np.float32)

    # Rolling z-score normalisation per feature across assets
    # Use expanding window up to 252 days to avoid lookahead
    window = 252
    for f in range(n_f):
        feat_slice = features_arr[:, :, f]  # (T, n_assets)
        for t in range(T):
            start = max(0, t - window + 1)
            hist = feat_slice[start : t + 1, :]
            mu = hist.mean()
            sigma = hist.std()
            if sigma > 1e-8:
                features_arr[t, :, f] = (feat_slice[t, :] - mu) / sigma
            else:
                features_arr[t, :, f] = 0.0

    features_arr = np.clip(features_arr, -5.0, 5.0)

    return returns_arr, features_arr, common_dates


# ── Current signal computers ──────────────────────────────────────────────────

def compute_current_signals_lstm(
    prices: pd.DataFrame,
    tickers: list[str],
    model: "LSTMModel",
    scaler: StandardScaler,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute LSTM P(up) signal for each ticker using the most recent SEQUENCE_LEN days.

    Parameters
    ----------
    prices : pd.DataFrame
    tickers : list[str]
    model : LSTMModel
    scaler : StandardScaler
    macro_df : pd.DataFrame or None

    Returns
    -------
    pd.DataFrame with columns: ticker, lstm_score, signal_label
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for compute_current_signals_lstm.")

    rows: list[dict] = []

    for ticker in tickers:
        subset = prices[prices["ticker"] == ticker][["date", "close", "volume"]].copy()
        if subset.empty:
            continue

        feat_df = compute_features_for_ticker(subset, macro_df)
        valid = (
            feat_df.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) // 2)
            .fillna(feat_df.median(numeric_only=True))
        )

        if len(valid) < SEQUENCE_LEN:
            continue

        window = valid[FEATURE_COLS].values[-SEQUENCE_LEN:]   # (SEQUENCE_LEN, N_FEATURES)
        n_f = window.shape[-1]

        # Normalise using fitted scaler
        window_scaled = scaler.transform(window).astype(np.float32)
        window_scaled = np.clip(window_scaled, -10.0, 10.0)

        x = torch.tensor(window_scaled).unsqueeze(0)  # (1, SEQUENCE_LEN, N_FEATURES)
        model.eval()
        with torch.no_grad():
            prob = float(model(x).item())

        if prob >= 0.70:
            label = "STRONG BUY"
        elif prob >= 0.55:
            label = "BUY"
        elif prob >= 0.45:
            label = "NEUTRAL"
        elif prob >= 0.30:
            label = "SELL"
        else:
            label = "STRONG SELL"

        rows.append({"ticker": ticker, "lstm_score": prob, "signal_label": label})

    return pd.DataFrame(rows)


def compute_current_signals_tft(
    prices: pd.DataFrame,
    tickers: list[str],
    model: "SimpleTFT",
    scaler: StandardScaler,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute TFT P10/P50/P90 signal for each ticker using the most recent SEQUENCE_LEN days.

    Parameters
    ----------
    prices : pd.DataFrame
    tickers : list[str]
    model : SimpleTFT
    scaler : StandardScaler
    macro_df : pd.DataFrame or None

    Returns
    -------
    pd.DataFrame with columns: ticker, tft_p10, tft_p50, tft_p90, signal_label
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for compute_current_signals_tft.")

    rows: list[dict] = []

    for ticker in tickers:
        subset = prices[prices["ticker"] == ticker][["date", "close", "volume"]].copy()
        if subset.empty:
            continue

        feat_df = compute_features_for_ticker(subset, macro_df)
        valid = (
            feat_df.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) // 2)
            .fillna(feat_df.median(numeric_only=True))
        )

        if len(valid) < SEQUENCE_LEN:
            continue

        window = valid[FEATURE_COLS].values[-SEQUENCE_LEN:]
        window_scaled = np.nan_to_num(
            np.clip(scaler.transform(window).astype(np.float32), -10.0, 10.0),
            nan=0.0, posinf=10.0, neginf=-10.0,
        )
        x = torch.tensor(window_scaled).unsqueeze(0)  # (1, SEQUENCE_LEN, N_FEATURES)
        model.eval()
        with torch.no_grad():
            p10_t, p50_t, p90_t, _, _ = model(x, return_weights=False)
            p10 = float(p10_t.item())
            p50 = float(p50_t.item())
            p90 = float(p90_t.item())

        # Signal from P50 direction
        if p50 > 0.02:
            label = "STRONG BUY"
        elif p50 > 0.0:
            label = "BUY"
        elif p50 > -0.01:
            label = "NEUTRAL"
        elif p50 > -0.02:
            label = "SELL"
        else:
            label = "STRONG SELL"

        rows.append({
            "ticker": ticker,
            "tft_p10": p10,
            "tft_p50": p50,
            "tft_p90": p90,
            "signal_label": label,
        })

    return pd.DataFrame(rows)
