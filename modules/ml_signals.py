"""ML feature engineering pipeline and model training utilities.

Used by pages/7_ML_Lab.py.  All computation is pure (no Streamlit).

Feature set follows Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine
Learning", Review of Financial Studies 33(5).

Walk-forward XGBoost training uses an expanding window retrained every
RETRAIN_EVERY_DAYS trading days — no look-ahead bias.

Label convention
----------------
Binary direction: 1 if the next FORECAST_HORIZON calendar-day forward return > 0,
else 0.  We avoid quintile ranking to prevent lookahead bias in small portfolios.

SHAP
----
TreeExplainer (exact, fast) on the fitted XGBClassifier.  Returns expected-value
baseline and per-feature additive contributions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FORECAST_HORIZON: int = 21      # ~one month
RETRAIN_EVERY_DAYS: int = 63    # ~one quarter
MIN_TRAIN_DAYS: int = 252       # at least 1 year of training data
RSI_PERIOD: int = 14
BB_PERIOD: int = 20
SMA_SHORT: int = 50
SMA_LONG: int = 200
VOL_SHORT: int = 20
VOL_LONG: int = 60
HIGH_52W: int = 252             # trading days in 52-week window
VOLUME_SHORT: int = 30
VOLUME_LONG: int = 90

# Feature display metadata  {col: (label, description)}
FEATURE_META: dict[str, tuple[str, str]] = {
    "mom_1m":       ("Momentum 1M",     "1-month return"),
    "mom_3m":       ("Momentum 3M",     "3-month return (skip 1M)"),
    "mom_6m":       ("Momentum 6M",     "6-month return (skip 1M)"),
    "mom_12m":      ("Momentum 12M",    "12-month return (skip 1M)"),
    "vol_20d":      ("Vol 20D",         "20-day realised volatility (ann.)"),
    "vol_60d":      ("Vol 60D",         "60-day realised volatility (ann.)"),
    "rsi_14":       ("RSI 14",          "14-day Relative Strength Index"),
    "bb_position":  ("BB Position",     "(Price − Lower BB) / BB Width"),
    "price_sma50":  ("Price/SMA50",     "Price ÷ 50-day SMA"),
    "price_sma200": ("Price/SMA200",    "Price ÷ 200-day SMA"),
    "high_52w":     ("52W High",        "Price ÷ 52-week high — George & Hwang (2004)"),
    "volume_trend": ("Volume Trend",    "30D avg volume ÷ 90D avg volume"),
    "vix":          ("VIX (z)",         "VIX level, z-scored over 252 days"),
    "yield_curve":  ("Yield Curve",     "10Y-2Y spread proxy (^TNX − ^IRX), z-scored"),
}

FEATURE_COLS: list[str] = list(FEATURE_META.keys())


# ── Per-ticker feature computation ────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI on a close-price series."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0.0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger_position(close: pd.Series, period: int = BB_PERIOD) -> pd.Series:
    """Normalised position within Bollinger Bands: (price − lower) / width.

    Returns 0.5 when price is at the mid band, approaching 1 at upper band.
    """
    mid   = close.rolling(period, min_periods=period).mean()
    std   = close.rolling(period, min_periods=period).std(ddof=0)
    lower = mid - 2.0 * std
    upper = mid + 2.0 * std
    width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / width


def compute_features_for_ticker(
    ohlcv: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute ML features for a single ticker.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Long-format slice for one ticker, columns: [date, close, volume].
        ``date`` must be UTC-aware timestamps, sorted ascending.
    macro_df : pd.DataFrame or None
        Index = UTC date.  Columns include 'vix' and 'yield_curve' (raw, not z-scored).
        If None, those columns are filled with NaN.

    Returns
    -------
    pd.DataFrame
        Index = date.  Columns = FEATURE_COLS.  All values are floats.
    """
    df = ohlcv.set_index("date").sort_index()
    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)

    ret = close.pct_change()

    # ── Momentum ──────────────────────────────────────────────────────────────
    mom_1m  = close.pct_change(21)
    mom_skip = close.shift(21)  # skip most-recent month (standard)
    mom_3m  = (close / close.shift(63)).sub(1).where(close.shift(21).notna())
    mom_6m  = (close / close.shift(126)).sub(1).where(close.shift(21).notna())
    mom_12m = (close / close.shift(252)).sub(1).where(close.shift(21).notna())
    # Simpler: raw period return skipping last month
    mom_3m  = (close.shift(21) / close.shift(63)).sub(1)
    mom_6m  = (close.shift(21) / close.shift(126)).sub(1)
    mom_12m = (close.shift(21) / close.shift(252)).sub(1)

    # ── Volatility ────────────────────────────────────────────────────────────
    vol_20d = ret.rolling(VOL_SHORT, min_periods=VOL_SHORT).std(ddof=1) * np.sqrt(252)
    vol_60d = ret.rolling(VOL_LONG, min_periods=VOL_LONG).std(ddof=1) * np.sqrt(252)

    # ── Technical ─────────────────────────────────────────────────────────────
    rsi_14     = _rsi(close, RSI_PERIOD)
    bb_pos     = _bollinger_position(close, BB_PERIOD)
    price_sma50  = close / close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
    price_sma200 = close / close.rolling(SMA_LONG, min_periods=SMA_LONG).mean()
    high_52w   = close / close.rolling(HIGH_52W, min_periods=HIGH_52W).max()
    vol_trend  = (
        volume.rolling(VOLUME_SHORT, min_periods=VOLUME_SHORT).mean()
        / volume.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).mean()
    )

    feat = pd.DataFrame(
        {
            "mom_1m":       mom_1m,
            "mom_3m":       mom_3m,
            "mom_6m":       mom_6m,
            "mom_12m":      mom_12m,
            "vol_20d":      vol_20d,
            "vol_60d":      vol_60d,
            "rsi_14":       rsi_14,
            "bb_position":  bb_pos,
            "price_sma50":  price_sma50,
            "price_sma200": price_sma200,
            "high_52w":     high_52w,
            "volume_trend": vol_trend,
        },
        index=close.index,
    )

    # ── Macro features ────────────────────────────────────────────────────────
    if macro_df is not None and not macro_df.empty:
        macro_aligned = macro_df.reindex(close.index, method="ffill")

        if "vix" in macro_aligned.columns:
            vix_raw = macro_aligned["vix"].astype(float)
            vix_mu  = vix_raw.rolling(252, min_periods=63).mean()
            vix_std = vix_raw.rolling(252, min_periods=63).std(ddof=1)
            feat["vix"] = (vix_raw - vix_mu) / vix_std.replace(0.0, np.nan)
        else:
            feat["vix"] = np.nan

        if "yield_curve" in macro_aligned.columns:
            yc_raw = macro_aligned["yield_curve"].astype(float)
            yc_mu  = yc_raw.rolling(252, min_periods=63).mean()
            yc_std = yc_raw.rolling(252, min_periods=63).std(ddof=1)
            feat["yield_curve"] = (yc_raw - yc_mu) / yc_std.replace(0.0, np.nan)
        else:
            feat["yield_curve"] = np.nan
    else:
        feat["vix"]         = np.nan
        feat["yield_curve"] = np.nan

    return feat


def build_ml_dataset(
    prices: pd.DataFrame,
    tickers: list[str],
    macro_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct the ML feature matrix and label vector.

    Each row = (ticker, date).  Label = 1 if the next FORECAST_HORIZON-day
    forward return is positive, else 0.

    Parameters
    ----------
    prices : pd.DataFrame
        Long-format OHLCV.  Columns: [date, ticker, close, volume].
    tickers : list[str]
        Subset of tickers to include.
    macro_df : pd.DataFrame or None
        Macro feature series aligned by date.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.  MultiIndex (ticker, date).
    y : pd.Series
        Binary label (0/1).  Same index.
    """
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        subset = prices[prices["ticker"] == ticker][["date", "close", "volume"]].copy()
        if subset.empty or len(subset) < MIN_TRAIN_DAYS + FORECAST_HORIZON + 50:
            continue

        feat = compute_features_for_ticker(subset, macro_df)

        # Forward return (label)
        close_idx = subset.set_index("date")["close"].sort_index()
        fwd_ret   = close_idx.pct_change(FORECAST_HORIZON).shift(-FORECAST_HORIZON)

        feat["fwd_ret"]  = fwd_ret.reindex(feat.index)
        feat["ticker"]   = ticker

        # Drop rows missing label or more than half the features
        feat = feat.dropna(subset=["fwd_ret"])
        feat = feat.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) // 2)
        feat = feat.fillna(feat.median(numeric_only=True))

        frames.append(feat)

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

    combined = pd.concat(frames)
    combined = combined.reset_index().rename(columns={"date": "date"})
    combined = combined.set_index(["ticker", "date"]).sort_index()

    X = combined[FEATURE_COLS]
    y = (combined["fwd_ret"] > 0.0).astype(int)
    return X, y


# ── Walk-forward XGBoost ──────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Results from a walk-forward XGBoost evaluation."""

    predictions:   pd.Series          # predicted probability (P(return > 0))
    actuals:       pd.Series          # actual labels (0/1)
    oos_accuracy:  float
    hit_rate:      float               # % of high-confidence calls that were correct
    feature_importance: pd.Series     # mean gain importance
    shap_values:   np.ndarray | None  # shape (n_samples, n_features)
    shap_expected: float              # SHAP expected value (base rate)
    final_model:   XGBClassifier      # last trained model (used for current signals)
    final_scaler:  StandardScaler     # scaler fitted on last training window


def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[XGBClassifier, StandardScaler]:
    """Fit a scaler + XGBClassifier on the given training data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y_train.values)

    return model, scaler


def run_walk_forward_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    compute_shap: bool = True,
) -> WalkForwardResult:
    """Walk-forward evaluation of XGBoost on the ML dataset.

    The index must have a 'date' level.  The split is temporal: each fold
    uses all data up to a cutoff date for training, and the next
    RETRAIN_EVERY_DAYS observations for testing.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with MultiIndex (ticker, date).
    y : pd.Series
        Binary labels, same index.
    compute_shap : bool
        Whether to compute SHAP values (can be slow for large datasets).

    Returns
    -------
    WalkForwardResult
    """
    # Sort by date (second index level)
    dates = X.index.get_level_values("date")
    order = dates.argsort()
    X = X.iloc[order]
    y = y.iloc[order]
    dates = X.index.get_level_values("date")

    unique_dates = pd.Series(dates).sort_values().unique()
    n = len(unique_dates)

    if n < MIN_TRAIN_DAYS:
        # Fallback: single 80/20 split
        split = int(len(X) * 0.8)
        model, scaler = _train_xgb(X.iloc[:split], y.iloc[:split])
        X_test_scaled = scaler.transform(X.iloc[split:].values)
        preds = pd.Series(
            model.predict_proba(X_test_scaled)[:, 1],
            index=y.index[split:],
        )
        actuals = y.iloc[split:]
    else:
        preds_list:   list[tuple] = []
        actuals_list: list[tuple] = []

        train_cutoff_idx = int(n * 0.6)  # start testing from 60% of timeline

        # Initial train on first 60% of dates
        cutoff_date = unique_dates[train_cutoff_idx]
        mask_train = dates < cutoff_date
        model, scaler = _train_xgb(X[mask_train], y[mask_train])

        # Walk forward: move cutoff every RETRAIN_EVERY_DAYS dates
        step = max(1, RETRAIN_EVERY_DAYS)
        for i in range(train_cutoff_idx, n, step):
            window_end = unique_dates[min(i + step, n - 1)]
            window_start = unique_dates[i]

            mask_test = (dates >= window_start) & (dates < window_end)
            if not mask_test.any():
                continue

            X_test = X[mask_test]
            y_test = y[mask_test]

            X_test_scaled = scaler.transform(X_test.values)
            prob = model.predict_proba(X_test_scaled)[:, 1]

            for idx_val, p, a in zip(y_test.index, prob, y_test.values):
                preds_list.append((idx_val, p))
                actuals_list.append((idx_val, int(a)))

            # Retrain on all data up to window_end
            if i + step < n:
                mask_retrain = dates < window_end
                model, scaler = _train_xgb(X[mask_retrain], y[mask_retrain])

        if not preds_list:
            # Fallback
            split = int(len(X) * 0.8)
            model, scaler = _train_xgb(X.iloc[:split], y.iloc[:split])
            X_test_scaled = scaler.transform(X.iloc[split:].values)
            preds = pd.Series(
                model.predict_proba(X_test_scaled)[:, 1],
                index=y.index[split:],
            )
            actuals = y.iloc[split:]
        else:
            idx_vals, prob_vals = zip(*preds_list)
            preds   = pd.Series(prob_vals, index=pd.MultiIndex.from_tuples(idx_vals))
            _, act_vals = zip(*actuals_list)
            actuals = pd.Series(act_vals, index=preds.index)

    # ── Metrics ───────────────────────────────────────────────────────────────
    pred_labels   = (preds >= 0.5).astype(int)
    oos_accuracy  = float(accuracy_score(actuals.values, pred_labels.values))

    # Hit rate: among high-confidence predictions (prob > 0.65), % correct
    high_conf = preds >= 0.65
    if high_conf.any():
        hit_rate = float(accuracy_score(actuals[high_conf].values, pred_labels[high_conf].values))
    else:
        hit_rate = float(accuracy_score(actuals.values, pred_labels.values))

    # ── Feature importance (gain) ─────────────────────────────────────────────
    fi = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS,
    ).sort_values(ascending=False)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_vals = None
    shap_exp  = float(y.mean())
    if compute_shap:
        try:
            import shap as shap_lib  # noqa: PLC0415

            explainer  = shap_lib.TreeExplainer(model)
            # Use last test window for SHAP (not the full dataset — too slow)
            sample_idx = min(500, len(X))
            X_sample   = X.iloc[-sample_idx:]
            X_scaled   = scaler.transform(X_sample.values)
            X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
            sv = explainer.shap_values(X_scaled_df)
            # XGB binary returns list [neg_class, pos_class] or single array
            if isinstance(sv, list):
                shap_vals = sv[1]
            else:
                shap_vals = sv
            shap_exp = float(explainer.expected_value if not isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value[1])
        except Exception:
            shap_vals = None

    return WalkForwardResult(
        predictions=preds,
        actuals=actuals,
        oos_accuracy=oos_accuracy,
        hit_rate=hit_rate,
        feature_importance=fi,
        shap_values=shap_vals,
        shap_expected=shap_exp,
        final_model=model,
        final_scaler=scaler,
    )


# ── Logistic regression baseline ─────────────────────────────────────────────

def run_logistic_baseline(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:
    """Fit logistic regression as a baseline comparison.

    Returns (oos_accuracy, hit_rate) using a simple 80/20 temporal split.
    """
    split = int(len(X) * 0.8)
    if split < 30:
        return float("nan"), float("nan")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[:split].values)
    X_test  = scaler.transform(X.iloc[split:].values)
    y_train = y.iloc[:split].values
    y_test  = y.iloc[split:].values

    model = LogisticRegression(max_iter=500, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    prob   = model.predict_proba(X_test)[:, 1]
    labels = (prob >= 0.5).astype(int)
    acc    = float(accuracy_score(y_test, labels))

    hc = prob >= 0.65
    hr = float(accuracy_score(y_test[hc], labels[hc])) if hc.any() else acc

    return acc, hr


# ── Current signals (most recent feature values per ticker) ───────────────────

def compute_current_signals(
    prices: pd.DataFrame,
    tickers: list[str],
    model: XGBClassifier,
    scaler: StandardScaler,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the XGBoost signal for each ticker using the most recent features.

    Returns a DataFrame with columns:
        ticker, <feature_cols>, xgb_score, signal_label
    """
    rows: list[dict] = []

    for ticker in tickers:
        subset = prices[prices["ticker"] == ticker][["date", "close", "volume"]].copy()
        if subset.empty:
            continue

        feat = compute_features_for_ticker(subset, macro_df)
        # Drop rows with too many NaNs
        valid = feat.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS) // 2)
        if valid.empty:
            continue

        last_row = valid.iloc[[-1]].fillna(valid.median(numeric_only=True))
        X_last = scaler.transform(last_row[FEATURE_COLS].values)
        prob   = float(model.predict_proba(X_last)[0, 1])

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

        row: dict = {"ticker": ticker, "xgb_score": prob, "signal_label": label}
        for col in FEATURE_COLS:
            val = last_row[col].values[0] if col in last_row.columns else float("nan")
            row[col] = float(val) if not (isinstance(val, float) and np.isnan(val)) else None
        rows.append(row)

    return pd.DataFrame(rows)


# ── Macro data loader ─────────────────────────────────────────────────────────

def _load_macro(start: str, end: str) -> pd.DataFrame:
    """Fetch ^VIX, ^TNX (10Y), and ^IRX (3M T-bill) from yfinance."""
    try:
        import yfinance as yf  # noqa: PLC0415

        raw = yf.download(
            ["^VIX", "^TNX", "^IRX"],
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()

        # Flatten MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, raw.columns.get_level_values(0) == "Close"]
        else:
            close = raw

        vix = close.get("^VIX", close.get("VIX")) if hasattr(close, "get") else None
        tnx = close.get("^TNX", close.get("TNX")) if hasattr(close, "get") else None
        irx = close.get("^IRX", close.get("IRX")) if hasattr(close, "get") else None

        macro = pd.DataFrame(index=close.index)
        macro.index = pd.to_datetime(macro.index, utc=True)

        if vix is not None:
            macro["vix"] = vix.values if hasattr(vix, "values") else vix
        if tnx is not None and irx is not None:
            tnx_vals = tnx.values if hasattr(tnx, "values") else tnx
            irx_vals = irx.values if hasattr(irx, "values") else irx
            macro["yield_curve"] = tnx_vals - irx_vals

        return macro.dropna(how="all")

    except Exception:
        return pd.DataFrame()


def load_macro_data(start: str, end: str) -> pd.DataFrame:
    """Public wrapper — loads macro features, returns empty DataFrame on failure."""
    try:
        return _load_macro(start, end)
    except Exception:
        return pd.DataFrame()
