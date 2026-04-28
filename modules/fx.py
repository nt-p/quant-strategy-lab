"""FX Analysis module — G10 carry, momentum, and PPP value signals.

All pairs expressed in USD terms (foreign currency per 1 USD).
Strategies:
  Carry   — Lustig, Roussanov & Verdelhan (2011), RFS
  Momentum — Moskowitz, Ooi & Pedersen (2012), JFE (TSMOM applied to FX)
  Value    — PPP deviation (long-run signal; Rogoff, 1996)

No Streamlit imports — pure computation only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from modules.macro import fetch_series

# ── G10 pair definitions ────────────────────────────────────────────────────
# yf:    yfinance ticker
# invert: True → yfinance quotes USD/FX; invert to get FX/USD
# rate:  FRED series for the foreign short rate (% p.a.)
# name:  display name
# ppp:   approximate OECD PPP fair value (FX/USD) as of 2024; very long-run
G10_PAIRS: dict[str, dict] = {
    "AUD": {
        "yf":     "AUDUSD=X",
        "invert": False,
        "rate":   "IR3TIB01AUM156N",
        "name":   "Australian Dollar",
        "ppp":    0.74,
    },
    "GBP": {
        "yf":     "GBPUSD=X",
        "invert": False,
        "rate":   "IR3TIB01GBM156N",
        "name":   "British Pound",
        "ppp":    1.38,
    },
    "EUR": {
        "yf":     "EURUSD=X",
        "invert": False,
        "rate":   "ECBDFR",
        "name":   "Euro",
        "ppp":    1.22,
    },
    "JPY": {
        "yf":     "USDJPY=X",
        "invert": True,
        "rate":   "IRSTCI01JPM156N",
        "name":   "Japanese Yen",
        "ppp":    0.0092,
    },
    "CAD": {
        "yf":     "USDCAD=X",
        "invert": True,
        "rate":   "IRSTCI01CAM156N",
        "name":   "Canadian Dollar",
        "ppp":    0.82,
    },
    "CHF": {
        "yf":     "USDCHF=X",
        "invert": True,
        "rate":   "IR3TIB01CHM156N",
        "name":   "Swiss Franc",
        "ppp":    0.89,
    },
    "NZD": {
        "yf":     "NZDUSD=X",
        "invert": False,
        "rate":   "IR3TIB01NZM156N",
        "name":   "New Zealand Dollar",
        "ppp":    0.67,
    },
    "NOK": {
        "yf":     "USDNOK=X",
        "invert": True,
        "rate":   "IRSTCI01NOM156N",
        "name":   "Norwegian Krone",
        "ppp":    0.105,
    },
    "SEK": {
        "yf":     "USDSEK=X",
        "invert": True,
        "rate":   "IR3TIB01SEM156N",
        "name":   "Swedish Krona",
        "ppp":    0.107,
    },
}

# USD short rate from FRED
_USD_RATE_SERIES = "DFF"

# Target annualised volatility for TSMOM vol-scaling
_TARGET_VOL = 0.10


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_fx_prices(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch daily FX spot prices for all G10 pairs, all in FX/USD terms.

    Parameters
    ----------
    start : str
        Start date YYYY-MM-DD.

    Returns
    -------
    pd.DataFrame
        Index = date, columns = currency codes (AUD, GBP, …), values = FX/USD.
    """
    tickers = [info["yf"] for info in G10_PAIRS.values()]
    raw = yf.download(
        tickers,
        start=start,
        progress=False,
        auto_adjust=True,
    )["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])

    # Drop tz info if present
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    frames: dict[str, pd.Series] = {}
    for ccy, info in G10_PAIRS.items():
        col = info["yf"]
        if col not in raw.columns:
            continue
        series = raw[col].dropna()
        if info["invert"]:
            series = 1.0 / series.replace(0, np.nan)
        frames[ccy] = series

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices.sort_index(inplace=True)
    # Forward-fill up to 5 days for holidays/gaps
    prices = prices.ffill(limit=5)
    return prices


def fetch_fx_rates(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch FRED short-rate series for each G10 currency plus USD.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame; index = date, columns = currency codes + 'USD'.
        Values are annualised percent (e.g. 5.0 = 5% p.a.).
    """
    fred_ids = {
        ccy: info["rate"]
        for ccy, info in G10_PAIRS.items()
        if info["rate"] is not None
    }
    all_ids = list(fred_ids.values()) + [_USD_RATE_SERIES]
    # deduplicate while preserving order
    seen: dict[str, None] = {}
    for s in all_ids:
        seen[s] = None
    unique_ids = list(seen.keys())

    try:
        raw = fetch_series(unique_ids, start=start)
    except ValueError:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}
    for ccy, sid in fred_ids.items():
        if sid in raw.columns:
            frames[ccy] = raw[sid]

    if _USD_RATE_SERIES in raw.columns:
        frames["USD"] = raw[_USD_RATE_SERIES]

    if not frames:
        return pd.DataFrame()

    rates = pd.concat(frames, axis=1)
    rates.sort_index(inplace=True)
    rates = rates.ffill()  # rates change infrequently; no limit needed
    return rates


# ── Return computation ───────────────────────────────────────────────────────

def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end and compute log returns.

    Returns
    -------
    pd.DataFrame
        Index = month-end date, columns = currency codes.
    """
    monthly = prices.resample("ME").last()
    returns = np.log(monthly / monthly.shift(1))
    return returns.dropna(how="all")


def compute_realized_vol(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Annualised daily realised volatility over a rolling window.

    Returns
    -------
    pd.DataFrame
        Same shape as prices, values annualised (std × sqrt(252)).
    """
    log_rets = np.log(prices / prices.shift(1))
    return log_rets.rolling(window).std() * np.sqrt(252)


# ── Carry signal ─────────────────────────────────────────────────────────────

def compute_carry_signals(
    rates: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute monthly carry differential (foreign rate − USD rate) for each pair.

    Parameters
    ----------
    rates : pd.DataFrame
        FRED rate data (columns include each ccy + 'USD').
    month_ends : pd.DatetimeIndex
        Month-end dates to align carry with return periods.

    Returns
    -------
    pd.DataFrame
        Index = month-end dates, columns = currency codes, values = carry %.
    """
    if rates.empty or "USD" not in rates.columns:
        return pd.DataFrame()

    aligned = rates.reindex(month_ends, method="ffill")
    usd_rate = aligned["USD"]

    ccys = [c for c in aligned.columns if c != "USD" and c in G10_PAIRS]
    carry = aligned[ccys].subtract(usd_rate, axis=0)
    return carry


def run_carry_backtest(
    monthly_returns: pd.DataFrame,
    carry: pd.DataFrame,
    n_long: int = 3,
    n_short: int = 3,
) -> pd.Series:
    """Long top-n carry pairs, short bottom-n carry pairs, equal-weighted.

    Signals are formed using carry known at the prior month-end (no lookahead).

    Returns
    -------
    pd.Series
        Monthly strategy returns.
    """
    common_ccys = [c for c in monthly_returns.columns if c in carry.columns]
    rets = monthly_returns[common_ccys]
    carry_aligned = carry[common_ccys].reindex(rets.index).shift(1)

    strategy_returns: list[float] = []
    dates: list[pd.Timestamp] = []

    for t in rets.index[1:]:
        c = carry_aligned.loc[t].dropna()
        if len(c) < n_long + n_short:
            strategy_returns.append(np.nan)
            dates.append(t)
            continue

        c_sorted = c.sort_values()
        shorts = c_sorted.index[:n_short]
        longs  = c_sorted.index[-n_long:]

        r = rets.loc[t]
        long_ret  = r[longs].mean()
        short_ret = r[shorts].mean()
        strat_ret = long_ret - short_ret
        strategy_returns.append(strat_ret)
        dates.append(t)

    return pd.Series(strategy_returns, index=dates, name="carry")


# ── Momentum signal (TSMOM) ──────────────────────────────────────────────────

def compute_momentum_signals(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
    target_vol: float = _TARGET_VOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute TSMOM signal for each pair: sign(12m return) × vol-scalar.

    Returns
    -------
    signals : pd.DataFrame
        Monthly signal (position size) per currency.
    monthly_rets : pd.DataFrame
        Monthly returns per currency.
    """
    monthly_px = prices.resample("ME").last()
    monthly_rets = np.log(monthly_px / monthly_px.shift(1))

    # 12-month return, skipping most recent month
    mom_12m = monthly_px.pct_change(lookback_months + skip_months).shift(skip_months)

    # Realised vol: 60-day daily vol → monthly resampled
    daily_rets = np.log(prices / prices.shift(1))
    daily_vol  = daily_rets.rolling(60).std() * np.sqrt(252)
    monthly_vol = daily_vol.resample("ME").last()
    monthly_vol = monthly_vol.reindex(monthly_rets.index, method="ffill")

    # Signal: sign of momentum × (target_vol / realized_vol)
    sign    = np.sign(mom_12m)
    vol_scale = (target_vol / monthly_vol).clip(0, 2.0)  # cap at 2× leverage
    signals = sign * vol_scale

    return signals, monthly_rets


def run_momentum_backtest(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
    target_vol: float = _TARGET_VOL,
) -> pd.Series:
    """Run TSMOM backtest. Signal is lagged 1 month (no lookahead).

    Returns
    -------
    pd.Series
        Monthly strategy returns.
    """
    signals, monthly_rets = compute_momentum_signals(
        prices, lookback_months, skip_months, target_vol
    )

    # Lag signal by 1 month (use signal from end of prior month)
    signal_lagged = signals.shift(1)

    # Strategy return: sum of (signal × pair return) normalised by number of pairs
    active_pairs = signal_lagged.notna() & monthly_rets.notna()
    n_active = active_pairs.sum(axis=1).replace(0, np.nan)

    strat_rets = (signal_lagged * monthly_rets).sum(axis=1, min_count=1) / n_active
    strat_rets.name = "momentum"
    return strat_rets.dropna()


# ── PPP value signal ──────────────────────────────────────────────────────────

def compute_ppp_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute PPP deviation for each pair at each month-end.

    PPP fair value is static (OECD/IMF estimates embedded in G10_PAIRS).
    Deviation = (current_spot − ppp_fair_value) / ppp_fair_value.
    Negative = currency is cheap vs PPP (potential buy).
    Positive = currency is expensive vs PPP (potential sell).

    Returns
    -------
    pd.DataFrame
        Index = month-end, columns = ccy, values = PPP deviation (decimal).
    """
    monthly_px = prices.resample("ME").last()
    result: dict[str, pd.Series] = {}
    for ccy, info in G10_PAIRS.items():
        if ccy not in monthly_px.columns:
            continue
        ppp = info["ppp"]
        deviation = (monthly_px[ccy] - ppp) / ppp
        result[ccy] = deviation
    if not result:
        return pd.DataFrame()
    return pd.concat(result, axis=1)


def run_value_backtest(prices: pd.DataFrame) -> pd.Series:
    """Long currencies most undervalued vs PPP, short most overvalued.

    Returns
    -------
    pd.Series
        Monthly strategy returns.
    """
    monthly_px = prices.resample("ME").last()
    monthly_rets = np.log(monthly_px / monthly_px.shift(1))
    ppp_dev = compute_ppp_signals(prices)

    common = [c for c in monthly_rets.columns if c in ppp_dev.columns]
    ppp_aligned = ppp_dev[common].reindex(monthly_rets.index).shift(1)
    rets = monthly_rets[common]

    n_each = 3
    strategy_returns: list[float] = []
    dates: list[pd.Timestamp] = []

    for t in rets.index[1:]:
        dev = ppp_aligned.loc[t].dropna()
        r   = rets.loc[t].dropna()
        shared = dev.index.intersection(r.index)
        if len(shared) < n_each * 2:
            strategy_returns.append(np.nan)
            dates.append(t)
            continue
        dev = dev[shared].sort_values()
        longs  = dev.index[:n_each]   # most negative = most undervalued = buy
        shorts = dev.index[-n_each:]  # most positive = most overvalued = sell
        strat_ret = r[longs].mean() - r[shorts].mean()
        strategy_returns.append(strat_ret)
        dates.append(t)

    return pd.Series(strategy_returns, index=dates, name="value").dropna()


# ── Combined signal ───────────────────────────────────────────────────────────

def run_combined_backtest(
    carry_rets: pd.Series,
    momentum_rets: pd.Series,
    value_rets: pd.Series,
) -> pd.Series:
    """Equal-weight average of carry, momentum, and value monthly returns.

    Returns
    -------
    pd.Series
        Monthly strategy returns for the blended portfolio.
    """
    df = pd.concat([carry_rets, momentum_rets, value_rets], axis=1)
    combined = df.mean(axis=1, skipna=False)
    combined.name = "combined"
    return combined.dropna()


# ── Performance metrics ───────────────────────────────────────────────────────

def backtest_metrics(rets: pd.Series) -> dict[str, float]:
    """Compute standard performance metrics from monthly return series.

    Parameters
    ----------
    rets : pd.Series
        Monthly returns (decimal, e.g. 0.01 = 1%).

    Returns
    -------
    dict with keys: ann_return, ann_vol, sharpe, max_dd, hit_rate.
    """
    rets = rets.dropna()
    if len(rets) < 12:
        return {}
    ann_ret = (1 + rets).prod() ** (12 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd   = drawdown.min()
    hit_rate = (rets > 0).mean()

    return {
        "ann_return": ann_ret,
        "ann_vol":    ann_vol,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "hit_rate":   hit_rate,
    }


# ── Carry crash visualisation helper ─────────────────────────────────────────

def vix_series(start: str = "2000-01-01") -> pd.Series:
    """Fetch VIX from FRED. Returns monthly resampled series."""
    try:
        df = fetch_series(["VIXCLS"], start=start)
        if df.empty or "VIXCLS" not in df.columns:
            return pd.Series(dtype=float)
        return df["VIXCLS"].resample("ME").last().dropna()
    except Exception:
        return pd.Series(dtype=float)


