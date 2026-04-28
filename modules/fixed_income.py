"""Fixed Income module — yield curve, duration/convexity, credit spreads.

Tab 1: Yield Curve          — interactive curve with date comparison, roll-down
Tab 2: Duration & Convexity — Macaulay, modified duration, convexity
Tab 3: Credit Spreads       — IG/HY OAS z-scores, Cochrane-Piazzesi factor

No Streamlit imports — pure computation only.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from modules.macro import fetch_series


# ── Shim to allow module-level @st_cache_shim decoration ────────────────────
# The real cache is applied at the page level via @st.cache_data; here we
# apply a no-op decorator so the module is importable without Streamlit.

def st_cache_shim(fn):  # type: ignore[no-untyped-def]
    """No-op decorator; replaced by @st.cache_data at the call site."""
    return fn

# ── Treasury yield curve maturities ─────────────────────────────────────────
TREASURY_TICKERS: dict[str, str] = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1":   "1Y",
    "DGS2":   "2Y",
    "DGS5":   "5Y",
    "DGS10":  "10Y",
    "DGS20":  "20Y",
    "DGS30":  "30Y",
}

# Tenor in years for each ticker (for duration/convexity calculations)
TENOR_YEARS: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 3 / 12,
    "DGS6MO": 6 / 12,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS5":   5.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}

CREDIT_SERIES: dict[str, str] = {
    "BAMLH0A0HYM2": "HY OAS Spread (bp)",
    "BAMLC0A0CM":   "IG OAS Spread (bp)",
    "T10Y2Y":       "10Y-2Y Spread",
    "USREC":        "NBER Recession",
}


# ── Yield curve fetching ─────────────────────────────────────────────────────

@st_cache_shim
def fetch_yield_curve(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch all Treasury constant-maturity yields from FRED.

    Returns
    -------
    pd.DataFrame
        Daily yields in percent, columns = TREASURY_TICKERS keys,
        index = date (DatetimeIndex).
    """
    return fetch_series(list(TREASURY_TICKERS.keys()), start=start)


def latest_curve(df: pd.DataFrame) -> pd.Series:
    """Return the most recent complete yield curve row."""
    return df.dropna(how="all").ffill().iloc[-1]


def curve_one_year_ago(df: pd.DataFrame) -> pd.Series:
    """Return the yield curve as of approximately one year prior."""
    cutoff = df.index[-1] - pd.DateOffset(years=1)
    subset = df.loc[df.index <= cutoff].dropna(how="all").ffill()
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[-1]


def spread_10y2y(df: pd.DataFrame) -> pd.Series:
    """Compute 10Y-2Y spread from the yield curve DataFrame."""
    return df["DGS10"] - df["DGS2"]


def roll_down(df: pd.DataFrame, horizon_months: int = 6) -> dict[str, float]:
    """Estimate roll-down return for each maturity point.

    Roll-down = (yield_at_T - yield_at_T-horizon) × duration_approx
    Assumes the curve shape is unchanged over the horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Full yield curve history.
    horizon_months : int
        Holding period in months.

    Returns
    -------
    dict[str, float]
        Approximate roll-down return (%) per maturity label.
    """
    latest = latest_curve(df)
    tickers = list(TREASURY_TICKERS.keys())
    labels  = list(TREASURY_TICKERS.values())
    tenors  = [TENOR_YEARS[t] for t in tickers]
    horizon = horizon_months / 12

    result: dict[str, float] = {}
    for i, (ticker, label, tenor) in enumerate(zip(tickers, labels, tenors)):
        current_yield = latest.get(ticker, np.nan)
        if np.isnan(current_yield):
            result[label] = np.nan
            continue
        # Find next shorter tenor point on the curve
        shorter_tenor = tenor - horizon
        if shorter_tenor <= 0:
            result[label] = np.nan
            continue
        # Interpolate yield at the shorter tenor
        shorter_yield = _interp_yield(latest, tickers, tenors, shorter_tenor)
        if np.isnan(shorter_yield):
            result[label] = np.nan
            continue
        # Approximate modified duration ≈ tenor (rough)
        mod_dur = tenor / (1 + current_yield / 200)
        roll_return = (current_yield - shorter_yield) / 100 * mod_dur
        result[label] = round(roll_return * 100, 3)  # in %

    return result


def _interp_yield(
    curve: pd.Series,
    tickers: list[str],
    tenors: list[float],
    target_tenor: float,
) -> float:
    """Linear interpolation to find yield at an arbitrary tenor."""
    vals  = [curve.get(t, np.nan) for t in tickers]
    pairs = [(t, v) for t, v in zip(tenors, vals) if not np.isnan(v)]
    if len(pairs) < 2:
        return np.nan
    pairs.sort()
    ts, vs = zip(*pairs)
    return float(np.interp(target_tenor, ts, vs))


# ── Duration & convexity ─────────────────────────────────────────────────────

def macaulay_duration(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity: float,
    freq: int = 2,
) -> float:
    """Compute Macaulay duration for a coupon-bearing bond.

    Parameters
    ----------
    face : float
        Face (par) value.
    coupon_rate : float
        Annual coupon rate as a decimal (e.g. 0.05 for 5%).
    ytm : float
        Annual yield-to-maturity as a decimal.
    maturity : float
        Years to maturity.
    freq : int
        Coupon payments per year (2 = semi-annual).

    Returns
    -------
    float
        Macaulay duration in years.
    """
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    y      = ytm / freq

    if n == 0:
        return 0.0

    pv_weighted = 0.0
    pv_total    = 0.0
    for t in range(1, n + 1):
        cf      = coupon if t < n else coupon + face
        pv      = cf / (1 + y) ** t
        pv_weighted += t * pv
        pv_total    += pv

    if pv_total == 0:
        return 0.0
    return (pv_weighted / pv_total) / freq


def modified_duration(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity: float,
    freq: int = 2,
) -> float:
    """Modified duration = Macaulay duration / (1 + ytm/freq)."""
    mac = macaulay_duration(face, coupon_rate, ytm, maturity, freq)
    return mac / (1 + ytm / freq)


def convexity(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity: float,
    freq: int = 2,
) -> float:
    """Convexity of a coupon bond.

    Convexity measures the curvature of the price-yield relationship —
    the second derivative of price with respect to yield.

    Returns
    -------
    float
        Convexity (dimensionless, scaled by 1/freq²).
    """
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    y      = ytm / freq
    price  = bond_price(face, coupon_rate, ytm, maturity, freq)

    if price == 0 or n == 0:
        return 0.0

    cx = 0.0
    for t in range(1, n + 1):
        cf   = coupon if t < n else coupon + face
        pv   = cf / (1 + y) ** t
        cx  += t * (t + 1) * pv

    return cx / (price * (1 + y) ** 2 * freq ** 2)


def bond_price(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity: float,
    freq: int = 2,
) -> float:
    """Price a coupon bond by discounting all cash flows."""
    n      = int(maturity * freq)
    coupon = face * coupon_rate / freq
    y      = ytm / freq

    if n == 0:
        return face

    price = 0.0
    for t in range(1, n + 1):
        cf    = coupon if t < n else coupon + face
        price += cf / (1 + y) ** t
    return price


def price_change_table(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity: float,
    freq: int = 2,
) -> pd.DataFrame:
    """Compute bond price changes for ±25/50/100/200 bp yield shocks.

    Returns
    -------
    pd.DataFrame
        Columns: Yield Shock, New YTM (%), New Price, $ Change, % Change,
                 Duration Approx, Convexity Adj.
    """
    mod_dur = modified_duration(face, coupon_rate, ytm, maturity, freq)
    cx      = convexity(face, coupon_rate, ytm, maturity, freq)
    base_px = bond_price(face, coupon_rate, ytm, maturity, freq)

    shocks_bp = [-200, -100, -50, -25, 25, 50, 100, 200]
    rows = []
    for bp in shocks_bp:
        dy       = bp / 10_000
        new_ytm  = max(ytm + dy, 0.0001)
        new_px   = bond_price(face, coupon_rate, new_ytm, maturity, freq)
        pct_chg  = (new_px - base_px) / base_px * 100
        dur_approx = -mod_dur * dy * 100  # % approx
        cx_adj     = 0.5 * cx * dy ** 2 * 100
        rows.append({
            "Yield Shock (bp)": f"{'+' if bp > 0 else ''}{bp}",
            "New YTM (%)":      round(new_ytm * 100, 3),
            "New Price ($)":    round(new_px, 4),
            "$ Change":         round(new_px - base_px, 4),
            "% Change":         round(pct_chg, 3),
            "Duration Est. (%)": round(dur_approx, 3),
            "Convexity Adj. (%)": round(cx_adj, 4),
        })
    return pd.DataFrame(rows)


def price_yield_curve_data(
    face: float,
    coupon_rate: float,
    maturity: float,
    ytm_center: float,
    freq: int = 2,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (ytm, price) pairs for the price-yield curve chart."""
    ytm_lo = max(0.0001, ytm_center - 0.05)
    ytm_hi = ytm_center + 0.05
    ytms   = np.linspace(ytm_lo, ytm_hi, n_points)
    prices = np.array([
        bond_price(face, coupon_rate, y, maturity, freq) for y in ytms
    ])
    return ytms * 100, prices  # return YTM in %


# ── Credit spreads ───────────────────────────────────────────────────────────

@st_cache_shim
def fetch_credit_data(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch HY OAS, IG OAS, 10Y-2Y spread, and NBER recession from FRED."""
    return fetch_series(list(CREDIT_SERIES.keys()), start=start)


def zscore_series(s: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score: (x - μ) / σ over a trailing window."""
    mu    = s.rolling(window).mean()
    sigma = s.rolling(window).std()
    return (s - mu) / sigma


def credit_zscore_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table of current spread levels and their z-scores.

    Returns
    -------
    pd.DataFrame
        Rows = HY OAS, IG OAS. Columns = current, 1Y mean, z-score (1Y window).
    """
    rows = []
    for sid, label in [("BAMLH0A0HYM2", "HY OAS (bp)"), ("BAMLC0A0CM", "IG OAS (bp)")]:
        if sid not in df.columns:
            continue
        s         = df[sid].dropna()
        z_series  = zscore_series(s, window=252)
        current   = s.iloc[-1]
        mean_1y   = s.tail(252).mean()
        z_current = z_series.iloc[-1]
        signal    = (
            "Wide / Cheap" if z_current > 1.5 else
            "Tight / Expensive" if z_current < -1.5 else
            "Neutral"
        )
        rows.append({
            "Series":      label,
            "Current":     round(current, 1),
            "1Y Average":  round(mean_1y, 1),
            "Z-Score (1Y)": round(z_current, 2),
            "Signal":      signal,
        })
    return pd.DataFrame(rows)


def cochrane_piazzesi_factor(df: pd.DataFrame) -> pd.Series:
    """Approximate Cochrane-Piazzesi (2005) tent-shaped factor.

    The CP factor predicts excess bond returns using a linear combination
    of forward rates with tent-shaped coefficients. We approximate it from
    available FRED yield data.

    Coefficients follow Cochrane & Piazzesi (2005) Table 1 (annualised %):
      CP ≈ -2.14 × y1 + 0.81 × y2 + 1.48 × y5 - 0.99 × y10 + 0.54 × y(2-5 avg)

    This is a rough replication from constant-maturity yields; the original
    paper uses Fama-Bliss zero-coupon data.

    Returns
    -------
    pd.Series
        Approximate CP factor (higher = more attractive bond excess return).
    """
    needed = {"DGS1", "DGS2", "DGS5", "DGS10"}
    if not needed.issubset(df.columns):
        return pd.Series(dtype=float)

    y1  = df["DGS1"].ffill()
    y2  = df["DGS2"].ffill()
    y5  = df["DGS5"].ffill()
    y10 = df["DGS10"].ffill()
    y25 = (y2 + y5) / 2

    cp = -2.14 * y1 + 0.81 * y2 + 1.48 * y5 - 0.99 * y10 + 0.54 * y25
    cp.name = "CP Factor"
    return cp
