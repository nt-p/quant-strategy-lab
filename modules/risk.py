"""Risk analytics computations for backtested strategies.

All functions operate on daily return series (floats).  The convention
matches engine/backtest.py: index 0 of ``returns`` is always 0.0 (the
first backtest day has no prior return), so every function skips it.

No Streamlit or Plotly imports here — this module is pure computation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from engine.backtest import BacktestResult


# ── Value-at-Risk & Expected Shortfall ───────────────────────────────────────

def compute_var(returns: list[float], confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at the given confidence level.

    Returns the loss (positive number) not exceeded on ``confidence`` fraction
    of days.  E.g. VaR(0.95) = 0.02 means only 5% of days had losses > 2%.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.  Index 0 (= 0.0) is skipped.
    confidence : float
        Confidence level in (0, 1).  Default 0.95.

    Returns
    -------
    float
        VaR as a positive number (loss magnitude).  NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    if len(r) < 10:
        return float("nan")
    return float(-np.percentile(r, (1.0 - confidence) * 100))


def compute_cvar(returns: list[float], confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) at the given confidence level.

    The average loss on the worst ``(1 - confidence)`` fraction of days.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.  Index 0 (= 0.0) is skipped.
    confidence : float
        Confidence level in (0, 1).  Default 0.95.

    Returns
    -------
    float
        CVaR as a positive number (loss magnitude).  NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    if len(r) < 10:
        return float("nan")
    var_threshold = -compute_var(r.tolist(), confidence)  # negative (loss side)
    tail = r[r <= var_threshold]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


# ── Distribution statistics ───────────────────────────────────────────────────

def compute_tail_stats(returns: list[float]) -> dict[str, float]:
    """Descriptive statistics of the return distribution.

    Parameters
    ----------
    returns : list[float]
        Daily portfolio returns.  Index 0 (= 0.0) is skipped.

    Returns
    -------
    dict[str, float]
        Keys:
        - ``skewness``      : Fisherian skewness (0 = symmetric).
        - ``excess_kurtosis``: Excess kurtosis (0 = normal tails).
        - ``tail_ratio``    : |95th pct| / |5th pct|.  > 1 = fatter right tail.
        - ``best_day``      : Best single-day return (as a decimal).
        - ``worst_day``     : Worst single-day return (as a decimal, negative).
        - ``pct_positive``  : Fraction of days with a positive return.
    """
    r = np.array(returns[1:], dtype=float)
    if len(r) < 4:
        nan = float("nan")
        return dict(
            skewness=nan,
            excess_kurtosis=nan,
            tail_ratio=nan,
            best_day=nan,
            worst_day=nan,
            pct_positive=nan,
        )

    skewness = float(stats.skew(r, bias=False))
    excess_kurtosis = float(stats.kurtosis(r, fisher=True, bias=False))  # Fisher → excess

    p95 = float(np.percentile(r, 95))
    p05 = float(np.percentile(r, 5))
    tail_ratio = abs(p95) / abs(p05) if abs(p05) > 1e-10 else float("nan")

    return dict(
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        tail_ratio=tail_ratio,
        best_day=float(r.max()),
        worst_day=float(r.min()),
        pct_positive=float(np.mean(r > 0)),
    )


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_correlation_matrix(results: list[BacktestResult]) -> pd.DataFrame:
    """Pairwise Pearson correlation of daily returns across strategies.

    Returns aligned to the intersection of all result date sets so every
    series has the same length.

    Parameters
    ----------
    results : list[BacktestResult]
        Results to correlate (strategies + benchmark).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix indexed and columned by ``strategy_name``.
        Empty DataFrame if fewer than 2 results are provided.
    """
    if len(results) < 2:
        return pd.DataFrame()

    # Build a DataFrame: index = dates (intersection), columns = strategy names
    series: dict[str, pd.Series] = {}
    for r in results:
        s = pd.Series(r.returns, index=r.dates, name=r.strategy_name)
        series[r.strategy_name] = s

    df = pd.DataFrame(series).dropna()
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    return df.iloc[1:].corr(method="pearson")   # skip day-0 zeros


# ── Market beta & alpha ───────────────────────────────────────────────────────

def compute_beta_alpha(
    returns: list[float],
    benchmark_returns: list[float],
) -> tuple[float, float]:
    """OLS market beta and annualised Jensen's alpha vs the benchmark.

    Parameters
    ----------
    returns : list[float]
        Daily strategy returns.  Index 0 skipped.
    benchmark_returns : list[float]
        Daily benchmark returns, same length and alignment as ``returns``.

    Returns
    -------
    tuple[float, float]
        ``(beta, annualised_alpha)``.  Both NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    b = np.array(benchmark_returns[1:], dtype=float)

    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]

    if min_len < 30:
        return float("nan"), float("nan")

    result = stats.linregress(b, r)
    beta = float(result.slope)
    alpha_daily = float(result.intercept)
    alpha_annualised = alpha_daily * 252
    return beta, alpha_annualised


def compute_r_squared(
    returns: list[float],
    benchmark_returns: list[float],
) -> float:
    """R² of strategy returns regressed against benchmark returns.

    Parameters
    ----------
    returns : list[float]
        Daily strategy returns.  Index 0 skipped.
    benchmark_returns : list[float]
        Daily benchmark returns, same alignment.

    Returns
    -------
    float
        Coefficient of determination in [0, 1].  NaN if insufficient data.
    """
    r = np.array(returns[1:], dtype=float)
    b = np.array(benchmark_returns[1:], dtype=float)

    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]

    if min_len < 30:
        return float("nan")

    result = stats.linregress(b, r)
    return float(result.rvalue ** 2)


# ── Drawdown detail ───────────────────────────────────────────────────────────

# ── Portfolio-level risk (active portfolio — pd.Series interface) ─────────────

STRESS_SCENARIOS: list[dict] = [
    {
        "name": "2008 Global Financial Crisis",
        "short": "Sep–Nov 2008",
        "start": "2008-09-01",
        "end": "2008-11-28",
    },
    {
        "name": "2020 COVID Crash",
        "short": "Feb–Mar 2020",
        "start": "2020-02-20",
        "end": "2020-03-23",
    },
    {
        "name": "2022 Rate Shock",
        "short": "Jan–Dec 2022",
        "start": "2022-01-03",
        "end": "2022-12-30",
    },
    {
        "name": "2000 Dot-com Bust",
        "short": "Mar 2000–Oct 2002",
        "start": "2000-03-24",
        "end": "2002-10-09",
    },
]


def compute_var_from_series(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk from a pd.Series of daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        VaR as a positive loss magnitude.  NaN if insufficient data.
    """
    r = returns.dropna()
    if len(r) < 10:
        return float("nan")
    return float(-np.percentile(r.values, (1.0 - confidence) * 100))


def compute_cvar_from_series(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) from a pd.Series of daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        CVaR as a positive loss magnitude.  NaN if insufficient data.
    """
    r = returns.dropna()
    if len(r) < 10:
        return float("nan")
    threshold = -compute_var_from_series(r, confidence)
    tail = r[r <= threshold]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def compute_parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (normal) VaR: μ + z(α) × σ.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        Parametric VaR as a positive loss magnitude.  NaN if insufficient data.
    """
    r = returns.dropna()
    if len(r) < 10:
        return float("nan")
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    z = float(stats.norm.ppf(1.0 - confidence))
    return float(-(mu + z * sigma))


def compute_rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
) -> pd.Series:
    """Rolling historical VaR over a sliding window.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    window : int
        Rolling window length in trading days.  Default 252 (1 year).
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    pd.Series
        Rolling VaR (positive loss magnitude), same index as ``returns``.
        NaN until ``window // 4`` observations are available.
    """

    def _var(x: np.ndarray) -> float:
        if len(x) < 20:
            return float("nan")
        return float(-np.percentile(x, (1.0 - confidence) * 100))

    return returns.rolling(window=window, min_periods=window // 4).apply(_var, raw=True)


def compute_drawdown_from_series(returns: pd.Series) -> pd.Series:
    """Peak-to-trough drawdown series from a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    pd.Series
        Drawdown at each date as a fraction (≤ 0).  Same index as ``returns``.
    """
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    hwm = equity.cummax()
    dd = equity / hwm - 1.0
    return dd.where(dd < -1e-9, 0.0)


def compute_worst_drawdown_periods(
    returns: pd.Series,
    n: int = 5,
) -> pd.DataFrame:
    """Identify the N worst drawdown periods from a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    n : int
        Number of worst periods to return.

    Returns
    -------
    pd.DataFrame
        Columns: Peak Date, Trough Date, Depth, Recovery Date, Duration (days).
        Sorted by Depth (deepest first).  Recovery Date is NaT if ongoing.
    """
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    uw = dd.values < -1e-6
    dates = dd.index
    eq_vals = equity.values
    dd_vals = dd.values

    # Vectorised boundary detection
    starts = list(np.where(~uw[:-1] & uw[1:])[0] + 1)
    ends_raw = list(np.where(uw[:-1] & ~uw[1:])[0] + 1)

    # Series starting in drawdown
    if uw[0]:
        starts = [0] + starts

    # Series ending in drawdown (sentinel = one past last index)
    if uw[-1]:
        ends_raw.append(len(uw))

    periods: list[dict] = []
    for start_idx in starts:
        candidates = [e for e in ends_raw if e >= start_idx]
        if not candidates:
            continue
        end_idx = candidates[0]

        # Trough: deepest point in [start, end)
        seg = dd_vals[start_idx:end_idx]
        trough_offset = int(np.argmin(seg))
        trough_idx = start_idx + trough_offset
        trough_depth = float(dd_vals[trough_idx])

        # Peak: last day before start where equity equalled its running max
        if start_idx == 0:
            peak_idx = 0
        else:
            prev_max = float(np.max(eq_vals[:start_idx]))
            peak_idx = start_idx - 1
            for k in range(start_idx - 1, -1, -1):
                if abs(eq_vals[k] - prev_max) / (prev_max + 1e-12) < 1e-6:
                    peak_idx = k
                    break

        peak_date = dates[peak_idx]
        trough_date = dates[trough_idx]
        recovery_date = dates[end_idx] if end_idx < len(dates) else pd.NaT
        duration = (trough_date - peak_date).days

        periods.append(
            {
                "Peak Date": peak_date,
                "Trough Date": trough_date,
                "Depth": trough_depth,
                "Recovery Date": recovery_date,
                "Duration (days)": duration,
            }
        )

    if not periods:
        return pd.DataFrame(
            columns=["Peak Date", "Trough Date", "Depth", "Recovery Date", "Duration (days)"]
        )

    df = pd.DataFrame(periods)
    return df.nsmallest(n, "Depth").reset_index(drop=True)


# ── Asset-class sensitivity lookup ────────────────────────────────────────────

#  (asset_class, sensitivity_param)
#  equity      → equity beta
#  fi_rate     → modified duration (years)
#  fi_credit   → spread duration (years); also has rate sensitivity
#  hy_credit   → HY spread duration; equity-correlated
#  commodity   → equity beta proxy
#  reit        → equity beta; also rate-sensitive
_ASSET_CLASS: dict[str, tuple[str, float]] = {
    # Broad equity
    "SPY": ("equity", 1.00), "VOO": ("equity", 1.00), "IVV": ("equity", 1.00),
    "VTI": ("equity", 1.00), "QQQ": ("equity", 1.20), "IWM": ("equity", 1.10),
    "EFA": ("equity", 0.90), "EEM": ("equity", 1.10), "VEA": ("equity", 0.90),
    "VWO": ("equity", 1.05), "ACWI": ("equity", 1.00), "VT": ("equity", 1.00),
    # Fixed income — rate risk (sensitivity = mod. duration)
    "AGG": ("fi_rate", 6.0), "BND": ("fi_rate", 6.0), "TLT": ("fi_rate", 17.0),
    "IEF": ("fi_rate", 8.0), "SHY": ("fi_rate", 2.0), "BIL": ("fi_rate", 0.2),
    "GOVT": ("fi_rate", 7.0), "VGLT": ("fi_rate", 17.0), "VGIT": ("fi_rate", 5.5),
    "SGOV": ("fi_rate", 0.1), "SCHO": ("fi_rate", 2.0),
    # IG credit — rate + spread
    "LQD": ("fi_credit", 8.0), "VCIT": ("fi_credit", 6.0), "VCLT": ("fi_credit", 14.0),
    # HY credit — equity-correlated + spread
    "HYG": ("hy_credit", 4.0), "JNK": ("hy_credit", 4.0), "USHY": ("hy_credit", 3.5),
    # Commodities
    "GLD": ("commodity", 0.00), "IAU": ("commodity", 0.00), "SLV": ("commodity", 0.00),
    "PDBC": ("commodity", 0.30), "DBC": ("commodity", 0.30), "USO": ("commodity", 0.50),
    # REITs
    "VNQ": ("reit", 1.00), "IYR": ("reit", 1.00), "XLRE": ("reit", 1.00),
    # AU market
    "STW.AX": ("equity", 1.00), "VAS.AX": ("equity", 1.00), "IOZ.AX": ("equity", 1.00),
}


def compute_sensitivity_impact(
    holdings: dict[str, float],
    equity_shock: float,
    yield_shock_bps: float,
    spread_shock_bps: float,
) -> dict[str, object]:
    """Linear sensitivity of portfolio to isolated macro shocks.

    Parameters
    ----------
    holdings : dict[str, float]
        Ticker → weight on the 0–100 scale.
    equity_shock : float
        Equity market return shock, e.g. ``-0.20`` for −20%.
    yield_shock_bps : float
        Yield level change in basis points (positive = rates rising).
    spread_shock_bps : float
        Credit spread widening in basis points.

    Returns
    -------
    dict with keys:
        ``total``        : estimated portfolio impact as a decimal.
        ``breakdown``    : dict[str, float] — per-ticker impact.
        ``classified_as``: dict[str, str]   — per-ticker asset class label.
    """
    yield_shock = yield_shock_bps / 10_000.0
    spread_shock = spread_shock_bps / 10_000.0

    total = 0.0
    breakdown: dict[str, float] = {}
    classified_as: dict[str, str] = {}

    for ticker, weight_pct in holdings.items():
        w = weight_pct / 100.0
        asset_class, sensitivity = _ASSET_CLASS.get(ticker, ("equity", 1.0))

        if asset_class == "equity":
            shock = equity_shock * sensitivity
            label = "Equity"
        elif asset_class == "fi_rate":
            shock = -sensitivity * yield_shock
            label = "Fixed Income (rate)"
        elif asset_class == "fi_credit":
            # Rate + IG spread contribution (roughly half spread duration)
            shock = -sensitivity * yield_shock - 0.5 * sensitivity * spread_shock
            label = "IG Credit"
        elif asset_class == "hy_credit":
            # HY: partial equity beta + spread sensitive
            shock = 0.5 * equity_shock - sensitivity * spread_shock
            label = "HY Credit"
        elif asset_class == "commodity":
            shock = equity_shock * sensitivity
            label = "Commodity"
        elif asset_class == "reit":
            # REITs: equity-like but also rate sensitive (~5yr duration proxy)
            shock = 0.80 * equity_shock - 5.0 * yield_shock
            label = "REIT"
        else:
            shock = equity_shock
            label = "Equity (default)"

        impact = w * shock
        total += impact
        breakdown[ticker] = impact
        classified_as[ticker] = label

    return {"total": total, "breakdown": breakdown, "classified_as": classified_as}


def apply_stress_scenarios(
    holdings: dict[str, float],
    benchmark_ticker: str,
) -> pd.DataFrame:
    """Apply historical stress scenarios to current portfolio weights.

    Fetches price data for each scenario period from yfinance and computes the
    portfolio return using current holdings weights.

    Parameters
    ----------
    holdings : dict[str, float]
        Ticker → portfolio weight on the 0–100 scale.
    benchmark_ticker : str
        Benchmark ticker (e.g. ``"SPY"``) for comparison.

    Returns
    -------
    pd.DataFrame
        Columns: Scenario, Period, Portfolio, Benchmark, Relative.
        NaN for scenarios where price data is unavailable.
    """
    import yfinance as yf  # lazy import

    tickers_set = set(holdings.keys()) | {benchmark_ticker}
    tickers_list = sorted(tickers_set)

    rows = []
    for sc in STRESS_SCENARIOS:
        try:
            raw = yf.download(
                tickers_list,
                start=sc["start"],
                end=sc["end"],
                progress=False,
                auto_adjust=True,
            )

            # Normalise column structure across yfinance versions
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                # Single-ticker string download returns regular DataFrame
                close = raw[["Close"]].rename(columns={"Close": tickers_list[0]})

            if isinstance(close, pd.Series):
                close = close.to_frame(name=tickers_list[0])

            close = close.dropna(how="all")
            if len(close) < 2:
                raise ValueError("insufficient price data for scenario period")

            first = close.iloc[0]
            last = close.iloc[-1]
            period_ret = (last - first) / first.replace(0, float("nan"))

            # Portfolio return (rescale if some tickers have no data)
            port_ret = 0.0
            covered_w = 0.0
            for ticker, wpct in holdings.items():
                w = wpct / 100.0
                if ticker in period_ret.index:
                    v = period_ret[ticker]
                    if not (isinstance(v, float) and math.isnan(v)):
                        port_ret += w * float(v)
                        covered_w += w

            if covered_w > 1e-6:
                port_ret = port_ret / covered_w  # normalise to covered portion

            bench_ret = float("nan")
            if benchmark_ticker in period_ret.index:
                v = period_ret[benchmark_ticker]
                if not (isinstance(v, float) and math.isnan(v)):
                    bench_ret = float(v)

            rows.append(
                {
                    "Scenario": sc["name"],
                    "Period": sc["short"],
                    "Portfolio": port_ret,
                    "Benchmark": bench_ret,
                    "Relative": (
                        port_ret - bench_ret if not math.isnan(bench_ret) else float("nan")
                    ),
                }
            )
        except Exception:  # noqa: BLE001
            rows.append(
                {
                    "Scenario": sc["name"],
                    "Period": sc["short"],
                    "Portfolio": float("nan"),
                    "Benchmark": float("nan"),
                    "Relative": float("nan"),
                }
            )

    return pd.DataFrame(rows)


# ── Strategy-level helpers (original interface, kept below) ───────────────────

def compute_drawdown_stats(drawdown: list[float]) -> dict[str, float]:
    """Extended drawdown statistics from a pre-computed drawdown series.

    Parameters
    ----------
    drawdown : list[float]
        Daily peak-to-trough drawdown series (values ≤ 0), as produced by
        ``engine.metrics.compute_drawdown``.

    Returns
    -------
    dict[str, float]
        Keys:
        - ``max_drawdown``       : Worst single drawdown (≤ 0).
        - ``avg_drawdown``       : Mean drawdown on days spent underwater (≤ 0).
        - ``pct_time_underwater``: Fraction of days with drawdown < 0.
        - ``n_drawdown_periods`` : Number of distinct drawdown episodes.
    """
    dd = np.array(drawdown, dtype=float)
    if len(dd) == 0:
        nan = float("nan")
        return dict(
            max_drawdown=nan,
            avg_drawdown=nan,
            pct_time_underwater=nan,
            n_drawdown_periods=nan,
        )

    underwater = dd < -1e-9
    underwater_vals = dd[underwater]

    max_dd = float(dd.min())
    avg_dd = float(underwater_vals.mean()) if len(underwater_vals) > 0 else 0.0
    pct_underwater = float(underwater.mean())

    # Count distinct episodes (transitions from above to below waterline)
    n_periods = 0
    in_period = False
    for val in dd:
        if val < -1e-9:
            if not in_period:
                n_periods += 1
                in_period = True
        else:
            in_period = False

    return dict(
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        pct_time_underwater=pct_underwater,
        n_drawdown_periods=float(n_periods),
    )
