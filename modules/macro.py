"""Macro data fetching, regime detection, and stress index computation.

Data source: FRED (Federal Reserve Economic Data) via fredapi.
Regime framework: Ilmanen (2011) growth/inflation quadrant — Bridgewater.
HMM classifier: Hamilton (1989) 2-state Gaussian HMM.

No Streamlit imports — pure computation only.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

# Load .env from project root at import time so the key is always available
def _load_env_file() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

_load_env_file()

import numpy as np
import pandas as pd

# ── FRED series definitions ─────────────────────────────────────────────────

FRED_SERIES: dict[str, str] = {
    "INDPRO":        "Industrial Production",
    "PAYEMS":        "Nonfarm Payrolls",
    "RECPROUSM156N": "Recession Probability",
    "USREC":         "NBER Recession Indicator",
    "CPIAUCSL":      "CPI All Items",
    "CPILFESL":      "Core CPI (ex Food & Energy)",
    "T10YIE":        "10Y Breakeven Inflation",
    "UNRATE":        "Unemployment Rate",
    "ICSA":          "Initial Jobless Claims",
    "DFF":           "Fed Funds Rate",
    "DGS2":          "2Y Treasury Yield",
    "DGS10":         "10Y Treasury Yield",
    "T10Y2Y":        "10Y-2Y Yield Spread",
    "BAMLH0A0HYM2":  "HY OAS Spread",
    "BAMLC0A0CM":    "IG OAS Spread",
    "VIXCLS":        "CBOE VIX",
    "SP500":         "S&P 500",
}

SNAPSHOT_SERIES: list[str] = [
    "INDPRO",
    "CPIAUCSL",
    "UNRATE",
    "DFF",
    "T10Y2Y",
    "VIXCLS",
]

YIELD_CURVE_SERIES: dict[str, str] = {
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


# ── FRED data fetching ───────────────────────────────────────────────────────

def get_fred_api_key() -> Optional[str]:
    """Return FRED API key from environment, or None if not set."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key if key else None


def _fetch_single_series(
    sid: str,
    key: str,
    start: str,
    end: Optional[str],
) -> Optional[pd.Series]:
    """Fetch one FRED series via the REST API (no fredapi dependency)."""
    import urllib.request
    import json

    params = f"series_id={sid}&api_key={key}&file_type=json&observation_start={start}"
    if end:
        params += f"&observation_end={end}"
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        obs = data.get("observations", [])
        if not obs:
            return None
        index = pd.to_datetime([o["date"] for o in obs])
        values = pd.to_numeric(
            [o["value"] for o in obs], errors="coerce"
        )
        s = pd.Series(values, index=index, name=sid)
        return s
    except Exception:  # noqa: BLE001
        return None


def fetch_series(
    series_ids: list[str],
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch multiple FRED series via REST API and return as a wide DataFrame.

    Parameters
    ----------
    series_ids : list[str]
        FRED series identifiers.
    start : str
        Start date ISO format.
    end : str | None
        End date ISO format. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame, index = date, columns = series_ids.

    Raises
    ------
    ValueError
        If FRED_API_KEY is not set.
    """
    key = get_fred_api_key()
    if not key:
        raise ValueError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html "
            "and add it to your .env file."
        )

    frames: dict[str, pd.Series] = {}
    for sid in series_ids:
        s = _fetch_single_series(sid, key, start, end)
        if s is not None:
            frames[sid] = s

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_yield_curve(start: str = "2000-01-01") -> pd.DataFrame:
    """Fetch Treasury yield curve maturities from FRED.

    Returns
    -------
    pd.DataFrame
        Columns named by maturity label (e.g. '1M', '2Y', '10Y').
    """
    series_ids = list(YIELD_CURVE_SERIES.keys())
    raw = fetch_series(series_ids, start=start)
    if raw.empty:
        return raw
    # Rename to maturity labels
    rename_map = {k: v for k, v in YIELD_CURVE_SERIES.items() if k in raw.columns}
    raw.rename(columns=rename_map, inplace=True)
    return raw


# ── Derived metrics ──────────────────────────────────────────────────────────

def compute_mom_change(series: pd.Series, months: int = 1) -> pd.Series:
    """Month-over-month percentage change, resampled to monthly if needed."""
    monthly = series.resample("ME").last().ffill()
    return monthly.pct_change(periods=months)


def compute_zscore_rolling(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    roll_mean = series.rolling(window, min_periods=60).mean()
    roll_std  = series.rolling(window, min_periods=60).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)


def compute_regime_coordinates(
    indpro: pd.Series,
    core_cpi: pd.Series,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """Compute (x, y) regime quadrant coordinates.

    X = IP 6-month momentum z-score (growth axis).
    Y = Core CPI 6-month momentum z-score (inflation axis).

    Both series are resampled monthly before differencing.

    Parameters
    ----------
    indpro : pd.Series
        Industrial production index (INDPRO).
    core_cpi : pd.Series
        Core CPI (CPILFESL).
    lookback_months : int
        Momentum window in months. Default 6.

    Returns
    -------
    pd.DataFrame
        Columns: ['x', 'y', 'date']
    """
    ip_mom   = compute_mom_change(indpro,   months=lookback_months)
    cpi_mom  = compute_mom_change(core_cpi, months=lookback_months)

    ip_z  = compute_zscore_rolling(ip_mom,  window=60)
    cpi_z = compute_zscore_rolling(cpi_mom, window=60)

    df = pd.DataFrame({"x": ip_z, "y": cpi_z}).dropna()
    df.index.name = "date"
    df = df.reset_index()
    return df


def regime_label(x: float, y: float) -> str:
    """Return the Ilmanen (2011) regime quadrant name."""
    if x >= 0 and y < 0:
        return "Goldilocks"
    elif x >= 0 and y >= 0:
        return "Reflation"
    elif x < 0 and y >= 0:
        return "Stagflation"
    else:
        return "Deflation"


# ── Macro Stress Index ────────────────────────────────────────────────────────

_STRESS_LEVELS = [
    (-np.inf, -0.5, "Low",      "#68d391"),   # green
    (-0.5,     0.5, "Neutral",  "#f6ad55"),   # amber
    ( 0.5,     1.5, "Elevated", "#ed8936"),   # orange
    ( 1.5,     2.5, "High",     "#f56565"),   # red
    ( 2.5,  np.inf, "Crisis",   "#9b2c2c"),   # deep red
]


def compute_stress_index(
    vix: pd.Series,
    hy_oas: pd.Series,
    t10y2y: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Composite macro stress index: mean of z(VIX) + z(HY OAS) + z(-T10Y2Y).

    Higher = more stress.  Curve inversion is negated so rising inversion
    (more negative T10Y2Y) increases stress.

    Parameters
    ----------
    vix : pd.Series     VIXCLS daily series.
    hy_oas : pd.Series  BAMLH0A0HYM2 daily series.
    t10y2y : pd.Series  T10Y2Y daily series.
    window : int        Rolling z-score window. Default 252.

    Returns
    -------
    pd.Series   Composite stress index.
    """
    z_vix   = compute_zscore_rolling(vix,      window)
    z_hy    = compute_zscore_rolling(hy_oas,   window)
    z_curve = compute_zscore_rolling(-t10y2y,  window)   # negate: inversion = stress
    return ((z_vix + z_hy + z_curve) / 3).dropna()


def stress_level(score: float) -> tuple[str, str]:
    """Return (label, color) for a composite stress score."""
    for lo, hi, label, color in _STRESS_LEVELS:
        if lo <= score < hi:
            return label, color
    return "Crisis", "#9b2c2c"


# ── HMM Regime Classifier ────────────────────────────────────────────────────

def fit_hmm_regime(
    indpro: pd.Series,
    core_cpi: pd.Series,
    vix: pd.Series,
    hy_oas: pd.Series,
    n_states: int = 2,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Fit 2-state Gaussian HMM on macro features and return regime probabilities.

    Features (monthly):
        - IP 1-month change
        - Core CPI 1-month change
        - VIX (monthly mean)
        - HY OAS (monthly mean)

    Parameters
    ----------
    indpro : pd.Series     INDPRO daily series.
    core_cpi : pd.Series   CPILFESL daily series.
    vix : pd.Series        VIXCLS daily series.
    hy_oas : pd.Series     BAMLH0A0HYM2 daily series.
    n_states : int         Number of HMM states. Default 2.

    Returns
    -------
    states : pd.Series          Most-likely state index per observation.
    probs : pd.DataFrame        State probability columns ['state_0', 'state_1'].
    risk_off_state : int        Which state index is "risk-off" (higher VIX mean).
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install hmmlearn: pip install hmmlearn") from exc

    # Resample to monthly
    ip_chg   = indpro.resample("ME").last().ffill().pct_change().dropna()
    cpi_chg  = core_cpi.resample("ME").last().ffill().pct_change().dropna()
    vix_m    = vix.resample("ME").mean()
    hy_m     = hy_oas.resample("ME").mean()

    common = ip_chg.index.intersection(cpi_chg.index).intersection(vix_m.index).intersection(hy_m.index)
    common = common[common >= pd.Timestamp("2005-01-01")]  # min 20yr history

    X = np.column_stack([
        ip_chg.reindex(common).fillna(0),
        cpi_chg.reindex(common).fillna(0),
        vix_m.reindex(common).fillna(vix_m.median()),
        hy_m.reindex(common).fillna(hy_m.median()),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)

    states_arr = model.predict(X)
    probs_arr  = model.predict_proba(X)

    states = pd.Series(states_arr, index=common, name="state")
    probs  = pd.DataFrame(
        probs_arr,
        index=common,
        columns=[f"state_{i}" for i in range(n_states)],
    )

    # Identify risk-off state: higher mean VIX
    vix_means = [
        X[states_arr == i, 2].mean() if (states_arr == i).sum() > 0 else 0.0
        for i in range(n_states)
    ]
    risk_off_state = int(np.argmax(vix_means))

    return states, probs, risk_off_state


# ── Snapshot computations ────────────────────────────────────────────────────

def compute_snapshot_metrics(df: pd.DataFrame) -> list[dict]:
    """Compute display metrics for the Economic Snapshot tiles.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame with FRED series columns.

    Returns
    -------
    list[dict]
        One dict per series with keys:
        name, series_id, latest_value, mom_change, zscore_1y, trend, unit
    """
    _META = {
        "INDPRO":   {"name": "Industrial Production",  "unit": "Index",   "invert_good": False},
        "CPIAUCSL": {"name": "CPI All Items",           "unit": "YoY %",   "invert_good": True},
        "UNRATE":   {"name": "Unemployment Rate",       "unit": "%",       "invert_good": True},
        "DFF":      {"name": "Fed Funds Rate",          "unit": "%",       "invert_good": False},
        "T10Y2Y":   {"name": "10Y–2Y Yield Spread",     "unit": "pp",      "invert_good": False},
        "VIXCLS":   {"name": "VIX",                     "unit": "Level",   "invert_good": True},
    }

    results = []
    for sid, meta in _META.items():
        if sid not in df.columns:
            continue
        s = df[sid].dropna()
        if len(s) < 13:
            continue

        latest = float(s.iloc[-1])

        # MoM change
        monthly = s.resample("ME").last().ffill()
        mom = float(monthly.pct_change(1).iloc[-1]) if len(monthly) >= 2 else np.nan

        # 1Y z-score: where does current value sit vs trailing 252 days
        tail_252 = s.tail(252)
        zscore = (
            (latest - tail_252.mean()) / tail_252.std()
            if tail_252.std() > 0 else 0.0
        )

        # Trend arrow: compare last 3 months direction
        if len(monthly) >= 4:
            trend_val = monthly.iloc[-1] - monthly.iloc[-4]
            trend = "↑" if trend_val > 0 else "↓" if trend_val < 0 else "→"
        else:
            trend = "→"

        # CPI: display as YoY %
        if sid == "CPIAUCSL":
            yoy = float(monthly.pct_change(12).iloc[-1]) if len(monthly) >= 13 else np.nan
            results.append({
                "name": meta["name"],
                "series_id": sid,
                "latest_value": yoy,
                "mom_change": mom,
                "zscore_1y": zscore,
                "trend": trend,
                "unit": meta["unit"],
                "invert_good": meta["invert_good"],
            })
        else:
            results.append({
                "name": meta["name"],
                "series_id": sid,
                "latest_value": latest,
                "mom_change": mom,
                "zscore_1y": zscore,
                "trend": trend,
                "unit": meta["unit"],
                "invert_good": meta["invert_good"],
            })

    return results


# ── Regime-conditional recommendation ───────────────────────────────────────

_REGIME_RECOMMENDATIONS: dict[str, dict] = {
    "Goldilocks": {
        "label": "Goldilocks — Growth ↑, Inflation ↓",
        "summary": (
            "Historically the best environment for risk assets. "
            "Favour equities (growth and cyclicals), real assets, and credit. "
            "Reduce duration in nominal bonds."
        ),
        "suggested": {"Equities": "+Overweight", "Credit": "+Overweight", "Duration": "Underweight", "Commodities": "Neutral"},
        "color": "#68d391",
    },
    "Reflation": {
        "label": "Reflation — Growth ↑, Inflation ↑",
        "summary": (
            "Strong nominal growth but rising inflation erodes real returns. "
            "Favour real assets, commodities, and inflation-linked bonds (TIPS). "
            "Trim long-duration nominal exposure."
        ),
        "suggested": {"Equities": "Neutral", "Commodities": "+Overweight", "TIPS": "+Overweight", "Duration": "Underweight"},
        "color": "#f6ad55",
    },
    "Stagflation": {
        "label": "Stagflation — Growth ↓, Inflation ↑",
        "summary": (
            "Historically the worst environment for 60/40. "
            "Reduce equities and duration. Increase commodities, gold, and real assets. "
            "Favour value over growth."
        ),
        "suggested": {"Equities": "Underweight", "Duration": "Underweight", "Gold": "+Overweight", "Commodities": "+Overweight"},
        "color": "#f56565",
    },
    "Deflation": {
        "label": "Deflation / Recession — Growth ↓, Inflation ↓",
        "summary": (
            "Risk-off. Favour long-duration Treasuries, quality bonds, and defensive equities. "
            "Reduce credit, cyclicals, and commodities. "
            "Gold can serve as a safe haven."
        ),
        "suggested": {"Duration": "+Overweight", "Equities": "Underweight", "Credit": "Underweight", "Gold": "+Overweight"},
        "color": "#81a4e8",
    },
}


def get_regime_recommendation(current_regime: str) -> dict:
    """Return allocation guidance for the current macro regime.

    Source: Ilmanen, A. (2011). Expected Returns. Wiley.
            Bridgewater All Weather framework.

    Parameters
    ----------
    current_regime : str
        One of: 'Goldilocks', 'Reflation', 'Stagflation', 'Deflation'.

    Returns
    -------
    dict with keys: label, summary, suggested, color
    """
    return _REGIME_RECOMMENDATIONS.get(
        current_regime,
        _REGIME_RECOMMENDATIONS["Goldilocks"],
    )
