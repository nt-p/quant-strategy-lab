"""Unit tests for engine/backtest.py."""

import math

import numpy as np
import pandas as pd
import pytest

from engine.backtest import BacktestResult, _get_rebalance_dates, run_backtest
from strategies.base import RebalanceFrequency, StrategyBase, StrategyCategory


# ── Minimal test strategies ───────────────────────────────────────────────────

class _SingleAssetBuyHold(StrategyBase):
    """Puts 100% into the first ticker in the universe.  NEVER rebalances."""

    strategy_id = "test_buy_hold"
    name = "Test Buy & Hold"
    category = StrategyCategory.PASSIVE
    description = "Test only"
    long_description = ""
    rebalance_frequency = RebalanceFrequency.NEVER

    def get_weights(self, prices, current_date, universe):
        return {universe[0]: 1.0}


class _EqualWeightMonthly(StrategyBase):
    strategy_id = "test_equal_monthly"
    name = "Test Equal Monthly"
    category = StrategyCategory.QUANTITATIVE
    description = "Test only"
    long_description = ""
    rebalance_frequency = RebalanceFrequency.MONTHLY

    def get_weights(self, prices, current_date, universe):
        return self._equal_weight(universe)


class _NeverTrades(StrategyBase):
    """Returns empty weights — all cash."""

    strategy_id = "test_cash"
    name = "Test Cash"
    category = StrategyCategory.PASSIVE
    description = "Test only"
    long_description = ""
    rebalance_frequency = RebalanceFrequency.MONTHLY

    def get_weights(self, prices, current_date, universe):
        return {}


class _HalfInvested(StrategyBase):
    """Puts 50% into the first ticker, leaving 50% as CASH."""

    strategy_id = "test_half_invested"
    name = "Test Half Invested"
    category = StrategyCategory.PASSIVE
    description = "Test only"
    long_description = ""
    rebalance_frequency = RebalanceFrequency.NEVER

    def get_weights(self, prices, current_date, universe):
        return {universe[0]: 0.5}


class _OverInvested(StrategyBase):
    """Returns weights summing to 1.5 — engine must normalise down."""

    strategy_id = "test_over_invested"
    name = "Test Over Invested"
    category = StrategyCategory.PASSIVE
    description = "Test only"
    long_description = ""
    rebalance_frequency = RebalanceFrequency.NEVER

    def get_weights(self, prices, current_date, universe):
        return {t: 0.75 for t in universe[:2]}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def single_ticker_prices():
    """AAAA with a deterministic linear price series."""
    dates = pd.bdate_range("2021-01-04", periods=252, tz="UTC")
    close = 100.0 + np.arange(len(dates)) * 0.1   # rises steadily
    rows = []
    for d, c in zip(dates, close):
        rows.append({"date": d, "ticker": "AAAA", "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000})
    return pd.DataFrame(rows)


@pytest.fixture
def two_ticker_prices():
    """AAAA and BBBB over 252 days."""
    dates = pd.bdate_range("2021-01-04", periods=252, tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for ticker, seed_ret in [("AAAA", 0.001), ("BBBB", -0.0005)]:
        close = 100.0 * np.cumprod(1 + rng.normal(seed_ret, 0.01, len(dates)))
        for d, c in zip(dates, close):
            rows.append({"date": d, "ticker": ticker, "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000})
    return pd.DataFrame(rows)


# ── _get_rebalance_dates ──────────────────────────────────────────────────────

def test_rebalance_never_returns_only_first():
    days = pd.bdate_range("2021-01-04", periods=50, tz="UTC").tolist()
    rdates = _get_rebalance_dates(days, RebalanceFrequency.NEVER)
    assert rdates == [days[0]]


def test_rebalance_weekly_every_5():
    days = pd.bdate_range("2021-01-04", periods=25, tz="UTC").tolist()
    rdates = _get_rebalance_dates(days, RebalanceFrequency.WEEKLY)
    expected = [days[i] for i in range(0, 25, 5)]
    assert rdates == expected


def test_rebalance_monthly_first_day_of_each_month():
    days = pd.bdate_range("2021-01-04", periods=252, tz="UTC").tolist()
    rdates = _get_rebalance_dates(days, RebalanceFrequency.MONTHLY)
    # First rebalance = first trading day in the window
    assert rdates[0] == days[0]
    # Every subsequent rebalance is in a new month
    months = [(d.year, d.month) for d in rdates]
    assert len(months) == len(set(months)), "Duplicate months in monthly rebalance schedule"


def test_rebalance_quarterly_4_per_year():
    days = pd.bdate_range("2021-01-04", periods=252, tz="UTC").tolist()
    rdates = _get_rebalance_dates(days, RebalanceFrequency.QUARTERLY)
    quarters = [(d.year, (d.month - 1) // 3) for d in rdates]
    assert len(quarters) == len(set(quarters)), "Duplicate quarters in quarterly schedule"
    assert len(rdates) <= 5   # at most 5 distinct quarters in ~252 trading days


# ── run_backtest ──────────────────────────────────────────────────────────────

def test_buy_hold_single_asset_equity_matches_price(single_ticker_prices):
    """Buy-and-hold of a single asset must track the price series exactly."""
    prices = single_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    universe = ["AAAA"]
    capital = 50_000.0

    result = run_backtest(
        _SingleAssetBuyHold(), prices, universe, start, end, capital
    )

    assert isinstance(result, BacktestResult)
    assert len(result.equity) == len(result.dates)
    assert result.equity[0] == pytest.approx(capital)

    # Equity should compound with price returns
    close_pivot = prices.pivot_table(values="close", index="date", columns="ticker")
    close_pivot = close_pivot.sort_index()
    price_returns = close_pivot["AAAA"].pct_change().fillna(0).tolist()

    expected_equity = capital
    for i, r in enumerate(price_returns):
        if i == 0:
            continue
        expected_equity *= 1 + r
        assert result.equity[i] == pytest.approx(expected_equity, rel=1e-6)


def test_initial_equity_is_capital(two_ticker_prices):
    prices = two_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    capital = 123_456.0

    result = run_backtest(
        _EqualWeightMonthly(), prices, ["AAAA", "BBBB"], start, end, capital
    )
    assert result.equity[0] == pytest.approx(capital)


def test_result_lengths_consistent(two_ticker_prices):
    prices = two_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    result = run_backtest(
        _EqualWeightMonthly(), prices, ["AAAA", "BBBB"], start, end
    )
    n = len(result.dates)
    assert len(result.equity) == n
    assert len(result.returns) == n
    assert len(result.drawdown) == n
    assert len(result.rolling_sharpe) == n
    assert len(result.rolling_vol) == n
    for ticker, wlist in result.weights_history.items():
        assert len(wlist) == n


def test_no_lookahead(two_ticker_prices):
    """get_weights must never be called with data after the rebalance date."""
    called_with_dates: list[pd.Timestamp] = []

    class _SpyingStrategy(StrategyBase):
        strategy_id = "spy"
        name = "Spy"
        category = StrategyCategory.PASSIVE
        description = ""
        long_description = ""
        rebalance_frequency = RebalanceFrequency.MONTHLY

        def get_weights(self, prices, current_date, universe):
            max_date = prices["date"].max()
            called_with_dates.append(max_date)
            return self._equal_weight(universe)

    prices = two_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    run_backtest(_SpyingStrategy(), prices, ["AAAA", "BBBB"], start, end)

    # Every prices_slice passed to get_weights must have max date <= the
    # corresponding rebalance signal date (never future data)
    for signal_date in called_with_dates:
        assert signal_date <= end


def test_returns_day0_is_zero(two_ticker_prices):
    prices = two_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    result = run_backtest(_EqualWeightMonthly(), prices, ["AAAA", "BBBB"], start, end)
    assert result.returns[0] == pytest.approx(0.0)


def test_strategy_metadata_preserved(two_ticker_prices):
    prices = two_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    result = run_backtest(_EqualWeightMonthly(), prices, ["AAAA", "BBBB"], start, end)
    assert result.strategy_id == "test_equal_monthly"
    assert result.strategy_name == "Test Equal Monthly"
    assert result.category == "quantitative"


def test_metrics_dict_has_all_keys(two_ticker_prices):
    prices = two_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    result = run_backtest(_EqualWeightMonthly(), prices, ["AAAA", "BBBB"], start, end)
    expected_keys = {
        "total_return", "cagr", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "max_drawdown_duration", "calmar",
        "win_rate", "avg_monthly_turnover", "excess_return", "information_ratio",
    }
    assert expected_keys.issubset(set(result.metrics.keys()))


def test_equity_never_negative(two_ticker_prices):
    """Portfolio value must always remain positive."""
    prices = two_ticker_prices
    start = prices["date"].min()
    end = prices["date"].max()
    result = run_backtest(_EqualWeightMonthly(), prices, ["AAAA", "BBBB"], start, end)
    assert all(e > 0 for e in result.equity)


# ── Phase 3 integration: buy_and_hold([SPY]) == benchmark ────────────────────

@pytest.fixture
def spy_prices():
    """Synthetic SPY-like price series over 252 trading days."""
    dates = pd.bdate_range("2021-01-04", periods=252, tz="UTC")
    rng = np.random.default_rng(99)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates)))
    rows = [
        {
            "date": d,
            "ticker": "SPY",
            "open": c,
            "high": c * 1.002,
            "low": c * 0.998,
            "close": c,
            "volume": 50_000_000,
        }
        for d, c in zip(dates, close)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def stw_prices():
    """Synthetic STW.AX-like price series over 252 trading days."""
    dates = pd.bdate_range("2021-01-04", periods=252, tz="UTC")
    rng = np.random.default_rng(42)
    close = 60.0 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(dates)))
    rows = [
        {
            "date": d,
            "ticker": "STW.AX",
            "open": c,
            "high": c * 1.002,
            "low": c * 0.998,
            "close": c,
            "volume": 5_000_000,
        }
        for d, c in zip(dates, close)
    ]
    return pd.DataFrame(rows)


def test_buy_and_hold_spy_matches_benchmark(spy_prices):
    """BuyAndHold on [SPY] must produce an equity curve identical to run_benchmark().

    This is the canonical Phase 3 smoke-test: when the universe is a single
    asset (SPY), the passive strategy and the benchmark are mathematically
    equivalent — both represent unlevered buy-and-hold of SPY.
    """
    from engine.benchmark import run_benchmark
    from strategies.buy_and_hold import BuyAndHold

    prices = spy_prices
    start = prices["date"].min()
    end = prices["date"].max()
    capital = 100_000.0

    strategy_result = run_backtest(
        BuyAndHold(), prices, ["SPY"], start, end, capital
    )
    benchmark_result = run_benchmark(prices, start, end, capital)

    assert len(strategy_result.equity) == len(benchmark_result.equity), (
        "Equity curves must have the same length"
    )

    for i, (s, b) in enumerate(zip(strategy_result.equity, benchmark_result.equity)):
        assert s == pytest.approx(b, rel=1e-6), (
            f"Equity mismatch on day {i}: strategy={s:.4f}, benchmark={b:.4f}"
        )


def test_benchmark_stw_ax_equity_starts_at_capital(stw_prices):
    """run_benchmark with STW.AX must produce a valid equity curve starting at initial_capital."""
    from engine.benchmark import run_benchmark

    prices = stw_prices
    start = prices["date"].min()
    end = prices["date"].max()
    capital = 100_000.0

    result = run_benchmark(prices, start, end, capital, benchmark_ticker="STW.AX")

    assert result.strategy_id == "benchmark_stw_ax"
    assert result.strategy_name == "S&P/ASX 200 (STW.AX)"
    assert result.category == "benchmark"
    assert result.equity[0] == pytest.approx(capital)
    assert len(result.equity) == len(result.dates)
    assert all(e > 0 for e in result.equity)


def test_buy_and_hold_stw_ax_matches_benchmark(stw_prices):
    """BuyAndHold on [STW.AX] must produce an equity curve matching run_benchmark(STW.AX)."""
    from engine.benchmark import run_benchmark
    from strategies.buy_and_hold import BuyAndHold

    prices = stw_prices
    start = prices["date"].min()
    end = prices["date"].max()
    capital = 100_000.0

    strategy_result = run_backtest(
        BuyAndHold(), prices, ["STW.AX"], start, end, capital
    )
    benchmark_result = run_benchmark(prices, start, end, capital, benchmark_ticker="STW.AX")

    assert len(strategy_result.equity) == len(benchmark_result.equity)
    for i, (s, b) in enumerate(zip(strategy_result.equity, benchmark_result.equity)):
        assert s == pytest.approx(b, rel=1e-6), (
            f"Equity mismatch on day {i}: strategy={s:.4f}, benchmark={b:.4f}"
        )


# ── Cash support tests ────────────────────────────────────────────────────────

@pytest.fixture
def volatile_single_ticker_prices():
    """Single ticker with realistic random returns (non-trivial volatility)."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=252, tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, len(dates)))
    rows = [
        {
            "date": d, "ticker": "AAAA",
            "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000,
        }
        for d, c in zip(dates, close)
    ]
    return pd.DataFrame(rows)


def test_half_invested_cash_in_weights_history(volatile_single_ticker_prices):
    """Strategy returning 0.5 weight must produce CASH key in weights_history."""
    prices = volatile_single_ticker_prices
    start, end = prices["date"].min(), prices["date"].max()
    result = run_backtest(_HalfInvested(), prices, ["AAAA"], start, end)
    assert "CASH" in result.weights_history, "CASH must appear in weights_history"
    # On day 0, CASH weight ≈ 0.5 (exact at first rebalance)
    assert result.weights_history["CASH"][0] == pytest.approx(0.5, abs=1e-6)


def test_half_invested_lower_vol_than_fully_invested(volatile_single_ticker_prices):
    """50% invested + 50% cash must have strictly lower volatility than 100% invested."""
    prices = volatile_single_ticker_prices
    start, end = prices["date"].min(), prices["date"].max()

    full_result = run_backtest(_SingleAssetBuyHold(), prices, ["AAAA"], start, end)
    half_result = run_backtest(_HalfInvested(), prices, ["AAAA"], start, end)

    full_vol = float(np.std(full_result.returns[1:]))
    half_vol = float(np.std(half_result.returns[1:]))
    assert half_vol < full_vol, (
        f"Half-invested vol ({half_vol:.6f}) must be < fully-invested vol ({full_vol:.6f})"
    )


def test_half_invested_first_day_return_is_half_asset(volatile_single_ticker_prices):
    """On day 1 (right after the initial 50/50 allocation) port return = 0.5 * asset return."""
    prices = volatile_single_ticker_prices
    start, end = prices["date"].min(), prices["date"].max()
    result = run_backtest(_HalfInvested(), prices, ["AAAA"], start, end)

    close = (
        prices.pivot_table(values="close", index="date", columns="ticker")
        .sort_index()
    )
    asset_ret_day1 = close["AAAA"].pct_change().iloc[1]
    assert result.returns[1] == pytest.approx(0.5 * asset_ret_day1, abs=1e-9), (
        f"Day 1: expected {0.5 * asset_ret_day1:.8f}, got {result.returns[1]:.8f}"
    )


def test_over_invested_normalised_and_warns(two_ticker_prices):
    """Weights summing > 1.0 must be normalised to 1.0 and emit a UserWarning."""
    prices = two_ticker_prices
    start, end = prices["date"].min(), prices["date"].max()

    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = run_backtest(_OverInvested(), prices, ["AAAA", "BBBB"], start, end)

    assert any("normalis" in str(w.message).lower() for w in caught), (
        "Expected a normalisation warning for over-invested strategy"
    )
    # No cash should be held when weights were normalised down
    cash_weights = result.weights_history.get("CASH", [0.0] * len(result.dates))
    assert all(abs(w) < 1e-9 for w in cash_weights), (
        "Over-invested strategy must not produce cash after normalisation"
    )


def test_all_cash_equity_stays_flat(volatile_single_ticker_prices):
    """A strategy returning {} must hold all cash and produce a flat equity curve."""
    prices = volatile_single_ticker_prices
    start, end = prices["date"].min(), prices["date"].max()
    capital = 50_000.0
    result = run_backtest(_NeverTrades(), prices, ["AAAA"], start, end, capital)

    assert all(e == pytest.approx(capital) for e in result.equity), (
        "All-cash portfolio equity must remain constant at initial_capital"
    )
    assert "CASH" in result.weights_history
    assert all(abs(w - 1.0) < 1e-6 for w in result.weights_history["CASH"])
