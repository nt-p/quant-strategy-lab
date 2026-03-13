"""Tests for strategy auto-discovery registry and StrategyBase contract."""

import pandas as pd
import pytest

from strategies.base import RebalanceFrequency, StrategyBase, StrategyCategory
from strategies.registry import discover_strategies


# ── Registry discovery ────────────────────────────────────────────────────────

def test_discover_finds_example_strategy():
    """The registry must discover the _example.py template strategy."""
    strategies = discover_strategies()
    ids = [s.strategy_id for s in strategies]
    assert "example_equal_weight" in ids, (
        f"Expected 'example_equal_weight' in discovered strategies, got: {ids}"
    )


def test_discover_returns_sorted_list():
    """Strategies must be sorted by (category, name) for stable sidebar ordering."""
    strategies = discover_strategies()
    keys = [(s.category.value, s.name) for s in strategies]
    assert keys == sorted(keys), "discover_strategies() must return a sorted list"


def test_discover_no_duplicates():
    """Each strategy_id must appear at most once."""
    strategies = discover_strategies()
    ids = [s.strategy_id for s in strategies]
    assert len(ids) == len(set(ids)), f"Duplicate strategy_ids found: {ids}"


def test_discover_skips_base_and_registry():
    """base.py and registry.py must never appear as strategies."""
    strategies = discover_strategies()
    ids = [s.strategy_id for s in strategies]
    # StrategyBase itself has strategy_id="" — confirm it is not in the list
    assert "" not in ids, "StrategyBase (strategy_id='') must not appear in discovered list"


# ── StrategyBase contract ─────────────────────────────────────────────────────

def test_example_strategy_metadata():
    """The example strategy must have all required metadata fields populated."""
    strategies = discover_strategies()
    example = next(s for s in strategies if s.strategy_id == "example_equal_weight")

    assert example.strategy_id
    assert example.name
    assert example.description
    assert example.long_description
    assert isinstance(example.category, StrategyCategory)
    assert isinstance(example.rebalance_frequency, RebalanceFrequency)


def test_example_strategy_get_weights(synthetic_prices):
    """get_weights() must return weights that sum to ~1.0 and cover all tickers."""
    strategies = discover_strategies()
    example = next(s for s in strategies if s.strategy_id == "example_equal_weight")

    universe = synthetic_prices["ticker"].unique().tolist()
    current_date = synthetic_prices["date"].max()

    weights = example.get_weights(synthetic_prices, current_date, universe)

    assert isinstance(weights, dict), "get_weights() must return a dict"
    assert len(weights) > 0, "weights dict must not be empty"
    assert all(isinstance(v, float) for v in weights.values()), "all weights must be floats"
    assert abs(sum(weights.values()) - 1.0) < 1e-6, (
        f"weights must sum to 1.0, got {sum(weights.values()):.6f}"
    )
    assert set(weights.keys()) == set(universe), (
        "weights must cover all tickers in the universe"
    )


def test_example_strategy_equal_weights(synthetic_prices):
    """The example strategy must assign identical weight to every ticker."""
    strategies = discover_strategies()
    example = next(s for s in strategies if s.strategy_id == "example_equal_weight")

    universe = synthetic_prices["ticker"].unique().tolist()
    current_date = synthetic_prices["date"].max()
    weights = example.get_weights(synthetic_prices, current_date, universe)

    values = list(weights.values())
    assert all(abs(v - values[0]) < 1e-9 for v in values), (
        "ExampleEqualWeight must assign equal weight to all tickers"
    )


def test_requires_training_default():
    """Non-ML strategies must return False from requires_training()."""
    strategies = discover_strategies()
    example = next(s for s in strategies if s.strategy_id == "example_equal_weight")
    assert example.requires_training() is False


def test_pivot_close_helper(synthetic_prices):
    """_pivot_close() helper must return a date×ticker DataFrame."""
    strategies = discover_strategies()
    example = next(s for s in strategies if s.strategy_id == "example_equal_weight")

    pivot = example._pivot_close(synthetic_prices)

    assert isinstance(pivot, pd.DataFrame)
    assert set(pivot.columns) == set(synthetic_prices["ticker"].unique())
    assert pivot.index.name == "date"


def test_equal_weight_helper():
    """_equal_weight() must return 1/N for N tickers and handle empty input."""
    strategies = discover_strategies()
    example = next(s for s in strategies if s.strategy_id == "example_equal_weight")

    w = example._equal_weight(["A", "B", "C", "D"])
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(abs(v - 0.25) < 1e-9 for v in w.values())

    assert example._equal_weight([]) == {}


def test_zscore_cross_sectional_helper():
    """_zscore_cross_sectional() must produce row-mean≈0 and row-std≈1."""
    strategies = discover_strategies()
    example = next(s for s in strategies if s.strategy_id == "example_equal_weight")

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]})
    zdf = example._zscore_cross_sectional(df)

    row_means = zdf.mean(axis=1)
    row_stds = zdf.std(axis=1)

    assert (row_means.abs() < 1e-10).all(), "Row means must be ~0 after z-scoring"
    assert ((row_stds - 1.0).abs() < 1e-10).all(), "Row stds must be ~1 after z-scoring"
