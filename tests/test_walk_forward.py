from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from backtester.optimizer import walk_forward_validate, walk_forward_search, optimize_parameters
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi import RSIStrategy
from strategies.bollinger import BollingerMeanReversion


def generate_synthetic_prices(periods: int = 240) -> pd.DataFrame:
    """Generate deterministic synthetic multi-asset price history."""
    dates = pd.date_range('2022-01-03', periods=periods, freq='B')
    t = np.arange(periods)
    # Asset AAA: upward trend with cyclical oscillation
    p_aaa = 100.0 + 0.25 * t + 3.0 * np.sin(t / 8.0)
    # Asset BBB: oscillating mean-reverting series
    p_bbb = 80.0 + 5.0 * np.cos(t / 12.0) + 0.05 * t
    return pd.DataFrame({'AAA': p_aaa, 'BBB': p_bbb}, index=dates)


def test_future_leakage_immunity():
    """Future test data must never influence earlier training fold parameter selection."""
    prices1 = generate_synthetic_prices(180)
    param_grid = {'short_window': [5, 10, 15], 'long_window': [25, 40]}

    res1 = walk_forward_validate(
        prices=prices1,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        window_type='rolling',
        include_hindsight=False,
    )

    # Construct prices2: exactly identical for the first 60 rows (Fold 1 training window),
    # but radically mutated across all subsequent rows (future/test data).
    prices2 = prices1.copy()
    mutated_slice = prices2.iloc[60:].copy()
    prices2.iloc[60:] = mutated_slice * 5.0 + 50.0

    res2 = walk_forward_validate(
        prices=prices2,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        window_type='rolling',
        include_hindsight=False,
    )

    # Fold 1 parameters MUST be 100% identical
    assert res1.windows.loc[0, 'Selected Parameters'] == res2.windows.loc[0, 'Selected Parameters']
    assert res1.windows.loc[0, 'short_window'] == res2.windows.loc[0, 'short_window']
    assert res1.windows.loc[0, 'long_window'] == res2.windows.loc[0, 'long_window']
    assert np.isclose(
        res1.windows.loc[0, 'IS Score'],
        res2.windows.loc[0, 'IS Score'],
        rtol=1e-7,
        atol=1e-7,
    )


def test_test_windows_partition_without_overlap_or_gaps():
    """Test windows must form an exact sequential partition of the out-of-sample period."""
    prices = generate_synthetic_prices(180)
    param_grid = {'short_window': [5, 10], 'long_window': [20, 30]}

    res = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        step_size=30,
        include_hindsight=False,
    )

    windows = res.windows
    assert len(windows) == 4

    for i in range(len(windows) - 1):
        curr_test_end = windows.loc[i, 'Test End']
        next_test_start = windows.loc[i + 1, 'Test Start']
        # The next test window starts strictly after the current test window ends
        assert next_test_start > curr_test_end
        # In business day sequence, curr_test_end index + 1 == next_test_start index
        curr_idx = prices.index.get_loc(curr_test_end)
        next_idx = prices.index.get_loc(next_test_start)
        assert next_idx == curr_idx + 1

    # Verify stitched history covers exactly the union of all test dates
    all_test_dates = []
    for i in range(len(windows)):
        sub = prices.loc[windows.loc[i, 'Test Start']:windows.loc[i, 'Test End']].index
        all_test_dates.extend(sub)

    assert res.oos_history.index.equals(pd.Index(all_test_dates))


def test_continuous_capital_and_exact_oos_reconciliation():
    """Stitched OOS total return must reconcile with compounded individual fold returns."""
    prices = generate_synthetic_prices(200)
    param_grid = {'short_window': [5, 15], 'long_window': [30, 50]}

    res = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=80,
        test_size=40,
        engine_kwargs={'initial_cash': 100_000, 'commission': 0.001, 'slippage': 0.0005},
        include_hindsight=False,
    )

    windows = res.windows
    assert len(windows) == 3

    # Capital continuity: Fold k entry equity must match Fold k-1 exit equity
    for k in range(1, len(windows)):
        assert np.isclose(
            windows.loc[k, 'Entry Equity'],
            windows.loc[k - 1, 'Exit Equity'],
            rtol=1e-7,
            atol=1e-7,
        )

    # Fold returns compounded must match ending stitched equity / initial cash - 1
    compounded_fold_return = np.prod(1.0 + windows['OOS Return'].values) - 1.0
    stitched_total_return = res.oos_history['Equity'].iloc[-1] / 100_000.0 - 1.0

    assert np.isclose(compounded_fold_return, stitched_total_return, rtol=1e-6, atol=1e-6)
    assert np.isclose(res.oos_metrics['Total Return'], stitched_total_return, rtol=1e-6, atol=1e-6)


def test_transaction_costs_and_slippage_are_actively_deducted():
    """Transaction costs must be deducted during test windows and fold rebalances."""
    prices = generate_synthetic_prices(180)
    param_grid = {'short_window': [5, 10], 'long_window': [20, 30]}

    res = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        engine_kwargs={'commission': 0.002, 'slippage': 0.001},
        include_hindsight=False,
    )

    assert not res.trades.empty
    assert res.oos_metrics['Total Trading Cost'] > 0.0
    assert res.oos_metrics['Total Cost Drag'] > 0.0


def test_rolling_vs_expanding_window_mechanics():
    """Expanding windows must keep train origin fixed at 0; rolling windows advance."""
    prices = generate_synthetic_prices(180)
    param_grid = {'short_window': [5, 10], 'long_window': [20, 30]}

    res_rolling = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        window_type='rolling',
        include_hindsight=False,
    )
    res_expanding = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        window_type='expanding',
        include_hindsight=False,
    )

    # Rolling: train start advances
    assert res_rolling.windows['Train Start'].nunique() > 1
    assert res_rolling.windows.loc[1, 'Train Start'] > res_rolling.windows.loc[0, 'Train Start']

    # Expanding: train start is anchored at the initial date
    assert res_expanding.windows['Train Start'].nunique() == 1
    assert (res_expanding.windows['Train Start'] == prices.index[0]).all()
    # But train end advances
    assert res_expanding.windows.loc[1, 'Train End'] > res_expanding.windows.loc[0, 'Train End']


def test_benchmark_alignment_and_reconciliation():
    """Benchmark curve must align with test dates and accurately track benchmark return."""
    prices = generate_synthetic_prices(180)
    benchmark = pd.DataFrame({'SPY': prices['AAA'] * 1.5}, index=prices.index)
    param_grid = {'short_window': [5, 10], 'long_window': [20, 30]}

    res = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        benchmark=benchmark,
        include_hindsight=False,
    )

    assert 'Benchmark' in res.oos_history
    assert res.oos_history['Benchmark'].notna().all()
    assert res.oos_history.index.equals(res.oos_history['Benchmark'].index)

    # Benchmark total return on test dates
    first_date = res.oos_history.index[0]
    last_date = res.oos_history.index[-1]
    expected_bench_ret = benchmark.loc[last_date, 'SPY'] / benchmark.loc[first_date, 'SPY'] - 1.0
    actual_bench_ret = res.oos_history['Benchmark'].iloc[-1] / res.oos_history['Benchmark'].iloc[0] - 1.0
    assert np.isclose(actual_bench_ret, expected_bench_ret, rtol=1e-5, atol=1e-5)


def test_parameter_stability_types():
    """Numeric parameters must have std/mean; discrete/categorical must have frequency/switch counts."""
    prices = generate_synthetic_prices(180)
    param_grid = {'short_window': [5, 10], 'long_window': [20, 30], 'long_only': [True]}

    res = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        include_hindsight=False,
    )

    stab = res.stability_summary.set_index('Parameter')
    assert 'short_window' in stab.index
    assert 'long_only' in stab.index

    # short_window is numeric
    assert stab.loc['short_window', 'Type'] == 'Numeric'
    assert np.isfinite(stab.loc['short_window', 'Mean'])
    assert np.isfinite(stab.loc['short_window', 'Std Dev'])

    # long_only is boolean/discrete
    assert stab.loc['long_only', 'Type'] == 'Discrete / Categorical'
    assert np.isnan(stab.loc['long_only', 'Std Dev'])
    assert stab.loc['long_only', 'Switches'] == 0


def test_determinism_and_reproducibility():
    """Identical runs must produce identical parameters and numerically tight metrics."""
    prices = generate_synthetic_prices(180)
    param_grid = {'short_window': [5, 10], 'long_window': [20, 30]}

    res1 = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        include_hindsight=False,
    )
    res2 = walk_forward_validate(
        prices=prices,
        strategy_factory=MovingAverageCrossover,
        param_grid=param_grid,
        train_size=60,
        test_size=30,
        include_hindsight=False,
    )

    assert res1.windows['Selected Parameters'].tolist() == res2.windows['Selected Parameters'].tolist()
    assert res1.windows['Test Start'].tolist() == res2.windows['Test Start'].tolist()
    assert np.allclose(res1.oos_history['Equity'].values, res2.oos_history['Equity'].values, rtol=1e-7, atol=1e-7)
    assert np.isclose(res1.oos_metrics['Total Return'], res2.oos_metrics['Total Return'], rtol=1e-7, atol=1e-7)
    assert np.isclose(res1.oos_metrics['Sharpe Ratio'], res2.oos_metrics['Sharpe Ratio'], rtol=1e-7, atol=1e-7)


def test_backward_compatibility_walk_forward_search():
    """Existing walk_forward_search wrapper must return DataFrame with expected columns."""
    prices = generate_synthetic_prices(120)
    param_grid = {'short_window': [5, 10], 'long_window': [20, 30]}
    df = walk_forward_search(prices, MovingAverageCrossover, param_grid, train_size=60, test_size=30)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'Fold' in df.columns
    assert 'Train Sharpe Ratio' in df.columns
    assert 'Test Sharpe Ratio' in df.columns
    assert 'Train Start' in df.columns
    assert 'Test End' in df.columns
