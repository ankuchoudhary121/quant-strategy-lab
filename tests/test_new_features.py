import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from backtester.engine import BacktestEngine
from backtester.optimizer import grid_search, walk_forward_search
from strategies.macd import MACDTrendStrategy
from strategies.bollinger import BollingerMeanReversion
from strategies.moving_average import MovingAverageCrossover
from strategies.moving_average import MovingAverageCrossover


def sample_prices():
    idx = pd.date_range('2024-01-01', periods=120)
    return pd.DataFrame({
        'AAA': [100 + i * 0.4 for i in range(120)],
        'BBB': [120 - i * 0.1 for i in range(120)],
        'CCC': [80 + (i % 10) for i in range(120)],
    }, index=idx)


def test_new_strategies_return_weight_frames():
    prices = sample_prices()
    for strategy in [MACDTrendStrategy(), BollingerMeanReversion(), MovingAverageCrossover(10, 30)]:
        weights = strategy.generate_signals(prices)
        assert weights.shape == prices.shape
        assert weights.index.equals(prices.index)


def test_engine_risk_controls_add_exposure_columns():
    prices = sample_prices()
    engine = BacktestEngine(max_weight=0.4, max_gross_exposure=0.8, rebalance_frequency='W')
    result = engine.run(prices, MovingAverageCrossover(10, 30))
    hist = result['history']
    assert 'Gross Exposure' in hist.columns
    assert hist['Gross Exposure'].max() <= 0.800001
    assert result['metrics']['Trade Count'] == len(result['trades'])
    assert 'Annualized Turnover' in result['metrics']
    assert 'Total Trading Cost' in result['metrics']


def test_grid_search_returns_ranked_results():
    prices = sample_prices()
    results = grid_search(prices, MovingAverageCrossover, {'short_window': [5, 10], 'long_window': [20, 30]})
    assert len(results) == 4
    assert 'Sharpe Ratio' in results.columns


def test_walk_forward_search_reports_only_test_metrics():
    prices = sample_prices()
    results = walk_forward_search(
        prices,
        MovingAverageCrossover,
        {'short_window': [5, 10], 'long_window': [20, 30]},
        train_size=60,
        test_size=30,
    )
    assert len(results) == 2
    assert list(results['Fold']) == [1, 2]
    assert 'Train Sharpe Ratio' in results.columns
    assert 'Test Sharpe Ratio' in results.columns
    assert (results['Test Start'] > results['Train End']).all()
