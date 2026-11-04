import numpy as np
import pandas as pd

from src.backtester.metrics import calculate_metrics, drawdown_series, max_drawdown_duration, value_at_risk, conditional_value_at_risk
from src.backtester.portfolio import Portfolio
from src.strategies.moving_average import MovingAverageCrossover
from src.strategies.indicators import rsi


def test_drawdown_known_case():
    equity = pd.Series([100, 120, 90, 150, 75], dtype=float)
    dd = drawdown_series(equity)
    assert dd.iloc[0] == 0
    assert dd.iloc[2] == -0.25
    assert dd.iloc[4] == -0.5
    assert max_drawdown_duration(equity) == 1


def test_drawdown_duration_and_ulcer_metrics():
    equity = pd.Series([100, 90, 80, 85, 101, 99, 102], dtype=float)
    metrics = calculate_metrics(equity)
    assert metrics['Max Drawdown Duration'] == 3
    assert metrics['Ulcer Index'] > 0
    assert np.isclose(metrics['Recovery Factor'], 0.02 / 0.20)


def test_total_return_and_cagr_known_case():
    equity = pd.Series([100.0, 101.0, 102.01])
    metrics = calculate_metrics(equity)
    assert np.isclose(metrics['Total Return'], 0.0201)
    expected_cagr = (1.0201) ** (252 / 2) - 1
    assert np.isclose(metrics['Annual Return'], expected_cagr)


def test_nonpositive_ending_equity_has_no_complex_cagr():
    metrics = calculate_metrics(pd.Series([100.0, -10.0]))
    assert np.isnan(metrics['Annual Return'])


def test_var_and_cvar_known_case():
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])
    assert np.isclose(value_at_risk(returns, 0.2), -0.06)
    assert np.isclose(conditional_value_at_risk(returns, 0.2), -0.10)


def test_rsi_edge_cases():
    dates = pd.date_range('2024-01-01', periods=20)
    up = pd.DataFrame({'A': np.arange(1, 21, dtype=float)}, index=dates)
    down = pd.DataFrame({'A': np.arange(20, 0, -1, dtype=float)}, index=dates)
    flat = pd.DataFrame({'A': np.ones(20) * 10}, index=dates)
    assert rsi(up, 14).iloc[-1, 0] == 100
    assert rsi(down, 14).iloc[-1, 0] == 0
    assert rsi(flat, 14).iloc[-1, 0] == 50


def test_strategy_signal_is_applied_exactly_one_period_later():
    dates = pd.date_range('2024-01-01', periods=6)
    prices = pd.DataFrame({'A': [1, 1, 2, 4, 8, 16]}, index=dates, dtype=float)
    signals = MovingAverageCrossover(short_window=1, long_window=2).generate_signals(prices)
    history, _ = Portfolio(initial_cash=100, commission=0, slippage=0).simulate(prices, signals)

    assert signals.loc[dates[2], 'A'] == 1.0
    assert history.loc[dates[2], 'Portfolio Return'] == 0.0
    assert history.loc[dates[3], 'Portfolio Return'] == 1.0


def test_rebalance_costs_match_trade_ledger_and_include_price_drift():
    dates = pd.date_range('2024-01-01', periods=4)
    prices = pd.DataFrame(
        {'A': [100.0, 110.0, 120.0, 130.0], 'B': [100.0, 100.0, 100.0, 100.0]},
        index=dates,
    )
    weights = pd.DataFrame({'A': 0.5, 'B': 0.5}, index=dates)
    portfolio = Portfolio(initial_cash=1_000, commission=0.01, slippage=0.0)
    history, trades = portfolio.simulate(prices, weights)

    pre_cost_equity = history['Equity'].shift(1).fillna(1_000) * (1 + history['Return Before Costs'])
    deducted_cost = float((pre_cost_equity * history['Trading Cost']).sum())

    assert (history['Trading Cost'].iloc[1:] > 0).all()
    assert np.isclose(trades['Estimated Cost'].sum(), deducted_cost)


def test_portfolio_rejects_mismatched_signal_assets():
    prices = pd.DataFrame({'A': [1.0, 2.0]}, index=pd.date_range('2024-01-01', periods=2))
    weights = pd.DataFrame({'B': [0.0, 1.0]}, index=prices.index)
    try:
        Portfolio().simulate(prices, weights)
    except ValueError as exc:
        assert 'columns must exactly match' in str(exc)
    else:
        raise AssertionError('mismatched strategy assets should be rejected')


def test_portfolio_rejects_unsorted_dates_and_bankruptcy():
    dates = pd.to_datetime(['2024-01-02', '2024-01-01'])
    prices = pd.DataFrame({'A': [1.0, 2.0]}, index=dates)
    weights = pd.DataFrame({'A': [0.0, 1.0]}, index=dates)
    try:
        Portfolio().simulate(prices, weights)
    except ValueError as exc:
        assert 'sorted' in str(exc)
    else:
        raise AssertionError('unsorted dates should be rejected')

    dates = pd.date_range('2024-01-01', periods=2)
    prices = pd.DataFrame({'A': [100.0, 1.0]}, index=dates)
    weights = pd.DataFrame({'A': [2.0, 2.0]}, index=dates)
    try:
        Portfolio(max_weight=2.0, max_gross_exposure=2.0, long_only=False).simulate(prices, weights)
    except ValueError as exc:
        assert 'lost 100%' in str(exc)
    else:
        raise AssertionError('bankrupt portfolios should stop instead of producing negative equity')
