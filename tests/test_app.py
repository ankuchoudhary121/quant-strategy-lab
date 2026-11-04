from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert any(button.label == 'Run Backtest' for button in app.button)
    assert any(box.label == 'Run MA parameter optimizer' for box in app.checkbox)

    optimizer_toggle = next(box for box in app.checkbox if box.label == 'Run MA parameter optimizer')
    optimizer_toggle.set_value(True)
    app.run(timeout=30)
    assert not app.exception
    assert any(select.label == 'Optimizer validation' for select in app.selectbox)


def test_dashboard_renders_validation_tab_with_results():
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / 'src'))

    import pandas as pd
    from backtester.engine import BacktestEngine
    from strategies.moving_average import MovingAverageCrossover
    from backtester.optimizer import walk_forward_validate

    prices = pd.DataFrame(
        {'AAPL': [100.0 + i * 0.4 for i in range(120)]},
        index=pd.date_range('2024-01-01', periods=120),
    )
    engine = BacktestEngine()
    res = engine.run(prices, MovingAverageCrossover(5, 10))
    wf = walk_forward_validate(
        prices,
        MovingAverageCrossover,
        {'short_window': [5, 10], 'long_window': [20, 30]},
        train_size=60,
        test_size=30,
    )

    app_path = ROOT / 'app.py'
    app = AppTest.from_file(str(app_path))
    app.session_state['result'] = res
    app.session_state['prices'] = prices
    app.session_state['benchmark'] = None
    app.session_state['engine_kwargs'] = {}
    app.session_state['strategy_name'] = 'Moving Average Crossover'
    app.session_state['validation_result'] = wf
    app.run(timeout=30)

    assert not app.exception
    tab_labels = [t.label for t in app.tabs]
    assert 'Validation' in tab_labels
    assert tab_labels == ['Backtest', 'Validation', 'Trades', 'Risk', 'Allocations', 'Signals', 'Optimizer', 'Reports', 'Strategy Comparison']


def test_switching_strategy_clears_stale_results():
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / 'src'))

    import pandas as pd
    from backtester.engine import BacktestEngine
    from strategies.moving_average import MovingAverageCrossover

    prices = pd.DataFrame(
        {'AAPL': [100.0 + i * 0.4 for i in range(120)]},
        index=pd.date_range('2024-01-01', periods=120),
    )
    engine = BacktestEngine()
    res = engine.run(prices, MovingAverageCrossover(5, 10))

    app_path = ROOT / 'app.py'
    app = AppTest.from_file(str(app_path))
    app.session_state['result'] = res
    app.session_state['prices'] = prices
    app.session_state['active_strategy'] = 'Moving Average Crossover'
    app.session_state['strategy_name'] = 'Moving Average Crossover'
    app.run(timeout=30)

    assert not app.exception
    assert 'result' in app.session_state

    # Switch strategy in the selectbox
    strategy_select = next(s for s in app.selectbox if s.label == 'Strategy')
    strategy_select.select('RSI Mean Reversion')
    app.run(timeout=30)

    assert not app.exception
    # Stale results must be cleared
    assert 'result' not in app.session_state
    assert 'validation_result' not in app.session_state
    assert any('Strategy switched to' in info.value for info in app.info)
