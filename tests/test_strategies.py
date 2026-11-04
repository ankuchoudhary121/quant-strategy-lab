import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from strategies.moving_average import MovingAverageCrossover


def test_moving_average_outputs_weights():
    prices = pd.DataFrame({'AAA': range(1, 101), 'BBB': range(101, 201)}, index=pd.date_range('2024-01-01', periods=100))
    strategy = MovingAverageCrossover(5, 20)
    signals = strategy.generate_signals(prices)
    assert signals.shape == prices.shape
    assert signals.max().max() <= 0.5
