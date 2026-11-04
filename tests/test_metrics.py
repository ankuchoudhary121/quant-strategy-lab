import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from backtester.metrics import calculate_metrics


def test_calculate_metrics_has_core_fields():
    curve = pd.Series([100, 102, 101, 106, 110], index=pd.date_range('2024-01-01', periods=5))
    metrics = calculate_metrics(curve)
    assert 'Total Return' in metrics
    assert 'Sharpe Ratio' in metrics
    assert round(metrics['Total Return'], 4) == 0.10
