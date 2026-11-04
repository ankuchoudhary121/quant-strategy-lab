from __future__ import annotations

import pandas as pd
from .base import Strategy
from .indicators import macd


class MACDTrendStrategy(Strategy):
    name = 'MACD Trend Following'

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, long_only: bool = True):
        if fast >= slow:
            raise ValueError('fast must be less than slow.')
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.long_only = long_only

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        line, signal_line, _ = macd(prices, self.fast, self.slow, self.signal)
        raw = (line > signal_line).astype(float)
        if not self.long_only:
            raw = raw.replace(0.0, -1.0)
        return raw.div(len(prices.columns))
