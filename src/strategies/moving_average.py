from __future__ import annotations

import pandas as pd
from .base import Strategy


class MovingAverageCrossover(Strategy):
    name = 'Moving Average Crossover'

    def __init__(self, short_window: int = 20, long_window: int = 50, long_only: bool = True):
        if short_window >= long_window:
            raise ValueError('short_window must be less than long_window.')
        self.short_window = short_window
        self.long_window = long_window
        self.long_only = long_only

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()
        raw = (short_ma > long_ma).astype(float)
        if not self.long_only:
            raw = raw.replace(0.0, -1.0)
        return raw.div(len(prices.columns))
