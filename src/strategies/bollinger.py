from __future__ import annotations

import pandas as pd
from .base import Strategy
from .indicators import bollinger_bands


class BollingerMeanReversion(Strategy):
    name = 'Bollinger Mean Reversion'

    def __init__(self, window: int = 20, num_std: float = 2.0, exit_on_mean: bool = True):
        self.window = window
        self.num_std = num_std
        self.exit_on_mean = exit_on_mean

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        lower, mid, upper = bollinger_bands(prices, self.window, self.num_std)
        buy = prices < lower
        sell = prices > mid if self.exit_on_mean else prices > upper
        # Store state changes, then carry the current position forward.
        events = pd.DataFrame(float('nan'), index=prices.index, columns=prices.columns)
        events = events.mask(buy, 1.0).mask(sell, 0.0)
        weights = events.ffill().fillna(0.0)
        return weights.div(len(prices.columns))
