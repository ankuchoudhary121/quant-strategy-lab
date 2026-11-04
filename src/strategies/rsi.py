from __future__ import annotations

import numpy as np
import pandas as pd
from .base import Strategy


class RSIStrategy(Strategy):
    name = 'RSI Mean Reversion'

    def __init__(self, window: int = 14, oversold: float = 30, overbought: float = 70):
        if oversold >= overbought:
            raise ValueError('oversold must be less than overbought.')
        self.window = int(window)
        self.oversold = float(oversold)
        self.overbought = float(overbought)

    def _rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        from .indicators import rsi
        return rsi(prices, self.window)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        rsi_values = self._rsi(prices)
        state = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        state[rsi_values < self.oversold] = 1.0
        state[rsi_values > self.overbought] = 0.0
        state = state.ffill().fillna(0.0)
        gross = state.abs().sum(axis=1).replace(0, np.nan)
        return state.div(gross, axis=0).fillna(0.0)
