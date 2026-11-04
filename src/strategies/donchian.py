from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class DonchianBreakout(Strategy):
    """Classic channel breakout: enter on new highs, exit on channel lows."""

    name = 'Donchian Channel Breakout'

    def __init__(self, entry_window: int = 55, exit_window: int = 20, long_only: bool = True):
        self.entry_window = int(entry_window)
        self.exit_window = int(exit_window)
        self.long_only = bool(long_only)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        entry_high = prices.shift(1).rolling(self.entry_window).max()
        exit_low = prices.shift(1).rolling(self.exit_window).min()
        entry_low = prices.shift(1).rolling(self.entry_window).min()
        exit_high = prices.shift(1).rolling(self.exit_window).max()

        state = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for i in range(1, len(prices)):
            current = state.iloc[i - 1].copy()
            current[prices.iloc[i] > entry_high.iloc[i]] = 1.0
            current[prices.iloc[i] < exit_low.iloc[i]] = 0.0
            if not self.long_only:
                current[prices.iloc[i] < entry_low.iloc[i]] = -1.0
                current[(prices.iloc[i] > exit_high.iloc[i]) & (current < 0)] = 0.0
            state.iloc[i] = current

        gross = state.abs().sum(axis=1).replace(0, np.nan)
        return state.div(gross, axis=0).fillna(0.0)
