from __future__ import annotations

import pandas as pd
from .base import Strategy


class BuyAndHold(Strategy):
    name = 'Buy and Hold'

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(
            0.0,
            index=prices.index,
            columns=prices.columns
        )

        if len(prices.columns) > 0:
            weights.iloc[0] = self.weight / len(prices.columns)

        return weights