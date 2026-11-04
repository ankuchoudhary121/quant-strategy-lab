from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Base class for signal-generating strategies.

    Signals are target weights from -1 to 1 for each asset. Long-only strategies
    should use 0 to 1. The engine converts weight changes into trades.
    """

    name = 'Base Strategy'

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
