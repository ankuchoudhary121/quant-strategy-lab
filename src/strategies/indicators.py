from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Rolling RSI with explicit zero-gain and zero-loss handling.

    - Average loss = 0 and average gain > 0 -> RSI 100
    - Average gain = 0 and average loss > 0 -> RSI 0
    - Both are 0, such as flat prices -> RSI 50
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    out = out.mask((gain == 0) & (loss > 0), 0.0)
    out = out.mask((gain == 0) & (loss == 0), 50.0)
    return out


def macd(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return line, signal_line, hist


def bollinger_bands(prices: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def zscore(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    mean = frame.rolling(window).mean()
    std = frame.rolling(window).std().replace(0, np.nan)
    return (frame - mean) / std


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).stack(),
        (high - prev_close).abs().stack(),
        (low - prev_close).abs().stack(),
    ], axis=1).max(axis=1).unstack()
    return tr.rolling(window).mean()


def rolling_beta(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, window: int = 60) -> pd.DataFrame:
    betas = {}
    bench_var = benchmark_returns.rolling(window).var()
    for col in asset_returns.columns:
        cov = asset_returns[col].rolling(window).cov(benchmark_returns)
        betas[col] = cov / bench_var
    return pd.DataFrame(betas, index=asset_returns.index)
