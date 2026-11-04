from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalize_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        normalized = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    else:
        normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
    return list(dict.fromkeys(normalized))


def load_prices(
    tickers: str | Iterable[str],
    start: str = '2018-01-01',
    end: str | None = None,
    source: str = 'yfinance',
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load adjusted close prices for one or more tickers.

    CSV mode expects either:
    - wide data with a Date column plus one column per ticker, or
    - long OHLCV rows with Date, Ticker and Adj Close/Close columns.
    """
    tickers_list = _normalize_tickers(tickers)
    if not tickers_list:
        raise ValueError('At least one ticker is required.')

    if source == 'csv':
        if csv_path is None:
            raise ValueError('csv_path is required when source="csv".')
        raw = pd.read_csv(csv_path)
        raw.columns = [str(c).strip() for c in raw.columns]
        date_col = next((c for c in raw.columns if c.lower() in {'date', 'datetime', 'timestamp'}), None)
        if date_col is None:
            raise ValueError('CSV must contain a Date, Datetime, or Timestamp column.')
        raw[date_col] = pd.to_datetime(raw[date_col])

        ticker_col = next((c for c in raw.columns if c.lower() == 'ticker'), None)
        if ticker_col:
            price_col = next((c for c in ['Adj Close', 'adj close', 'Adjusted Close', 'Close', 'close'] if c in raw.columns), None)
            if price_col is None:
                raise ValueError('Long CSV must contain an Adj Close or Close column.')
            raw[ticker_col] = raw[ticker_col].astype(str).str.upper().str.strip()
            prices = raw.pivot(index=date_col, columns=ticker_col, values=price_col)
        else:
            prices = raw.set_index(date_col)
            prices = prices.apply(pd.to_numeric, errors='coerce')
            prices.columns = [str(c).upper().strip() for c in prices.columns]
            prices = prices[[c for c in prices.columns if c in tickers_list]]
    elif source == 'yfinance':
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError('yfinance is required for Yahoo Finance data. Install it with: pip install yfinance, or choose CSV source.') from exc
        raw = yf.download(tickers_list, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError('No data returned. Check tickers and date range.')
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']].rename(columns={'Close': tickers_list[0]})
    else:
        raise ValueError('source must be "yfinance" or "csv".')

    prices = prices.sort_index().ffill().dropna(how='all')
    if start is not None:
        prices = prices.loc[prices.index >= pd.to_datetime(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.to_datetime(end)]
    prices.columns = [str(c).upper() for c in prices.columns]
    missing = [t for t in tickers_list if t not in prices.columns]
    if missing:
        raise ValueError(f'Missing price data for: {missing}')
    prices = prices[tickers_list].dropna()
    if prices.empty:
        raise ValueError('No usable price rows after cleaning. Check tickers, CSV columns, and date range.')
    return prices
