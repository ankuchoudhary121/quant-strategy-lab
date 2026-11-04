from __future__ import annotations

import pandas as pd

from .metrics import calculate_metrics, drawdown_series
from .portfolio import Portfolio

# Support both documented ``backtester.*`` imports (with src on sys.path) and
# repository-root ``src.backtester.*`` imports used by library consumers.
try:
    from strategies.base import Strategy
except ModuleNotFoundError:
    from src.strategies.base import Strategy


class BacktestEngine:
    def __init__(
        self,
        initial_cash: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_weight: float = 1.0,
        max_gross_exposure: float = 1.0,
        rebalance_frequency: str = 'D',
        long_only: bool = True,
        volatility_target: float | None = None,
        risk_free_rate: float = 0.0,
    ):
        if risk_free_rate <= -1:
            raise ValueError('risk_free_rate must be greater than -1.0.')
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            max_weight=max_weight,
            max_gross_exposure=max_gross_exposure,
            rebalance_frequency=rebalance_frequency,
            long_only=long_only,
            volatility_target=volatility_target,
        )
        self.risk_free_rate = risk_free_rate

    def run(self, prices: pd.DataFrame, strategy: Strategy, benchmark: pd.DataFrame | None = None) -> dict:
        signals = strategy.generate_signals(prices)
        history, trades = self.portfolio.simulate(prices, signals)

        benchmark_curve = None
        if benchmark is not None and not benchmark.empty:
            bench_prices = benchmark.iloc[:, 0].reindex(history.index).ffill().dropna()
            if not bench_prices.empty:
                benchmark_curve = history['Equity'].loc[bench_prices.index[0]] * bench_prices / bench_prices.iloc[0]

        metrics = calculate_metrics(history['Equity'], benchmark_curve, risk_free_rate=self.risk_free_rate)
        metrics.update({
            'Trade Count': int(len(trades)),
            'Total Trading Cost': float(trades['Estimated Cost'].sum()) if not trades.empty else 0.0,
            'Total Cost Drag': float(history['Trading Cost'].sum()),
            'Annualized Turnover': float(history['Turnover'].mean() * 252),
        })
        history['Drawdown'] = drawdown_series(history['Equity'])
        if benchmark_curve is not None:
            history['Benchmark'] = benchmark_curve

        return {
            'strategy': strategy.name,
            'history': history,
            'trades': trades,
            'signals': signals,
            'metrics': metrics,
        }
