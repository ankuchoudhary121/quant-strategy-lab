from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return percentage drawdown from the running peak.

    Drawdown is 0 at new highs and negative below the peak.
    """
    equity_curve = equity_curve.astype(float).dropna()
    return equity_curve / equity_curve.cummax() - 1


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Return the longest number of consecutive observations below a prior peak."""
    underwater = drawdown_series(equity_curve) < 0
    if underwater.empty or not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max())


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical one-period VaR as a return threshold.

    For 95% VaR, pass level=0.05. The result is usually negative, e.g. -0.025
    means the 5th-percentile daily return is -2.5%.
    """
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(np.quantile(clean, level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical expected shortfall / CVaR as average return in the VaR tail."""
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    var = value_at_risk(clean, level)
    if np.isnan(var):
        return np.nan
    tail = clean[clean <= var]
    return float(tail.mean()) if not tail.empty else np.nan


def annualized_return_from_equity(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualized return from an equity curve."""
    equity_curve = equity_curve.astype(float).dropna()
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0 or equity_curve.iloc[-1] <= 0:
        return np.nan
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = max((len(equity_curve) - 1) / periods_per_year, 1 / periods_per_year)
    return float((1 + total_return) ** (1 / years) - 1)


def calculate_metrics(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate backtest performance metrics.

    Notes:
    - Total/annual return are geometric.
    - Sharpe and Sortino use periodic excess returns annualized by sqrt(252),
      which is the standard backtesting convention.
    - VaR/CVaR are reported as return thresholds, not positive loss amounts.
    """
    equity_curve = equity_curve.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(equity_curve) < 2 or returns.empty:
        return {}

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    annual_return = annualized_return_from_equity(equity_curve)
    return_std = returns.std(ddof=0)
    annual_volatility = return_std * np.sqrt(TRADING_DAYS)

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess_daily = returns - daily_rf
    excess_annual_return = (1 + excess_daily.mean()) ** TRADING_DAYS - 1
    sharpe = (excess_daily.mean() / return_std) * np.sqrt(TRADING_DAYS) if return_std != 0 else np.nan

    # Downside deviation uses all observations, with non-negative excess returns
    # contributing zero. Taking the std of only losing days understates risk.
    downside_deviation = np.sqrt(np.mean(np.minimum(excess_daily, 0.0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = excess_annual_return / downside_deviation if downside_deviation > 0 else np.nan

    dd = drawdown_series(equity_curve)
    max_drawdown = dd.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    ulcer_index = float(np.sqrt(np.mean(dd**2)))
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    positive_days = returns[returns > 0]
    negative_days = returns[returns < 0]
    gross_profit = positive_days.sum()
    gross_loss = abs(negative_days.sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    metrics = {
        'Start Value': float(equity_curve.iloc[0]),
        'End Value': float(equity_curve.iloc[-1]),
        'Total Return': float(total_return),
        'Annual Return': float(annual_return),
        'Annual Volatility': float(annual_volatility),
        'Sharpe Ratio': float(sharpe),
        'Sortino Ratio': float(sortino),
        'Max Drawdown': float(max_drawdown),
        'Max Drawdown Duration': max_drawdown_duration(equity_curve),
        'Ulcer Index': ulcer_index,
        'Recovery Factor': float(recovery_factor),
        'Calmar Ratio': float(calmar),
        'Positive Day Rate': float((returns > 0).mean()),
        'Profit Factor': float(profit_factor),
        'Daily VaR 95%': value_at_risk(returns, 0.05),
        'Daily CVaR 95%': conditional_value_at_risk(returns, 0.05),
        'Best Day': float(returns.max()),
        'Worst Day': float(returns.min()),
    }
    # Backward-compatible alias for the dashboard/tests; this is daily-return hit rate, not trade win rate.
    metrics['Win Rate'] = metrics['Positive Day Rate']

    if benchmark_curve is not None and not benchmark_curve.empty:
        benchmark_curve = benchmark_curve.astype(float).reindex(equity_curve.index).ffill().dropna()
        bench_returns = benchmark_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        aligned = returns.reindex(bench_returns.index).dropna()
        bench_returns = bench_returns.reindex(aligned.index).dropna()
        if len(aligned) > 2 and len(bench_returns) == len(aligned):
            bench_total = benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1
            strategy_overlap = equity_curve.loc[benchmark_curve.index]
            strategy_total = strategy_overlap.iloc[-1] / strategy_overlap.iloc[0] - 1
            metrics['Benchmark Total Return'] = float(bench_total)
            metrics['Excess Return'] = float(strategy_total - bench_total)
            bench_var = bench_returns.var(ddof=0)
            if bench_var != 0:
                beta = aligned.cov(bench_returns, ddof=0) / bench_var
                alpha = (aligned.mean() - beta * bench_returns.mean()) * TRADING_DAYS
                corr = aligned.corr(bench_returns)
                active = aligned - bench_returns
                tracking_error = active.std(ddof=0) * np.sqrt(TRADING_DAYS)
                information_ratio = (active.mean() * TRADING_DAYS) / tracking_error if tracking_error else np.nan
                metrics.update({
                    'Beta': float(beta),
                    'Alpha': float(alpha),
                    'Correlation': float(corr),
                    'Tracking Error': float(tracking_error),
                    'Information Ratio': float(information_ratio),
                })
    return metrics
