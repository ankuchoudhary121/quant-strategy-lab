from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

from .engine import BacktestEngine
from .metrics import calculate_metrics, drawdown_series
from .portfolio import Portfolio

try:
    from strategies.base import Strategy
except ModuleNotFoundError:
    from src.strategies.base import Strategy


MINIMIZE_OBJECTIVES = {
    'ulcer index',
    'max drawdown duration',
    'annual volatility',
    'tracking error',
}


@dataclass
class WalkForwardResult:
    """Container for walk-forward validation results."""

    windows: pd.DataFrame
    oos_history: pd.DataFrame
    oos_metrics: dict[str, float]
    is_vs_oos: pd.DataFrame
    parameter_stability: pd.DataFrame
    stability_summary: pd.DataFrame
    trades: pd.DataFrame
    metadata: dict[str, Any]
    hindsight_history: pd.DataFrame | None = None
    hindsight_metrics: dict[str, float] | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return the window-by-window DataFrame for backward compatibility."""
        return self.windows


def _is_numeric_parameter(values: list[Any]) -> bool:
    """Return True if all values are non-boolean numbers."""
    return all(
        isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
        for v in values
    )


def optimize_parameters(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list[Any]],
    warmup_prices: pd.DataFrame | None = None,
    engine_kwargs: dict[str, Any] | None = None,
    benchmark: pd.DataFrame | None = None,
    objective: str = 'Sharpe Ratio',
) -> dict[str, Any]:
    """Optimize parameters on `prices`, using optional prior rows for warm-up only."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError('prices must be a non-empty pandas DataFrame.')
    if not param_grid:
        raise ValueError('param_grid must not be empty.')

    engine_kwargs = engine_kwargs or {}
    engine = BacktestEngine(**engine_kwargs)
    rows: list[dict[str, Any]] = []
    results_by_candidate: list[dict[str, Any]] = []
    keys = list(param_grid)

    for values in product(*[param_grid[k] for k in keys]):
        candidate_params = dict(zip(keys, values))
        try:
            strategy = strategy_factory(**candidate_params)
            if warmup_prices is not None and not warmup_prices.empty:
                full_prices = pd.concat([warmup_prices, prices])
                full_signals = strategy.generate_signals(full_prices)
                target_weights = full_signals.loc[prices.index]
                history, trades = engine.portfolio.simulate(prices, target_weights)
                benchmark_curve = None
                if benchmark is not None and not benchmark.empty:
                    bench_prices = benchmark.iloc[:, 0].reindex(history.index).ffill().dropna()
                    if not bench_prices.empty:
                        benchmark_curve = (
                            history['Equity'].loc[bench_prices.index[0]]
                            * bench_prices
                            / bench_prices.iloc[0]
                        )
                metrics = calculate_metrics(
                    history['Equity'],
                    benchmark_curve,
                    risk_free_rate=engine.risk_free_rate,
                )
                metrics.update({
                    'Trade Count': int(len(trades)),
                    'Total Trading Cost': (
                        float(trades['Estimated Cost'].sum()) if not trades.empty else 0.0
                    ),
                    'Total Cost Drag': float(history['Trading Cost'].sum()),
                    'Annualized Turnover': float(history['Turnover'].mean() * 252),
                })
                history['Drawdown'] = drawdown_series(history['Equity'])
                if benchmark_curve is not None:
                    history['Benchmark'] = benchmark_curve
                run_res = {
                    'strategy': strategy.name,
                    'history': history,
                    'trades': trades,
                    'signals': target_weights,
                    'metrics': metrics,
                }
            else:
                run_res = engine.run(prices, strategy, benchmark=benchmark)

            row = {**candidate_params, **run_res['metrics']}
            rows.append(row)
            results_by_candidate.append({
                'params': candidate_params,
                'metrics': run_res['metrics'],
                'result': run_res,
            })
        except Exception as exc:
            rows.append({**candidate_params, 'error': str(exc)})
            results_by_candidate.append({
                'params': candidate_params,
                'error': str(exc),
            })

    ranked = pd.DataFrame(rows)
    ascending = objective.lower() in MINIMIZE_OBJECTIVES
    if objective in ranked.columns and not ranked[objective].dropna().empty:
        ranked = ranked.sort_values(objective, ascending=ascending).reset_index(drop=True)
        best_row = ranked.loc[ranked[objective].first_valid_index()]
        best_params = {
            key: next(candidate for candidate in param_grid[key] if candidate == best_row[key])
            for key in param_grid
        }
        best_score = float(best_row[objective])
        best_run = next(
            item for item in results_by_candidate
            if item.get('params') == best_params and 'metrics' in item
        )
        best_metrics = best_run['metrics']
        best_result = best_run['result']
    else:
        best_params = {k: param_grid[k][0] for k in keys}
        best_score = float('nan')
        best_metrics = {}
        best_result = None

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_metrics': best_metrics,
        'ranked': ranked,
        'best_result': best_result,
    }


def grid_search(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list[Any]],
    engine_kwargs: dict[str, Any] | None = None,
    benchmark: pd.DataFrame | None = None,
    objective: str = 'Sharpe Ratio',
) -> pd.DataFrame:
    """Run a parameter sweep and rank strategies by objective metric."""
    opt_res = optimize_parameters(
        prices=prices,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        engine_kwargs=engine_kwargs,
        benchmark=benchmark,
        objective=objective,
    )
    return opt_res['ranked']


def walk_forward_validate(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list[Any]],
    train_size: int = 252,
    test_size: int = 63,
    step_size: int | None = None,
    window_type: str = 'rolling',
    engine_kwargs: dict[str, Any] | None = None,
    benchmark: pd.DataFrame | None = None,
    objective: str = 'Sharpe Ratio',
    include_hindsight: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> WalkForwardResult:
    """Run rolling or expanding walk-forward validation."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError('prices must be a non-empty pandas DataFrame.')
    if train_size < 2 or test_size < 2:
        raise ValueError('train_size and test_size must both be at least 2.')
    step = test_size if step_size is None else int(step_size)
    if step < 1:
        raise ValueError('step_size must be positive.')
    if train_size + test_size > len(prices):
        raise ValueError('Not enough price rows for one train/test window.')
    if window_type not in {'rolling', 'expanding'}:
        raise ValueError("window_type must be either 'rolling' or 'expanding'.")

    engine_kwargs = engine_kwargs or {}
    engine = BacktestEngine(**engine_kwargs)

    window_bounds: list[tuple[int, int, int]] = []
    fold_idx = 0
    while True:
        train_start = 0 if window_type == 'expanding' else fold_idx * step
        train_end = train_size + fold_idx * step
        test_end = train_end + test_size
        if test_end > len(prices):
            break
        window_bounds.append((train_start, train_end, test_end))
        fold_idx += 1

    if not window_bounds:
        raise ValueError('No valid walk-forward windows could be constructed.')

    total_windows = len(window_bounds)
    fold_records: list[dict[str, Any]] = []
    oos_signal_slices: list[pd.DataFrame] = []
    train_metrics_list: list[dict[str, float]] = []

    for fold_num, (train_start, train_end, test_end) in enumerate(window_bounds, start=1):
        if progress_callback:
            progress_callback(fold_num, total_windows, f'Optimizing fold {fold_num}/{total_windows}')

        train_prices = prices.iloc[train_start:train_end]
        train_warmup = prices.iloc[0:train_start] if train_start > 0 else None
        train_benchmark = benchmark.reindex(train_prices.index) if benchmark is not None else None

        opt_res = optimize_parameters(
            prices=train_prices,
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            warmup_prices=train_warmup,
            engine_kwargs=engine_kwargs,
            benchmark=train_benchmark,
            objective=objective,
        )
        best_params = opt_res['best_params']
        best_score = opt_res['best_score']
        train_metrics = opt_res['best_metrics']
        train_metrics_list.append(train_metrics)

        test_dates = prices.index[train_end:test_end]
        context_prices = prices.iloc[0:test_end]
        frozen_strategy = strategy_factory(**best_params)
        context_signals = frozen_strategy.generate_signals(context_prices)
        test_signals = context_signals.loc[test_dates]
        oos_signal_slices.append(test_signals)

        param_str = ', '.join(f'{k}={v}' for k, v in best_params.items())
        fold_records.append({
            'Fold': fold_num,
            'Window': fold_num,
            'Train Start': train_prices.index[0],
            'Train End': train_prices.index[-1],
            'Test Start': test_dates[0],
            'Test End': test_dates[-1],
            'Train Period': f'{train_prices.index[0]:%Y-%m-%d} to {train_prices.index[-1]:%Y-%m-%d}',
            'Test Period': f'{test_dates[0]:%Y-%m-%d} to {test_dates[-1]:%Y-%m-%d}',
            'Selected Parameters': param_str,
            'IS Score': best_score,
            f'Train {objective}': best_score,
            **best_params,
            '_best_params': best_params,
            '_train_metrics': train_metrics,
        })

    # Concatenate test signals so the portfolio carries capital across fold boundaries.
    stitched_signals = pd.concat(oos_signal_slices)
    oos_dates = stitched_signals.index
    oos_prices = prices.loc[oos_dates]
    oos_benchmark = benchmark.reindex(oos_dates) if benchmark is not None else None

    oos_history, oos_trades = engine.portfolio.simulate(oos_prices, stitched_signals)

    bench_curve = None
    if oos_benchmark is not None and not oos_benchmark.empty:
        bench_prices = oos_benchmark.iloc[:, 0].reindex(oos_history.index).ffill().dropna()
        if not bench_prices.empty:
            bench_curve = oos_history['Equity'].iloc[0] * bench_prices / bench_prices.iloc[0]
            oos_history['Benchmark'] = bench_curve

    oos_history['Drawdown'] = drawdown_series(oos_history['Equity'])

    # Add a pre-OOS baseline so the first test-day return is measured correctly.
    pre_oos_date = prices.index[window_bounds[0][1] - 1]
    metrics_equity = pd.concat([
        pd.Series([engine.portfolio.initial_cash], index=[pre_oos_date]),
        oos_history['Equity'],
    ])
    metrics_bench = None
    if bench_curve is not None:
        metrics_bench = pd.concat([
            pd.Series([engine.portfolio.initial_cash], index=[pre_oos_date]),
            bench_curve,
        ])

    oos_metrics = calculate_metrics(
        metrics_equity,
        metrics_bench,
        risk_free_rate=engine.risk_free_rate,
    )
    oos_metrics.update({
        'Trade Count': int(len(oos_trades)),
        'Total Trading Cost': (
            float(oos_trades['Estimated Cost'].sum()) if not oos_trades.empty else 0.0
        ),
        'Total Cost Drag': float(oos_history['Trading Cost'].sum()),
        'Annualized Turnover': float(oos_history['Turnover'].mean() * 252),
    })

    # 4. Measure per-fold test metrics from continuous history so folds reconcile
    test_metrics_list: list[dict[str, float]] = []
    for idx, fold_dict in enumerate(fold_records):
        t_start = fold_dict['Test Start']
        t_end = fold_dict['Test End']
        fold_test_dates = prices.loc[t_start:t_end].index

        if idx == 0:
            entry_equity = engine.portfolio.initial_cash
        else:
            prev_t_end = fold_records[idx - 1]['Test End']
            entry_equity = float(oos_history.loc[prev_t_end, 'Equity'])

        fold_equity_slice = oos_history.loc[fold_test_dates, 'Equity']
        exit_equity = float(fold_equity_slice.iloc[-1])
        fold_oos_return = exit_equity / entry_equity - 1.0

        # Anchor each fold curve to its entry capital for comparable metrics.
        sub_equity = fold_equity_slice / entry_equity * engine.portfolio.initial_cash
        sub_bench = bench_curve.loc[fold_test_dates] if bench_curve is not None else None
        fold_m = calculate_metrics(sub_equity, sub_bench, risk_free_rate=engine.risk_free_rate)
        fold_trades = oos_trades[oos_trades['Date'].isin(fold_test_dates)]
        fold_trade_count = int(len(fold_trades))

        fold_dict['OOS Return'] = fold_oos_return
        fold_dict['Test Total Return'] = fold_oos_return
        fold_dict['OOS Sharpe'] = fold_m.get('Sharpe Ratio', np.nan)
        fold_dict['Test Sharpe Ratio'] = fold_m.get('Sharpe Ratio', np.nan)
        fold_dict['OOS Drawdown'] = fold_m.get('Max Drawdown', np.nan)
        fold_dict['Test Max Drawdown'] = fold_m.get('Max Drawdown', np.nan)
        fold_dict['Trades'] = fold_trade_count
        fold_dict['Test Trade Count'] = fold_trade_count
        fold_dict['Entry Equity'] = entry_equity
        fold_dict['Exit Equity'] = exit_equity

        for m_key, m_val in fold_m.items():
            fold_dict[f'Test {m_key}'] = m_val

        fold_m['Total Return'] = fold_oos_return
        fold_m['Trade Count'] = fold_trade_count
        test_metrics_list.append(fold_m)

    windows_df = pd.DataFrame(fold_records)

    is_vs_oos = _build_is_vs_oos_comparison(train_metrics_list, test_metrics_list, oos_metrics)

    param_stability, stability_summary = _build_parameter_stability(
        windows_df=windows_df,
        param_grid=param_grid,
    )

    hindsight_history = None
    hindsight_metrics = None
    if include_hindsight:
        try:
            hindsight_opt = optimize_parameters(
                prices=prices,
                strategy_factory=strategy_factory,
                param_grid=param_grid,
                engine_kwargs=engine_kwargs,
                benchmark=benchmark,
                objective=objective,
            )
            hs_strategy = strategy_factory(**hindsight_opt['best_params'])
            hs_res = engine.run(prices, hs_strategy, benchmark=benchmark)
            hs_slice = hs_res['history'].reindex(oos_dates)
            hs_slice['Equity'] = (
                engine.portfolio.initial_cash
                * (hs_slice['Equity'] / hs_slice['Equity'].iloc[0])
            )
            hs_slice['Drawdown'] = drawdown_series(hs_slice['Equity'])
            if 'Benchmark' in hs_res['history']:
                hs_slice['Benchmark'] = bench_curve
            hindsight_history = hs_slice
            hindsight_metrics = calculate_metrics(
                hs_slice['Equity'],
                bench_curve,
                risk_free_rate=engine.risk_free_rate,
            )
        except Exception:
            hindsight_history = None
            hindsight_metrics = None

    metadata = {
        'strategy_name': strategy_factory(**param_grid_defaults(param_grid)).name,
        'param_grid': param_grid,
        'train_size': train_size,
        'test_size': test_size,
        'step_size': step,
        'window_type': window_type,
        'objective': objective,
        'engine_kwargs': engine_kwargs,
        'start_date': str(prices.index[0].date()),
        'end_date': str(prices.index[-1].date()),
        'oos_start_date': str(oos_dates[0].date()),
        'oos_end_date': str(oos_dates[-1].date()),
        'total_windows': total_windows,
    }

    return WalkForwardResult(
        windows=windows_df,
        oos_history=oos_history,
        oos_metrics=oos_metrics,
        is_vs_oos=is_vs_oos,
        parameter_stability=param_stability,
        stability_summary=stability_summary,
        trades=oos_trades,
        metadata=metadata,
        hindsight_history=hindsight_history,
        hindsight_metrics=hindsight_metrics,
    )


def walk_forward_search(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list[Any]],
    train_size: int,
    test_size: int,
    engine_kwargs: dict[str, Any] | None = None,
    benchmark: pd.DataFrame | None = None,
    objective: str = 'Sharpe Ratio',
    step_size: int | None = None,
    window_type: str = 'rolling',
) -> pd.DataFrame:
    """Backward-compatible wrapper returning the window results DataFrame."""
    res = walk_forward_validate(
        prices=prices,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        window_type=window_type,
        engine_kwargs=engine_kwargs,
        benchmark=benchmark,
        objective=objective,
        include_hindsight=False,
    )
    return res.windows


def _build_is_vs_oos_comparison(
    train_metrics: list[dict[str, float]],
    test_metrics: list[dict[str, float]],
    stitched_oos_metrics: dict[str, float],
) -> pd.DataFrame:
    """Build train/test fold summaries alongside the stitched OOS statistics."""
    metric_keys = [
        ('Annual Return', 'CAGR / Annual Return', '{:+.2%}'),
        ('Annual Volatility', 'Annual Volatility', '{:.2%}'),
        ('Sharpe Ratio', 'Sharpe Ratio', '{:.2f}'),
        ('Sortino Ratio', 'Sortino Ratio', '{:.2f}'),
        ('Max Drawdown', 'Max Drawdown', '{:.2%}'),
        ('Total Return', 'Period Return (Fold Avg)', '{:+.2%}'),
        ('Trade Count', 'Trades (Fold Avg)', '{:.1f}'),
    ]
    rows: list[dict[str, Any]] = []

    for key, label, fmt in metric_keys:
        train_vals = [m.get(key, np.nan) for m in train_metrics if key in m and np.isfinite(m.get(key, np.nan))]
        test_vals = [m.get(key, np.nan) for m in test_metrics if key in m and np.isfinite(m.get(key, np.nan))]

        train_mean = float(np.mean(train_vals)) if train_vals else np.nan
        test_mean = float(np.mean(test_vals)) if test_vals else np.nan
        diff = test_mean - train_mean if np.isfinite(train_mean) and np.isfinite(test_mean) else np.nan
        stitched_val = stitched_oos_metrics.get(key, np.nan)

        rows.append({
            'Metric': label,
            'Train Fold (Mean)': train_mean,
            'Test Fold (Mean)': test_mean,
            'Fold Degradation': diff,
            'Stitched OOS (Full)': stitched_val,
            '_format': fmt,
        })

    return pd.DataFrame(rows)


def _build_parameter_stability(
    windows_df: pd.DataFrame,
    param_grid: dict[str, list[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize parameter changes across windows."""
    timeline_cols = ['Window', 'Test Start', 'Test End'] + list(param_grid)
    available_cols = [c for c in timeline_cols if c in windows_df.columns]
    param_timeline = windows_df[available_cols].copy()

    summary_rows: list[dict[str, Any]] = []
    for key in param_grid:
        if key not in windows_df.columns:
            continue
        series = windows_df[key]
        vals = list(series)
        switches = int((series != series.shift(1)).iloc[1:].sum()) if len(series) > 1 else 0
        mode_val = series.mode().iloc[0] if not series.empty else np.nan

        if _is_numeric_parameter(vals):
            numeric_vals = [float(v) for v in vals]
            summary_rows.append({
                'Parameter': key,
                'Type': 'Numeric',
                'Switches': switches,
                'Most Frequent': mode_val,
                'Mean': float(np.mean(numeric_vals)),
                'Std Dev': float(np.std(numeric_vals)),
                'Min': float(np.min(numeric_vals)),
                'Max': float(np.max(numeric_vals)),
            })
        else:
            summary_rows.append({
                'Parameter': key,
                'Type': 'Discrete / Categorical',
                'Switches': switches,
                'Most Frequent': mode_val,
                'Mean': np.nan,
                'Std Dev': np.nan,
                'Min': np.nan,
                'Max': np.nan,
            })

    stability_summary = pd.DataFrame(summary_rows)
    return param_timeline, stability_summary


def param_grid_defaults(param_grid: dict[str, list[Any]]) -> dict[str, Any]:
    """Extract default first candidate from each parameter grid entry."""
    return {k: v[0] for k, v in param_grid.items() if v}
