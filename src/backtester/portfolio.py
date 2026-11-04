from __future__ import annotations

import numpy as np
import pandas as pd


class Portfolio:
    """Simulate target-weight rebalancing with share-level accounting."""

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
        volatility_window: int = 20,
    ):
        numeric_settings = [initial_cash, commission, slippage, max_weight, max_gross_exposure]
        if not all(np.isfinite(value) for value in numeric_settings):
            raise ValueError('Portfolio settings must be finite numbers.')
        if initial_cash <= 0:
            raise ValueError('initial_cash must be positive.')
        if commission < 0 or slippage < 0:
            raise ValueError('commission and slippage cannot be negative.')
        if commission + slippage >= 1:
            raise ValueError('combined commission and slippage must be less than 1.0.')
        if max_weight <= 0 or max_gross_exposure <= 0:
            raise ValueError('max_weight and max_gross_exposure must be positive.')
        if volatility_target is not None and volatility_target <= 0:
            raise ValueError('volatility_target must be positive when provided.')
        if volatility_window < 2:
            raise ValueError('volatility_window must be at least 2.')
        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.max_weight = float(max_weight)
        self.max_gross_exposure = float(max_gross_exposure)
        self.rebalance_frequency = rebalance_frequency
        self.long_only = bool(long_only)
        self.volatility_target = volatility_target
        self.volatility_window = int(volatility_window)

    def _apply_risk_controls(self, weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
        weights = weights.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        lower = 0.0 if self.long_only else -self.max_weight
        weights = weights.clip(lower=lower, upper=self.max_weight)

        gross = weights.abs().sum(axis=1).replace(0, np.nan)
        scale = (self.max_gross_exposure / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(scale, axis=0)

        if self.volatility_target:
            raw_returns = (weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
            realized_vol = raw_returns.rolling(self.volatility_window).std() * np.sqrt(252)
            vol_scale = (self.volatility_target / realized_vol).clip(upper=1.5).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            weights = weights.mul(vol_scale, axis=0)
            gross = weights.abs().sum(axis=1).replace(0, np.nan)
            scale = (self.max_gross_exposure / gross).clip(upper=1.0).fillna(1.0)
            weights = weights.mul(scale, axis=0)
        return weights

    def _apply_rebalance_frequency(self, weights: pd.DataFrame) -> pd.DataFrame:
        mask = self._rebalance_mask(weights.index)
        if mask.all():
            return weights
        rebalanced = weights.where(mask).ffill().fillna(0.0)
        return rebalanced

    def _rebalance_mask(self, index: pd.Index) -> pd.Series:
        freq = self.rebalance_frequency.upper()

        if freq in {'ONCE', 'INITIAL'}:
            mask = pd.Series(False, index=index)
            mask.iloc[0] = True
            return mask

        if freq in {'D', 'DAILY'}:
            return pd.Series(True, index=index)

        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError('Weekly and monthly rebalancing require a DatetimeIndex.')

        if freq in {'W', 'WEEKLY'}:
            periods = index.to_series().dt.to_period('W')
        elif freq in {'M', 'MONTHLY'}:
            periods = index.to_series().dt.to_period('M')
        else:
            raise ValueError('rebalance_frequency must be D, W, M, or ONCE.')

        return periods.ne(periods.shift(1))

    def simulate(self, prices: pd.DataFrame, target_weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            raise ValueError('prices must be a non-empty pandas DataFrame.')
        if not isinstance(target_weights, pd.DataFrame):
            raise TypeError('target_weights must be a pandas DataFrame.')
        if prices.index.has_duplicates or prices.columns.has_duplicates:
            raise ValueError('prices index and columns must be unique.')
        if not prices.index.is_monotonic_increasing:
            raise ValueError('prices index must be sorted in increasing order.')

        prices = prices.copy().apply(pd.to_numeric, errors='coerce')
        prices = prices.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        if prices.empty:
            raise ValueError('prices contain no complete, usable rows.')
        if (prices <= 0).any().any():
            raise ValueError('prices must be strictly positive.')

        asset_returns = prices.pct_change().fillna(0.0)
        # Require an exact asset match so exposure statistics cannot omit signals.
        if target_weights.index.has_duplicates or target_weights.columns.has_duplicates:
            raise ValueError('target_weights index and columns must be unique.')
        if set(target_weights.columns) != set(prices.columns):
            raise ValueError('target_weights columns must exactly match prices columns.')
        if not prices.index.isin(target_weights.index).all():
            raise ValueError('target_weights must contain every usable price date.')
        weights = target_weights.reindex(index=prices.index, columns=prices.columns)
        weights = weights.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        weights = self._apply_risk_controls(weights, asset_returns)

        # Apply the rebalance schedule
        rebalance_mask = self._rebalance_mask(prices.index)
        weights = self._apply_rebalance_frequency(weights)

        cost_rate = self.commission + self.slippage
        equity_values = []
        portfolio_returns_before_costs = []
        portfolio_returns = []
        turnover = []
        trading_costs = []
        share_history = []
        previous_shares = pd.Series(0.0, index=prices.columns)
        cash = self.initial_cash

        for date in prices.index:
            current_prices = prices.loc[date]
            current_asset_value = float((previous_shares * current_prices).sum())
            pre_cost_equity = cash + current_asset_value
            if not np.isfinite(pre_cost_equity) or pre_cost_equity <= 0:
                raise ValueError(f'Portfolio lost 100% or more before trading on {date}; reduce leverage or trading costs.')

            previous_equity = self.initial_cash if not equity_values else equity_values[-1]
            before_cost_return = pre_cost_equity / previous_equity - 1.0
            if rebalance_mask.loc[date]:
                target_values = weights.loc[date] * pre_cost_equity
                target_shares = target_values / current_prices
                traded_notional = float(((target_shares - previous_shares).abs() * current_prices).sum())
                cost_dollars = traded_notional * cost_rate
                cash = pre_cost_equity - float(target_values.sum()) - cost_dollars
                previous_shares = target_shares
            else:
                traded_notional = 0.0
                cost_dollars = 0.0
                cash = pre_cost_equity - current_asset_value

            end_equity = cash + float((previous_shares * current_prices).sum())
            if not np.isfinite(end_equity) or end_equity <= 0:
                raise ValueError(f'Portfolio lost 100% or more on {date}; reduce leverage or trading costs.')
            cost_return = cost_dollars / pre_cost_equity
            equity_values.append(end_equity)
            portfolio_returns_before_costs.append(before_cost_return)
            portfolio_returns.append(end_equity / previous_equity - 1.0)
            turnover.append(traded_notional / pre_cost_equity)
            trading_costs.append(cost_return)
            share_history.append(previous_shares.copy())

        equity = pd.Series(equity_values, index=prices.index, dtype=float)
        portfolio_returns_before_costs = pd.Series(portfolio_returns_before_costs, index=prices.index, dtype=float)
        portfolio_returns = pd.Series(portfolio_returns, index=prices.index, dtype=float)
        turnover = pd.Series(turnover, index=prices.index, dtype=float)
        trading_costs = pd.Series(trading_costs, index=prices.index, dtype=float)
        shares = pd.DataFrame(share_history, index=prices.index, columns=prices.columns)
        trades = shares.diff().fillna(shares)
        trade_log = trades.stack().reset_index()
        trade_log.columns = ['Date', 'Ticker', 'Shares']
        price_frame = prices.copy()
        price_frame.index.name = 'Date'
        price_frame.columns.name = 'Ticker'
        price_lookup = price_frame.stack().rename('Price')
        trade_log = trade_log.merge(price_lookup.reset_index(), on=['Date', 'Ticker'], how='left')
        trade_log['Side'] = np.where(trade_log['Shares'] > 0, 'BUY', 'SELL')
        trade_log['Notional'] = trade_log['Shares'] * trade_log['Price']
        trade_log['Estimated Cost'] = trade_log['Notional'].abs() * (self.commission + self.slippage)
        trade_log = trade_log[trade_log['Shares'].abs() > 1e-8]

        history = pd.DataFrame({
            'Equity': equity,
            'Portfolio Return': portfolio_returns,
            'Return Before Costs': portfolio_returns_before_costs,
            'Turnover': turnover,
            'Trading Cost': trading_costs,
            'Cash Weight': 1 - weights.sum(axis=1),
            'Gross Exposure': weights.abs().sum(axis=1),
        })
        for col in weights.columns:
            history[f'Weight {col}'] = weights[col]
        return history, trade_log
