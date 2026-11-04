from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from backtester.engine import BacktestEngine
from backtester.optimizer import (
    WalkForwardResult,
    grid_search,
    optimize_parameters,
    walk_forward_search,
    walk_forward_validate,
)
from backtester.reporting import export_report
from data.loader import load_prices
from strategies.buy_hold import BuyAndHold
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi import RSIStrategy
from strategies.macd import MACDTrendStrategy
from strategies.bollinger import BollingerMeanReversion
from strategies.donchian import DonchianBreakout

st.set_page_config(
    page_title='Quant Strategy Lab',
    layout='wide',
    initial_sidebar_state='expanded',
)

APP_CSS = """
<style>
:root {
    --bg-app: #080d16;
    --surface: #111a29;
    --surface-raised: #172235;
    --border-subtle: #26354a;
    --border-strong: #344760;
    --text-primary: #edf3fa;
    --text-secondary: #a5b4c7;
    --text-muted: #74859b;
    --accent-blue: #38bdf8;
    --gain-green: #34d399;
    --loss-red: #fb7185;
}

/* Main application */
.stApp {
    background: var(--bg-app);
    color: var(--text-primary);
}

.block-container {
    padding: 1.8rem 2rem 2rem;
    max-width: 1600px;
}

/* Typography */
body, .stMarkdown, p, span, div {
    font-variant-numeric: tabular-nums;
}

h1, h2, h3 {
    color: var(--text-primary);
    letter-spacing: -0.03em;
}

/* Application header */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    padding: 12px 0 18px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 20px;
}

.header-title {
    font-size: 23px;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.8px;
}

.header-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 11px;
    color: var(--text-muted);
}

/* Context information */
.context-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    padding: 13px 16px;
    margin-bottom: 14px;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    color: var(--text-secondary);
    font-size: 11px;
}

.context-label {
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .06em;
}

.context-project {
    color: var(--accent-blue);
    font-weight: 700;
}

.context-value {
    color: var(--text-primary);
}

.context-sep {
    color: var(--border-strong);
}

/* Performance KPI cards */
.summary-row {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 18px;
    background: transparent;
    border: none;
}

.summary-cell {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
    padding: 17px 16px;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    transition: border-color .2s ease;
}

.summary-cell:hover {
    border-color: var(--accent-blue);
}

.summary-cell:last-child {
    border-right: 1px solid var(--border-subtle);
}

.summary-label {
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
}

.summary-value {
    font-family: 'SF Mono', Consolas, monospace;
    font-size: 21px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.summary-value.pos {
    color: var(--gain-green);
}

.summary-value.neg {
    color: var(--loss-red);
}

/* Benchmark strip */
.benchmark-substrip {
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    padding: 12px 16px;
    margin-bottom: 18px;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    font-size: 11px;
    color: var(--text-muted);
}

.bench-tag {
    font-weight: 700;
    color: var(--accent-blue);
    text-transform: uppercase;
}

.bench-item b {
    color: var(--text-primary);
    font-weight: 600;
}

.bench-item b.pos {
    color: var(--gain-green);
}

.bench-item b.neg {
    color: var(--loss-red);
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #0c1421 !important;
    border-right: 1px solid var(--border-subtle);
}

div[data-testid="stSidebar"] > div:first-child {
    width: 340px;
}

div[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

.sidebar-group-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--accent-blue);
    margin-top: 8px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: .08em;
}

div[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    background: var(--surface);
}

div[data-testid="stSidebar"] [data-testid="stExpander"] details summary {
    padding: 12px 14px;
    font-size: 12px;
    color: var(--text-primary);
    font-weight: 600;
}

div[data-testid="stSidebar"] [data-testid="stExpander"] details > div {
    padding: 4px 14px 12px;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: var(--surface-raised) !important;
    border-radius: 8px !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 5px;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    border-radius: 9px;
    font-weight: 600;
    transition: all .2s ease;
}

div[data-testid="stButton"] button[kind="primary"] {
    background: #087eae;
    border: 1px solid #159acb;
    color: white;
    min-height: 2.8rem;
    font-size: 13px;
    border-radius: 10px;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #0b9bd0;
    border-color: #38bdf8;
}

/* Navigation tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 18px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 12px;
    font-weight: 600;
    padding: 11px 16px;
    color: var(--text-muted);
}

.stTabs [aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
}

/* Section headings */
.section-label {
    color: var(--accent-blue);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    margin: 12px 0 5px;
}

.section-note {
    color: var(--text-muted);
    font-size: 12px;
    margin-bottom: 12px;
}

/* General containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
}

/* Responsive KPI layout */
@media (max-width: 1100px) {
    .summary-row {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}

@media (max-width: 650px) {
    .block-container {
        padding: 1rem;
    }

    .summary-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .summary-value {
        font-size: 17px;
    }

    .app-header {
        align-items: flex-start;
        flex-direction: column;
    }
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


def parse_tickers(raw: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in raw.replace('\n', ',').split(','):
        ticker = token.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def safe_top_n_slider(label: str, asset_count: int, default: int = 2) -> int:
    max_assets = max(1, int(asset_count))
    return st.slider(label, 1, max_assets, min(default, max_assets))


def get_strategy_param_grid(name: str) -> tuple[Any, dict[str, list[Any]]]:
    """Return the strategy factory and validation parameter candidates."""
    if name == 'Moving Average Crossover':
        return MovingAverageCrossover, {'short_window': [10, 20, 30], 'long_window': [50, 100, 150]}
    elif name == 'RSI Mean Reversion':
        return RSIStrategy, {'window': [10, 14, 21], 'oversold': [25, 30, 35], 'overbought': [65, 70, 75]}
    elif name == 'MACD Trend Following':
        return MACDTrendStrategy, {'fast': [8, 12], 'slow': [21, 26], 'signal': [7, 9]}
    elif name == 'Bollinger Mean Reversion':
        return BollingerMeanReversion, {'window': [15, 20, 30], 'num_std': [1.5, 2.0, 2.5]}
    return MovingAverageCrossover, {'short_window': [10, 20, 30], 'long_window': [50, 100, 150]}


def build_monthly_return_table(equity: pd.Series) -> pd.DataFrame:
    if len(equity) < 20:
        return pd.DataFrame()
    monthly = equity.resample('ME').last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values,
    })
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    month_map = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
    }
    pivot = pivot.rename(columns=month_map)
    for col in month_map.values():
        if col not in pivot.columns:
            pivot[col] = np.nan
    ordered_months = list(month_map.values())

    year_totals = []
    for yr in pivot.index:
        yr_equity = equity.loc[equity.index.year == yr]
        if len(yr_equity) >= 2 and yr_equity.iloc[0] > 0:
            year_totals.append((yr_equity.iloc[-1] / yr_equity.iloc[0]) - 1.0)
        else:
            year_totals.append(np.nan)
    pivot['YTD'] = year_totals
    return pivot[ordered_months + ['YTD']]

def build_comparison_strategies(long_only: bool = True) -> dict:
    """Create the core strategies with consistent default parameters."""

    return {
        'Moving Average Crossover': MovingAverageCrossover(
            short_window=20,
            long_window=50,
            long_only=long_only,
        ),

        'RSI Mean Reversion': RSIStrategy(
            window=14,
            oversold=30,
            overbought=70,
        ),

        'MACD Trend Following': MACDTrendStrategy(
            fast=12,
            slow=26,
            signal=9,
            long_only=long_only,
        ),

        'Bollinger Mean Reversion': BollingerMeanReversion(
            window=20,
            num_std=2.0,
        ),

        'Donchian Channel Breakout': DonchianBreakout(
            entry_window=55,
            exit_window=20,
            long_only=long_only,
        ),

        'Buy and Hold': BuyAndHold(),
    }

if st.query_params.get('preload_validation') == '1' and 'result' not in st.session_state:
    demo_csv = ROOT / 'reports' / 'market_data_2019_2026.csv'
    demo_pkl = ROOT / 'reports' / 'validation_demo_result.pkl'
    if demo_csv.exists():
        prices = load_prices(['AAPL', 'MSFT', 'NVDA', 'SPY'], source='csv', csv_path=demo_csv)
        benchmark = load_prices(['SPY'], source='csv', csv_path=demo_csv)
        engine_kwargs = dict(
            initial_cash=100000,
            commission=0.0010,
            slippage=0.0005,
            max_weight=0.50,
            max_gross_exposure=1.0,
            rebalance_frequency='D',
            long_only=True,
            volatility_target=None,
            risk_free_rate=0.0,
        )
        engine = BacktestEngine(**engine_kwargs)
        strat_name = 'Moving Average Crossover'
        strategy = MovingAverageCrossover(short_window=20, long_window=50)
        result = engine.run(prices, strategy, benchmark=benchmark)
        st.session_state['result'] = result
        st.session_state['prices'] = prices
        st.session_state['benchmark'] = benchmark
        st.session_state['engine_kwargs'] = engine_kwargs
        st.session_state['active_strategy'] = strat_name
        st.session_state['strategy_name'] = strat_name
        st.session_state['validation_strategy'] = strat_name
        st.session_state['ticker_list'] = ['AAPL', 'MSFT', 'NVDA', 'SPY']
        st.session_state['benchmark_ticker'] = 'SPY'
        st.session_state['initial_cash'] = 100000

        if demo_pkl.exists():
            import pickle
            with open(demo_pkl, 'rb') as f:
                st.session_state['validation_result'] = pickle.load(f)


STRATEGY_OPTIONS = [
    'Moving Average Crossover',
    'RSI Mean Reversion',
    'MACD Trend Following',
    'Bollinger Mean Reversion',
    'Donchian Channel Breakout',
    'Buy and Hold',
]

with st.sidebar:
    st.markdown('<div class="sidebar-group-title">Research setup</div>', unsafe_allow_html=True)
    with st.expander('Strategy', expanded=True):
        active_strat_default = st.session_state.get('active_strategy', st.session_state.get('strategy_name', 'Moving Average Crossover'))
        strat_idx = STRATEGY_OPTIONS.index(active_strat_default) if active_strat_default in STRATEGY_OPTIONS else 0
        strategy_name = st.selectbox(
            'Strategy',
            STRATEGY_OPTIONS,
            index=strat_idx,
            label_visibility='collapsed',
        )

    if 'active_strategy' in st.session_state and st.session_state['active_strategy'] != strategy_name:
        st.session_state['last_changed_strategy'] = strategy_name
        st.session_state.pop('result', None)
        st.session_state.pop('validation_result', None)
        st.session_state.pop('active_strategy', None)
        st.session_state.pop('validation_strategy', None)

    input_errors: list[str] = []
    with st.expander('Universe & Benchmark', expanded=True):
        tickers = st.text_input('Universe symbols', value='AAPL,MSFT,NVDA,SPY', help='Comma-separated symbols')
        ticker_list = parse_tickers(tickers)
        benchmark_ticker = st.text_input('Benchmark symbol', value='SPY')
        c_d1, c_d2 = st.columns(2)
        start = c_d1.date_input('Start date', value=pd.to_datetime('2019-01-01'))
        end = c_d2.date_input('End date', value=pd.Timestamp.today())
        data_source = st.radio('Price data source', ['Yahoo Finance', 'CSV upload'], horizontal=True)
        csv_file = st.file_uploader('Upload price CSV', type=['csv'], help='Wide CSV: Date,AAPL,MSFT... or long CSV: Date,Ticker,Close/Adj Close.') if data_source == 'CSV upload' else None

        if not ticker_list:
            input_errors.append('Enter at least one ticker.')
        if pd.to_datetime(end) <= pd.to_datetime(start):
            input_errors.append('End date must be after start date.')
        if data_source == 'CSV upload' and csv_file is None:
            input_errors.append('Upload a CSV file or switch back to Yahoo Finance.')

    with st.expander('Capital & Execution', expanded=False):
        c_c1, c_c2 = st.columns(2)
        initial_cash = c_c1.number_input('Initial cash ($)', min_value=1_000, value=100_000, step=10_000)
        rebalance_frequency = c_c2.selectbox('Rebalance', ['D', 'W', 'M'], format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x])
        c_fee1, c_fee2 = st.columns(2)
        commission = c_fee1.number_input('Commission rate', min_value=0.0, value=0.0010, step=0.0005, format='%.4f')
        slippage = c_fee2.number_input('Slippage rate', min_value=0.0, value=0.0005, step=0.0005, format='%.4f')

    with st.expander('Risk Constraints', expanded=False):
        c_r1, c_r2 = st.columns(2)
        long_only = c_r1.checkbox('Long only', value=True)
        use_vol_target = c_r2.checkbox('Vol targeting', value=False)
        c_w1, c_w2 = st.columns(2)
        max_weight = c_w1.slider('Max asset wt', 0.05, 1.0, 0.50, 0.05)
        max_gross = c_w2.slider('Max gross exp', 0.1, 2.0, 1.0, 0.1)
        vol_target = st.slider('Annual vol target', 0.05, 0.50, 0.15, 0.01) if use_vol_target else None
        risk_free_rate = st.number_input('Risk-free rate', min_value=0.0, value=0.0, step=0.005, format='%.3f')

    with st.expander('Strategy Parameters', expanded=True):
        if strategy_name == 'Moving Average Crossover':
            short = st.slider('Short MA window', 5, 100, 20)
            long = st.slider('Long MA window', short + 1, 250, 50)
            strategy = MovingAverageCrossover(short_window=short, long_window=long, long_only=long_only)
        elif strategy_name == 'RSI Mean Reversion':
            window = st.slider('RSI window', 5, 40, 14)
            c_rsi1, c_rsi2 = st.columns(2)
            oversold = c_rsi1.slider('Oversold', 10, 45, 30)
            overbought = c_rsi2.slider('Overbought', 55, 90, 70)
            strategy = RSIStrategy(window=window, oversold=oversold, overbought=overbought)
        elif strategy_name == 'MACD Trend Following':
            c_m1, c_m2 = st.columns(2)
            fast = c_m1.slider('Fast EMA', 5, 20, 12)
            slow = c_m2.slider('Slow EMA', fast + 1, 60, 26)
            signal = st.slider('Signal EMA', 3, 20, 9)
            strategy = MACDTrendStrategy(fast=fast, slow=slow, signal=signal, long_only=long_only)
        elif strategy_name == 'Bollinger Mean Reversion':
            window = st.slider('Bollinger window', 10, 80, 20)
            num_std = st.slider('Standard deviations', 1.0, 3.5, 2.0, 0.25)
            strategy = BollingerMeanReversion(window=window, num_std=num_std)
        else:
            strategy = BuyAndHold()

    if input_errors:
        for error in input_errors:
            st.warning(error)

    st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)
    run = st.button('Run Backtest', type='primary', disabled=bool(input_errors), width='stretch')
    with st.expander('Optimizer & Validation', expanded=False):
        run_optimizer = st.checkbox('Run MA parameter optimizer')
        optimizer_mode = st.selectbox('Optimizer validation', ['Walk-forward', 'In-sample grid search']) if run_optimizer else 'Walk-forward'
        wf_sidebar_mode = st.selectbox('Window structure', ['rolling', 'expanding'], index=0, format_func=lambda x: x.capitalize()) if run_optimizer and optimizer_mode == 'Walk-forward' else 'rolling'


st.markdown(f"""
<div class="app-header">
    <div class="header-title">Quant Strategy Lab</div>
    <div class="header-meta">
        <span class="meta-item">Research output</span>
        <span class="meta-sep">·</span>
        <span class="meta-item">{data_source}</span>
    </div>
</div>
""", unsafe_allow_html=True)


if run:
    try:
        source = 'csv' if data_source == 'CSV upload' else 'yfinance'
        prices = load_prices(ticker_list, start=str(start), end=str(end), source=source, csv_path=csv_file)
        benchmark_symbols = parse_tickers(benchmark_ticker)
        benchmark = None
        if benchmark_symbols:
            try:
                if csv_file is not None and hasattr(csv_file, 'seek'):
                    csv_file.seek(0)
                benchmark = load_prices(benchmark_symbols[:1], start=str(start), end=str(end), source=source, csv_path=csv_file)
            except Exception as bench_exc:
                st.warning(f'Benchmark could not be loaded: {bench_exc}')

        engine_kwargs = dict(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            max_weight=max_weight,
            max_gross_exposure=max_gross,
            rebalance_frequency=(
                      'ONCE' if strategy_name == 'Buy and Hold'
                            else rebalance_frequency
        ),
            long_only=long_only,
            volatility_target=vol_target,
            risk_free_rate=risk_free_rate,
        )
        engine = BacktestEngine(**engine_kwargs)
        result = engine.run(prices, strategy, benchmark=benchmark)

        st.session_state['result'] = result
        st.session_state['prices'] = prices
        st.session_state['benchmark'] = benchmark
        st.session_state['engine_kwargs'] = engine_kwargs
        st.session_state['strategy_name'] = strategy_name
        st.session_state['active_strategy'] = strategy_name
        st.session_state['ticker_list'] = ticker_list
        st.session_state['benchmark_ticker'] = benchmark_ticker
        st.session_state['initial_cash'] = initial_cash
        st.session_state.pop('last_changed_strategy', None)

        if st.session_state.get('validation_strategy') != strategy_name:
            st.session_state.pop('validation_result', None)
            st.session_state.pop('validation_strategy', None)

        if run_optimizer and optimizer_mode == 'Walk-forward' and len(prices) >= 120:
            train_size_sb = min(252, len(prices) // 2)
            test_size_sb = min(63, max(20, (len(prices) - train_size_sb) // 2))
            opt_factory, opt_grid = get_strategy_param_grid(strategy_name)
            wf_res = walk_forward_validate(
                prices=prices,
                strategy_factory=opt_factory,
                param_grid=opt_grid,
                train_size=train_size_sb,
                test_size=test_size_sb,
                window_type=wf_sidebar_mode,
                engine_kwargs=engine_kwargs,
                benchmark=benchmark,
                objective='Sharpe Ratio',
                include_hindsight=True,
            )
            st.session_state['validation_result'] = wf_res
            st.session_state['validation_strategy'] = strategy_name
    except Exception as exc:
        st.error(f'Backtest execution failed: {exc}')


if 'result' in st.session_state:
    result = st.session_state['result']
    hist = result['history']
    metrics = result['metrics']
    trades = result['trades']
    signals = result['signals']
    prices = st.session_state.get('prices')
    benchmark = st.session_state.get('benchmark')
    engine_kwargs = st.session_state.get('engine_kwargs', {})

    tot_ret = metrics.get('Total Return', 0.0)
    cagr = metrics.get('Annual Return', 0.0)
    vol = metrics.get('Annual Volatility', 0.0)
    sharpe = metrics.get('Sharpe Ratio', 0.0)
    sortino = metrics.get('Sortino Ratio', 0.0)
    max_dd = metrics.get('Max Drawdown', 0.0)
    win_rate = metrics.get('Positive Day Rate', metrics.get('Win Rate', 0.0))
    trade_count = metrics.get('Trade Count', len(trades))

    tot_class = 'pos' if tot_ret > 0 else ('neg' if tot_ret < 0 else '')
    cagr_class = 'pos' if cagr > 0 else ('neg' if cagr < 0 else '')
    sharpe_class = 'pos' if sharpe > 1.0 else ('neg' if sharpe < 0 else '')

    # Show the period represented by the loaded history, not only the requested
    # controls, because a source may return fewer rows than requested.
    effective_start = hist.index.min().date()
    effective_end = hist.index.max().date()

    active_strat_display = st.session_state.get('active_strategy', strategy_name)
    active_tickers_display = st.session_state.get('ticker_list', ticker_list)
    active_bench_display = st.session_state.get('benchmark_ticker', benchmark_ticker)
    active_cash_display = st.session_state.get('initial_cash', initial_cash)

    st.markdown(f"""
    <div class="context-bar">
        <span class="context-project">Quant Strategy Lab</span><span class="context-sep">·</span>
        <span class="context-label">Strategy</span><span class="context-value">{active_strat_display}</span>
        <span class="context-sep">/</span>
        <span class="context-label">Universe</span><span class="context-value">{','.join(active_tickers_display)}</span>
        <span class="context-sep">/</span>
        <span class="context-label">Benchmark</span><span class="context-value">{active_bench_display or 'None'}</span>
        <span class="context-sep">/</span>
        <span class="context-label">Data period</span><span class="context-value">{effective_start} to {effective_end}</span>
        <span class="context-sep">/</span>
        <span class="context-label">Capital</span><span class="context-value">${active_cash_display:,.0f}</span>
    </div>
    <div class="summary-row">
        <div class="summary-cell"><span class="summary-label">Total return</span><span class="summary-value {tot_class}">{tot_ret:+.2%}</span></div>
        <div class="summary-cell"><span class="summary-label">CAGR</span><span class="summary-value {cagr_class}">{cagr:+.2%}</span></div>
        <div class="summary-cell"><span class="summary-label">Volatility</span><span class="summary-value">{vol:.2%}</span></div>
        <div class="summary-cell"><span class="summary-label">Sharpe</span><span class="summary-value {sharpe_class}">{sharpe:.2f}</span></div>
        <div class="summary-cell"><span class="summary-label">Max drawdown</span><span class="summary-value neg">{max_dd:.2%}</span></div>
        <div class="summary-cell"><span class="summary-label">Trades</span><span class="summary-value">{trade_count:,}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if 'Benchmark Total Return' in metrics:
        b_ret = metrics.get('Benchmark Total Return', 0.0)
        exc_ret = metrics.get('Excess Return', 0.0)
        alpha = metrics.get('Alpha', 0.0)
        beta = metrics.get('Beta', 0.0)
        te = metrics.get('Tracking Error', 0.0)
        ir = metrics.get('Information Ratio', 0.0)
        exc_class = 'pos' if exc_ret > 0 else ('neg' if exc_ret < 0 else '')
        alpha_class = 'pos' if alpha > 0 else ('neg' if alpha < 0 else '')

        st.markdown(f"""
        <div class="benchmark-substrip">
            <span class="bench-tag">Benchmark ({st.session_state.get('benchmark_ticker', 'SPY')})</span>
            <span class="bench-item">Return: <b>{b_ret:+.2%}</b></span>
            <span class="bench-item">Excess: <b class="{exc_class}">{exc_ret:+.2%}</b></span>
            <span class="bench-item">Alpha: <b class="{alpha_class}">{alpha:+.2%}</b></span>
            <span class="bench-item">Beta: <b>{beta:.2f}</b></span>
            <span class="bench-item">Tracking Error: <b>{te:.2%}</b></span>
            <span class="bench-item">Info Ratio: <b>{ir:.2f}</b></span>
        </div>
        """, unsafe_allow_html=True)

    tabs = st.tabs([
    'Backtest',
    'Validation',
    'Trades',
    'Risk',
    'Allocations',
    'Signals',
    'Optimizer',
    'Reports',
    'Strategy Comparison'
])

    with tabs[8]:
        st.markdown(
            '<div class="section-label">Compare trading strategies</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="section-note">Run all six strategies using the same market data and compare their performance.</p>',
            unsafe_allow_html=True
        )

        if st.button('Run Strategy Comparison', type='primary'):
            comparison_results = []
            comparison_strategies = build_comparison_strategies(
                long_only=engine_kwargs.get('long_only', True)
            )

            comparison_engine_kwargs = dict(engine_kwargs)

            # Use the selected rebalance frequency for all strategies.
            if comparison_engine_kwargs.get('rebalance_frequency') == 'ONCE':
                comparison_engine_kwargs['rebalance_frequency'] = 'D'

            progress = st.progress(0)
            status = st.empty()

            for i, (name, comparison_strategy) in enumerate(
                comparison_strategies.items()
            ):
                status.write(f'Running {name}...')

                try:
                    strategy_engine_kwargs = dict(comparison_engine_kwargs)

                    if name == 'Buy and Hold':
                         strategy_engine_kwargs['rebalance_frequency'] = 'ONCE'
                    
                    comparison_engine = BacktestEngine(
                         **strategy_engine_kwargs
                    )
                    comparison_result = comparison_engine.run(
                        prices,
                        comparison_strategy,
                        benchmark=benchmark
                    )

                    m = comparison_result['metrics']

                    comparison_results.append({
                        'Strategy': name,
                        'Total Return': m.get('Total Return', 0),
                        'Annual Return': m.get('Annual Return', 0),
                        'Annual Volatility': m.get('Annual Volatility', 0),
                        'Sharpe Ratio': m.get('Sharpe Ratio', 0),
                        'Sortino Ratio': m.get('Sortino Ratio', 0),
                        'Max Drawdown': m.get('Max Drawdown', 0),
                        'Trade Count': m.get(
                            'Trade Count',
                            len(comparison_result.get('trades', []))
                        )
                    })

                except Exception as exc:
                    st.warning(f'{name} could not run: {exc}')

                progress.progress((i + 1) / len(comparison_strategies))

            status.empty()
            progress.empty()

            if comparison_results:
                st.session_state['comparison_results'] = pd.DataFrame(
                    comparison_results
                )
                st.success('Strategy comparison completed.')

        if 'comparison_results' in st.session_state:
            comparison_df = st.session_state['comparison_results']

            st.markdown('#### Performance comparison')

            display_df = comparison_df.copy()

            percentage_columns = [
                'Total Return',
                'Annual Return',
                'Annual Volatility',
                'Max Drawdown'
            ]

            for column in percentage_columns:
                display_df[column] = display_df[column].map(
                    lambda x: f'{x:.2%}'
                )

            display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].map(
                lambda x: f'{x:.2f}'
            )
            display_df['Sortino Ratio'] = display_df['Sortino Ratio'].map(
                lambda x: f'{x:.2f}'
            )

            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True
            )
    with tabs[0]:
        st.markdown('<div class="section-label">Equity / benchmark / drawdown</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-note">Cumulative portfolio value with the selected benchmark rebased to the same starting capital.</p>', unsafe_allow_html=True)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            row_heights=[0.72, 0.28],
        )

        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Equity'],
                name='Strategy Equity',
                line=dict(color='#38bdf8', width=2),
                hovertemplate='%{x|%Y-%m-%d}<br>Strategy: $%{y:,.0f}<extra></extra>',
            ),
            row=1,
            col=1,
        )

        if 'Benchmark' in hist:
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Benchmark'],
                    name=f'Benchmark ({st.session_state.get("benchmark_ticker", "SPY")})',
                    line=dict(color='#94a3b8', width=1.5, dash='dot'),
                    hovertemplate='%{x|%Y-%m-%d}<br>Benchmark: $%{y:,.0f}<extra></extra>',
                ),
                row=1,
                col=1,
            )

        if not trades.empty and len(trades) <= 400:
            buys = trades[trades['Side'] == 'BUY']
            sells = trades[trades['Side'] == 'SELL']
            if not buys.empty:
                buy_points = hist['Equity'].reindex(buys['Date']).dropna()
                fig.add_trace(
                    go.Scatter(
                        x=buy_points.index,
                        y=buy_points.values,
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=5, color='#22c55e', line=dict(width=0)),
                        hoverinfo='skip',
                    ),
                    row=1,
                    col=1,
                )
            if not sells.empty:
                sell_points = hist['Equity'].reindex(sells['Date']).dropna()
                fig.add_trace(
                    go.Scatter(
                        x=sell_points.index,
                        y=sell_points.values,
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=5, color='#ef4444', line=dict(width=0)),
                        hoverinfo='skip',
                    ),
                    row=1,
                    col=1,
                )

        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Drawdown'],
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.15)',
                line=dict(color='#ef4444', width=1.2),
                hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>',
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=525,
            margin=dict(l=48, r=18, t=16, b=18),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='left',
                x=0.01,
                font=dict(size=11, color='#94a3b8'),
            ),
            font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace', size=11, color='#94a3b8'),
        )
        fig.update_yaxes(
            row=1, col=1, title='Portfolio ($)', showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False, tickformat='$,.0f'
        )
        fig.update_yaxes(
            row=2, col=1, title='Drawdown', showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False, tickformat='.0%'
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False)

        st.plotly_chart(fig, width='stretch', config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

        col_c1, col_c2 = st.columns([1.1, 0.9])
        with col_c1:
            st.markdown('<div class="section-label">Exposure</div>', unsafe_allow_html=True)
            fig_exp = px.line(
                hist,
                y=['Gross Exposure', 'Cash Weight'],
                color_discrete_map={'Gross Exposure': '#38bdf8', 'Cash Weight': '#64748b'},
            )
            fig_exp.update_layout(
                height=240,
                margin=dict(l=42, r=15, t=8, b=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10.5, color='#94a3b8'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            )
            fig_exp.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickformat='.0%')
            fig_exp.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
            st.plotly_chart(fig_exp, width='stretch', config={'displaylogo': False})

        with col_c2:
            st.markdown('<div class="section-label">Return distribution</div>', unsafe_allow_html=True)
            fig_dist = px.histogram(
                hist,
                x='Portfolio Return',
                nbins=50,
                color_discrete_sequence=['#38bdf8'],
            )
            fig_dist.update_layout(
                height=240,
                margin=dict(l=42, r=15, t=8, b=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10.5, color='#94a3b8'),
                showlegend=False,
            )
            fig_dist.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickformat='.1%')
            fig_dist.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
            st.plotly_chart(fig_dist, width='stretch', config={'displaylogo': False})

    with tabs[1]:
        st.markdown('<div class="section-label">Walk-forward validation</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">Out-of-sample evaluation across rolling or expanding windows. '
            'Parameters are selected on each training window and held fixed for the following test period while capital carries forward.</p>',
            unsafe_allow_html=True
        )

        active_val_strategy = st.session_state.get('active_strategy', strategy_name)
        wf_res: WalkForwardResult | None = st.session_state.get('validation_result')
        if wf_res is not None and st.session_state.get('validation_strategy') not in (None, active_val_strategy):
            st.session_state.pop('validation_result', None)
            wf_res = None

        with st.expander('Validation settings', expanded=(wf_res is None)):
            c_wf1, c_wf2, c_wf3, c_wf4 = st.columns(4)
            n_bars = len(prices) if prices is not None else 500
            max_train = max(20, n_bars - 10)
            default_train = min(252, max(20, n_bars // 2))
            default_train = min(default_train, max_train)

            max_test = max(10, n_bars - default_train)
            default_test = min(63, max(10, (n_bars - default_train) // 2))
            default_test = min(default_test, max_test)

            wf_window_type = c_wf1.selectbox('Window structure', ['rolling', 'expanding'], index=0, format_func=lambda x: x.capitalize(), key='wf_tab_struct')
            wf_train_size = c_wf2.number_input('Training window (bars)', min_value=20, max_value=max_train, value=default_train, step=21, key='wf_tab_train')
            wf_test_size = c_wf3.number_input('Test window (bars)', min_value=10, max_value=max_test, value=default_test, step=21, key='wf_tab_test')
            wf_objective = c_wf4.selectbox('Optimization objective', ['Sharpe Ratio', 'Sortino Ratio', 'Total Return', 'Annual Return'], index=0, key='wf_tab_obj')

            strategy_factory, default_grid = get_strategy_param_grid(active_val_strategy)
            st.caption(f'Strategy: **{active_val_strategy}**. Candidates: `{default_grid}`')

            if st.button('Run Walk-Forward Validation', type='primary', width='stretch', key='wf_run_btn'):
                prog_bar = st.progress(0.0)
                status_text = st.empty()

                def update_prog(curr, total, msg):
                    prog_bar.progress(curr / total)
                    status_text.caption(f'{msg} ({curr}/{total})')

                try:
                    wf_res = walk_forward_validate(
                        prices=prices,
                        strategy_factory=strategy_factory,
                        param_grid=default_grid,
                        train_size=int(wf_train_size),
                        test_size=int(wf_test_size),
                        window_type=wf_window_type,
                        engine_kwargs=engine_kwargs,
                        benchmark=benchmark,
                        objective=wf_objective,
                        include_hindsight=True,
                        progress_callback=update_prog,
                    )
                    st.session_state['validation_result'] = wf_res
                    st.session_state['validation_strategy'] = active_val_strategy
                    prog_bar.empty()
                    status_text.empty()
                except Exception as wf_exc:
                    st.error(f'Walk-forward validation failed: {wf_exc}')

        if wf_res is not None:
            oos_m = wf_res.oos_metrics
            oos_ret = oos_m.get('Total Return', 0.0)
            oos_cagr = oos_m.get('Annual Return', 0.0)
            oos_vol = oos_m.get('Annual Volatility', 0.0)
            oos_sharpe = oos_m.get('Sharpe Ratio', 0.0)
            oos_dd = oos_m.get('Max Drawdown', 0.0)
            oos_trades_count = oos_m.get('Trade Count', len(wf_res.trades))

            oos_ret_cls = 'pos' if oos_ret > 0 else ('neg' if oos_ret < 0 else '')
            oos_cagr_cls = 'pos' if oos_cagr > 0 else ('neg' if oos_cagr < 0 else '')
            oos_sharpe_cls = 'pos' if oos_sharpe > 1.0 else ('neg' if oos_sharpe < 0 else '')

            st.markdown(f"""
            <div class="summary-row">
                <div class="summary-cell"><span class="summary-label">Stitched OOS return</span><span class="summary-value {oos_ret_cls}">{oos_ret:+.2%}</span></div>
                <div class="summary-cell"><span class="summary-label">Stitched CAGR</span><span class="summary-value {oos_cagr_cls}">{oos_cagr:+.2%}</span></div>
                <div class="summary-cell"><span class="summary-label">OOS Volatility</span><span class="summary-value">{oos_vol:.2%}</span></div>
                <div class="summary-cell"><span class="summary-label">OOS Sharpe</span><span class="summary-value {oos_sharpe_cls}">{oos_sharpe:.2f}</span></div>
                <div class="summary-cell"><span class="summary-label">OOS Max drawdown</span><span class="summary-value neg">{oos_dd:.2%}</span></div>
                <div class="summary-cell"><span class="summary-label">OOS Trades</span><span class="summary-value">{oos_trades_count:,}</span></div>
            </div>
            """, unsafe_allow_html=True)

            c_chk1, c_chk2 = st.columns([1, 1])
            show_hindsight = c_chk1.checkbox(
                'Show full-sample reference curve',
                value=False,
                key='wf_hindsight_chk',
                help='Uses parameters selected on the full dataset; it is a reference, not an out-of-sample result.'
            )
            show_folds = c_chk2.checkbox('Show fold boundary markers', value=True, key='wf_folds_chk')

            fig_wf = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.035,
                row_heights=[0.72, 0.28],
            )

            fig_wf.add_trace(
                go.Scatter(
                    x=wf_res.oos_history.index,
                    y=wf_res.oos_history['Equity'],
                    name='Stitched OOS Strategy Equity',
                    line=dict(color='#38bdf8', width=2.5),
                    hovertemplate='%{x|%Y-%m-%d}<br>Stitched OOS: $%{y:,.0f}<extra></extra>',
                ),
                row=1,
                col=1,
            )

            if 'Benchmark' in wf_res.oos_history:
                fig_wf.add_trace(
                    go.Scatter(
                        x=wf_res.oos_history.index,
                        y=wf_res.oos_history['Benchmark'],
                        name=f'Benchmark ({st.session_state.get("benchmark_ticker", "SPY")})',
                        line=dict(color='#94a3b8', width=1.5, dash='dot'),
                        hovertemplate='%{x|%Y-%m-%d}<br>Benchmark: $%{y:,.0f}<extra></extra>',
                    ),
                    row=1,
                    col=1,
                )

            if show_hindsight and wf_res.hindsight_history is not None:
                fig_wf.add_trace(
                    go.Scatter(
                        x=wf_res.hindsight_history.index,
                        y=wf_res.hindsight_history['Equity'],
                        name='Hindsight Reference (Full-Sample Overfit)',
                        line=dict(color='#a855f7', width=1.3, dash='dash'),
                        hovertemplate='%{x|%Y-%m-%d}<br>Hindsight (Overfit): $%{y:,.0f}<extra></extra>',
                    ),
                    row=1,
                    col=1,
                )

            fig_wf.add_trace(
                go.Scatter(
                    x=wf_res.oos_history.index,
                    y=wf_res.oos_history['Drawdown'],
                    name='OOS Drawdown',
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.15)',
                    line=dict(color='#ef4444', width=1.2),
                    hovertemplate='%{x|%Y-%m-%d}<br>OOS Drawdown: %{y:.2%}<extra></extra>',
                ),
                row=2,
                col=1,
            )

            if show_folds:
                for _, w_row in wf_res.windows.iterrows():
                    fig_wf.add_vline(
                        x=w_row['Test Start'],
                        line_width=1,
                        line_dash='dash',
                        line_color='rgba(255, 255, 255, 0.15)',
                        row=1,
                        col=1,
                    )

            fig_wf.update_layout(
                height=520,
                margin=dict(l=48, r=18, t=16, b=18),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='left',
                    x=0.01,
                    font=dict(size=11, color='#94a3b8'),
                ),
                font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace', size=11, color='#94a3b8'),
            )
            fig_wf.update_yaxes(
                row=1, col=1, title='Portfolio ($)', showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False, tickformat='$,.0f'
            )
            fig_wf.update_yaxes(
                row=2, col=1, title='Drawdown', showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False, tickformat='.0%'
            )
            fig_wf.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False)

            st.plotly_chart(fig_wf, width='stretch', config={'displaylogo': False})

            st.markdown('<div class="section-label">Walk-Forward Window Results</div>', unsafe_allow_html=True)
            st.markdown('<p class="section-note">Per-fold parameter selection on training data and continuous out-of-sample evaluation on the subsequent unseen test period.</p>', unsafe_allow_html=True)

            disp_windows = wf_res.windows[[
                'Window', 'Train Period', 'Test Period', 'Selected Parameters',
                'IS Score', 'OOS Return', 'OOS Sharpe', 'OOS Drawdown', 'Trades'
            ]].copy()

            st.dataframe(
                disp_windows,
                width='stretch',
                hide_index=True,
                column_config={
                    'Window': st.column_config.NumberColumn('Window', format='%d'),
                    'Train Period': st.column_config.TextColumn('Train Period'),
                    'Test Period': st.column_config.TextColumn('Test Period'),
                    'Selected Parameters': st.column_config.TextColumn('Selected Parameters'),
                    'IS Score': st.column_config.NumberColumn('IS Score', format='%.2f'),
                    'OOS Return': st.column_config.NumberColumn('OOS Return', format='%+.2%'),
                    'OOS Sharpe': st.column_config.NumberColumn('OOS Sharpe', format='%.2f'),
                    'OOS Drawdown': st.column_config.NumberColumn('OOS Drawdown', format='%.2%'),
                    'Trades': st.column_config.NumberColumn('Trades', format='%d'),
                }
            )

            c_cmp, c_stab = st.columns([1.1, 0.9])

            with c_cmp:
                st.markdown('<div class="section-label">In-sample vs out-of-sample metrics</div>', unsafe_allow_html=True)
                st.markdown('<p class="section-note">Train-fold averages are shown alongside test-fold averages and the stitched out-of-sample result.</p>', unsafe_allow_html=True)

                is_oos_display = wf_res.is_vs_oos.copy()
                formatted_rows = []
                for _, r in is_oos_display.iterrows():
                    fmt = r['_format']
                    def _safe_fmt(val, f):
                        return f.format(val) if np.isfinite(val) else 'N/A'
                    formatted_rows.append({
                        'Metric': r['Metric'],
                        'Train Fold (Mean)': _safe_fmt(r['Train Fold (Mean)'], fmt),
                        'Test Fold (Mean)': _safe_fmt(r['Test Fold (Mean)'], fmt),
                        'Fold Degradation': _safe_fmt(r['Fold Degradation'], fmt),
                        'Stitched OOS (Full)': _safe_fmt(r['Stitched OOS (Full)'], fmt),
                    })
                st.dataframe(pd.DataFrame(formatted_rows), width='stretch', hide_index=True)

                train_sharpes = wf_res.windows['IS Score']
                test_sharpes = wf_res.windows['OOS Sharpe'].dropna()
                mean_train_s = train_sharpes.mean()
                mean_test_s = test_sharpes.mean() if not test_sharpes.empty else np.nan
                ratio = mean_test_s / mean_train_s if mean_train_s != 0 and np.isfinite(mean_test_s) else np.nan
                st.caption(f'**Overfitting Ratio**: Test Sharpe / Train Sharpe = **{ratio:.2f}**. Lower OOS performance confirms genuine parameter friction and market regime shifts on unseen data.')

            with c_stab:
                st.markdown('<div class="section-label">Parameter selection by window</div>', unsafe_allow_html=True)
                st.markdown('<p class="section-note">Shows how selected parameters change across validation windows.</p>', unsafe_allow_html=True)

                st.dataframe(
                    wf_res.stability_summary,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        'Parameter': st.column_config.TextColumn('Parameter'),
                        'Type': st.column_config.TextColumn('Type'),
                        'Switches': st.column_config.NumberColumn('Switches', format='%d'),
                        'Most Frequent': st.column_config.TextColumn('Mode'),
                        'Mean': st.column_config.NumberColumn('Mean', format='%.1f'),
                        'Std Dev': st.column_config.NumberColumn('Std Dev', format='%.2f'),
                        'Min': st.column_config.NumberColumn('Min', format='%.1f'),
                        'Max': st.column_config.NumberColumn('Max', format='%.1f'),
                    }
                )

                numeric_keys = [
                    col for col in wf_res.parameter_stability.columns
                    if col not in {'Window', 'Test Start', 'Test End'}
                    and pd.api.types.is_numeric_dtype(wf_res.parameter_stability[col])
                ]
                if numeric_keys:
                    fig_ps = px.line(
                        wf_res.parameter_stability,
                        x='Window',
                        y=numeric_keys,
                        markers=True,
                        title='Parameter Trajectory by Window',
                    )
                    fig_ps.update_layout(
                        height=220,
                        margin=dict(l=40, r=15, t=28, b=12),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=10.5, color='#94a3b8'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                    )
                    fig_ps.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', dtick=1)
                    fig_ps.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
                    st.plotly_chart(fig_ps, width='stretch', config={'displaylogo': False})
        else:
            st.info('Run walk-forward validation above to populate this tab.')

    with tabs[2]:
        if trades.empty:
            st.info('No trades were generated during this backtest.')
        else:
            buy_count = (trades['Side'] == 'BUY').sum()
            st.markdown(f"""
            <div class="stat-strip">
                <span class="stat-item">Executions: <b>{len(trades):,}</b></span>
                <span class="stat-item">Notional Traded: <b>${trades["Notional"].abs().sum():,.0f}</b></span>
                <span class="stat-item">Estimated Costs: <b>${trades["Estimated Cost"].sum():,.2f}</b></span>
                <span class="stat-item">Buy / Sell: <b>{buy_count} / {len(trades) - buy_count}</b></span>
            </div>
            """, unsafe_allow_html=True)

            display_trades = trades.copy()
            display_trades['Date'] = pd.to_datetime(display_trades['Date']).dt.strftime('%Y-%m-%d')
            display_trades = display_trades.sort_values('Date', ascending=False)

            st.dataframe(
                display_trades,
                width='stretch',
                hide_index=True,
                column_config={
                    'Date': st.column_config.TextColumn('Execution Date'),
                    'Ticker': st.column_config.TextColumn('Symbol'),
                    'Side': st.column_config.TextColumn('Side'),
                    'Shares': st.column_config.NumberColumn('Quantity', format='%.2f'),
                    'Price': st.column_config.NumberColumn('Execution Price', format='$%.2f'),
                    'Notional': st.column_config.NumberColumn('Notional Value', format='$%.2f'),
                    'Estimated Cost': st.column_config.NumberColumn('Fee / Slippage', format='$%.2f'),
                },
            )

    with tabs[3]:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown('**Returns**')
            ret_metrics = {
                'Initial Value': f"${metrics.get('Start Value', 0.0):,.2f}",
                'Ending Value': f"${metrics.get('End Value', 0.0):,.2f}",
                'Total Return': f"{metrics.get('Total Return', 0.0):+.2%}",
                'CAGR': f"{metrics.get('Annual Return', 0.0):+.2%}",
                'Positive Day Rate': f"{metrics.get('Positive Day Rate', 0.0):.1%}",
                'Best Single Day': f"{metrics.get('Best Day', 0.0):+.2%}",
                'Worst Single Day': f"{metrics.get('Worst Day', 0.0):+.2%}",
                'Profit Factor': f"{metrics.get('Profit Factor', np.nan):.2f}",
                'Trade Count': f"{int(metrics.get('Trade Count', len(trades))):,}",
            }
            ret_df = pd.DataFrame({'Statistic': list(ret_metrics.keys()), 'Value': list(ret_metrics.values())})
            st.dataframe(ret_df, width='stretch', hide_index=True)

        with col_r2:
            st.markdown('**Risk & Volatility**')
            risk_stat = {
                'Annual Volatility': f"{metrics.get('Annual Volatility', 0.0):.2%}",
                'Sharpe Ratio': f"{metrics.get('Sharpe Ratio', 0.0):.2f}",
                'Sortino Ratio': f"{metrics.get('Sortino Ratio', 0.0):.2f}",
                'Max Drawdown': f"{metrics.get('Max Drawdown', 0.0):.2%}",
                'Max Drawdown Duration': f"{int(metrics.get('Max Drawdown Duration', 0))} trading days",
                'Calmar Ratio': f"{metrics.get('Calmar Ratio', np.nan):.2f}",
                'Ulcer Index': f"{metrics.get('Ulcer Index', 0.0):.4f}",
                'Daily VaR (95%)': f"{metrics.get('Daily VaR 95%', np.nan):.2%}",
                'Daily CVaR (95%)': f"{metrics.get('Daily CVaR 95%', np.nan):.2%}",
            }
            risk_df = pd.DataFrame({'Statistic': list(risk_stat.keys()), 'Value': list(risk_stat.values())})
            st.dataframe(risk_df, width='stretch', hide_index=True)

        st.markdown('**Monthly Returns**')
        monthly_table = build_monthly_return_table(hist['Equity'])
        if not monthly_table.empty:
            def color_returns(val):
                if pd.isna(val):
                    return 'color: #64748b;'
                if val > 0:
                    return 'color: #22c55e; background-color: rgba(34, 197, 94, 0.07); font-family: monospace;'
                elif val < 0:
                    return 'color: #ef4444; background-color: rgba(239, 68, 68, 0.07); font-family: monospace;'
                return 'color: #94a3b8; font-family: monospace;'

            styled_monthly = monthly_table.style.format('{:+.2%}', na_rep='-').map(color_returns)
            st.dataframe(styled_monthly, width='stretch')
        else:
            st.caption('Insufficient history for monthly breakdown.')

    with tabs[4]:
        weight_cols = [c for c in hist.columns if c.startswith('Weight ')]
        if weight_cols:
            fig_w = px.area(
                hist[weight_cols],
                title='Historical Allocations',
                labels={'value': 'Portfolio Weight', 'Date': 'Date', 'variable': 'Asset'},
            )
            fig_w.update_layout(
                height=390,
                margin=dict(l=45, r=15, t=28, b=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color='#94a3b8'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            )
            fig_w.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickformat='.0%')
            fig_w.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
            st.plotly_chart(fig_w, width='stretch')

            st.markdown('**Recent Weights**')
            recent_weights = hist[weight_cols].tail(100).copy()
            recent_weights.index = recent_weights.index.strftime('%Y-%m-%d')
            st.dataframe(
                recent_weights.style.format('{:.2%}'),
                width='stretch',
            )
        else:
            st.info('No weight time series available for this strategy.')

    with tabs[5]:
        signal_sample = signals.tail(252)
        if not signal_sample.empty:
            active_days = (signals.abs().sum(axis=1) > 0).mean()
            avg_turnover = hist['Turnover'].mean()
            cost_drag = hist['Trading Cost'].sum()

            st.markdown(f"""
            <div class="stat-strip">
                <span class="stat-item">Active Days: <b>{active_days:.1%}</b></span>
                <span class="stat-item">Avg Daily Turnover: <b>{avg_turnover:.2%}</b></span>
                <span class="stat-item">Total Cost Drag: <b>{cost_drag:.2%}</b></span>
            </div>
            """, unsafe_allow_html=True)

            fig_heat = px.imshow(
                signal_sample.T,
                aspect='auto',
                title='Signal Weights (Trailing 252 Days)',
                color_continuous_scale=[[0, '#0b0f17'], [0.5, '#1e293b'], [1, '#0284c7']],
            )
            fig_heat.update_layout(
                height=250,
                margin=dict(l=45, r=15, t=28, b=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10.5, color='#94a3b8'),
            )
            st.plotly_chart(fig_heat, width='stretch')

            avg_names = signals.abs().mean().sort_values(ascending=False).reset_index()
            avg_names.columns = ['Ticker', 'Average Weight']
            avg_names['Average Weight'] = avg_names['Average Weight'].map(lambda x: f'{x:.2%}')
            st.dataframe(avg_names, width='stretch', hide_index=True)

        else:
            st.info('No signal history available.')

    with tabs[6]:
        if run_optimizer:
            parameter_grid = {'short_window': [10, 20, 30], 'long_window': [50, 100, 150]}
            if optimizer_mode == 'Walk-forward':
                st.info('Walk-forward parameter validation is actively analyzed in the primary **Validation** tab above.')
                if prices is not None and len(prices) < 120:
                    st.warning('Walk-forward optimization requires at least 120 rows of price history.')
                elif prices is not None:
                    train_size = min(252, len(prices) // 2)
                    test_size = min(63, max(20, (len(prices) - train_size) // 2))
                    opt = walk_forward_search(
                        prices,
                        MovingAverageCrossover,
                        parameter_grid,
                        train_size=train_size,
                        test_size=test_size,
                        engine_kwargs=engine_kwargs,
                        benchmark=benchmark,
                        objective='Sharpe Ratio',
                    )
                    st.caption(f'Rolling walk-forward: {train_size} in-sample training rows, {test_size} out-of-sample test rows per fold.')
                    st.dataframe(opt, width='stretch')
                    if not opt.empty and 'Test Total Return' in opt:
                        fig_opt = px.bar(opt, x='Fold', y='Test Total Return', title='Out-of-Sample Return by Fold', color_discrete_sequence=['#38bdf8'])
                        fig_opt.update_layout(
                            height=290,
                            margin=dict(l=40, r=15, t=28, b=12),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=10.5, color='#94a3b8'),
                        )
                        fig_opt.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickformat='.1%')
                        st.plotly_chart(fig_opt, width='stretch')
            else:
                if prices is not None:
                    opt = grid_search(
                        prices,
                        MovingAverageCrossover,
                        parameter_grid,
                        engine_kwargs=engine_kwargs,
                        benchmark=benchmark,
                        objective='Sharpe Ratio',
                    )
                    st.dataframe(opt, width='stretch')
                    if not opt.empty and 'Sharpe Ratio' in opt:
                        fig_grid = px.scatter(
                            opt,
                            x='Annual Volatility',
                            y='Annual Return',
                            size='Sharpe Ratio',
                            hover_data=['short_window', 'long_window'],
                            title='Parameter Search: Risk vs Return',
                            color_discrete_sequence=['#38bdf8'],
                        )
                        fig_grid.update_layout(
                            height=310,
                            margin=dict(l=40, r=15, t=28, b=12),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=10.5, color='#94a3b8'),
                        )
                        fig_grid.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickformat='.1%')
                        fig_grid.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickformat='.1%')
                        st.plotly_chart(fig_grid, width='stretch')
        else:
            st.info('Enable "Run MA parameter optimizer" in the sidebar for a parameter sweep, or use the **Validation** tab for walk-forward testing.')

    with tabs[7]:
        report_paths = export_report(result, ROOT / 'reports')
        st.markdown('**Export Results**')
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        c_exp1.download_button('Trade Log (CSV)', trades.to_csv(index=False), file_name='trade_ledger.csv', width='stretch')
        c_exp2.download_button('Equity Curve (CSV)', hist.to_csv(index=True), file_name='equity_curve.csv', width='stretch')
        c_exp3.download_button('Metrics (CSV)', pd.Series(metrics).to_csv(), file_name='performance_metrics.csv', width='stretch')
        st.caption(f'HTML report saved to: `{report_paths["html"]}`')
else:
    if 'last_changed_strategy' in st.session_state:
        st.info(f"Strategy switched to **{st.session_state['last_changed_strategy']}**. Adjust parameters in the sidebar and click **Run Backtest**.")
    else:
        st.info('Select a strategy in the sidebar and click **Run Backtest**.')
