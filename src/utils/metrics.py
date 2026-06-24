import numpy as np 
import pandas as pd 

def calculate_annualized_moments(portfolio_returns: pd.Series, trading_days: int = 252) -> tuple:
    daily_mean = portfolio_returns.mean()
    daily_vol = portfolio_returns.std()
    
    ann_returns = daily_mean * trading_days
    ann_volatility = daily_vol * np.sqrt(trading_days)

    return ann_returns, ann_volatility

def calculate_risk_adjusted_ratios(portfolio_returns: pd.Series, ann_return: float, ann_volatility: float, risk_free_rate: float = 0.0) -> tuple: 
    if ann_volatility == 0: 
        return 0.0, 0.0
    
    sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility
    downside_returns = portfolio_returns[portfolio_returns < 0]

    if len(downside_returns) == 0: 
        sortino_ratio = 0.0 
    else: 
        ann_downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (ann_return - risk_free_rate) / ann_downside_vol if ann_downside_vol > 0 else 0.0 
    
    return sharpe_ratio, sortino_ratio

def calculate_maximum_drawdown(cumulative_growth_series: pd.Series) -> float: 
    running_peak = cumulative_growth_series.cummax()
    drawdown_series = (cumulative_growth_series - running_peak) / running_peak
    max_drawdown = drawdown_series.min()

    return max_drawdown

def generate_performance_summary_report(backtest_ledger: pd.DataFrame) -> dict: 
    p_rets = backtest_ledger['Portfolio_Return']
    p_growth = backtest_ledger['Cumulative_Growth']

    ann_ret, ann_vol = calculate_annualized_moments(p_rets)
    sharpe, sortino = calculate_risk_adjusted_ratios(p_rets, ann_ret, ann_vol)
    max_dd = calculate_maximum_drawdown(p_growth)

    print("\n" + "="*70)
    print("          GLOBAL PORTFOLIO RISK & PERFORMANCE STATISTICS AUDIT")
    print("="*70)
    print(f"  ├─ Annualized Strategic Return : {ann_ret*100:.2f}%")
    print(f"  ├─ Annualized Volatility Risk  : {ann_vol*100:.2f}%")
    print(f"  ├─ Strategy Sharpe Ratio       : {sharpe:.4f}")
    print(f"  ├─ Strategy Sortino Ratio      : {sortino:.4f}")
    print(f"  └─ Maximum Timeline Drawdown   : {max_dd*100:.2f}%")
    print("="*70)

    print("\nREGIME DECOMPOSED PERFORMANCE METRICS BREAKDOWN:")
    print(f"{'Regime ID':<10} | {'Days Allocated':<15} | {'Ann. Return':<13} | {'Ann. Volatility':<16} | {'Max Drawdown'}")
    print("-" * 75)

    regime_metrics = {}
    regime_col = 'Regime' if 'Regime' in backtest_ledger.columns else 'Regime_Cluster'

    for r_id in sorted(backtest_ledger[regime_col].unique()):
        slice_df = backtest_ledger[backtest_ledger[regime_col] == r_id]
        s_rets = slice_df['Portfolio_Return']

        r_ret, r_vol = calculate_annualized_moments(s_rets)
        r_sharpe, r_sortino = calculate_risk_adjusted_ratios(s_rets, r_ret, r_vol)
        r_dd = calculate_maximum_drawdown(slice_df['Cumulative_Growth'])

        print(f"Regime {r_id:<3} | {len(slice_df):<15} | {r_ret*100:<11.2f}% | {r_vol*100:<14.2f}% | {r_dd*100:.2f}%")

        regime_metrics[f"regime_{r_id}"] = {
            'return': r_ret, 'volatility': r_vol, 'sharpe': r_sharpe, 'sortino': r_sortino, 'max_drawdown': r_dd
        }

    return {
        'global_annualized_return': ann_ret, 'global_annualized_volatility': ann_vol,
        'global_sharpe': sharpe, 'global_sortino': sortino, 'global_max_drawdown': max_dd,
        'regime_breakdown': regime_metrics
    }