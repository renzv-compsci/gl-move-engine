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