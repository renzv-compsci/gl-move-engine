import numpy as np 
import pandas as pd 

def simulate_daily_portfolio_returns(raw_prices_df: pd.DataFrame, weights: list, target_tickers: list) -> pd.DataFrame:
    cleaned_prices = raw_prices_df[target_tickers].ffill().dropna()
    daily_log_returns = np.log(cleaned_prices / cleaned_prices.shift(1)).dropna()

    allocation_weights = np.array(weights, dtype=float)
    if not np.isclose(np.sum(allocation_weights), 1.0):
        raise ValueError("Portfolio weights must sum to exactly 1.0")
    
    sim_book = daily_log_returns.copy()
    sim_book['Portfolio_Return'] = daily_log_returns.dot(allocation_weights)
    return sim_book[['Portfolio_Return']]

def synchronize_portfolio_regimes(portfolio_returns_df: pd.DataFrame, cluster_labels: list) -> pd.DataFrame: 
    ledger_df = portfolio_returns_df.copy()
    if len(ledger_df) != len(cluster_labels): 
        sliced_labels = cluster_labels[-len(ledger_df):]
    else: 
        sliced_labels = cluster_labels
    
    ledger_df['Regime_Cluster'] = sliced_labels 
    ledger_df['Cumulative_Growth'] = (1 + ledger_df['Portfolio_Return']).cumprod()
    return ledger_df

def apply_transaction_costs(synchronized_ledger: pd.DataFrame, transaction_fee_bps: float = 5.0) -> pd.DataFrame: 
    friction_df = synchronized_ledger.copy()
    regime_shift_mask = friction_df['Regime_Cluster'] != friction_df['Regime_Cluster'].shift(1)
    regime_shift_mask.iloc[0] = False 

    cost_factor = transaction_fee_bps / 10000.0
    friction_df.loc[regime_shift_mask, 'Portfolio_Return'] -= cost_factor
    friction_df['Cumulative_Growth'] = (1 + friction_df['Portfolio_Return']).cumprod()
    return friction_df

def run_portfolio_backtest(raw_prices_df: pd.DataFrame, weights: list, target_tickers: list, cluster_labels: list, transaction_fee_bps: float = 5.0) -> pd.DataFrame: 
    print("Executing master historical backtest simulation")
    base_returns_df = simulate_daily_portfolio_returns(
        raw_prices_df=raw_prices_df,
        weights=weights,
        target_tickers=target_tickers
    )

    synchronized_leadger = synchronize_portfolio_regimes(
        portfolio_returns_df=base_returns_df,
        cluster_labels=cluster_labels
    )

    final_net_ledger = apply_transaction_costs(
        synchronized_ledger=synchronized_leadger,
        transaction_fee_bps=transaction_fee_bps
    )
    print(f"\nPerformance ledger generated successfully. Final cumulative capital multiplier: {final_net_ledger['Cumulative_Growth'].iloc[-1]:.4f}")

    return final_net_ledger