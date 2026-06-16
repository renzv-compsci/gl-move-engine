from src.data.ingestion import fetch_synchronized_data
from src.utils.preprocessing import run_preprocessing_pipeline
from src.models.clustering import run_pca_decomposition, evaluate_elbow_optimization, fit_final_regime_model, generate_regime_profiles
from src.portfolio.execution import allocate_portfolio_by_regime
import pandas as pd 


def run_ingestion_pipeline() -> pd.DataFrame: 
    print(f"Initializing ingestion")
    target_tickers = ["SPY", "0005.HK", "EPHE", "GLD", "QQQ"]
    raw_prices = fetch_synchronized_data(tickers=target_tickers)

    if raw_prices.empty: 
        print("Ingestion returned empty")
        return pd.DataFrame()
    
    rows, cols = raw_prices.shape
    print(f"Orch complete. Matrix Dimension: {rows} Rows x {cols} Cols")
    return raw_prices

def main(): 
    raw_prices = run_ingestion_pipeline()
    if raw_prices.empty: 
        print("Pipeline cancelled. Missing raw data")
        return 
    
    print("\nInitializing Signal Engineering Pipeline")
    scaled_X, financial_scaler = run_preprocessing_pipeline(raw_prices)

    p_rows, p_cols = scaled_X.shape
    print(f"S.E Complete. Feature Matrix X: {p_rows} Rows x {p_cols} Cols")
    print("\nFeature Matrix X Head (Stationary & Scaled)")
    print(scaled_X.head())

    print("\n" + "="*50)
    print("Decomp & Regime Training")
    print("="*50)

    pca_df, trained_pca = run_pca_decomposition(scaled_X, n_components=3)
    inertia_map = evaluate_elbow_optimization(pca_df, max_k=8)
    labeled_timeline, final_kmeans_model = fit_final_regime_model(pca_df, n_clusters=4)
    backtest_history, current_live_orders = allocate_portfolio_by_regime(labeled_timeline, total_capital=150000.0)
    regime_profiles = generate_regime_profiles(scaled_X, labeled_timeline)
    print("Regime Profile Matrix")
    print(regime_profiles.round(3))
    
    print("Recent tabled timeline")
    print(labeled_timeline.tail(10))

    print("Model Diagnostics Matrix")
    print("Review these numbers to locate where the 'Elbow' point forms:")
    for k, score in inertia_map.items():
        print(f"  └─ Clusters (k): {k} | Internal Distance Metric (Inertia): {score:.2f}")

if __name__ == "__main__": 
    # price_matrix = run_ingestion_pipeline()
    # if not price_matrix.empty: 
    #     print(price_matrix.head())
    main()