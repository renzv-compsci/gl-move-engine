import pandas as pd 
import sys 
import io
import numpy as np 

if sys.platform.startswith('win'): 
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    print("Terminal pipe configured to utf-8")

from src.data.ingestion import fetch_synchronized_data
from src.utils.preprocessing import run_preprocessing_pipeline
from src.models.clustering import run_pca_decomposition, evaluate_elbow_optimization, fit_final_regime_model, generate_regime_profiles
from src.portfolio.execution import allocate_portfolio_by_regime
from src.portfolio.backtest import run_portfolio_backtest
from src.utils.metrics import generate_performance_summary_report
from src.surveillance.anomaly import MarketSurveillanceEngine

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
    
    raw_prices.index = pd.to_datetime(raw_prices.index).tz_localize(None)
    
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

    print("\nFitting Definitive K-Means Model")
    labeled_timeline, final_kmeans_model = fit_final_regime_model(pca_df, n_clusters=4)

    target_cluster_col = None

    if isinstance(labeled_timeline, pd.DataFrame): 
        cluster_col = None
        for col in labeled_timeline.columns:
            unique_vals = labeled_timeline[col].dropna().unique()
            if len(unique_vals) <= 5 and all(isinstance(v, (int, np.integer)) or (hasattr(v, 'is_integer') and v.is_integer()) for v in unique_vals):
                cluster_col = col
                break
        
        if cluster_col is not None:
            print(f"Detected Cluster Column: '{cluster_col}'")
            target_cluster_col = cluster_col
            raw_values = labeled_timeline[cluster_col].values
        else:
            matched_col = None 
            for col in labeled_timeline.columns: 
                if any(k in str(col).lower() for k in ['cluster', 'regime', 'label', 'pred']): 
                    matched_col = col 
                    break 
            target_cluster_col = matched_col if matched_col else labeled_timeline.columns[-1]
            raw_values = labeled_timeline[target_cluster_col].values
    elif isinstance(labeled_timeline, pd.Series):
        raw_values = labeled_timeline.values 
    else: 
        raw_values = np.array(labeled_timeline)

    final_labels = pd.Series(raw_values, index=scaled_X.index).dropna().astype(int)

    if isinstance(labeled_timeline, (pd.DataFrame, pd.Series)):
        labeled_timeline.index = scaled_X.index

    SMOOTHING_WINDOW = 5  # 5 Trading Days = 1 Business Week
    print(f"\n[⚡] Applying Temporal Smoothing Filter (Window: {SMOOTHING_WINDOW} days)...")
    
    final_labels = (
        final_labels.rolling(window=SMOOTHING_WINDOW, min_periods=1)
        .apply(lambda x: pd.Series(x).mode().iloc[0])
        .astype(int)
    )
    
    if isinstance(labeled_timeline, pd.DataFrame) and target_cluster_col is not None:
        labeled_timeline[target_cluster_col] = final_labels
    elif isinstance(labeled_timeline, pd.Series):
        labeled_timeline = final_labels
        
    print("      Status: High-frequency noise filtered. Macro signals stabilized.")

    print("\nCluster Distribution Sent to Backtester:")
    print(final_labels.value_counts().sort_index())
    print("="*50)

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

    print("\n" + "="*50)
    print("Backtest Simulation")
    print("="*50)

    target_tickers = ["SPY", "0005.HK", "EPHE", "GLD", "QQQ"]
    equally_weighted_allocation = [0.20, 0.20, 0.20, 0.20, 0.20]

    performance_ledger = run_portfolio_backtest(
        raw_prices_df=raw_prices,
        weights=equally_weighted_allocation,
        target_tickers=target_tickers,
        cluster_labels=final_labels, 
        transaction_fee_bps=5.0
    )
    print("\nPipeline executed completely")

    report_dictionary = generate_performance_summary_report(performance_ledger)
    print("\nPipeline executed completely")

    print("\n" + "="*50)
    print("Deploying Live Market Surveillance Machine Learning Layer")
    print("="*50)

    surveillance_system = MarketSurveillanceEngine(n_clusters=4)
    print("Engineering rolling predictive feature matrices from raw assets.")
    surveillance_features = surveillance_system.generate_surveillance_features(raw_prices)
    
    surveillance_system.train_supervisor(X=surveillance_features, y=final_labels)

    print("Running live surveillance scan across full asset timelines.")
    surveillance_results = surveillance_system.analyze_live_market(
        current_features=surveillance_features, 
        uncertainty_threshold=0.65
    )

    total_days_scanned = len(surveillance_results)
    total_anomalies_flagged = surveillance_results['Anomaly_Flag'].sum()
    anomaly_rate = (total_anomalies_flagged / total_days_scanned) * 100 if total_days_scanned > 0 else 0.0

    print("\n" + "="*50)
    print("          REAL-TIME ANOMALY DETECTOR SUMMARY AUDIT")
    print("="*50)
    print(f"  ├─ Total Timesteps Scanned   : {total_days_scanned} Days")
    print(f"  ├─ Structural Anomalies Found: {total_anomalies_flagged} Days")
    print(f"  └─ Portfolio Anomaly Rate    : {anomaly_rate:.2f}%")
    print("="*50)

    print("\nPipeline executed completely with Live XGBoost Surveillance online\n")

if __name__ == "__main__": 
    main()