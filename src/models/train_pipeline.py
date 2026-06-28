import os 
import joblib
import pandas as pd 
import numpy as np 

from src.utils.preprocessing import run_preprocessing_pipeline
from src.models.clustering import run_pca_decomposition, fit_final_regime_model
from src.surveillance.anomaly import MarketSurveillanceEngine
from src.portfolio.backtest import run_portfolio_backtest
from src.utils.metrics import generate_performance_summary_report
from src.data.ingestion import fetch_synchronized_data

def run_and_serialize_prod_pipeline(raw_prices_df: pd.DataFrame, target_dir: str = "models"):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    export_path = os.path.join(project_root, target_dir)
    os.makedirs(export_path, exist_ok=True)

    print("\n" + "="*60)
    print(f"Initializing Production Training Pipeline | System Date: 2026-06-27")
    print("="*60)

    try:
        raw_prices_df.index = pd.to_datetime(raw_prices_df.index).tz_localize(None)

        print("  ├─ [1/4] Running Log Returns & Feature Scaling Pipeline")
        scaled_X, fitted_scaler = run_preprocessing_pipeline(raw_prices_df)
        
        
        print("  ├─ [2/4] Executing PCA Principal Component Decomposition")
        pca_df, trained_pca = run_pca_decomposition(scaled_X, n_components=3)

        print("  ├─ [3/4] Fitting Analytically Aware K-Means & Sorting Labels")
        labeled_timeline, fitted_kmeans = fit_final_regime_model(pca_df, n_clusters=4)

        target_cluster_col = None 
        if isinstance(labeled_timeline, pd.DataFrame): 
            cluster_col = None 
            for col in labeled_timeline.columns: 
                unique_vals = labeled_timeline[col].dropna().unique()
                if len(unique_vals) <= 5 and all(isinstance(v, (int, np.integer)) or (hasattr(v, 'is_integer') and v.is_integer()) for v in unique_vals):
                    cluster_col = col 
                    break 

            if cluster_col is not None: 
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

        SMOOTHING_WINDOW = 5 
        print(f"  ├─ Applying Temporal Smoothing Filter (Window: {SMOOTHING_WINDOW} days).")
        final_labels = (
            final_labels.rolling(window=SMOOTHING_WINDOW, min_periods=1)
            .apply(lambda x: pd.Series(x).mode().iloc[0])
            .astype(int)
        )
        print("       Status: High-frequency noise filtered. Macro signals stabilized.")

        print("\n" + "="*60)
        print("Backtest Simulation Audit")
        print("="*60)
        target_tickers = ["SPY", "0005.HK", "EPHE", "GLD", "QQQ"]
        equally_weighted_allocation = [0.20, 0.20, 0.20, 0.20, 0.20]

        performance_ledger = run_portfolio_backtest(
            raw_prices_df=raw_prices_df,
            weights=equally_weighted_allocation,
            target_tickers=target_tickers,
            cluster_labels=final_labels, 
            transaction_fee_bps=5.0
        )
        _ = generate_performance_summary_report(performance_ledger)

        print("\n" + "="*60)
        print("Deploying Live Market Surveillance Machine Learning Layer")
        print("="*60)

        surveillance_system = MarketSurveillanceEngine(n_clusters=4)
        print("Engineering rolling predictive feature matrices from raw assets.")
        surveillance_features = surveillance_system.generate_surveillance_features(raw_prices_df)
        
        common_idx = surveillance_features.index.intersection(final_labels.index)
        X_input = surveillance_features.loc[common_idx]
        y_input = final_labels.loc[common_idx]
        
        surveillance_system.train_supervisor(X=X_input, y=y_input)

        print("="*60)
        print(f" Exporting Serialized Artifact to: '{export_path}/'")
        print("="*60)

        joblib.dump(fitted_scaler, os.path.join(export_path, "scaler.joblib"))
        print("  ├─ [✔] StandardScaler serialized -> scaler.joblib")
        
        joblib.dump(trained_pca, os.path.join(export_path, "pca.joblib"))
        print("  ├─ [✔] PCA Engine serialized -> pca.joblib")
        
        joblib.dump(fitted_kmeans, os.path.join(export_path, "kmeans.joblib"))
        print("  ├─ [✔] Sorted K-Means Model serialized -> kmeans.joblib")
        
        joblib.dump(surveillance_system.classifier, os.path.join(export_path, "xgboost_surv.joblib"))
        print("  └─ [✔] XGBoost Surveillance Engine serialized -> xgboost_surv.joblib")

        print("="*60)
        print("Training, Metrics Audit, and Serialization pipeline finished successfully.")
        
    except Exception as e:
        print(f"CRITICAL PIPELINE TRAINING FAILURE: {str(e)}")
        raise e
        
if __name__ == "__main__": 
    print("Fetching live synchronized data stream for production training...")
    target_tickers = ["SPY", "0005.HK", "EPHE", "GLD", "QQQ"]
    live_market_df = fetch_synchronized_data(tickers=target_tickers)
    
    if not live_market_df.empty:
        run_and_serialize_prod_pipeline(raw_prices_df=live_market_df)
    else:
        print("Critical Error: Live ingestion returned an empty dataset.")