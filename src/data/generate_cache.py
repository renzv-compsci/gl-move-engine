import os 
import pandas as pd 
from main import run_ingestion_pipeline
from src.utils.preprocessing import run_preprocessing_pipeline

def main(): 
    print("Cache generation trace")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    raw_prices = run_ingestion_pipeline()
    if raw_prices.empty:
        print("Ingestion failed")
        return 
    
    print("\nExecuting Signal Transformation")
    scaled_X, financial_scaler = run_preprocessing_pipeline(raw_prices)
    raw_path = "data/raw/raw_prices_cache.csv"
    processed_path = "data/processed/stationary_scaled_features.csv"
    raw_prices.to_csv(raw_path)
    scaled_X.to_csv(processed_path)

    print("Cache system sync")
    print(f"Raw Price Cache Saved to: {raw_path} ({raw_prices.shape[0]} rows)")
    print(f"Processed Feature Matrix X Saved to: {processed_path} ({scaled_X.shape[0]} rows)")

if __name__ == "__main__": 
    main()