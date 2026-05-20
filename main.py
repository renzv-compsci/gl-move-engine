from src.data.ingestion import fetch_synchronized_data
from src.utils.preprocessing import run_preprocessing_pipeline
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

if __name__ == "__main__": 
    # price_matrix = run_ingestion_pipeline()
    # if not price_matrix.empty: 
    #     print(price_matrix.head())
    main()