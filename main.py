from src.data.ingestion import fetch_synchronized_data
import pandas as pd 

def run_ingestion_pipeline(): 
    print(f"Initializing ingestion")

    target_tickers = ["SPY", "0005.HK", "EPHE", "GLD", "QQQ"]
    raw_prices = fetch_synchronized_data(target_tickers)
    if raw_prices.empty: 
        print("Ingestion returned empty")
        return pd.DataFrame()
    rows, cols = raw_prices.shape
    print(f"Orch complete. Matrix Dimension: {rows} Rows x {cols} Cols")
    return raw_prices

if __name__ == "__main__": 
    price_matrix = run_ingestion_pipeline()
    if not price_matrix.empty: 
        print(price_matrix.head())