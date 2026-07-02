import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

TICKERS = ["SPY", "QQQ", "GLD", "0005.HK", "EPHE"]
API_URL = "http://localhost:8000/api/v1/surveillance/predict"
OUTPUT_FILE = "data/market_regime_history_3y.csv"
WINDOW_SIZE = 32

def main():
    print("Initializing Multi-Year Historical Regime Generation Pipeline...")
    
    print(f"Downloading historical daily close records for: {TICKERS}")
    raw_data = yf.download(TICKERS, period="3y")['Close']
    
    raw_data = raw_data.ffill().bfill().dropna()
    
    total_available_days = len(raw_data)
    print(f"Data sync complete. Total synchronized trading days retrieved: {total_available_days}")
    
    if total_available_days < WINDOW_SIZE:
        print(f"CRITICAL ERROR: Total records ({total_available_days}) are less than minimum required window size ({WINDOW_SIZE}).")
        return

    historical_records = []
    
    print(f"Slicing timeline into rolling {WINDOW_SIZE}-day historical matrix frames...")
    
    for i in range(WINDOW_SIZE, total_available_days + 1):
        window_chunk = raw_data.iloc[i - WINDOW_SIZE:i]
        target_date = window_chunk.index[-1] 
        
        payload = {
            "ticker_data": {
                "SPY": window_chunk["SPY"].tolist(),
                "QQQ": window_chunk["QQQ"].tolist(),
                "GLD": window_chunk["GLD"].tolist(),
                "0005.HK": window_chunk["0005.HK"].tolist(),
                "EPHE": window_chunk["EPHE"].tolist()
            }
        }
        
        current_prices = {ticker: window_chunk[ticker].iloc[-1] for ticker in TICKERS}
    
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                res_json = response.json()
                inference = res_json["inference_results"]
                portfolio = res_json["portfolio_allocation"]
                weights = portfolio["weights"]
                
                record_row = {
                    "Date": target_date.strftime("%Y-%m-%d"),
                    "Inferred_Regime": f"State {inference['current_regime']}",
                    "Regime_Description": inference['regime_description'],
                    "Anomaly_Risk_Score": float(inference['anomaly_probability']) * 100,
                    "Strategy_Action": portfolio['strategy_action'],
                    "Price_SPY": current_prices["SPY"],
                    "Price_QQQ": current_prices["QQQ"],
                    "Price_GLD": current_prices["GLD"],
                    "Price_HK0005": current_prices["0005.HK"],
                    "Price_EPHE": current_prices["EPHE"],
                    "Weight_SPY": float(weights.get("SPY", 0)) * 100,
                    "Weight_QQQ": float(weights.get("QQQ", 0)) * 100,
                    "Weight_GLD": float(weights.get("GLD", 0)) * 100,
                    "Weight_HK0005": float(weights.get("0005.HK", 0)) * 100,
                    "Weight_EPHE": float(weights.get("EPHE", 0)) * 100
                }
                historical_records.append(record_row)
            else:
                print(f"Skipping date {target_date.strftime('%Y-%m-%d')} | Backend HTTP Error {response.status_code}")
                
        except Exception as e:
            print(f"CRITICAL: Transmission loop broken on date {target_date.strftime('%Y-%m-%d')}. Check server status. Error: {str(e)}")
            break
            
        processed_count = len(historical_records)
        if processed_count % 50 == 0 and processed_count > 0:
            print(f"Progress Update: Inferred and compiled {processed_count} trading sessions successfully.")

    if historical_records:
        final_df = pd.DataFrame(historical_records)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("\n==================================================================")
        print(f"PIPELINE RUN SUCCESSFUL: Master historical dataset compiled.")
        print(f"File Saved As: {os.path.abspath(OUTPUT_FILE)}")
        print(f"Total Rows Saved: {len(final_df)}")
        print("==================================================================")
        print("Ready for ingestion. Open Power BI and point your Data Source to this CSV file.")
    else:
        print("Pipeline execution resulted in zero records saved. Verify backend server logs.")

if __name__ == "__main__":
    main()