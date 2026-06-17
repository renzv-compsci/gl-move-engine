import yfinance as yf 
import pandas as pd 
from typing import List 

def fetch_synchronized_data(tickers: List[str], start: str = "2023-01-01", end: str = None) -> pd.DataFrame:
    try: 
        print(f"Ingesting Proxy Tickers: {tickers}")

        raw_data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False, 
            threads=False
        )

        if raw_data is None or raw_data.empty:
            print("yfinance returned no data")
            return pd.DataFrame()
        
        if len(tickers) > 1: 
            price_matrix = raw_data['Close']
        else: 
            price_matrix = raw_data[['Close']]
            price_matrix.columns = tickers
        
        clean_matrix = price_matrix.ffill().dropna()
        if clean_matrix.index.tz is None: 
            clean_matrix.index = clean_matrix.index.tz_localize('UTC')
        else: 
            clean_matrix.index = clean_matrix.index.tz_convert('UTC')
        print(f"Synch complete. Rows: {len(clean_matrix)} | Features {len(clean_matrix.columns)}")
        return clean_matrix
    
    except Exception as e: 
        print(f"yfinance data ingest failure: {str(e)}")
        return pd.DataFrame() 