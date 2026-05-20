import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame: 
    cleaned_prices = price_df.ffill().dropna()
    log_returns = np.log(cleaned_prices / cleaned_prices.shift(1))
    return log_returns.dropna()

def scale_features(returns_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]: 
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(returns_df)
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=returns_df.columns,
        index=returns_df.index 
    )
    return scaled_df, scaler

def run_preprocessing_pipeline(raw_price_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]: 
    log_returns = compute_log_returns(raw_price_df)
    scaled_X, fitted_scaler = scale_features(log_returns)
    return scaled_X, fitted_scaler