import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_normalized_trends(df: pd.DataFrame):
    normalized_df = (df / df.iloc[0]) * 100
    
    plt.figure(figsize=(14, 7))
    for column in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[column], label=column)
        
    plt.title("Normalized Asset Trends (Base 100)")
    plt.xlabel("Date")
    plt.ylabel("Relative Growth (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()