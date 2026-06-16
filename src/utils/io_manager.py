import os 
import pandas as pd 
from datetime import datetime 

def log_daily_trading_signal(live_signals: dict, file_path: str = "data/logs/trading_signals.csv") -> None:
    print("\nRecording daily execution signals to persistent storage")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_data = {
        "Execution_Timestamp": [timestamp],
        "Detected_Regime": [live_signals["Active_Regime"]],
        "Risk_Target_USD": [round(live_signals["Risk_Target_USD"], 2)],
        "Cash_Target_USD": [round(live_signals["Cash_Target_USD"], 2)]
    }

    new_entry_df = pd.DataFrame(log_data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    new_entry_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
    print(f"  └─ Audit log successfully updated at: {file_path}")