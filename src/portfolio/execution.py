import pandas as pd 
from typing import Dict, Tuple

def allocate_portfolio_by_regime(labeled_df: pd.DataFrame, total_capital: float = 100000.0) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print("\nInitializing Portfolio Execution Layer")

    PORTFOLIO_PLAYBOOKS = {
        0: {"Risk_Allocation": 0.00, "Cash_Allocation": 1.00, "Condition": "Market Liquidation State (Defensive Risk-Off)"}, # Was Regime 3
        1: {"Risk_Allocation": 0.40, "Cash_Allocation": 0.60, "Condition": "Chop / Neutral Consolidation Phase"},            # Was Regime 0
        2: {"Risk_Allocation": 1.00, "Cash_Allocation": 0.00, "Condition": "Standard Bull Grind Phase"},                     # Was Regime 1
        3: {"Risk_Allocation": 1.00, "Cash_Allocation": 0.00, "Condition": "Parabolic Alpha Outlier Phase (Max Growth)"}     # Was Regime 2
    }

    latest_record = labeled_df.iloc[-1]
    latest_date = labeled_df.index[-1]
    current_regime = int(latest_record['Regime'])
    active_rules = PORTFOLIO_PLAYBOOKS[current_regime]

    risk_dollar = total_capital * active_rules["Risk_Allocation"]
    cash_dollar = total_capital * active_rules["Cash_Allocation"]

    print("=" * 60)
    print(f" LIVE TRADING DISPATCH SIGNAL | RUN DATE: {latest_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    print(f"  ├─ Current Market State     : Regime {current_regime} ({active_rules['Condition']})")
    print(f"  ├─ Target Risk Allocation   : {active_rules['Risk_Allocation'] * 100:.1f}% (${risk_dollar:,.2f})")
    print(f"  └─ Target Safety Cash Base  : {active_rules['Cash_Allocation'] * 100:.1f}% (${cash_dollar:,.2f})")
    print("=" * 60)

    execution_timeline = labeled_df.copy()
    execution_timeline['Target_Risk_Weight'] = execution_timeline['Regime'].map(lambda r: PORTFOLIO_PLAYBOOKS[r]["Risk_Allocation"])
    execution_timeline['Target_Cash_Weight'] = execution_timeline['Regime'].map(lambda r: PORTFOLIO_PLAYBOOKS[r]["Cash_Allocation"])

    live_signals = {
        "Active_Regime": current_regime, 
        "Risk_Target_USD": risk_dollar,
        "Cash_Target_USD": cash_dollar
    }
    return execution_timeline, live_signals