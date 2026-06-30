import json 
import urllib.request
from datetime import datetime

def run_api_integration_test():
    base_url = "http://localhost:8000/api/v1"
    
    print("=" * 60)
    print("Executing GL-Move Engine Endpoint Validation Suite")
    print("=" * 60)
    
    try:
        print("├─ Testing Service Health Endpoint...")
        with urllib.request.urlopen(f"{base_url}/health") as response:
            health_data = json.loads(response.read().decode())
            print(f"│  STATUS: {health_data['status'].upper()}")
            print(f"│  MODELS CACHED: {health_data['artifacts_cached']}")
            print(f"│  ACTIVE SYSTEMS: {health_data['active_models']}")
    except Exception as e:
        print(f"└─ CRITICAL: Health check failed. Is the API server running? Error: {e}")
        return

    print("│")
    print("├─ Compiling Synthetic Valid Matrix (6 Trading Days)...")
    
    valid_payload = {
        "ticker_data": {
            "SPY": [415.20, 416.80, 414.10, 417.30, 418.50, 419.10],
            "QQQ": [320.50, 322.10, 319.40, 324.00, 325.60, 326.40],
            "GLD": [182.10, 183.40, 181.90, 184.00, 184.20, 183.80],
            "0005.HK": [60.10, 60.40, 59.80, 61.20, 60.90, 61.05],
            "EPHE": [28.40, 28.10, 28.50, 29.00, 28.80, 28.95]
        }
    }
    
    try:
        print("├─ Transmitting Prediction Request payload...")
        req = urllib.request.Request(
            f"{base_url}/surveillance/predict",
            data=json.dumps(valid_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            
            print("│")
            print("├─ [SUCCESS] Surveillance Engine Response Received:")
            print(f"│  ├─ System Status: {res_data['status']}")
            print(f"│  ├─ Inferred Regime: {res_data['inference_results']['current_regime']}")
            print(f"│  ├─ Profile Description: {res_data['inference_results']['regime_description']}")
            print(f"│  ├─ Anomaly Signal Flag: {res_data['inference_results']['anomaly_detected']}")
            print(f"│  └─ Anomaly Probability: {res_data['inference_results']['anomaly_probability']:.4f}")
            print("│")
            print("├─ Execution Portfolio Allocations:")
            print(f"│  ├─ Strategy Action Call: {res_data['portfolio_allocation']['strategy_action']}")
            for ticker, weight in res_data['portfolio_allocation']['weights'].items():
                print(f"│  │  └─ {ticker}: {weight * 100:.1f}%")
                
    except Exception as e:
        print(f"│  └─ [ERROR] Inference transmission failure: {e}")

    print("=" * 60)

if __name__ == "__main__":
    run_api_integration_test()