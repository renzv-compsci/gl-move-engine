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
    print("├─ Compiling Synthetic Valid Matrix (32 Trading Days)...")
    
    valid_payload = {
        "ticker_data": {
            "SPY": [
                415.2, 416.8, 414.1, 417.3, 418.5, 419.1, 420.0, 421.2, 
                419.8, 422.5, 423.1, 421.9, 424.0, 425.3, 423.8, 426.1, 
                427.5, 425.9, 428.4, 429.1, 428.0, 430.5, 431.8, 430.2, 
                432.9, 434.0, 432.5, 435.1, 436.5, 434.9, 437.2, 438.5
            ],
            "QQQ": [
                320.5, 322.1, 319.4, 324.0, 325.6, 326.4, 327.5, 328.9, 
                327.1, 330.2, 331.4, 329.6, 332.8, 334.1, 332.0, 335.7, 
                337.2, 335.0, 338.6, 340.0, 338.1, 341.5, 342.9, 340.8, 
                344.4, 346.0, 343.7, 347.2, 348.8, 346.4, 350.0, 351.5
            ],
            "GLD": [
                182.1, 183.4, 181.9, 184.0, 184.2, 183.8, 184.5, 185.1, 
                184.6, 185.9, 186.4, 185.8, 187.0, 187.6, 186.9, 188.2, 
                188.9, 188.1, 189.4, 190.0, 189.3, 190.7, 191.2, 190.5, 
                191.9, 192.4, 191.8, 193.1, 193.7, 192.9, 194.3, 194.8
            ],
            "0005.HK": [
                60.10, 60.40, 59.80, 61.20, 60.90, 61.05, 61.30, 61.60, 
                61.25, 61.90, 62.10, 61.75, 62.30, 62.60, 62.15, 62.80, 
                63.10, 62.65, 63.30, 63.50, 63.10, 63.70, 64.00, 63.55, 
                64.20, 64.50, 64.05, 64.70, 64.95, 64.50, 65.10, 65.40
            ],
            "EPHE": [
                28.40, 28.10, 28.50, 29.00, 28.80, 28.95, 29.10, 29.30, 
                29.15, 29.45, 29.60, 29.40, 29.75, 29.90, 29.70, 30.05, 
                30.20, 30.00, 30.35, 30.50, 30.30, 30.65, 30.80, 30.60, 
                30.95, 31.10, 30.90, 31.25, 31.40, 31.15, 31.55, 31.70
            ]
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