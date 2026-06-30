import os 
import joblib 
import pandas as pd 
import numpy as np 
from datetime import datetime 
from fastapi import APIRouter, HTTPException

from api.schemas import MarketDataRequest, MarketSurveillanceResponse, HealthCheckResponse
from src.models.clustering import run_pca_decomposition

router = APIRouter(prefix="/api/v1")

ARTIFACTS = {}
ARTIFACT_NAMES = ["scaler.joblib", "pca.joblib", "kmeans.joblib", "xgboost_surv.joblib"]

def get_project_root() -> str: 
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_file_dir, ".."))

@router.on_event("startup")
def load_production_artifacts(): 
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models")

    for artificat_name in ARTIFACT_NAMES: 
        artifact_path = os.path.join(models_dir, artificat_name)
        if not os.path.exists(artifact_path): 
            raise RuntimeError(f"Required production binary missing: '{artifact_path}'")

        key = artificat_name.replace(".joblib", "")
        ARTIFACTS[key] = joblib.load(artifact_path)
    ARTIFACTS["initialized"] = True

@router.get("/health", response_model=HealthCheckResponse)
def health_check(): 
    is_cached = ARTIFACTS.get("initialized", False)
    active = [f"{k}.joblib" for k in ARTIFACTS.keys() if k != "initialized"] if is_cached else []
    
    return HealthCheckResponse(
        status="healthy",
        system_date=datetime.utcnow().strftime("%Y-%m-%d"),
        artifacts_cached=is_cached,
        active_models=active
    )

@router.post("/surveillance/predict", response_model=MarketSurveillanceResponse)
def predict_market_state(request: MarketDataRequest): 
    if not ARTIFACTS.get("initialized", False):
        raise HTTPException(status_code=503, detail="Model artifacts are uninitialized")
    
    try: 
        raw_df = pd.DataFrame(request.ticker_data)
        raw_df.index = pd.date_range(end=datetime.utcnow(), periods=len(raw_df), freq="D")
        log_returns = np.log(raw_df / raw_df.shift(1)).dropna()

        scaler = ARTIFACTS["scaler"]
        if hasattr(scaler, "feature_names_in_"):
            log_returns = log_returns[list(scaler.feature_names_in_)]
            
        scaled_features = scaler.transform(log_returns)
        scaled_df = pd.DataFrame(scaled_features, index=log_returns.index, columns=log_returns.columns)

        pca_df, _ = run_pca_decomposition(scaled_df, n_components=3)
        if not isinstance(pca_df, pd.DataFrame):
            pca_df = pd.DataFrame(pca_df, index=scaled_df.index, columns=[f"PC{i+1}" for i in range(pca_df.shape[1])])

        kmeans = ARTIFACTS["kmeans"]
        raw_cluster_labels = kmeans.predict(pca_df.values)

        cluster_series = pd.Series(raw_cluster_labels, index=pca_df.index)
        smoothed_labels = (
            cluster_series.rolling(window=5, min_periods=1)
            .apply(lambda x: pd.Series(x).mode().iloc[0])
            .astype(int)
        )

        current_regime = int(smoothed_labels.iloc[-1])
        xgboost_model = ARTIFACTS["xgboost_surv"]

        if hasattr(xgboost_model, "feature_names_in_"):
            model_features = [str(f) for f in xgboost_model.feature_names_in_]
        else:
            model_features = [str(f) for f in xgboost_model.get_booster().feature_names]

        features_df = pd.DataFrame(index=log_returns.index)
        tickers = ["SPY", "0005.HK", "EPHE", "GLD", "QQQ"]
        
        for t in tickers:
            features_df[f"{t}_ret_5d"] = log_returns[t].rolling(window=5, min_periods=1).sum()
            features_df[f"{t}_ret_21d"] = log_returns[t].rolling(window=21, min_periods=1).sum()
            features_df[f"{t}_vol_10d"] = log_returns[t].rolling(window=10, min_periods=1).std().fillna(0)
            features_df[f"{t}_vol_30d"] = log_returns[t].rolling(window=30, min_periods=1).std().fillna(0)
            
            roll_max = raw_df[t].rolling(window=21, min_periods=1).max()
            features_df[f"{t}_drawdown_21d"] = ((raw_df[t] - roll_max) / roll_max).loc[log_returns.index]

        features_df["cross_asset_dispersion_5d"] = log_returns[tickers].std(axis=1).rolling(window=5, min_periods=1).mean()
        
        missing_cols = [col for col in model_features if col not in features_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline Alignment Failure. Missing Generated Columns: {missing_cols}. Available Columns: {list(features_df.columns)}"
            )

        features_df = features_df[model_features]
        latest_feature_row = features_df.iloc[[-1]]
        
        anomaly_pred = xgboost_model.predict(latest_feature_row)[0]
        anomaly_prob = xgboost_model.predict_proba(latest_feature_row)[0]

        regime_descriptions = {
            0: "Low-Volatility Bullish Growth State",
            1: "High-Volatility Capital Flight State",
            2: "Moderate-Volatility Structural Variance State",
            3: "Extreme Tail-Risk Liquidity Anomaly"
        }
        
        strategy_actions = {
            0: "NOMINAL_EXECUTION",
            1: "CAPITAL_PRESERVATION",
            2: "DYNAMIC_REBALANCING",
            3: "CAPITAL_PRESERVATION"
        }
        
        allocations = {
            0: {"SPY": 0.20, "0005.HK": 0.20, "EPHE": 0.20, "GLD": 0.20, "QQQ": 0.20},
            1: {"SPY": 0.05, "0005.HK": 0.05, "EPHE": 0.00, "GLD": 0.70, "QQQ": 0.20},
            2: {"SPY": 0.15, "0005.HK": 0.25, "EPHE": 0.10, "GLD": 0.30, "QQQ": 0.20},
            3: {"SPY": 0.00, "0005.HK": 0.00, "EPHE": 0.00, "GLD": 1.00, "QQQ": 0.00}
        }

        return MarketSurveillanceResponse(
            status="success",
            system_date=datetime.utcnow().strftime("%Y-%m-%d"),
            inference_results={
                "current_regime": current_regime,
                "regime_description": regime_descriptions.get(current_regime, "Unknown State"),
                "anomaly_detected": int(anomaly_pred),
                "anomaly_probability": float(anomaly_prob[1] if len(anomaly_prob) > 1 else anomaly_prob[0])
            },
            portfolio_allocation={
                "strategy_action": strategy_actions.get(current_regime, "DYNAMIC_REBALANCING"),
                "weights": allocations.get(current_regime, allocations[0])
            }
        )
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Inference Engine Processing Failure: {str(e)}")