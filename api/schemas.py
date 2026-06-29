from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Literal

class MarketDataRequest(BaseModel): 
    ticker_data: Dict[str, List[float]] = Field(
        ...,
        description="Dictionary containing asset price series arrays"
    )

    @field_validator("ticker_data")
    @classmethod
    def validate_ticker_matrix(cls, v: Dict[str, List[float]]) -> Dict[str, List[float]]:
        required_tickers = {"SPY", "0005.HK", "EPHE", "GLD", "QQQ"}
        provided_tickers = set(v.keys())

        if provided_tickers != required_tickers: 
            missing = required_tickers - provided_tickers
            extra = provided_tickers - required_tickers
            error_msg = "Invalid ticker keys matrix configuration"
            if missing: 
                error_msg += f"Missing required targets: {list(missing)}"
            if extra: 
                error_msg += f"Disallowed extra targets found: {list(extra)}"
            raise ValueError(error_msg)

        lengths = {ticker: len(prices) for ticker, prices in v.items()}
        distinct_lengths = set(lengths.values())

        if len(distinct_lengths) != 1: 
            raise ValueError(
                f"Mismatched time-series lenghts detected across asset nodes: {lengths}"
            )
        series_length = distinct_lengths.pop()
        if series_length < 6: 
            raise ValueError(
                f"Insufficient historical timeline window. Provided {series_length} values, "
                f"but a minimum of 6 continuous trading days is structurally required to satisfy "
                f"the 5-day temporal smoothing matrix filter after computing log returns."
            )
        return v 
    
class InferenceMetrics(BaseModel): 
    current_regime: int 
    regime_description: str 
    anomaly_detected: int 
    anomaly_probability: float 

class PortfolioAllocation(BaseModel):
    strategy_action: Literal["NOMINAL_EXECUTION", "DYNAMIC_REBALANCING", "CAPITAL_PRESERVATION"]
    weights: Dict[str, float]

class MarketSurveillanceResponse(BaseModel): 
    status: str
    system_date: str
    inference_results: InferenceMetrics
    portfolio_allocation: PortfolioAllocation

class HealthCheckResponse(BaseModel): 
    status: str
    system_date: str
    artifacts_cached: bool 
    active_models: List[str]