import numpy as np 
import pandas as pd 
import xgboost as xgb 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class MarketSurveillanceEngine: 
    def __init__(self, n_clusters: int=4): 
        self.n_clusters = n_clusters
        self.classifier = None 
        self.feature_columns = None 

    def generate_surveillance_features(self, raw_prices: pd.DataFrame) -> pd.DataFrame: 
        clean_prices = raw_prices.apply(pd.to_numeric, errors='coerce').astype(float)
        features = pd.DataFrame(index=clean_prices.index)
        log_rets = np.log(clean_prices / clean_prices.shift(1))

        for col in clean_prices.columns: 
            features[f'{col}_ret_5d'] = log_rets[col].rolling(5).sum()
            features[f'{col}_ret_21d'] = log_rets[col].rolling(21).sum()

            features[f'{col}_vol_10d'] = log_rets[col].rolling(10).std() * np.sqrt(252)
            features[f'{col}_vol_30d'] = log_rets[col].rolling(30).std() * np.sqrt(252)

            rolling_peak = clean_prices[col].rolling(21, min_periods=1).max()
            features[f'{col}_drawdown_21d'] = (clean_prices[col] - rolling_peak) / rolling_peak
        
        features['cross_asset_dispersion_5d'] = log_rets.rolling(5).mean().std(axis=1)
        return features.dropna().astype(float)

    def train_supervisor(self, X: pd.DataFrame, y: pd.Series): 
        self.feature_columns = X.columns.tolist()

        common_idx = X.index.intersection(y.index)
        X_clean = X.loc[common_idx]
        y_clean = y.loc[common_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=42
        )

        print(f"Training XGBoost Surveillance Engine on {X_train.shape[0]} samples")

        self.classifier = xgb.XGBClassifier( 
            objective='multi:softprob',
            num_class=self.n_clusters,
            eval_metric='mlogloss',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.classifier.fit(X_train, y_train)

        preds = self.classifier.predict(X_test)
        print("\n" + "="*60)
        print("          SURVEILLANCE CLASSIFIER EVALUATION AUDIT")
        print("="*60)
        print(f"Out-of-Sample Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
        print("\nDetailed Performance Matrix:")
        print(classification_report(y_test, preds, zero_division=0))
        print("="*60)
        
        return self 
    
    def analyze_live_market(self, current_features: pd.DataFrame, uncertainty_threshold: float = 0.65) -> pd.DataFrame:
        if self.classifier is None: 
            raise ValueError("Surveillance model must be trained before calling live analysis.")
        
        X = current_features[self.feature_columns]
        prob_matrix = self.classifier.predict_proba(X)
        predicted_regimes = self.classifier.predict(X)
        max_probabilities = np.max(prob_matrix, axis=1)

        results = pd.DataFrame(index=X.index)
        results['Predicted_Regime'] = predicted_regimes
        results['Confidence_Score'] = max_probabilities
        results['Anomaly_Flag'] = (max_probabilities < uncertainty_threshold).astype(int)

        return results