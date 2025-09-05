"""
File-based ML forecaster that works with NDJSON data instead of database.
Provides equivalent functionality to ml_forecaster.py using file storage.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings

from storage.ndjson_storage import get_storage_instance
from modeling.profitability.file_feature_engineering import create_feature_dataframe_from_files, validate_feature_dataframe

warnings.filterwarnings('ignore')


def load_config_for_file_forecaster() -> Dict[str, Any]:
    """Load configuration for file-based forecasting."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def fetch_multivariate_series_from_files(storage, product_id: str, min_points: int = 500) -> Optional[pd.DataFrame]:
    """
    File-based equivalent of fetch_multivariate_series.
    Creates feature-rich DataFrame from NDJSON files.
    """
    
    # Create feature DataFrame from files
    df = create_feature_dataframe_from_files(
        storage, 
        product_id, 
        hours_back=168,  # 7 days of data
        min_points=min_points
    )
    
    if df is None or len(df) < min_points:
        return None
    
    # Validate we have all required features
    validate_feature_dataframe(df)
    
    return df


def create_features_targets_from_files(df: pd.DataFrame, horizons: List[int] = [15, 60, 240]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Create feature matrices and target vectors from file-based DataFrame.
    Equivalent to create_features_targets but works with file data.
    """
    
    feature_cols = [
        'ma_15', 'ma_60', 'spread_bps', 'vol_window_30',
        'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5',
        'ma_crossover', 'ma_ratio', 'vol_price_ratio',
        'hour_of_day', 'day_of_week', 'day_of_month',
        'market_volatility', 'market_momentum'
    ]
    
    # Ensure all feature columns exist and have numeric types
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0.0
    
    X = df[feature_cols].values
    
    datasets = {}
    for horizon in horizons:
        # Create target: price horizon minutes in the future
        horizon_steps = max(1, horizon // 5)  # Assuming 5-minute intervals
        y = df['mid_price'].shift(-horizon_steps).values
        
        # Remove rows where target is NaN (end of series)
        valid_indices = ~np.isnan(y)
        X_valid = X[valid_indices]
        y_valid = y[valid_indices]
        
        if len(X_valid) > 100:  # Minimum data requirement
            datasets[horizon] = (X_valid, y_valid)
    
    return datasets


class FileBasedMLForecaster:
    """
    File-based ML forecaster using LightGBM and XGBoost.
    Works with NDJSON files instead of database.
    """
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.models = {}  # horizon -> trained model
        self.feature_names = [
            'ma_15', 'ma_60', 'spread_bps', 'vol_window_30',
            'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5', 
            'ma_crossover', 'ma_ratio', 'vol_price_ratio',
            'hour_of_day', 'day_of_week', 'day_of_month',
            'market_volatility', 'market_momentum'
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, horizon: int) -> Dict[str, float]:
        """Train model for specific horizon."""
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        if self.model_type == 'lightgbm':
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            )
        else:  # xgboost
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        
        # Validation scores
        mae_scores = []
        mse_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            y_pred = model_copy.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            mse_scores.append(mean_squared_error(y_val, y_pred))
        
        # Final training on all data
        model.fit(X, y)
        self.models[horizon] = model
        
        return {
            'mae': np.mean(mae_scores),
            'rmse': np.sqrt(np.mean(mse_scores)),
            'horizon': horizon
        }
    
    def predict(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Make predictions for specific horizon."""
        if horizon not in self.models:
            raise ValueError(f"No trained model for horizon {horizon}")
        
        return self.models[horizon].predict(X)
    
    def get_feature_importance(self, horizon: int) -> Dict[str, float]:
        """Get feature importance for specific horizon."""
        if horizon not in self.models:
            return {}
        
        model = self.models[horizon]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return {}
        
        return dict(zip(self.feature_names, importances))
    
    def save_model(self, filepath: str, horizon: int):
        """Save trained model."""
        if horizon in self.models:
            joblib.dump(self.models[horizon], f"{filepath}_h{horizon}_{self.model_type}.pkl")
    
    def load_model(self, filepath: str, horizon: int):
        """Load trained model."""
        model_path = f"{filepath}_h{horizon}_{self.model_type}.pkl"
        if os.path.exists(model_path):
            self.models[horizon] = joblib.load(model_path)


def write_ml_forecast_to_file(
    storage, 
    ts: datetime, 
    product_id: str, 
    horizon: int, 
    price: float, 
    model_type: str, 
    metrics: Dict[str, Any] = None
):
    """Write ML forecast to NDJSON file instead of database."""
    
    forecast_record = {
        'ts': ts.isoformat(),
        'product_id': product_id,
        'horizon_minutes': horizon,
        'forecast_price': price,
        'model_version': f"{model_type}-v1",
        'model_metrics': metrics,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    # Store forecasts in a separate data type
    storage.append_record("forecasts", forecast_record)


def get_latest_forecast_from_files(
    storage,
    product_id: str,
    horizon_minutes: int
) -> Optional[Dict[str, Any]]:
    """Retrieve latest forecast for a product and horizon from files."""
    
    # Read recent forecast records
    records = list(storage.read_records("forecasts", hours_back=24))
    
    # Filter for specific product and horizon
    matching_records = [
        r for r in records 
        if r.get('product_id') == product_id 
        and r.get('horizon_minutes') == horizon_minutes
    ]
    
    if not matching_records:
        return None
    
    # Return the most recent
    return max(matching_records, key=lambda x: x.get('ts', ''))


def train_and_forecast_ml_from_files(
    product_id: str, 
    model_type: str = 'lightgbm', 
    horizons: Tuple[int, ...] = (15, 60, 240)
) -> bool:
    """
    File-based equivalent of train_and_forecast_ml.
    Train ML model and generate forecasts using NDJSON files.
    """
    
    cfg = load_config_for_file_forecaster()
    storage = get_storage_instance(cfg)
    
    if storage is None:
        print("No-database mode not enabled")
        return False
    
    try:
        # Fetch multivariate data from files
        df = fetch_multivariate_series_from_files(storage, product_id)
        if df is None or df.empty:
            print(f"Not enough data for {product_id}")
            return False
        
        print(f"Training {model_type} model for {product_id} with {len(df)} data points")
        
        # Prepare feature sets for different horizons
        datasets = create_features_targets_from_files(df, list(horizons))
        
        forecaster = FileBasedMLForecaster(model_type=model_type)
        last_ts = df["ts"].iloc[-1]
        
        # Get the latest features for prediction
        latest_features = df[forecaster.feature_names].iloc[-1:].values
        
        for horizon in horizons:
            if horizon not in datasets:
                print(f"Insufficient data for horizon {horizon}min")
                continue
                
            X, y = datasets[horizon]
            print(f"Training horizon {horizon}min: {len(X)} samples, {X.shape[1]} features")
            
            # Train model
            metrics = forecaster.train(X, y, horizon)
            print(f"Horizon {horizon}min - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
            
            # Make prediction
            pred = forecaster.predict(latest_features, horizon)[0]
            
            # Save forecast to file
            write_ml_forecast_to_file(storage, last_ts, product_id, horizon, pred, model_type, metrics)
            
            # Save model
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            forecaster.save_model(f"{model_dir}/{product_id}", horizon)
            
            # Show feature importance
            importance = forecaster.get_feature_importance(horizon)
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top features for {horizon}min: {top_features}")
        
        print(f"ML forecasts written for {product_id}: horizons={list(datasets.keys())}")
        return True
    
    except Exception as e:
        print(f"Error training ML model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    product_id = sys.argv[1] if len(sys.argv) > 1 else "ENCHANTED_LAPIS_BLOCK"
    model_type = sys.argv[2] if len(sys.argv) > 2 else "lightgbm"
    train_and_forecast_ml_from_files(product_id, model_type)