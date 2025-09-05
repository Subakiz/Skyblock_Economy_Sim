"""
Phase 3: Advanced Machine Learning Forecasters
Implements LightGBM and XGBoost models for multivariate price prediction.
"""

import os
import yaml
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_config() -> Dict[str, Any]:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def fetch_multivariate_series(conn, product_id: str, min_points: int = 500) -> Optional[pd.DataFrame]:
    """Fetch enhanced feature set for multivariate modeling."""
    q = """
      WITH cross_item_features AS (
        -- Get features for the target item
        SELECT 
            bf.ts, bf.product_id, bf.mid_price, bf.ma_15, bf.ma_60, 
            bf.spread_bps, bf.vol_window_30,
            -- Cross-item dependencies (related items)
            lag(bf.mid_price, 1) OVER (PARTITION BY bf.product_id ORDER BY bf.ts) as price_lag_1,
            lag(bf.mid_price, 5) OVER (PARTITION BY bf.product_id ORDER BY bf.ts) as price_lag_5,
            -- Price momentum features
            (bf.mid_price - lag(bf.mid_price, 1) OVER (PARTITION BY bf.product_id ORDER BY bf.ts)) / 
                NULLIF(lag(bf.mid_price, 1) OVER (PARTITION BY bf.product_id ORDER BY bf.ts), 0) * 100 as momentum_1,
            (bf.mid_price - lag(bf.mid_price, 5) OVER (PARTITION BY bf.product_id ORDER BY bf.ts)) / 
                NULLIF(lag(bf.mid_price, 5) OVER (PARTITION BY bf.product_id ORDER BY bf.ts), 0) * 100 as momentum_5,
            -- MA crossovers and ratios
            CASE WHEN bf.ma_15 > bf.ma_60 THEN 1 ELSE 0 END as ma_crossover,
            bf.ma_15 / NULLIF(bf.ma_60, 0) as ma_ratio,
            -- Volume-price indicators  
            bf.vol_window_30 / NULLIF(bf.mid_price, 0) as vol_price_ratio,
            -- Time-based features
            EXTRACT(hour FROM bf.ts) as hour_of_day,
            EXTRACT(dow FROM bf.ts) as day_of_week,
            EXTRACT(day FROM bf.ts) as day_of_month
        FROM bazaar_features bf
        WHERE bf.product_id = %s
      ),
      market_context AS (
        -- Market-wide indicators
        SELECT 
            cif.ts,
            cif.product_id,
            cif.*,
            -- Market volatility context (average volatility across all items at this timestamp)
            (SELECT AVG(vol_window_30) FROM bazaar_features bf2 WHERE bf2.ts = cif.ts) as market_volatility,
            -- Market trend (average price change across all items)
            (SELECT AVG((mid_price - lag(mid_price, 1) OVER (PARTITION BY product_id ORDER BY ts)) / 
                       NULLIF(lag(mid_price, 1) OVER (PARTITION BY product_id ORDER BY ts), 0) * 100) 
             FROM bazaar_features bf3 WHERE bf3.ts = cif.ts) as market_momentum
        FROM cross_item_features cif
      )
      SELECT * FROM market_context 
      ORDER BY ts ASC
    """
    df = pd.read_sql(q, conn, params=(product_id,))
    if len(df) < min_points:
        return None
    
    # Fill NaN values with forward-fill and then backward-fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def create_features_targets(df: pd.DataFrame, horizons: List[int] = [15, 60, 240]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create feature matrices and target vectors for different prediction horizons."""
    
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

class MLForecaster:
    """Enhanced ML forecaster using LightGBM and XGBoost."""
    
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

def write_ml_forecast(conn, ts, product_id, horizon, price, model_type, metrics=None):
    """Write ML forecast to database."""
    version = f"{model_type}-v1"
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO model_forecasts (ts, product_id, horizon_minutes, forecast_price, model_version, model_metrics)
          VALUES (%s, %s, %s, %s, %s, %s)
          ON CONFLICT (ts, product_id, horizon_minutes) DO UPDATE
          SET forecast_price = EXCLUDED.forecast_price,
              model_version = EXCLUDED.model_version,
              model_metrics = EXCLUDED.model_metrics
        """, (ts, product_id, horizon, price, version, metrics))

def train_and_forecast_ml(product_id: str, model_type: str = 'lightgbm', horizons=(15, 60, 240)):
    """Train ML model and generate forecasts."""
    cfg = load_config()
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    
    try:
        # Fetch multivariate data
        df = fetch_multivariate_series(conn, product_id)
        if df is None or df.empty:
            print(f"Not enough data for {product_id}")
            return
        
        print(f"Training {model_type} model for {product_id} with {len(df)} data points")
        
        # Prepare feature sets for different horizons
        datasets = create_features_targets(df, horizons)
        
        forecaster = MLForecaster(model_type=model_type)
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
            
            # Save forecast
            write_ml_forecast(conn, last_ts, product_id, horizon, pred, model_type, metrics)
            
            # Save model
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            forecaster.save_model(f"{model_dir}/{product_id}", horizon)
            
            # Show feature importance
            importance = forecaster.get_feature_importance(horizon)
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top features for {horizon}min: {top_features}")
        
        print(f"ML forecasts written for {product_id}: horizons={list(datasets.keys())}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    product_id = sys.argv[1] if len(sys.argv) > 1 else "ENCHANTED_LAPIS_BLOCK"
    model_type = sys.argv[2] if len(sys.argv) > 2 else "lightgbm"
    train_and_forecast_ml(product_id, model_type)