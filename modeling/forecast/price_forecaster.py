import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import joblib
import lightgbm as lgb
import warnings

# File-based imports for no-database mode
from storage.ndjson_storage import get_storage_instance
from modeling.forecast.file_ml_forecaster import (
    fetch_multivariate_series_from_files, 
    create_features_targets_from_files,
    write_ml_forecast_to_file,
    FileBasedMLForecaster
)

warnings.filterwarnings('ignore')

def load_config() -> Dict[str, Any]:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def is_no_database_mode(cfg: Dict[str, Any]) -> bool:
    """Check if no-database mode is enabled."""
    return cfg.get("no_database_mode", {}).get("enabled", False)

# Database-based functions (original logic)
def fetch_series(conn, product_id: str, min_points: int = 200):
    q = """
      SELECT ts, mid_price, ma_15, ma_60, spread_bps, vol_window_30
      FROM bazaar_features
      WHERE product_id = %s
      ORDER BY ts ASC
    """
    df = pd.read_sql(q, conn, params=(product_id,))
    if len(df) < min_points:
        return None
    return df

def naive_forecast(df: pd.DataFrame, horizon_minutes: int) -> float:
    # Simple baseline: forward-fill last MA15 as forecast
    last = df.iloc[-1]
    return float(last["ma_15"] if pd.notnull(last["ma_15"]) else last["mid_price"])

def ml_forecast(df: pd.DataFrame, product_id: str, horizon_minutes: int) -> float:
    """
    LightGBM-based forecasting to replace naive approach.
    """
    try:
        # Prepare features - use last few rows for context
        feature_cols = ['ma_15', 'ma_60', 'spread_bps', 'vol_window_30']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 2 or len(df) < 10:
            # Fallback to naive if insufficient features
            return naive_forecast(df, horizon_minutes)
        
        # Create simple features from available data
        features_df = df[available_cols].copy()
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Add simple derived features
        if 'ma_15' in features_df.columns and 'ma_60' in features_df.columns:
            features_df['ma_ratio'] = features_df['ma_15'] / features_df['ma_60'].replace(0, np.nan)
            features_df['ma_ratio'] = features_df['ma_ratio'].fillna(1.0)
        
        # Simple target: next mid_price (approximation for different horizons)
        if 'mid_price' not in df.columns:
            return naive_forecast(df, horizon_minutes)
            
        target = df['mid_price'].copy()
        
        # Create training data (use historical lookback)
        lookback = min(100, len(df) - 5)  # Use up to 100 historical points
        if lookback < 5:
            return naive_forecast(df, horizon_minutes)
        
        X = features_df.iloc[-lookback:-1].values  # All but last row
        y = target.iloc[-lookback+1:].values      # Shifted by 1
        
        if len(X) != len(y) or len(X) < 5:
            return naive_forecast(df, horizon_minutes)
        
        # Quick LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbosity=-1
        )
        
        model.fit(X, y)
        
        # Predict using latest features
        latest_features = features_df.iloc[-1:].values
        prediction = model.predict(latest_features)[0]
        
        return float(prediction)
        
    except Exception as e:
        print(f"ML forecast failed for {product_id}: {e}")
        return naive_forecast(df, horizon_minutes)

def write_forecast(conn, ts, product_id, horizon, price, version="naive-v0"):
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO model_forecasts (ts, product_id, horizon_minutes, forecast_price, model_version)
          VALUES (%s, %s, %s, %s, %s)
          ON CONFLICT (ts, product_id, horizon_minutes) DO UPDATE
          SET forecast_price = EXCLUDED.forecast_price,
              model_version = EXCLUDED.model_version
        """, (ts, product_id, horizon, price, version))

def write_forecast_to_file(storage, ts, product_id: str, horizon: int, price: float, version: str = "ml-v1"):
    """Write forecast to NDJSON file for no-database mode."""
    forecast_record = {
        'ts': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
        'product_id': product_id,
        'horizon_minutes': horizon,
        'forecast_price': float(price),
        'model_version': version,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    storage.append_record("forecasts", forecast_record)
    return True

def train_and_forecast_database(product_id: str, horizons=(15, 60, 240)):
    """Original database-based training and forecasting."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 is required for database mode but not installed.")
        print("Install with: pip install psycopg2-binary")
        return False
        
    cfg = load_config()
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

    df = fetch_series(conn, product_id)
    if df is None or df.empty:
        print(f"Not enough data for {product_id}")
        conn.close()
        return False

    last_ts = df["ts"].iloc[-1]
    for h in horizons:
        # Use ML forecast instead of naive
        pred = ml_forecast(df, product_id, h)
        write_forecast(conn, last_ts, product_id, h, pred, version="lgbm-v1")

    conn.close()
    print(f"Database forecasts written for {product_id}: horizons={horizons}")
    return True

def train_and_forecast_files(product_id: str, horizons=(15, 60, 240)):
    """File-based training and forecasting using proper ML models."""
    cfg = load_config()
    storage = get_storage_instance(cfg)
    
    if storage is None:
        print("No-database mode not enabled or storage initialization failed")
        return False
    
    try:
        # Try to use the advanced file-based ML forecaster first
        from modeling.forecast.file_ml_forecaster import train_and_forecast_ml_from_files
        success = train_and_forecast_ml_from_files(product_id, "lightgbm", horizons)
        if success:
            return True
    except Exception as e:
        print(f"Advanced ML forecaster failed, falling back to simple ML: {e}")
    
    # Fallback to simpler ML approach using available data
    try:
        # Load feature data from bazaar_features.ndjson 
        feature_records = []
        try:
            feature_records = list(storage.read_records("bazaar_features", hours_back=168))
            # Filter for the specific product
            feature_records = [r for r in feature_records if r.get('product_id') == product_id]
        except Exception as e:
            print(f"Could not load from bazaar_features: {e}, trying direct bazaar data")
            
        if not feature_records:
            # Try to create basic features from bazaar data
            bazaar_records = list(storage.read_records("bazaar", hours_back=168))
            product_records = [r for r in bazaar_records if r.get('product_id') == product_id]
            
            if len(product_records) < 50:
                print(f"Not enough bazaar data for {product_id} (need 50+, got {len(product_records)})")
                return False
            
            # Convert to DataFrame and create basic features
            df = pd.DataFrame(product_records)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts')
            
            # Create mid_price and basic features
            df['mid_price'] = (df['buy_price'] + df['sell_price']) / 2
            df['spread_bps'] = ((df['sell_price'] - df['buy_price']) / df['mid_price'] * 10000).fillna(0)
            
            # Calculate moving averages
            df['ma_15'] = df['mid_price'].rolling(window=15, min_periods=1).mean()
            df['ma_60'] = df['mid_price'].rolling(window=60, min_periods=1).mean()
            df['vol_window_30'] = df['mid_price'].rolling(window=30, min_periods=1).std()
            
        else:
            # Use feature records directly
            df = pd.DataFrame(feature_records)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts')
        
        if len(df) < 50:
            print(f"Not enough data for {product_id} (need 50+, got {len(df)})")
            return False
        
        print(f"Training ML models for {product_id} with {len(df)} data points")
        
        last_ts = df["ts"].iloc[-1]
        models_saved = 0
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        for h in horizons:
            try:
                # Use improved ML forecast
                pred = ml_forecast(df, product_id, h)
                
                # Save forecast to file
                write_forecast_to_file(storage, last_ts, product_id, h, pred, version="lgbm-v1")
                
                # Train and save a more robust model for this horizon
                try:
                    model_path = f"models/lgbm_forecaster_{product_id}_h{h}.joblib"
                    
                    # Prepare features for a proper model
                    feature_cols = ['ma_15', 'ma_60', 'spread_bps', 'vol_window_30']
                    available_cols = [col for col in feature_cols if col in df.columns]
                    
                    if len(available_cols) >= 2 and len(df) > 10:
                        X = df[available_cols].fillna(method='ffill').fillna(0).iloc[:-1].values
                        y = df['mid_price'].iloc[1:].values
                        
                        if len(X) == len(y) and len(X) > 5:
                            model = lgb.LGBMRegressor(
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=5,
                                random_state=42,
                                verbosity=-1
                            )
                            model.fit(X, y)
                            joblib.dump(model, model_path)
                            models_saved += 1
                            print(f"Model saved for horizon {h}min: {model_path}")
                
                except Exception as e:
                    print(f"Could not save model for horizon {h}: {e}")
                
            except Exception as e:
                print(f"Failed to forecast for horizon {h}: {e}")
        
        print(f"File-based forecasts written for {product_id}: horizons={horizons}")
        if models_saved > 0:
            print(f"Saved {models_saved} model files to models/ directory")
        return True
        
    except Exception as e:
        print(f"File-based forecasting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_and_forecast(product_id: str, horizons=(15, 60, 240)):
    """Main training function with dual-mode support."""
    cfg = load_config()
    
    if is_no_database_mode(cfg):
        print(f"Using no-database mode for {product_id}")
        return train_and_forecast_files(product_id, horizons)
    else:
        print(f"Using database mode for {product_id}")
        return train_and_forecast_database(product_id, horizons)

if __name__ == "__main__":
    # Example single-product training. For batch, iterate distinct product_ids from features.
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "ENCHANTED_LAPIS_BLOCK"
    train_and_forecast(pid)