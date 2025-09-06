"""
Feature Pipeline for Skyblock Economy Sim

This module builds features for machine learning models from bazaar price data.
Supports both database and no-database (file-based) modes based on configuration.

In no-database mode, reads from NDJSON files and outputs features to bazaar_features.ndjson.
In database mode, uses PostgreSQL with the original SQL-based feature calculations.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import json

def load_config() -> Dict[str, Any]:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_features_from_database():
    """Original database-based feature building logic."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 is required for database mode but not installed.")
        print("Install with: pip install psycopg2-binary")
        return
    
    cfg = load_config()
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    with conn, conn.cursor() as cur:
        # Compute mid, spread, spread_bps and simple moving averages using SQL window functions
        cur.execute("""
        WITH base AS (
          SELECT
            ts, product_id,
            (buy_price + sell_price)/2.0 AS mid_price,
            (sell_price - buy_price) AS spread,
            CASE WHEN (buy_price + sell_price) > 0 THEN
              (sell_price - buy_price) / ((buy_price + sell_price)/2.0) * 10000
            ELSE NULL END AS spread_bps
          FROM bazaar_snapshots
        ),
        ranked AS (
          SELECT
            ts, product_id, mid_price, spread, spread_bps,
            ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY ts) AS rn
          FROM base
        )
        INSERT INTO bazaar_features (ts, product_id, mid_price, spread, spread_bps, vol_window_30, ma_5, ma_15, ma_60)
        SELECT
          r.ts,
          r.product_id,
          r.mid_price,
          r.spread,
          r.spread_bps,
          -- Volatility proxy: stddev over a recent window (30 points)
          STDDEV_SAMP(r2.mid_price) OVER (PARTITION BY r.product_id ORDER BY r.ts ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS vol_window_30,
          AVG(r2.mid_price) OVER (PARTITION BY r.product_id ORDER BY r.ts ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma_5,
          AVG(r2.mid_price) OVER (PARTITION BY r.product_id ORDER BY r.ts ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS ma_15,
          AVG(r2.mid_price) OVER (PARTITION BY r.product_id ORDER BY r.ts ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma_60
        FROM ranked r
        JOIN ranked r2
          ON r.product_id = r2.product_id AND r.ts = r2.ts
        ON CONFLICT (ts, product_id) DO NOTHING;
        """)
    conn.close()
    print("Features built from database.")

def load_bazaar_data_from_files(data_directory: str) -> Optional[pd.DataFrame]:
    """Load bazaar data from NDJSON files."""
    data_path = Path(data_directory)
    
    # Look for bazaar files - first check for direct bazaar_snapshots.ndjson
    potential_files = [
        data_path / "bazaar_snapshots.ndjson",
        data_path / "bazaar" / "bazaar_snapshots.ndjson"
    ]
    
    # Also look for timestamped bazaar files
    bazaar_dir = data_path / "bazaar"
    if bazaar_dir.exists():
        import glob
        pattern_files = glob.glob(str(bazaar_dir / "bazaar_*.ndjson"))
        potential_files.extend(pattern_files)
    
    records = []
    files_found = 0
    
    for file_path in potential_files:
        if Path(file_path).exists():
            files_found += 1
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            records.append(record)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
    
    if files_found == 0:
        print(f"ERROR: No bazaar data files found in {data_directory}")
        print("Expected files: bazaar_snapshots.ndjson or bazaar/bazaar_*.ndjson")
        return None
    
    if not records:
        print("ERROR: No valid records found in bazaar data files")
        return None
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(records)} records from {files_found} file(s)")
    return df

def calculate_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features from raw bazaar data using pandas."""
    
    # Ensure required columns exist
    required_cols = ['product_id', 'buy_price', 'sell_price']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert timestamp column
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
    elif 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
        df = df.drop('timestamp', axis=1)
    else:
        # Use current time as approximation if no timestamp
        df['ts'] = pd.date_range(end=datetime.now(timezone.utc), periods=len(df), freq='5T')
    
    # Sort by timestamp and product_id
    df = df.sort_values(['product_id', 'ts']).reset_index(drop=True)
    
    # Calculate basic price features
    df['mid_price'] = (df['buy_price'] + df['sell_price']) / 2.0
    df['spread'] = df['sell_price'] - df['buy_price']
    
    # Calculate spread_bps - handle division by zero
    df['spread_bps'] = np.where(
        df['mid_price'] > 0,
        (df['spread'] / df['mid_price']) * 10000,
        0
    )
    
    # Calculate features per product using groupby
    feature_dfs = []
    
    for product_id, group in df.groupby('product_id'):
        group = group.sort_values('ts').reset_index(drop=True)
        
        # Calculate moving averages
        group['ma_5'] = group['mid_price'].rolling(window=5, min_periods=1).mean()
        group['ma_15'] = group['mid_price'].rolling(window=15, min_periods=1).mean()
        group['ma_60'] = group['mid_price'].rolling(window=60, min_periods=1).mean()
        
        # Calculate volatility (30-period rolling standard deviation)
        group['vol_window_30'] = group['mid_price'].rolling(window=30, min_periods=1).std()
        
        feature_dfs.append(group)
    
    # Combine all products back together
    result_df = pd.concat(feature_dfs, ignore_index=True)
    
    # Fill any remaining NaN values
    result_df = result_df.fillna(0)
    
    return result_df

def save_features_to_file(df: pd.DataFrame, output_path: str):
    """Save features DataFrame to NDJSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert DataFrame to NDJSON format
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            # Convert timestamps to ISO format for JSON serialization
            record = row.to_dict()
            if 'ts' in record and pd.notna(record['ts']):
                record['ts'] = record['ts'].isoformat()
            
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved {len(df)} feature records to {output_path}")

def build_features_from_files():
    """Build features from NDJSON files in no-database mode."""
    cfg = load_config()
    data_directory = cfg.get("no_database_mode", {}).get("data_directory", "data")
    
    # Load raw bazaar data
    print("Loading bazaar data from files...")
    df = load_bazaar_data_from_files(data_directory)
    
    if df is None:
        print("ERROR: Could not load bazaar data from files")
        return
    
    # Calculate features
    print("Calculating features...")
    features_df = calculate_features_from_dataframe(df)
    
    # Save to output file
    output_path = Path(data_directory) / "bazaar_features.ndjson"
    save_features_to_file(features_df, str(output_path))
    
    print("Features built from files.")

def build_features():
    """Build features - supports both database and no-database modes."""
    cfg = load_config()
    no_db_mode = cfg.get("no_database_mode", {}).get("enabled", False)
    
    if no_db_mode:
        print("Using no-database mode for feature building")
        build_features_from_files()
    else:
        print("Using database mode for feature building")
        build_features_from_database()

if __name__ == "__main__":
    build_features()