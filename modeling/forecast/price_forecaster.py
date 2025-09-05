import os
import yaml
import psycopg2
import pandas as pd
from datetime import timedelta

from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

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

def write_forecast(conn, ts, product_id, horizon, price, version="naive-v0"):
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO model_forecasts (ts, product_id, horizon_minutes, forecast_price, model_version)
          VALUES (%s, %s, %s, %s, %s)
          ON CONFLICT (ts, product_id, horizon_minutes) DO UPDATE
          SET forecast_price = EXCLUDED.forecast_price,
              model_version = EXCLUDED.model_version
        """, (ts, product_id, horizon, price, version))

def train_and_forecast(product_id: str, horizons=(15, 60, 240)):
    cfg = load_config()
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    conn = psycopg2.connect(db_url)
    conn.autocommit = True

    df = fetch_series(conn, product_id)
    if df is None or df.empty:
        print(f"Not enough data for {product_id}")
        return

    last_ts = df["ts"].iloc[-1]
    for h in horizons:
        pred = naive_forecast(df, h)
        write_forecast(conn, last_ts, product_id, h, pred)

    conn.close()
    print(f"Forecasts written for {product_id}: horizons={horizons}")

if __name__ == "__main__":
    # Example single-product training. For batch, iterate distinct product_ids from features.
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "ENCHANTED_LAPIS_BLOCK"
    train_and_forecast(pid)