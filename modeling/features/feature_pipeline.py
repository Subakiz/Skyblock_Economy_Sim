import os
import yaml
import psycopg2
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_features():
    cfg = load_config()
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    lookbacks = cfg["features"]["lookbacks"]

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
    print("Features built.")

if __name__ == "__main__":
    build_features()