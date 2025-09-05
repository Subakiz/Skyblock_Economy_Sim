-- Core time-series table for Bazaar "quick_status" snapshots
CREATE TABLE IF NOT EXISTS bazaar_snapshots (
  ts timestamptz NOT NULL,
  product_id text NOT NULL,
  buy_price numeric,
  sell_price numeric,
  buy_volume numeric,
  sell_volume numeric,
  buy_orders integer,
  sell_orders integer,
  PRIMARY KEY (ts, product_id)
);

-- Optional: Convert to hypertable if TimescaleDB is available
-- SELECT create_hypertable('bazaar_snapshots', 'ts', if_not_exists => TRUE);

-- Derived features table (one row per ts, product_id)
CREATE TABLE IF NOT EXISTS bazaar_features (
  ts timestamptz NOT NULL,
  product_id text NOT NULL,
  mid_price numeric,            -- (buy + sell)/2
  spread numeric,               -- sell - buy
  spread_bps numeric,           -- (sell - buy)/mid * 10000
  vol_window_30 numeric,        -- realized volatility proxy
  ma_5 numeric,
  ma_15 numeric,
  ma_60 numeric,
  PRIMARY KEY (ts, product_id)
);

-- Simple model registry for price forecasting
CREATE TABLE IF NOT EXISTS model_forecasts (
  ts timestamptz NOT NULL,
  product_id text NOT NULL,
  horizon_minutes integer NOT NULL,
  forecast_price numeric,
  model_version text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (ts, product_id, horizon_minutes)
);