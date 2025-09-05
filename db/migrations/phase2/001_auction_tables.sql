-- Phase 2 Database Schema: Auction House Tables and Aggregates

-- Core auctions table for active/ongoing auctions
CREATE TABLE IF NOT EXISTS auctions (
    uuid text PRIMARY KEY,
    item_id text,
    item_name text,
    tier text,
    bin boolean DEFAULT false,
    starting_bid numeric,
    highest_bid numeric,
    start_time timestamptz,
    end_time timestamptz,
    seller text,
    bids_count integer DEFAULT 0,
    category text,
    attributes jsonb,
    raw_data jsonb, -- optional raw JSON capture when enabled
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Finalized auctions table
CREATE TABLE IF NOT EXISTS auctions_ended (
    uuid text PRIMARY KEY,
    item_id text,
    item_name text,
    tier text,
    bin boolean DEFAULT false,
    starting_bid numeric,
    highest_bid numeric,
    sale_price numeric, -- final sale price
    start_time timestamptz,
    end_time timestamptz,
    seller text,
    buyer text,
    bids_count integer DEFAULT 0,
    category text,
    attributes jsonb,
    raw_data jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_auctions_item_id_end_time ON auctions (item_id, end_time);
CREATE INDEX IF NOT EXISTS idx_auctions_bin_item_id ON auctions (bin, item_id) WHERE bin = true;
CREATE INDEX IF NOT EXISTS idx_auctions_end_time ON auctions (end_time);
CREATE INDEX IF NOT EXISTS idx_auctions_item_name ON auctions USING gin (to_tsvector('english', item_name));

CREATE INDEX IF NOT EXISTS idx_auctions_ended_item_id_end_time ON auctions_ended (item_id, end_time);
CREATE INDEX IF NOT EXISTS idx_auctions_ended_bin_item_id ON auctions_ended (bin, item_id) WHERE bin = true;
CREATE INDEX IF NOT EXISTS idx_auctions_ended_end_time ON auctions_ended (end_time);
CREATE INDEX IF NOT EXISTS idx_auctions_ended_item_name ON auctions_ended USING gin (to_tsvector('english', item_name));

-- Optional: Convert to hypertables if TimescaleDB is available
-- SELECT create_hypertable('auctions', 'end_time', if_not_exists => TRUE);
-- SELECT create_hypertable('auctions_ended', 'end_time', if_not_exists => TRUE);

-- Aggregated AH price views (15-minute intervals)
CREATE VIEW IF NOT EXISTS ah_prices_15m AS
WITH bins_15m AS (
    SELECT
        date_trunc('hour', end_time) + INTERVAL '15 minutes' * floor(extract(MINUTE FROM end_time) / 15) as time_bucket,
        item_id,
        sale_price
    FROM auctions_ended
    WHERE bin = true AND sale_price IS NOT NULL AND sale_price > 0
    AND end_time >= now() - INTERVAL '7 days'
)
SELECT
    time_bucket,
    item_id,
    count(*) as sale_count,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY sale_price) as p25_price,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) as median_price,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY sale_price) as p75_price,
    avg(sale_price) as avg_price,
    stddev(sale_price) as price_stddev,
    min(sale_price) as min_price,
    max(sale_price) as max_price
FROM bins_15m
GROUP BY time_bucket, item_id
ORDER BY time_bucket DESC, item_id;

-- Aggregated AH price views (1-hour intervals)  
CREATE VIEW IF NOT EXISTS ah_prices_1h AS
WITH bins_1h AS (
    SELECT
        date_trunc('hour', end_time) as time_bucket,
        item_id,
        sale_price
    FROM auctions_ended
    WHERE bin = true AND sale_price IS NOT NULL AND sale_price > 0
    AND end_time >= now() - INTERVAL '30 days'
)
SELECT
    time_bucket,
    item_id,
    count(*) as sale_count,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY sale_price) as p25_price,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) as median_price,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY sale_price) as p75_price,
    avg(sale_price) as avg_price,
    stddev(sale_price) as price_stddev,
    min(sale_price) as min_price,
    max(sale_price) as max_price
FROM bins_1h
GROUP BY time_bucket, item_id
ORDER BY time_bucket DESC, item_id;