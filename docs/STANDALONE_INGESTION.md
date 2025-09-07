# Standalone Data Ingestion Service

This document describes the new standalone data ingestion architecture for the Skyblock Economy Simulator.

## Architecture Overview

The system is now split into two main components:

1. **Standalone Data Ingestion Service** (`ingestion/standalone_ingestion_service.py`)
   - Continuously fetches data from Hypixel API
   - Processes and cleans the data
   - Writes to partitioned Parquet datasets
   - Runs independently of the Discord bot

2. **Read-Only Auction Sniper Cog** (`cogs/auction_sniper.py`)  
   - No longer fetches auction data directly
   - Reads from Parquet datasets created by ingestion service
   - Maintains all sniper functionality
   - Hunter task still fetches page 0 for real-time detection

## Key Features

### Canonical Item Loader
- Fetches valid item names from `/resources/skyblock/items` endpoint
- Caches items to disk for offline operation  
- Falls back to cache when API is unavailable

### Intelligent Base Item ID Cleaner
- `_get_base_item_id(item_name, canonical_items)` function
- Finds longest matching substring from canonical list
- Handles complex auction titles with enchants, stars, etc.

### Concurrent Data Ingestion
- **Auction Loop**: Runs every 90 seconds
  - Fetches all pages from `/skyblock/auctions`
  - Processes and cleans auction data
  - Determines `base_item_id` using intelligent cleaner
  - Writes to `data/auction_history` Parquet dataset

- **Bazaar Loop**: Runs every 60 seconds
  - Fetches from `/skyblock/bazaar` endpoint
  - Flattens product data with buy/sell information
  - Writes to `data/bazaar_history` Parquet dataset

## Usage

### Starting the Standalone Ingestion Service

```bash
# Method 1: Direct execution
cd Skyblock_Economy_Sim
python ingestion/standalone_ingestion_service.py

# Method 2: Using the start script
python start_ingestion_service.py
```

**Requirements:**
- `HYPIXEL_API_KEY` environment variable must be set
- Dependencies: `pandas`, `pyarrow`, `requests`, `pyyaml`

### Using with Discord Bot

The Discord bot's auction sniper cog will automatically:
1. Read from the Parquet datasets created by the ingestion service
2. Update market intelligence every 60 seconds
3. Continue snipe detection using cached FMV data

**No changes needed** - the cog is backward compatible but optimized for the new architecture.

## Data Structure

### Auction History Dataset (`data/auction_history`)
```
├── year=2025/
│   ├── month=9/
│   │   ├── day=7/
│   │   │   ├── part-0.parquet
│   │   │   └── ...
```

**Schema:**
- `uuid`: Auction UUID
- `item_name`: Original item name from auction
- `base_item_id`: Cleaned item ID using canonical matching
- `price`: Final price (BIN price or highest bid)
- `tier`: Item tier (LEGENDARY, EPIC, etc.)
- `bin`: Boolean indicating Buy It Now auction
- `seller`: Seller UUID
- `bids`: Number of bids placed
- `start_timestamp`: Auction start time
- `end_timestamp`: Auction end time  
- `scan_timestamp`: When data was collected

### Bazaar History Dataset (`data/bazaar_history`)
```
├── year=2025/
│   ├── month=9/
│   │   ├── day=7/
│   │   │   ├── part-0.parquet
```

**Schema:**
- `product_id`: Bazaar product ID
- `buy_price`: Instant buy price
- `sell_price`: Instant sell price
- `buy_volume`: Total buy volume
- `sell_volume`: Total sell volume
- `buy_orders`: Number of buy orders
- `sell_orders`: Number of sell orders
- `top_buy_price`: Highest buy order price
- `top_buy_amount`: Amount in highest buy order
- `top_sell_price`: Lowest sell order price
- `top_sell_amount`: Amount in lowest sell order
- `scan_timestamp`: When data was collected

## Configuration

The service uses the existing `config/config.yaml` file:

```yaml
hypixel:
  base_url: "https://api.hypixel.net"
  max_requests_per_minute: 120
  timeout_seconds: 10
  endpoints:
    bazaar: "/skyblock/bazaar"
    auctions: "/skyblock/auctions"

auction_house:
  max_pages_per_cycle: 100
```

## Benefits

1. **Separation of Concerns**: Data collection is separate from application logic
2. **Scalability**: Ingestion service can run on different hardware/schedule
3. **Reliability**: Data collection continues even if Discord bot is offline
4. **Performance**: Bot cog is more responsive (no API delays)
5. **Data Consistency**: Single source of truth in Parquet datasets
6. **Analysis Ready**: Data is already cleaned and structured for analysis

## Migration Notes

- **Backward Compatible**: Existing Discord commands work unchanged
- **No Data Loss**: FMV calculations continue using Parquet data
- **Improved Performance**: Market intelligence updates faster (60s vs 90s)
- **Hunter Task Unchanged**: Still scans page 0 every 2 seconds for snipes

## Monitoring

Check logs for:
- Ingestion service: API connectivity, data processing rates
- Discord cog: Parquet read success, watchlist updates, FMV calculations

The system will gracefully handle API outages and missing data scenarios.