# Phase 2: Auction House Integration & Strategy Backtesting

This document outlines the Phase 2 implementation, which extends the SkyBlock Economic Modeling system with Auction House data ingestion, crafting profitability analysis, and a comprehensive backtesting framework.

## Overview

Phase 2 delivers real-time Auction House analytics, intelligent crafting profit calculations, and robust strategy backtesting capabilities. The system now supports advanced trading strategies with realistic execution modeling.

## Components

### 1. Auction House Data Ingestion

**Module**: `ingestion/auction_collector.py`

Continuously polls Hypixel's Auction House endpoints with smart pagination and rate limiting:

- **Active Auctions** (`/skyblock/auctions`): Streams all active listings with full pagination support
- **Ended Auctions** (`/skyblock/auctions_ended`): Captures completed sales with final prices
- **Features**: UUID deduplication, configurable raw data capture, exponential backoff on errors
- **Performance**: Handles 100+ pages per cycle within API rate limits

**Usage**:
```bash
python -m ingestion.auction_collector
```

### 2. Database Schema Extensions  

**Location**: `db/migrations/phase2/001_auction_tables.sql`

New tables and indexes for auction data:

```sql
-- Core auction tracking
auctions           -- Active listings
auctions_ended     -- Completed sales  

-- Price aggregation views
ah_prices_15m      -- 15-minute OHLC with percentiles  
ah_prices_1h       -- Hourly price summaries
```

**Key Features**:
- Optimized indexes for time-series queries
- TimescaleDB hypertable support (optional)
- Configurable price aggregation windows
- Efficient storage with JSONB attributes

### 3. Craft Profitability Engine

**Module**: `modeling/profitability/crafting_profit.py`

Intelligent profit analysis combining item ontology with live market data:

**Algorithm**:
1. Parse crafting recipes from item ontology
2. Fetch input costs from Bazaar/AH (recursive for complex items)
3. Calculate expected sale prices using configurable pricing methods
4. Apply realistic fees and compute ROI metrics

**Metrics**:
- Craft cost (materials + fees)
- Expected sale price (median/P25/P75)  
- Gross/net margins after fees
- ROI percentage
- Turnover-adjusted profitability
- Best crafting path

**Usage**:
```bash
# Basic analysis
python -m modeling.profitability.crafting_profit ENCHANTED_LAPIS_BLOCK

# Advanced options
python -m modeling.profitability.crafting_profit ENCHANTED_LAPIS_BLOCK \
  --horizon 1h --pricing median
```

### 4. Backtesting Framework

**Modules**: `backtesting/`

Comprehensive strategy simulation with realistic execution modeling:

#### Core Components:

**Data Feed** (`data_feed.py`):
- Streams historical Bazaar + AH snapshots  
- Configurable time intervals (15min default)
- Combined market data for strategy decisions

**Execution Engine** (`execution.py`):
- Simulates order fills with latency and slippage
- Separate fee structures for Bazaar vs AH
- Realistic constraints (inventory limits, capital)

**Strategies** (`strategies/`):
- **Flip Strategy**: Buy underpriced BINs, resell at market
- **Craft & Sell**: Profitable crafting with material sourcing  
- **Arbitrage**: Exploit Bazaar-AH price differences

**Engine** (`engine.py`):
- Portfolio tracking with inventory management
- Performance metrics (Sharpe, drawdown, win rate)
- Transaction history and P&L attribution

#### Backtesting Example:

```bash
# Run craft-and-sell strategy
python -m backtesting.run \
  --strategy craft_and_sell \
  --item ENCHANTED_LAPIS_BLOCK \
  --start 2025-08-01 \
  --end 2025-09-01 \
  --capital 10000000

# Output:
# ============================================================
# BACKTEST RESULTS  
# ============================================================
# Duration:              31 days
# Total Return:          1,247,350 coins (12.47%)
# Max Drawdown:          385,220 coins (3.85%)  
# Sharpe Ratio:          1.834
# Total Trades:          156
# Win Rate:              72.4%
# Final Value:           11,247,350 coins
```

### 5. API Extensions

**Enhanced FastAPI endpoints** in `services/api/app.py`:

#### New Endpoints:

**AH Prices**: `GET /prices/ah/{product_id}?window=1h`
```json
{
  "product_id": "ENCHANTED_LAPIS_BLOCK",
  "window": "1h", 
  "median_price": 150000,
  "p25_price": 135000,
  "p75_price": 165000,
  "sale_count": 24,
  "last_updated": "2025-01-15T14:30:00Z"
}
```

**Craft Profitability**: `GET /profit/craft/{product_id}?horizon=1h&pricing=median`
```json
{
  "product_id": "ENCHANTED_LAPIS_BLOCK",
  "craft_cost": 125000,
  "expected_sale_price": 150000,
  "gross_margin": 25000,
  "net_margin": 23750,
  "roi_percent": 19.0,
  "turnover_adj_profit": 21375,
  "best_path": "buy(ENCHANTED_LAPIS_LAZULI)",
  "sell_volume": 24,
  "data_age_minutes": 15
}
```

**Backtest Runner**: `POST /backtest/run`  
```json
{
  "strategy": "flip_bin",
  "params": {"min_profit": 50000, "max_position": 10},
  "start_date": "2025-08-01", 
  "end_date": "2025-09-01",
  "capital": 10000000,
  "item_id": "ENCHANTED_LAPIS_BLOCK"
}
```

## Configuration

**Enhanced `config/config.yaml`**:

```yaml
# Auction House settings
auction_house:
  poll_interval_seconds: 300    # 5-minute cycles
  max_pages_per_cycle: 100      # Pagination limit
  enable_raw_capture: false     # Store full JSON
  fee_bps: 100                 # 1% AH fee
  tax_bps: 0                   # Additional taxes

# Profitability defaults  
profitability:
  default_horizon: "1h"        # Price window
  default_pricing: "median"    # Pricing method
  min_sell_volume: 10          # Liquidity threshold

# Backtesting parameters
backtesting:
  default_capital: 10000000    # Starting capital
  execution_latency_ms: 500    # Order delay
  slippage_bps: 10            # 0.1% slippage  
  max_inventory_slots: 1000    # Storage limit
```

## Deployment

### Database Migration:

```bash
# Apply Phase 2 schema
psql "$DATABASE_URL" -f db/migrations/phase2/001_auction_tables.sql

# Optional: Enable TimescaleDB 
psql "$DATABASE_URL" -c "
SELECT create_hypertable('auctions', 'end_time', if_not_exists => TRUE);
SELECT create_hypertable('auctions_ended', 'end_time', if_not_exists => TRUE);
"
```

### Data Collection:

```bash
# Start auction data collection
python -m ingestion.auction_collector

# Continue bazaar collection
python -m ingestion.bazaar_collector
```

### API Server:

```bash  
uvicorn services.api.app:app --host 0.0.0.0 --port 8000
```

## Testing

**Unit Tests**:
```bash
python -m unittest tests.test_phase2 -v
python -m unittest tests.test_api_phase2 -v
```

**Integration Test**:
```bash
# Test craft profitability
python -m modeling.profitability.crafting_profit ENCHANTED_LAPIS_BLOCK

# Test backtesting
python -m backtesting.run \
  --strategy flip_bin \
  --item ENCHANTED_LAPIS_BLOCK \
  --start 2025-01-01 --end 2025-01-07 \
  --capital 1000000
```

## Architecture Notes

### Performance Considerations:
- **Pagination**: AH collector handles 100+ pages efficiently
- **Rate Limiting**: Respects Hypixel's 120 requests/minute limit  
- **Indexing**: Optimized queries for time-series data
- **Memory**: Streaming data processing prevents memory bloat

### Error Handling:
- **Resilience**: Exponential backoff on API failures
- **Deduplication**: UUID-based duplicate prevention
- **Validation**: Input sanitization and type checking
- **Graceful Degradation**: Fallback pricing when data unavailable

### Extensibility:
- **Modular Strategies**: Easy to add new backtesting strategies
- **Configurable Fees**: All fees and taxes are configurable
- **Multiple Pricing**: Support for various pricing methodologies
- **Optional Features**: Raw data capture, TimescaleDB, etc.

## Known Limitations

1. **Historical Data**: Backtesting requires existing data (cold start problem)
2. **Market Impact**: Execution model doesn't account for large order impact  
3. **Inventory Tracking**: Real inventory status not synced with game
4. **Strategy Complexity**: Current strategies are relatively simple
5. **Real-time Execution**: Actual trading requires manual intervention

## Future Enhancements (Phase 3)

- **Advanced ML Models**: LightGBM/XGBoost for price prediction
- **Agent-Based Modeling**: Multi-agent market simulation
- **Portfolio Optimization**: Risk-adjusted position sizing  
- **Real-time Alerts**: Price alerts and opportunity notifications
- **Advanced Strategies**: Mean reversion, momentum, seasonal patterns

---

**Phase 2 Status**: ✅ Complete  
**Next Phase**: Advanced ML and Agent-Based Modeling