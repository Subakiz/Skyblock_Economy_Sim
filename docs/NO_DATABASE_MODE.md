# No-Database Mode

This document describes the no-database mode feature that allows the Skyblock Economy Sim to operate without PostgreSQL by storing data as NDJSON files.

## Overview

No-database mode provides a lightweight alternative to the full database setup, storing ingested data as NDJSON (Newline Delimited JSON) files in the `data/` directory. This mode is:

- **Additive**: Does not modify existing database workflows
- **Opt-in**: Disabled by default, enabled through configuration
- **Compatible**: Works with existing collectors and API endpoints

## Configuration

Enable no-database mode in `config/config.yaml`:

```yaml
no_database_mode:
  enabled: true              # Enable file-based storage instead of database
  data_directory: "data"     # Directory to store NDJSON files
  max_file_size_mb: 100      # Max size per NDJSON file before rotation
  retention_hours: 168       # Keep data for 7 days (168 hours)
```

You can also enable it via environment variable:
```bash
export NO_DATABASE_MODE_ENABLED=true
```

## Data Storage

### Directory Structure
```
data/
├── bazaar/
│   ├── bazaar_20250905_12.ndjson
│   ├── bazaar_20250905_13.ndjson.gz
│   └── ...
├── auctions/
│   ├── auctions_20250905_12.ndjson
│   └── ...
└── auctions_ended/
    ├── auctions_ended_20250905_12.ndjson
    └── ...
```

### File Format

Each NDJSON file contains one JSON object per line:

**Bazaar data (`data/bazaar/`):**
```json
{"ts": "2025-09-05T12:00:00Z", "product_id": "ENCHANTED_LAPIS_LAZULI", "buy_price": 1000.0, "sell_price": 950.0, "buy_volume": 500, "sell_volume": 300}
```

**Ended auctions (`data/auctions_ended/`):**
```json
{"uuid": "abc123", "item_id": "ENCHANTED_LAPIS_BLOCK", "sale_price": 150000, "end_time": "2025-09-05T12:00:00Z", "timestamp": "2025-09-05T12:01:00Z"}
```

### File Management

- **Hourly rotation**: New files created each hour
- **Automatic compression**: Files compressed to `.gz` when they exceed size limit
- **Automatic cleanup**: Files older than retention period are deleted
- **Thread-safe**: Multiple collectors can write simultaneously

## Usage

### Data Collection

Collectors automatically detect the mode and write to both database and files when enabled:

```bash
# Bazaar data collection (dual mode)
python -m ingestion.bazaar_collector

# Auction data collection (dual mode)  
python -m ingestion.auction_collector
```

### API Service

The FastAPI service automatically detects the mode at startup:

```bash
uvicorn services.api.app:app --host 0.0.0.0 --port 8000
```

**Health endpoint shows current mode:**
```json
GET /healthz
{
  "status": "ok",
  "mode": "file-based",
  "data_source": "NDJSON files"
}
```

### Craft Profitability Analysis

The command-line tool works with both modes:

```bash
python -m modeling.profitability.crafting_profit ENCHANTED_LAPIS_BLOCK
```

Output will indicate which mode is being used:
```
Using file-based mode
```

### ML Model Training

Train ML models using file data:

```bash
python -m modeling.forecast.file_ml_forecaster ENCHANTED_LAPIS_BLOCK lightgbm
```

Output shows training progress:
```
Training lightgbm model for ENCHANTED_LAPIS_BLOCK with 600 data points
Training horizon 15min: 597 samples, 16 features
Horizon 15min - MAE: 8198.14, RMSE: 9698.70
Top features for 15min: [('momentum_1', 228), ('ma_ratio', 204), ...]
ML forecasts written for ENCHANTED_LAPIS_BLOCK: horizons=[15, 60, 240]
```

## API Endpoints

All existing endpoints now work in file-based mode:

### Available Endpoints

- ✅ `GET /healthz` - Health check with mode information
- ✅ `GET /prices/ah/{product_id}` - Auction house price statistics
- ✅ `GET /profit/craft/{product_id}` - Craft profitability analysis
- ✅ `GET /forecast/{product_id}` - ML-based price forecasting (now supported!)
- ✅ `POST /ml/train` - Train ML models using file data (now supported!)
- ✅ `POST /analysis/predictive` - Comprehensive market analysis (now supported!)
- ✅ `POST /simulation/market` - Agent-based market simulation
- ✅ `POST /backtest/run` - Backtesting strategies
- ✅ `GET /scenarios/available` - List available market scenarios
- ✅ `POST /scenarios/compare` - Compare market scenarios
- ✅ `GET /models/status` - Check trained ML model status

### Example Usage

**Get auction house prices:**
```bash
curl http://localhost:8000/prices/ah/ENCHANTED_LAPIS_BLOCK
```

**Get craft profitability:**
```bash
curl http://localhost:8000/profit/craft/ENCHANTED_LAPIS_BLOCK?horizon=1h&pricing=median
```

**Get ML-based forecast:** (now supported in file mode!)
```bash
curl http://localhost:8000/forecast/ENCHANTED_LAPIS_BLOCK?horizon_minutes=60
```

**Train ML models:**
```bash
curl -X POST http://localhost:8000/ml/train -H "Content-Type: application/json" \
     -d '{"product_id": "ENCHANTED_LAPIS_BLOCK", "model_type": "lightgbm", "horizons": [15, 60, 240]}'
```

**Run predictive analysis:**
```bash
curl -X POST http://localhost:8000/analysis/predictive -H "Content-Type: application/json" \
     -d '{"items": ["ENCHANTED_LAPIS_BLOCK"], "model_type": "lightgbm", "include_opportunities": true}'
```

## Performance Characteristics

### Advantages
- **No database setup required**
- **Lightweight and portable**
- **Human-readable data format**
- **Easy backup and synchronization**
- **No connection limits or locks**

### Limitations  
- **Simplified market context** - Cross-item features are approximated rather than calculated across all products
- **Memory usage** - Data loaded into memory for analysis (but optimized for efficiency)
- **File I/O overhead** - Complex analyses may be slower than database queries for large datasets

## Development and Testing

### Running Tests

```bash
# Test no-database mode functionality
python test_no_database_mode.py

# Manual API testing with file mode
python test_api_manual.py
```

### Creating Test Data

```python
from storage.ndjson_storage import NDJSONStorage

storage = NDJSONStorage()

# Add bazaar data
storage.append_record("bazaar", {
    "product_id": "ENCHANTED_LAPIS_LAZULI",
    "buy_price": 1000.0,
    "sell_price": 950.0,
    "ts": datetime.now(timezone.utc).isoformat()
})

# Add auction data  
storage.append_record("auctions_ended", {
    "item_id": "ENCHANTED_LAPIS_BLOCK",
    "sale_price": 150000,
    "timestamp": datetime.now(timezone.utc).isoformat()
})
```

## Migration

### From Database to Files

1. Enable no-database mode in config
2. Run collectors in dual mode to build file data
3. Verify API endpoints work with file data
4. Disable database mode if desired

### From Files to Database

1. Set up PostgreSQL and apply schema
2. Disable no-database mode in config
3. Import historical data if needed
4. Collectors will automatically use database

## Troubleshooting

### Common Issues

**"No price data available"**
- Ensure collectors have run and generated data files
- Check `data/` directory exists and contains recent files
- Verify retention settings haven't cleaned up recent data

**"File storage should be None when disabled"**
- Check configuration file has correct YAML syntax
- Verify `no_database_mode.enabled: false` if disabling
- Restart services after configuration changes

**Performance issues**
- Reduce retention period to limit data volume
- Increase file rotation size to reduce file count
- Consider database mode for high-volume scenarios

### Debug Commands

```bash
# Check current configuration mode
python -c "from modeling.profitability.file_data_access import get_file_storage_if_enabled; print('File mode:', get_file_storage_if_enabled() is not None)"

# List data files
find data/ -name "*.ndjson*" -exec ls -lh {} \;

# Check file contents
head data/bazaar/bazaar_*.ndjson
```

## Architecture Details

### Key Components

- **`storage/ndjson_storage.py`** - Core file storage and retrieval (enhanced with forecasts support)
- **`modeling/profitability/file_data_access.py`** - File-based data access layer
- **`modeling/profitability/file_feature_engineering.py`** - File-based feature engineering for ML
- **`modeling/forecast/file_ml_forecaster.py`** - File-based ML forecasting engine
- **Updated collectors** - Dual-mode support for database + files
- **Updated API service** - Full feature parity between modes

### Data Flow

```
Hypixel API → Collectors → NDJSON Files → Feature Engineering → ML Models → API Endpoints
                      ↘ Database (if enabled)              ↗
                                                      File-based Analysis
```

This architecture ensures the no-database mode is a complete replacement for all use cases while maintaining compatibility with the full database mode.