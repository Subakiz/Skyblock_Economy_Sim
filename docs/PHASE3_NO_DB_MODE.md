# Phase 3 CLI - No-Database Mode

The Phase 3 CLI now supports a complete "no-database" mode that works entirely with NDJSON files, enabling advanced ML analysis and market simulation for Auction House items without requiring PostgreSQL.

## Key Features

### 🎯 Auction House Data Support
- Processes raw auction data from `data/auctions.ndjson` and `data/auctions_ended/` files
- Supports high-value items like **HYPERION** and **NECRON_CHESTPLATE**
- Calculates auction-specific features:
  - `price_per_hour`: BIN price divided by auction duration
  - `is_clean`: Boolean indicating items without extra enchantments
  - `rolling_avg_price`: Average prices over time windows

### 🤖 File-Based ML Forecasting
- Trains LightGBM/XGBoost models directly from NDJSON files
- Generates price predictions for multiple horizons (15min, 60min, 240min)
- Saves trained models locally in `models/` directory
- No database connection required

### 📊 Market Simulation Integration
- Combines ML predictions with agent-based market simulation
- Tests multiple market scenarios (normal, volatile, stable)
- Provides trading opportunities with confidence scores
- Generates comprehensive market insights

## Usage Examples

### Analyze Auction Items
```bash
# Complete analysis with ML predictions and market simulation
python phase3_cli.py analyze "HYPERION,NECRON_CHESTPLATE" --model-type lightgbm

# Save results to file
python phase3_cli.py analyze "HYPERION" --output hyperion_analysis.json
```

### Train Individual Models
```bash
# Train model for specific item
python phase3_cli.py train HYPERION --model-type lightgbm --horizons 15 60 240

# Train with XGBoost
python phase3_cli.py train NECRON_CHESTPLATE --model-type xgboost
```

### Check System Status
```bash
python phase3_cli.py status
```

## Sample Output

```
Running predictive analysis for items: ['HYPERION', 'NECRON_CHESTPLATE']
Starting comprehensive market analysis...

Step 1: Training ML models...
Training predictive model for HYPERION...
HYPERION - Horizon 15min: MAE=93451012.38
HYPERION - Horizon 60min: MAE=93451012.38
HYPERION - Horizon 240min: MAE=99049238.23

Step 2: Generating ML predictions...
Step 3: Running market simulations...
Step 4: Generating market insights...

Predictive Analysis Summary:
Items analyzed: 2

Price Predictions:
  HYPERION:
    15min: 820512584.09 coins
    60min: 820512584.09 coins
    240min: 782902158.14 coins
  NECRON_CHESTPLATE:
    15min: 54503581.83 coins
    60min: 54503581.83 coins
    240min: 52590400.04 coins

Market outlook: bearish
Trading opportunities found: 2
  1. NECRON_CHESTPLATE: buy (12.4% return, 0.20 confidence)
  2. HYPERION: buy (7.8% return, 0.10 confidence)
```

## Data Requirements

### Input Files
- `data/auctions_ended/*.ndjson` - Ended auction records
- `data/auctions/*.ndjson` - Active auction records (optional)

### Generated Files
- `data/auction_features.ndjson` - Processed auction features
- `models/*.pkl` - Trained ML models

## Configuration

Ensure `config/config.yaml` has no-database mode enabled:

```yaml
no_database_mode:
  enabled: true
  data_directory: "data"
  max_file_size_mb: 100
  retention_hours: 168
```

## Testing

Run the comprehensive test suite:

```bash
python test_phase3_no_db.py
```

This validates:
- ✅ Auction feature pipeline
- ✅ ML model training from files
- ✅ Price prediction generation
- ✅ Market simulation integration
- ✅ No database dependencies

## Architecture

The no-database mode consists of:

1. **Auction Feature Pipeline** (`modeling/features/auction_feature_pipeline.py`)
   - Loads raw auction data
   - Extracts item-specific features
   - Handles item name analysis for "clean" detection

2. **File-Based Predictive Engine** (`modeling/simulation/file_predictive_engine.py`)
   - Replaces database-dependent PredictiveMarketEngine
   - Uses NDJSON storage throughout
   - Integrates with existing agent simulation

3. **Updated Phase 3 CLI** (`phase3_cli.py`)
   - Removes all PostgreSQL dependencies
   - Enhanced error handling
   - Better progress reporting

This implementation enables sophisticated economic analysis of SkyBlock auction items without requiring database infrastructure.