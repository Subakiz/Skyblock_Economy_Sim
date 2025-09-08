# Enhanced Features: Snipes & Plot

This document describes the newly implemented features for restoring snipes (feature summaries) and enhancing the /plot command.

## Overview

The implementation addresses the core issue where the sniper loop was logging "No recent summaries found" due to missing feature summary data. The solution provides:

1. **Configurable feature summaries path** with robust fallback handling
2. **Continuous feature ingestion script** for uninterrupted data collection  
3. **Enhanced /plot command** with professional market visualization

## Feature Consumer Enhancements

### Configuration
The `FeatureConsumer` now reads the feature summaries path from `config.yaml`:

```yaml
market:
  feature_summaries_path: data/feature_summaries  # Configurable path
```

### Enhanced Logging
The consumer now provides detailed logging:
- Resolved path information
- Number of files found and loaded
- Clear error messages for missing data
- Graceful fallback when PyArrow filtering fails

### Robust Data Loading
- Primary: PyArrow dataset with `hour_start` filtering
- Fallback: Manual directory scanning for partitioned data
- Tolerant to missing columns and malformed partitions

## Continuous Feature Ingestor

### Usage
```bash
# Set required environment variable
export HYPIXEL_API_KEY="your_api_key_here"

# Run the continuous ingestor
python scripts/run_feature_ingestor.py
```

### Features
- **Continuous Operation**: Long-running loop with configurable intervals
- **Interim Flushing**: Data written every 2.5 minutes for real-time availability
- **Memory Monitoring**: Built-in memory usage tracking with configurable limits
- **Graceful Shutdown**: Handles SIGINT/SIGTERM for clean exit
- **Partitioned Storage**: Data organized as `year=/month=/day=/hour=/summary.parquet`

### Data Format
Each summary includes:
- `hour_start`: UTC timestamp for the hour
- `item_name`: Skyblock item identifier  
- `prices`: Array of price levels (lowest ladder)
- `counts`: Array of auction counts per price level
- `total_count`: Total auctions for the item
- `floor_price`: Lowest available price
- `second_lowest_price`: Second tier pricing
- `auction_count`: Total auction volume

## Enhanced /plot Command

### Usage
```
/plot item:WHEAT hours:3
/plot item:ENCHANTED_FLINT hours:6
```

### Features
- **Two-Panel Layout**: 
  - Top: Buy/sell prices with mid line and spread band
  - Bottom: Volume bars and order count lines
- **Data Processing**:
  - 1-minute median resampling for smooth visualization
  - 1-99 percentile clipping to eliminate spikes
  - Robust timestamp and column detection
- **Professional Visualization**:
  - Shaded spread bands for market gap analysis
  - Dual-axis plotting for volumes and order counts
  - Smart axis formatting (k/M notation)
  - Market statistics in Discord embed

### Data Sources
The plot command automatically detects and loads data from:
1. `data/bazaar/` directory (Parquet files)
2. `data/bazaar_snapshots.ndjson` file
3. Flexible column name detection for various data formats

### Computed Metrics
- **Mid Price**: Average of buy and sell prices
- **Spread**: Absolute difference between sell and buy prices  
- **Spread BPS**: Spread as basis points (10000 * spread / mid)
- **Volume Imbalance**: Ratio of buy vs sell volume

## Integration Testing

### Test Feature Consumer
```python
from ingestion.feature_consumer import FeatureConsumer
import yaml

config = yaml.safe_load(open('config/config.yaml'))
fc = FeatureConsumer(config)
intelligence = fc.generate_market_intelligence()
print(f"Watchlist: {len(intelligence['watchlist'])} items")
```

### Test Plot Functionality  
Create test data and verify the complete pipeline works end-to-end.

## Production Deployment

### Step 1: Configure Environment
```bash
export HYPIXEL_API_KEY="your_key"
export DISCORD_BOT_TOKEN="your_token"  
```

### Step 2: Start Feature Ingestion
```bash
# Start in background with nohup
nohup python scripts/run_feature_ingestor.py > logs/ingestor.log 2>&1 &
```

### Step 3: Start Discord Bot
```bash
python bot.py
```

### Step 4: Verify Operation
- Monitor logs for "Market intelligence generated" messages
- Use `/plot` commands to test visualization
- Check `data/feature_summaries/` for hourly Parquet files

## Monitoring

### Logs to Watch
- `FeatureConsumer` logs showing successful data loading
- `AuctionSniper` logs changing from "0 watchlist" to positive numbers
- Plot command successful responses with market statistics

### Data Verification
```bash
# Check for recent feature summaries
find data/feature_summaries -name "*.parquet" -mtime -1

# Verify data structure
python -c "import pandas as pd; print(pd.read_parquet('path/to/summary.parquet'))"
```

## Troubleshooting

### Common Issues
1. **No HYPIXEL_API_KEY**: Feature ingestor won't start
2. **Missing directories**: Auto-created by the system
3. **Permission errors**: Ensure write access to data/ directory
4. **Memory usage**: Monitor with built-in psutil tracking

### Performance Tuning
- Adjust `intel_interval_seconds` in config for ingestion frequency
- Configure memory limits in `guards` section
- Set appropriate retention policies in `storage` section

The implementation provides a robust, production-ready solution for continuous market intelligence with professional-quality visualizations.