# SkyBlock Economy Discord Bot

A Discord bot that provides automated data collection and interactive analysis for the Hypixel SkyBlock economy.

## Features

### Background Tasks
- **Automated Data Collection**: Continuously collects auction and bazaar data from the Hypixel API
- **Periodic Feature Building**: Automatically processes raw data into ML features every hour
- **No-Database Mode**: Uses file-based storage (NDJSON format) for easy deployment

### Discord Commands

#### `/status` 
Check bot health and data pipeline status
- Shows bot uptime
- Displays data collection statistics  
- Reports ML model availability
- Shows last feature build time

#### `/analyze <item_name>`
Comprehensive market analysis for a specific item
- ML-powered price predictions for multiple horizons (15min, 1h, 4h)
- Market opportunities with confidence scores
- Training data statistics
- Market outlook assessment

#### `/predict <item_name> <time_horizon>`
Focused price prediction for a specific item and time horizon
- Choose from 15 minutes, 1 hour, or 4 hours
- Uses LightGBM models trained on historical data
- Provides training status feedback

#### `/plot <item_name>` 
Generate price history chart for a specific item
- Creates visual charts showing price trends over time
- Displays both buy and sell prices when available
- Shows average prices and data point statistics
- Returns chart as a PNG image attachment

#### `/retrain <item_name>` 
Admin-only command to manually retrain models
- Requires administrator permissions
- Retrains models for all supported horizons
- Useful when new data is available

## Setup

### Prerequisites
- Python 3.8+
- Discord bot token
- Hypixel API key (optional, for data collection)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DISCORD_BOT_TOKEN="your_discord_bot_token"
export HYPIXEL_API_KEY="your_hypixel_api_key"  # Optional

# Run the bot
python bot.py
```

### Configuration
Bot configuration is managed through `config/config.yaml`. Key settings:

```yaml
discord_bot:
  feature_build_interval_hours: 1
  admin_roles: ["Administrator", "Bot Admin"]
  max_analysis_items: 5

no_database_mode:
  enabled: true
  data_directory: "data"
```

## Architecture

The bot is built as a wrapper around the existing SkyBlock Economy Sim project:

- **Data Ingestion**: Uses existing `ingestion/` modules for API data collection
- **Feature Engineering**: Leverages `modeling/features/feature_pipeline.py`
- **ML Models**: Integrates with Phase 3 LightGBM/XGBoost forecasting
- **Analysis**: Wraps `phase3_cli.py` functionality in Discord commands

### Background Tasks
- `feature_builder`: Runs every hour to process raw data into ML features
- Data collection tasks run continuously (in production, these would be separate containers)

### File Structure
```
data/                 # Raw data storage (NDJSON files)
models/              # Trained ML models  
config/config.yaml   # Configuration
bot.py              # Main bot script
```

## Development

### Testing
```bash
# Test bot functionality without connecting to Discord
python /tmp/test_bot.py
```

### Adding New Commands
1. Create new command function with `@bot.tree.command()` decorator
2. Add appropriate error handling and logging
3. Use Discord embeds for rich responses
4. Leverage existing project modules where possible

## Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "bot.py"]
```

### Environment Variables
- `DISCORD_BOT_TOKEN`: Required - Your Discord bot token
- `HYPIXEL_API_KEY`: Optional - For live data collection

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Phase 3 Unavailable**: Check that ML libraries (lightgbm, xgboost, mesa) are installed
3. **No Data**: Bot works in no-database mode by default, check `data/` directory for files
4. **Permission Errors**: Ensure bot has proper Discord permissions in your server

### Logging
The bot uses Python's logging module. Check console output for detailed error messages.