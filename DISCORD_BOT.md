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

#### `/compare <item_a> <item_b>` ⭐ NEW
Cross-item arbitrage analysis comparing two items
- Price predictions for both items
- Relative value assessment and investment recommendations
- Crafting arbitrage opportunities detection
- Price correlation analysis with statistical measures
- Example: `/compare HYPERION NECRON_CHESTPLATE`

#### `/event_impact <event_name>` ⭐ NEW
Analyze the impact of historical SkyBlock events on market prices
- Event details (type, dates, description)
- Market effects on different item categories
- List of affected items
- Investment recommendations based on event impact
- Autocomplete support for event names
- Example: `/event_impact MAYOR_DERPY`

#### `/market_pulse` ⭐ NEW
Holistic market overview and sentiment analysis
- Overall market sentiment (Bullish/Bearish/Neutral)
- Sector performance across 6 key market baskets:
  - Dungeons Index (high-tier dungeon items)
  - Farming Index (farming tools and crops)
  - Mining Index (mining equipment and ores)
  - Skill Items Index (experience and training items)
  - Luxury Items Index (rare pets and collectibles)
  - Commodities Index (basic materials and enchanted blocks)
- Market insights and trend analysis
- Summary statistics and positive sector count

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

## Enhanced Features (Part Two Expansion)

### Event-Aware Analysis
The bot now includes historical SkyBlock event data to provide context-aware market analysis:

- **Events Database**: `data/events.json` contains 8+ major historical events including mayors, festivals, updates, and market incidents
- **Event Features**: Auction data is enhanced with active event information (mayor, festival status, update periods)
- **Event Impact Analysis**: Quantified effects on different item categories and market sectors

### Market Baskets System
Six key economic baskets track different market sectors:

- **Dungeons Index**: HYPERION, NECRON items, Shadow Fury, etc.
- **Farming Index**: WHEAT, farming tools, ELEPHANT_PET, etc. 
- **Mining Index**: Gemstones, DIVAN_DRILL, ores, etc.
- **Skill Items Index**: Experience bottles, cookies, boosters, etc.
- **Luxury Items Index**: Dragon pets, rare items, collectibles, etc.
- **Commodities Index**: Enchanted blocks, basic materials, etc.

### Cross-Item Analysis
Advanced comparison capabilities:
- Price correlation analysis between items
- Crafting arbitrage detection using item ontology
- Relative value assessments for investment decisions

### Market Intelligence
The `/market_pulse` command provides a "SkyBlock S&P 500" style overview:
- Weighted basket indices with 24-hour performance
- Market sentiment classification (Bullish/Bearish/Neutral)
- Sector rotation insights and market trend analysis

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