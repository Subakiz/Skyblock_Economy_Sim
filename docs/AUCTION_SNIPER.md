# Hypixel Auction Sniper Documentation

## Overview

The Hypixel Auction Sniper is a production-level Discord bot cog that automatically identifies undervalued items on the Hypixel Skyblock Auction House and alerts users of profitable opportunities.

## Architecture

### Two-Speed Design

The sniper uses a sophisticated two-speed architecture:

1. **Hunter Task** (2-second interval)
   - High-frequency scanning of auction house page 0 only
   - Optimized for speed and real-time snipe detection
   - Only processes items on the watchlist that are "Buy It Now"
   - Performs intensive verification on potential snipes

2. **Analyst Task** (90-second interval)  
   - Full market analysis with complete pagination
   - Updates the dynamic watchlist based on item volume
   - Calculates Fair Market Value (FMV) for all watchlist items
   - Maintains market intelligence for the Hunter task

### Data Source Integrity

**Critical Design Principle**: The watchlist is derived exclusively from Auction House data. The sniper does NOT use the Bazaar API to determine item demand, ensuring accurate market analysis.

## Configuration

### Config File (config.yaml)

```yaml
auction_sniper:
  profit_threshold: 100000        # Minimum profit in coins
  min_auction_count: 50          # Min auctions for watchlist inclusion  
  hunter_interval_seconds: 2     # Hunter task frequency
  analyst_interval_seconds: 90   # Analyst task frequency
  max_fmv_multiplier: 1.1       # Max price vs FMV ratio
```

### Environment Variables

- `HYPIXEL_API_KEY` - Required for auction data access
- `DISCORD_BOT_TOKEN` - Required for Discord bot functionality

## Discord Commands

### Admin Commands (Requires Administrator permission)

#### `/sniper_channel <channel>`
Sets the Discord channel for snipe alerts.
```
/sniper_channel #auction-alerts
```

#### `/sniper_config [profit_threshold] [min_auction_count]`
Configure sniper parameters.
```
/sniper_config profit_threshold:500000 min_auction_count:30
```

### Public Commands

#### `/sniper_status`
Shows current sniper status, configuration, and statistics.

## Sniper Logic

### Watchlist Generation

Items are automatically added to the watchlist if they appear in more than `min_auction_count` auctions (default: 50). This ensures the sniper only monitors liquid items with sufficient trading volume.

### Snipe Verification (Three-Tier System)

1. **Initial Filters** (Fast)
   - Must be "Buy It Now" auction
   - Item name must be on watchlist

2. **Manipulation Check**
   - Price must not exceed FMV × `max_fmv_multiplier`
   - Protects against overpriced items

3. **Attribute Check** 
   - Parses item lore for critical attributes
   - Weapons: Checks for ultimate enchantments, good reforges
   - Armor: Checks for stars, beneficial reforges
   - Prevents sniping items with missing key attributes

4. **Profitability Check**
   - Estimated profit = FMV - Price - Auction House Fee (1%)
   - Must exceed configured `profit_threshold`

### Fair Market Value (FMV) Calculation

- Collects all BIN auction prices for each item
- Removes outliers (top/bottom 10%)
- Calculates trimmed mean as FMV
- Updates every analyst cycle (90 seconds)
- Minimum 3 data points required

## Alert System

When a valid snipe is detected, the bot sends a rich Discord embed containing:

- Item name and estimated profit
- Current price vs Fair Market Value  
- Copy-paste ready `/viewauction [uuid]` command
- Timestamp and sniper branding

## File Structure

```
data/sniper/
├── auction_watchlist.json    # Persistent watchlist storage
├── fmv_cache.json           # Fair Market Value cache  
└── sniper_config.json       # User configuration
```

## Performance

- **Hunter Task**: Processes ~64 auctions per scan (page 0 only)
- **Analyst Task**: Processes 100+ pages with full pagination
- **Memory Efficient**: Uses sets for O(1) watchlist lookups
- **Rate Limited**: Respects Hypixel's 120 requests/minute limit
- **Error Resilient**: Graceful handling of API failures

## Getting Started

1. Set environment variables:
   ```bash
   export DISCORD_BOT_TOKEN="your_discord_bot_token"
   export HYPIXEL_API_KEY="your_hypixel_api_key"  
   ```

2. Run the demo:
   ```bash
   python demo_auction_sniper.py
   ```

3. Configure in Discord:
   ```
   /sniper_channel #alerts
   /sniper_config profit_threshold:200000
   ```

4. Monitor the `#alerts` channel for snipe notifications!

## Monitoring and Logs

The sniper provides comprehensive logging:

- **INFO**: Task completions, snipe discoveries, configuration changes
- **DEBUG**: Detailed operation logs, data persistence
- **ERROR**: API failures, processing errors with graceful recovery

Use `/sniper_status` to monitor:
- Task health (Hunter/Analyst running status)
- Watchlist size and FMV cache status  
- API connectivity and configuration

## Troubleshooting

### Common Issues

1. **"No API Key" Error**
   - Set `HYPIXEL_API_KEY` environment variable
   - Verify API key is valid on Hypixel Developer Portal

2. **No Snipes Found**  
   - Check `/sniper_status` for watchlist size
   - Ensure `min_auction_count` isn't too high
   - Lower `profit_threshold` if market is competitive

3. **Tasks Not Running**
   - Check bot logs for initialization errors
   - Verify Discord bot permissions
   - Ensure sufficient API rate limit headroom

### Performance Tuning

- **Increase Speed**: Lower `hunter_interval_seconds` (minimum 1-2 seconds)
- **Reduce Load**: Increase `min_auction_count` to shrink watchlist
- **More Opportunities**: Lower `profit_threshold` for smaller flips
- **Quality Focus**: Increase `profit_threshold` for only high-value snipes

## Security and Best Practices

- Never commit API keys to version control
- Use environment variables for all sensitive data
- Monitor API usage to stay within rate limits
- Set reasonable profit thresholds to avoid spam
- Use dedicated Discord channels for alerts
- Regularly review sniper logs for anomalies

---

*The Hypixel Auction Sniper is designed for educational and legitimate trading purposes. Always follow Hypixel's Terms of Service and maintain fair trading practices.*