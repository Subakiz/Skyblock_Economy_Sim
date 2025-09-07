# Discord Bot Notification Troubleshooting Guide

## Problem Diagnosis: Silent Alert Failure

### Issue Description
The Hypixel Auction Sniper successfully detects profitable auction opportunities and logs them:
```
INFO:cogs.auction_sniper.AuctionSniper:Valid snipe found: [Lvl 1] Dolphina at 150,000 coins (FMV: 2,500,000, Profit: 2,348,500, Margin: 1565.7%)
```

However, no Discord messages appear in the alert channel.

### Root Cause
The issue occurs in the `_alert_snipe` method at this critical line:
```python
if not self.alert_channel_id:
    return  # <--- Silent failure here
```

When `self.alert_channel_id` is `None` (unconfigured), the function exits silently without sending any Discord message or error.

## Solution 1: Proper Configuration (Recommended)

### Step-by-Step Instructions

1. **Ensure Bot Permissions**: Make sure you have Administrator permissions in your Discord server.

2. **Use the Slash Command**: In your Discord server, run:
   ```
   /sniper_channel #your-alert-channel
   ```
   Replace `#your-alert-channel` with your desired channel (e.g., `#auction-alerts`).

3. **Verify Configuration**: Check that the bot responds with:
   ```
   ✅ Sniper alerts will be sent to #your-alert-channel
   ```

4. **Test the Setup**: Use `/sniper_status` to verify the configuration:
   - Alert Channel should show your channel
   - Tasks should be running (🟢 Running)

### How It Works
- The `/sniper_channel` command sets `self.alert_channel_id = channel.id`
- Configuration is saved to `data/sniper/sniper_config.json`
- On bot restart, `_load_saved_config()` restores the setting

### Troubleshooting
- **Missing `/sniper_channel` command**: Bot may not be loaded properly
- **Permission errors**: You need Administrator permissions
- **Channel not found**: Ensure the channel exists and bot can see it

## Solution 2: Code Modification (Quick Fix)

If you cannot use slash commands or need immediate functionality, you can modify the code:

### Code Change
In `cogs/auction_sniper.py`, locate the `__init__` method and add this after `self._load_saved_config()`:

```python
# Hardcoded fallback for alert channel if not configured
# This ensures notifications work immediately without requiring manual configuration
if not self.alert_channel_id:
    self.alert_channel_id = 1414187136609030154  # Your channel ID
    self.logger.info(f"Using hardcoded fallback alert channel: {self.alert_channel_id}")
```

### Finding Your Channel ID
1. Enable Developer Mode in Discord (User Settings → Advanced → Developer Mode)
2. Right-click your desired channel
3. Select "Copy ID"
4. Replace `1414187136609030154` with your channel ID

### Implementation
The modification has been applied to ensure the bot works immediately with the provided channel ID (1414187136609030154).

## Why Solution 1 is Better

### Long-term Maintainability
- **Flexibility**: Easy to change channels without code modification
- **Multi-server**: Different servers can use different channels
- **User-friendly**: Non-developers can configure the bot
- **Persistence**: Configuration survives code updates

### Professional Best Practices
- **Separation of Concerns**: Configuration separate from code
- **No Hardcoded Values**: Avoids maintenance issues
- **Audit Trail**: Configuration changes are logged
- **Scalability**: Supports multiple configuration options

## Verification

### Expected Behavior After Fix
1. **Immediate**: Bot logs show hardcoded channel being used
2. **Snipe Detection**: Existing snipe detection continues working
3. **Discord Messages**: Rich embeds appear in the configured channel
4. **User Override**: `/sniper_channel` command still works to change the channel

### Log Messages to Look For
```
INFO:cogs.auction_sniper.AuctionSniper:Using hardcoded fallback alert channel: 1414187136609030154
INFO:cogs.auction_sniper.AuctionSniper:Sent snipe alert for [Item Name] to channel auction-alerts
```

### Discord Embed Format
The bot sends rich embeds containing:
- 🎯 Auction Snipe Detected!
- Item name and details
- 💰 Current price
- 📊 Fair Market Value (FMV)
- 💸 Estimated profit
- 🔗 Copy-paste `/viewauction [uuid]` command

## Additional Commands

### Configuration Commands
- `/sniper_channel #channel` - Set alert channel
- `/sniper_config profit_threshold:500000` - Set minimum profit
- `/sniper_status` - View current configuration and status

### Monitoring
- Check bot logs for snipe detection messages
- Verify tasks are running with `/sniper_status`
- Monitor the configured Discord channel for alerts

## Summary

The silent notification failure was caused by an unconfigured alert channel (`self.alert_channel_id = None`). The hardcoded fallback solution provides immediate functionality, while the slash command configuration offers the proper long-term solution. Both approaches are now available to ensure reliable auction snipe notifications.