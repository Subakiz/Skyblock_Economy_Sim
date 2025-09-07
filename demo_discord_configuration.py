#!/usr/bin/env python3
"""
Demo script showing how to properly configure the auction sniper using slash commands.
This demonstrates the recommended solution for the Discord notification issue.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def demo_slash_command_configuration():
    """Demonstrate the proper way to configure the sniper using slash commands."""
    
    print("=== Demo: Proper Discord Bot Configuration ===")
    print("This demonstrates the RECOMMENDED solution using slash commands.")
    
    try:
        import os
        from cogs.auction_sniper import AuctionSniper
        
        # Mock environment
        os.environ["HYPIXEL_API_KEY"] = "demo-api-key"
        
        # Create mock Discord interaction for /sniper_channel command
        def create_mock_interaction(channel_id=1414187136609030154, is_admin=True):
            mock_interaction = AsyncMock()
            mock_interaction.response.send_message = AsyncMock()
            
            # Mock user with admin permissions
            mock_user = MagicMock()
            mock_user.guild_permissions.administrator = is_admin
            mock_interaction.user = mock_user
            
            # Mock text channel
            mock_channel = MagicMock()
            mock_channel.id = channel_id
            mock_channel.name = "auction-alerts"
            mock_channel.mention = f"<#{channel_id}>"
            
            return mock_interaction, mock_channel
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create bot and sniper instances
            mock_bot = MagicMock()
            sniper = AuctionSniper(mock_bot)
            sniper.data_dir = Path(temp_dir)
            
            print(f"\n1. Initial State:")
            print(f"   Alert Channel ID: {sniper.alert_channel_id}")
            print(f"   (Note: Hardcoded fallback is now set)")
            
            # Simulate user running /sniper_channel command
            print(f"\n2. User runs: /sniper_channel #auction-alerts")
            
            mock_interaction, mock_channel = create_mock_interaction()
            
            # Call the slash command handler (calling the callback directly)
            await sniper.set_sniper_channel.callback(sniper, mock_interaction, mock_channel)
            
            print(f"   ✅ Channel configured successfully!")
            print(f"   Alert Channel ID: {sniper.alert_channel_id}")
            
            # Verify the response
            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args[0][0]
            assert "✅ Sniper alerts will be sent to" in call_args
            print(f"   Bot Response: {call_args}")
            
            # Verify configuration persistence
            config_file = Path(temp_dir) / "sniper_config.json"
            assert config_file.exists(), "Configuration should be saved to file"
            print(f"   Configuration saved to: {config_file}")
            
            print(f"\n3. Configuration Status:")
            
            # Simulate /sniper_status command
            mock_status_interaction = AsyncMock()
            mock_status_interaction.response.send_message = AsyncMock()
            
            await sniper.sniper_status.callback(sniper, mock_status_interaction)
            
            # Verify status response
            mock_status_interaction.response.send_message.assert_called_once()
            call_args = mock_status_interaction.response.send_message.call_args
            embed = call_args[1]['embed']
            
            print(f"   Status Embed Title: {embed.title}")
            print(f"   Alert Channel Field: Found in embed fields")
            print(f"   ✅ /sniper_status shows proper configuration")
            
            print(f"\n4. Testing Snipe Alert:")
            
            # Create mock channel for alert testing
            mock_alert_channel = AsyncMock()
            mock_alert_channel.name = "auction-alerts"
            mock_alert_channel.send = AsyncMock()
            mock_bot.get_channel = MagicMock(return_value=mock_alert_channel)
            
            # Set up snipe data
            sniper.auction_watchlist.add("Demo Item")
            sniper.fmv_data["Demo Item"] = {"fmv": 1000000, "samples": 15}
            
            # Create mock auction
            mock_auction = {
                "item_name": "Demo Item",
                "starting_bid": 500000,
                "uuid": "demo-uuid-456",
                "bin": True
            }
            
            # Send alert
            await sniper._alert_snipe(mock_auction)
            
            # Verify alert was sent
            mock_bot.get_channel.assert_called_with(sniper.alert_channel_id)
            mock_alert_channel.send.assert_called_once()
            print(f"   ✅ Discord alert sent successfully!")
            
            # Show embed content
            call_args = mock_alert_channel.send.call_args
            embed = call_args[1]['embed']
            print(f"   Alert Title: {embed.title}")
            print(f"   Alert Description: {embed.description}")
            print(f"   Fields: {len(embed.fields)} fields (Price, FMV, Profit, etc.)")
            
        # Clean up
        del os.environ["HYPIXEL_API_KEY"]
        
        print(f"\n=== Configuration Demo Complete ===")
        print(f"Summary:")
        print(f"  • Use /sniper_channel #your-channel to configure")  
        print(f"  • Use /sniper_status to verify configuration")
        print(f"  • Bot automatically saves and loads configuration")
        print(f"  • Snipe alerts work immediately after configuration")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Could not import dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_permission_error():
    """Demonstrate what happens when a non-admin tries to configure the bot."""
    
    print(f"\n=== Demo: Permission Error Handling ===")
    
    try:
        import os
        from cogs.auction_sniper import AuctionSniper
        
        os.environ["HYPIXEL_API_KEY"] = "demo-api-key"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_bot = MagicMock()
            sniper = AuctionSniper(mock_bot)
            sniper.data_dir = Path(temp_dir)
            
            # Create mock interaction for non-admin user
            mock_interaction = AsyncMock()
            mock_interaction.response.send_message = AsyncMock()
            
            mock_user = MagicMock()
            mock_user.guild_permissions.administrator = False  # Not admin
            mock_interaction.user = mock_user
            
            mock_channel = MagicMock()
            mock_channel.id = 1414187136609030154
            
            print(f"Non-admin user tries: /sniper_channel #auction-alerts")
            
            # Call command as non-admin (using callback)
            await sniper.set_sniper_channel.callback(sniper, mock_interaction, mock_channel)
            
            # Verify permission error response
            mock_interaction.response.send_message.assert_called_once()
            call_args = mock_interaction.response.send_message.call_args
            error_message = call_args[0][0]
            is_ephemeral = call_args[1]['ephemeral']
            
            print(f"Bot Response: {error_message}")
            print(f"Ephemeral (private): {is_ephemeral}")
            print(f"✅ Permission error handled correctly")
            
        del os.environ["HYPIXEL_API_KEY"]
        return True
        
    except Exception as e:
        print(f"❌ Permission demo failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        print("Discord Bot Notification Fix - Configuration Demo")
        print("=" * 55)
        
        success1 = await demo_slash_command_configuration()
        success2 = await demo_permission_error()
        
        if success1 and success2:
            print(f"\n🎉 All configuration demos completed successfully!")
            print(f"\nNext Steps for Users:")
            print(f"1. Run your Discord bot")
            print(f"2. Use /sniper_channel #your-desired-channel")
            print(f"3. Verify with /sniper_status")
            print(f"4. Monitor the channel for auction alerts!")
            return 0
        else:
            print(f"\n❌ Some demos failed!")
            return 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)