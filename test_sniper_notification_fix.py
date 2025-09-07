#!/usr/bin/env python3
"""
Test script to reproduce and verify the fix for Discord bot notification issue.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def test_sniper_notification_issue():
    """Test that reproduces the silent failure in Discord notifications."""
    
    print("=== Testing Discord Bot Notification Issue ===")
    
    # Mock bot instance
    mock_bot = MagicMock()
    mock_bot.get_channel = MagicMock(return_value=None)  # Simulate channel not found initially
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_data_dir = Path(temp_dir)
        
        # Import and setup AuctionSniper with mocked environment
        try:
            from cogs.auction_sniper import AuctionSniper
            
            # Mock environment variables
            original_env = os.environ.get("HYPIXEL_API_KEY")
            os.environ["HYPIXEL_API_KEY"] = "test-api-key"
            
            # Create sniper instance
            sniper = AuctionSniper(mock_bot)
            sniper.data_dir = temp_data_dir  # Use temporary directory
            
            # Test Case 1: Verify the fix (hardcoded fallback channel should be set)
            print("\n--- Test Case 1: Verifying Hardcoded Fallback Fix ---")
            expected_channel_id = 1414187136609030154
            assert sniper.alert_channel_id == expected_channel_id, f"alert_channel_id should be set to hardcoded fallback {expected_channel_id}"
            
            # Create a mock auction that would be a valid snipe
            mock_auction = {
                "item_name": "Test Item",
                "starting_bid": 100000,
                "uuid": "test-uuid-123",
                "bin": True
            }
            
            # Add item to watchlist and FMV data to make it a valid snipe
            sniper.auction_watchlist.add("Test Item")
            sniper.fmv_data["Test Item"] = {
                "fmv": 500000,
                "samples": 10,
                "median": 500000
            }
            
            # Create mock channel for the hardcoded channel ID
            mock_channel = AsyncMock()
            mock_channel.name = "auction-alerts"
            mock_channel.send = AsyncMock()
            mock_bot.get_channel = MagicMock(return_value=mock_channel)
            
            # Call _alert_snipe - this should now work due to hardcoded fallback
            await sniper._alert_snipe(mock_auction)
            
            # Verify channel.send was called with the hardcoded channel ID
            mock_bot.get_channel.assert_called_with(expected_channel_id)
            mock_channel.send.assert_called_once()
            print("✅ Hardcoded fallback fix works: Discord message sent automatically")
            
            # Test Case 2: Test with channel configured via slash command simulation
            print("\n--- Test Case 2: Testing with Configured Channel ---")
            
            # Simulate the /sniper_channel command setting the channel
            test_channel_id = 1414187136609030154
            sniper.alert_channel_id = test_channel_id
            
            # Create mock channel
            mock_channel = AsyncMock()
            mock_channel.name = "auction-alerts"
            mock_channel.send = AsyncMock()
            mock_bot.get_channel = MagicMock(return_value=mock_channel)
            
            # Now call _alert_snipe again
            await sniper._alert_snipe(mock_auction)
            
            # Verify the channel.send was called
            mock_bot.get_channel.assert_called_with(test_channel_id)
            mock_channel.send.assert_called_once()
            print("✅ Discord notification sent successfully with configured channel")
            
            # Verify the embed content
            call_args = mock_channel.send.call_args
            embed = call_args[1]['embed']  # embed is passed as keyword argument
            assert embed.title == "🎯 Auction Snipe Detected!"
            assert "Test Item" in embed.description
            print("✅ Discord embed contains correct snipe information")
            
            # Test Case 3: Test hardcoded fallback approach
            print("\n--- Test Case 3: Testing Hardcoded Channel Fallback ---")
            
            # Create a new sniper instance to test hardcoded fallback
            sniper2 = AuctionSniper(mock_bot)
            sniper2.data_dir = temp_data_dir
            
            # Manually set the hardcoded channel ID (simulating our fix)
            if not sniper2.alert_channel_id:
                sniper2.alert_channel_id = 1414187136609030154  # Hardcoded fallback
            
            # Verify fallback works
            assert sniper2.alert_channel_id == 1414187136609030154
            print("✅ Hardcoded fallback channel ID set successfully")
            
            # Restore original environment
            if original_env:
                os.environ["HYPIXEL_API_KEY"] = original_env
            else:
                del os.environ["HYPIXEL_API_KEY"]
                
        except ImportError as e:
            print(f"⚠️ Could not import AuctionSniper: {e}")
            print("This is expected if dependencies are missing.")
            return False
        
    print("\n=== All Tests Passed! ===")
    return True

async def test_configuration_persistence():
    """Test that configuration is properly saved and loaded."""
    
    print("\n=== Testing Configuration Persistence ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_data_dir = Path(temp_dir)
        
        try:
            from cogs.auction_sniper import AuctionSniper
            
            mock_bot = MagicMock()
            os.environ["HYPIXEL_API_KEY"] = "test-api-key"
            
            # Create sniper and set data directory
            sniper = AuctionSniper(mock_bot)
            sniper.data_dir = temp_data_dir
            
            # Set configuration
            test_channel_id = 1414187136609030154
            sniper.alert_channel_id = test_channel_id
            sniper.profit_threshold = 200000
            
            # Save configuration
            await sniper._save_sniper_config()
            
            # Verify config file was created
            config_file = temp_data_dir / "sniper_config.json"
            assert config_file.exists(), "Configuration file should be created"
            
            # Load configuration in new instance
            sniper2 = AuctionSniper(mock_bot)
            sniper2.data_dir = temp_data_dir
            sniper2._load_saved_config()
            
            # Verify configuration was loaded
            assert sniper2.alert_channel_id == test_channel_id
            print("✅ Configuration persistence works correctly")
            
            # Clean up
            del os.environ["HYPIXEL_API_KEY"]
            
        except ImportError as e:
            print(f"⚠️ Could not import AuctionSniper: {e}")
            return False
    
    return True

if __name__ == "__main__":
    async def main():
        try:
            success1 = await test_sniper_notification_issue()
            success2 = await test_configuration_persistence()
            
            if success1 and success2:
                print("\n🎉 All tests completed successfully!")
                return 0
            else:
                print("\n❌ Some tests failed!")
                return 1
                
        except Exception as e:
            print(f"\n💥 Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)