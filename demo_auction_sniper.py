#!/usr/bin/env python3
"""
Demo script showing the auction sniper in action.
Run this with DISCORD_BOT_TOKEN and HYPIXEL_API_KEY environment variables set.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main demo function."""
    
    # Check environment variables
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    hypixel_key = os.getenv("HYPIXEL_API_KEY")
    
    if not discord_token:
        print("❌ DISCORD_BOT_TOKEN environment variable not set")
        print("   Set it with: export DISCORD_BOT_TOKEN=your_bot_token")
        return
        
    if not hypixel_key:
        print("⚠️  HYPIXEL_API_KEY environment variable not set")
        print("   The sniper will work but won't be able to fetch auction data")
        print("   Set it with: export HYPIXEL_API_KEY=your_api_key")
        print("   Continuing without API key...\n")
    
    print("🎯 Starting SkyBlock Economy Bot with Auction Sniper")
    print("   Discord commands available:")
    print("   • /sniper_status - Check sniper status")
    print("   • /sniper_channel #channel - Set alert channel")
    print("   • /sniper_config [profit] [min_count] - Configure settings")
    print("   • All existing bot commands also available")
    print()
    
    # Import and run the bot
    try:
        from bot import main as bot_main
        bot_main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot failed: {e}")

if __name__ == "__main__":
    main()