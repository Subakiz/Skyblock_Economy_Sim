#!/usr/bin/env python3
"""
Test script to verify the two-process architecture implementation.
Verifies that the mandated requirements have been met.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def test_data_ingestion_entry_point():
    """Test that data_ingestion/run_ingestion.py exists and is executable."""
    script_path = Path("data_ingestion/run_ingestion.py")
    
    if not script_path.exists():
        return False, "data_ingestion/run_ingestion.py does not exist"
    
    if not os.access(script_path, os.X_OK):
        return False, "data_ingestion/run_ingestion.py is not executable"
    
    # Test that it imports correctly
    try:
        sys.path.insert(0, '.')
        from data_ingestion.run_ingestion import main
        return True, "Entry point exists and imports successfully"
    except Exception as e:
        return False, f"Entry point import failed: {e}"

def test_master_startup_script():
    """Test that start_bot.sh implements the two-process architecture."""
    script_path = Path("start_bot.sh")
    
    if not script_path.exists():
        return False, "start_bot.sh does not exist"
    
    if not os.access(script_path, os.X_OK):
        return False, "start_bot.sh is not executable"
    
    # Read the script content to verify it implements the required functionality
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for required components
    required_components = [
        "nohup python3 data_ingestion/run_ingestion.py",  # Background ingestion service
        "logs/ingestion.log",  # Ingestion logging
        "cleanup()",  # Cleanup function
        "trap cleanup EXIT INT TERM",  # Signal handling
        "kill \"$INGESTION_PID\"",  # Process killing
        "exec python3 bot.py",  # Foreground bot
    ]
    
    missing = []
    for component in required_components:
        if component not in content:
            missing.append(component)
    
    if missing:
        return False, f"Missing required components: {missing}"
    
    return True, "Master startup script implements two-process architecture"

def test_bot_data_collection_removed():
    """Test that data collection tasks have been removed from bot.py."""
    try:
        sys.path.insert(0, '.')
        import bot
        
        # Check that data collection imports are commented out
        with open("bot.py", 'r') as f:
            content = f.read()
        
        if "from ingestion.auction_collector import run as auction_collector_run" in content and \
           "# from ingestion.auction_collector import run as auction_collector_run" not in content:
            return False, "Data collection imports not properly commented out"
        
        if "await self.start_data_collectors()" in content and \
           "# await self.start_data_collectors()" not in content:
            return False, "start_data_collectors call not properly commented out"
        
        return True, "Data collection tasks properly removed from Discord bot"
        
    except Exception as e:
        return False, f"Failed to verify bot changes: {e}"

def test_auction_sniper_parquet_reading():
    """Test that auction sniper cog reads from Parquet data."""
    try:
        sys.path.insert(0, '.')
        from cogs.auction_sniper import AuctionSniper
        
        # Check that the update_market_intelligence method exists
        sniper_class = AuctionSniper
        if not hasattr(sniper_class, 'update_market_intelligence'):
            return False, "update_market_intelligence method not found"
        
        # Check that PARQUET_DATA_PATH is defined
        if not hasattr(sniper_class, 'PARQUET_DATA_PATH'):
            return False, "PARQUET_DATA_PATH not defined"
        
        # Verify it points to the correct path
        if str(sniper_class.PARQUET_DATA_PATH) != "data/auction_history":
            return False, f"PARQUET_DATA_PATH incorrect: {sniper_class.PARQUET_DATA_PATH}"
        
        return True, "Auction sniper properly configured to read Parquet data"
        
    except Exception as e:
        return False, f"Failed to verify auction sniper: {e}"

def test_hunter_task_preserved():
    """Test that the hunter task (high-frequency scanning) is preserved."""
    try:
        sys.path.insert(0, '.')
        
        # Check the source code directly
        with open("cogs/auction_sniper.py", 'r') as f:
            source_code = f.read()
        
        # Verify Hunter Task is documented
        if "Hunter Task: High-frequency scanning" not in source_code:
            return False, "Hunter Task documentation not found"
        
        # Verify it scans page 0 only
        if '"page": 0' not in source_code:
            return False, "Hunter task does not scan page 0 as required"
        
        # Verify the method exists and is a task loop
        if "async def high_frequency_snipe_scan" not in source_code:
            return False, "high_frequency_snipe_scan method not found"
        
        if "@tasks.loop(seconds=2)" not in source_code:
            return False, "Hunter task not configured as 2-second loop"
        
        return True, "Hunter task (high-frequency scanning) properly preserved"
        
    except Exception as e:
        return False, f"Failed to verify hunter task: {e}"

def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Two-Process Architecture Verification Tests")
    print("=" * 70)
    
    tests = [
        ("Data Ingestion Entry Point", test_data_ingestion_entry_point),
        ("Master Startup Script", test_master_startup_script),
        ("Bot Data Collection Removed", test_bot_data_collection_removed),
        ("Auction Sniper Parquet Reading", test_auction_sniper_parquet_reading),
        ("Hunter Task Preserved", test_hunter_task_preserved),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success, message = test_func()
            if success:
                print(f"  ✅ {message}")
                passed += 1
            else:
                print(f"  ❌ {message}")
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Two-process architecture successfully implemented.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())