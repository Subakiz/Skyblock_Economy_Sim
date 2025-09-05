#!/usr/bin/env python3
"""
Tests for the no-database mode functionality.
Tests NDJSON storage, file-based data access, and API endpoints.
"""

import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.ndjson_storage import NDJSONStorage
from modeling.profitability.file_data_access import (
    get_bazaar_price_from_files, 
    get_ah_price_stats_from_files,
    get_file_storage_if_enabled
)


def test_ndjson_storage():
    """Test basic NDJSON storage functionality."""
    print("\nTesting NDJSON storage...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = NDJSONStorage(data_directory=temp_dir, max_file_size_mb=1)
        
        # Test bazaar data
        bazaar_record = {
            "product_id": "ENCHANTED_LAPIS_LAZULI",
            "buy_price": 1000.0,
            "sell_price": 950.0,
            "buy_volume": 500,
            "sell_volume": 300,
            "ts": datetime.now(timezone.utc).isoformat()
        }
        
        storage.append_record("bazaar", bazaar_record)
        
        # Test auction data
        auction_record = {
            "uuid": "test-uuid-123",
            "item_id": "ENCHANTED_LAPIS_BLOCK",
            "sale_price": 150000,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        storage.append_record("auctions_ended", auction_record)
        
        # Test reading back
        bazaar_data = list(storage.read_records("bazaar", hours_back=1))
        auction_data = list(storage.read_records("auctions_ended", hours_back=1))
        
        if len(bazaar_data) != 1:
            print("✗ Failed to read bazaar record")
            return False
            
        if len(auction_data) != 1:
            print("✗ Failed to read auction record")
            return False
        
        # Test price retrieval
        latest_prices = storage.get_latest_bazaar_prices("ENCHANTED_LAPIS_LAZULI")
        if not latest_prices or latest_prices.get("buy_price") != 1000.0:
            print("✗ Failed to get latest bazaar prices")
            return False
        
        # Test auction stats
        stats = storage.get_auction_price_stats("ENCHANTED_LAPIS_BLOCK")
        if not stats or stats.get("sale_count") != 1:
            print("✗ Failed to get auction price stats")
            return False
        
        print("✓ NDJSON storage working correctly")
        return True


def test_file_config_detection():
    """Test configuration detection for file mode."""
    print("\nTesting file mode configuration...")
    
    # This should return None since no_database_mode.enabled is false in config
    storage = get_file_storage_if_enabled()
    if storage is not None:
        print("✗ File storage should be None when disabled in config")
        return False
    
    print("✓ File mode configuration detection working")
    return True


def test_file_based_api_mode():
    """Test that the API can detect file mode when enabled."""
    print("\nTesting API mode detection...")
    
    # Create a temporary config file with no-database mode enabled
    original_config_path = "config/config.yaml"
    backup_config_path = "config/config.yaml.backup"
    
    try:
        # Backup original config
        shutil.copy(original_config_path, backup_config_path)
        
        # Modify config to enable no-database mode
        with open(original_config_path, 'r') as f:
            config_content = f.read()
        
        # Replace the no_database_mode section
        modified_config = config_content.replace(
            "no_database_mode:\n  enabled: false",
            "no_database_mode:\n  enabled: true"
        )
        
        with open(original_config_path, 'w') as f:
            f.write(modified_config)
        
        # Test that the API would detect file mode
        storage = get_file_storage_if_enabled()
        if storage is None:
            print("✗ File storage should be detected when enabled in config")
            return False
        
        # Test that we can create the data directory
        data_dir = Path("data")
        if data_dir.exists():
            shutil.rmtree(data_dir)
        
        # Initialize storage should create directories
        test_storage = NDJSONStorage()
        
        if not (data_dir / "bazaar").exists():
            print("✗ Bazaar directory not created")
            return False
        
        if not (data_dir / "auctions").exists():
            print("✗ Auctions directory not created")
            return False
        
        print("✓ File mode API detection working")
        return True
        
    finally:
        # Restore original config
        if os.path.exists(backup_config_path):
            shutil.move(backup_config_path, original_config_path)
        
        # Clean up test data directory
        data_dir = Path("data")
        if data_dir.exists():
            shutil.rmtree(data_dir)


def test_integration():
    """Test integration between components."""
    print("\nTesting integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = NDJSONStorage(data_directory=temp_dir)
        
        # Add some test data
        bazaar_record = {
            "product_id": "ENCHANTED_LAPIS_LAZULI",
            "buy_price": 1200.0,
            "sell_price": 1100.0,
            "ts": datetime.now(timezone.utc).isoformat()
        }
        
        auction_records = [
            {
                "item_id": "ENCHANTED_LAPIS_BLOCK",
                "sale_price": 145000,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "item_id": "ENCHANTED_LAPIS_BLOCK", 
                "sale_price": 155000,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "item_id": "ENCHANTED_LAPIS_BLOCK",
                "sale_price": 150000,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        storage.append_record("bazaar", bazaar_record)
        for record in auction_records:
            storage.append_record("auctions_ended", record)
        
        # Test file-based data access functions
        buy_price = get_bazaar_price_from_files(storage, "ENCHANTED_LAPIS_LAZULI", "buy")
        if buy_price != 1200.0:
            print(f"✗ Expected buy price 1200.0, got {buy_price}")
            return False
        
        sell_price = get_bazaar_price_from_files(storage, "ENCHANTED_LAPIS_LAZULI", "sell")
        if sell_price != 1100.0:
            print(f"✗ Expected sell price 1100.0, got {sell_price}")
            return False
        
        ah_price, count = get_ah_price_stats_from_files(storage, "ENCHANTED_LAPIS_BLOCK", "1h", "median")
        if count != 3:
            print(f"✗ Expected 3 auctions, got {count}")
            return False
        
        if ah_price != 150000:  # median of [145000, 150000, 155000]
            print(f"✗ Expected median price 150000, got {ah_price}")
            return False
        
        print("✓ Integration tests passed")
        return True


def main():
    """Run all tests."""
    print("No-Database Mode Tests")
    print("=" * 50)
    
    tests = [
        test_ndjson_storage,
        test_file_config_detection,
        test_file_based_api_mode,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All no-database mode tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())