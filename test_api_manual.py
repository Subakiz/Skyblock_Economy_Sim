#!/usr/bin/env python3
"""
Manual test of the API in file-based mode.
Creates some test data and starts the API server briefly to test endpoints.
"""

import os
import sys
import json
import tempfile
import shutil
import time
import requests
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.ndjson_storage import NDJSONStorage


def create_test_data():
    """Create some test data in the data/ directory."""
    print("Creating test data...")
    
    # Remove existing data directory
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    
    storage = NDJSONStorage()
    
    # Add bazaar test data
    bazaar_records = [
        {
            "product_id": "ENCHANTED_LAPIS_LAZULI",
            "buy_price": 1000.0,
            "sell_price": 950.0,
            "buy_volume": 500,
            "sell_volume": 300,
            "ts": datetime.now(timezone.utc).isoformat()
        },
        {
            "product_id": "INK_SACK:4",  # Regular lapis
            "buy_price": 10.0,
            "sell_price": 8.0,
            "buy_volume": 10000,
            "sell_volume": 5000,
            "ts": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    # Add auction test data
    auction_records = [
        {
            "uuid": "test-uuid-1",
            "item_id": "ENCHANTED_LAPIS_BLOCK",
            "sale_price": 145000,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "uuid": "test-uuid-2", 
            "item_id": "ENCHANTED_LAPIS_BLOCK",
            "sale_price": 155000,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "uuid": "test-uuid-3",
            "item_id": "ENCHANTED_LAPIS_BLOCK",
            "sale_price": 150000,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "uuid": "test-uuid-4",
            "item_id": "ENCHANTED_LAPIS_BLOCK",
            "sale_price": 148000,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    for record in bazaar_records:
        storage.append_record("bazaar", record)
    
    for record in auction_records:
        storage.append_record("auctions_ended", record)
    
    print(f"Created test data in {data_dir}")
    return True


def enable_file_mode():
    """Temporarily enable file mode in config."""
    print("Enabling file mode in config...")
    
    config_path = "config/config.yaml"
    backup_path = "config/config.yaml.backup"
    
    # Backup original config
    shutil.copy(config_path, backup_path)
    
    # Modify config
    with open(config_path, 'r') as f:
        content = f.read()
    
    modified_content = content.replace(
        "no_database_mode:\n  enabled: false",
        "no_database_mode:\n  enabled: true"
    )
    
    with open(config_path, 'w') as f:
        f.write(modified_content)
    
    return backup_path


def restore_config(backup_path):
    """Restore original config."""
    print("Restoring original config...")
    if os.path.exists(backup_path):
        shutil.move(backup_path, "config/config.yaml")


def test_api_endpoints():
    """Test API endpoints with file-based data."""
    print("Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/healthz", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check: {data}")
            if data.get("mode") != "file-based":
                print("✗ Expected file-based mode")
                return False
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
        
        # Test AH prices endpoint  
        response = requests.get(f"{base_url}/prices/ah/ENCHANTED_LAPIS_BLOCK", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ AH prices: {data}")
            if data.get("sale_count") != 4:
                print(f"✗ Expected 4 sales, got {data.get('sale_count')}")
                return False
        else:
            print(f"✗ AH prices failed: {response.status_code}")
            return False
        
        # Test craft profitability (this should fail due to missing craft recipe)
        response = requests.get(f"{base_url}/profit/craft/ENCHANTED_LAPIS_BLOCK", timeout=5)
        if response.status_code == 400:  # Expected to fail - no craft recipe
            print("✓ Craft profitability correctly rejected invalid item")
        else:
            print(f"? Craft profitability: {response.status_code}")
        
        # Test forecast (should be disabled in file mode)
        response = requests.get(f"{base_url}/forecast/ENCHANTED_LAPIS_BLOCK", timeout=5)
        if response.status_code == 501:  # Not implemented in file mode
            print("✓ Forecast correctly disabled in file mode")
        else:
            print(f"? Forecast: {response.status_code}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {e}")
        return False


def main():
    """Run the manual test."""
    print("Manual API Test in File-Based Mode")
    print("=" * 50)
    
    backup_path = None
    server_process = None
    
    try:
        # Create test data
        if not create_test_data():
            return 1
        
        # Enable file mode
        backup_path = enable_file_mode()
        
        # Start API server in background
        print("Starting API server...")
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "services.api.app:app", 
            "--host", "127.0.0.1",
            "--port", "8000",
            "--log-level", "warning"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Test endpoints
        success = test_api_endpoints()
        
        if success:
            print("\n🎉 Manual API test passed!")
            return 0
        else:
            print("\n❌ Manual API test failed!")
            return 1
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        return 1
        
    finally:
        # Clean up
        if server_process:
            print("Stopping API server...")
            server_process.terminate()
            server_process.wait(timeout=10)
        
        if backup_path:
            restore_config(backup_path)
        
        # Clean up test data
        data_dir = Path("data")
        if data_dir.exists():
            shutil.rmtree(data_dir)
            print("Cleaned up test data")


if __name__ == "__main__":
    sys.exit(main())