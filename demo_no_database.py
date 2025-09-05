#!/usr/bin/env python3
"""
Demo script for no-database mode.
Shows how to enable file-based storage and use the API.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.ndjson_storage import NDJSONStorage


def demo_file_storage():
    """Demonstrate NDJSON file storage."""
    print("\n=== NDJSON File Storage Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating storage in: {temp_dir}")
        storage = NDJSONStorage(data_directory=temp_dir)
        
        # Create some sample data
        print("\n1. Adding sample bazaar data...")
        bazaar_data = [
            {
                "product_id": "ENCHANTED_LAPIS_LAZULI",
                "buy_price": 1000.0,
                "sell_price": 950.0,
                "buy_volume": 500,
                "sell_volume": 300
            },
            {
                "product_id": "INK_SACK:4",
                "buy_price": 10.0,
                "sell_price": 8.5,
                "buy_volume": 50000,
                "sell_volume": 30000
            }
        ]
        
        for item in bazaar_data:
            storage.append_record("bazaar", item)
            print(f"   Added: {item['product_id']} (buy: {item['buy_price']}, sell: {item['sell_price']})")
        
        print("\n2. Adding sample auction data...")
        auction_data = [
            {"item_id": "ENCHANTED_LAPIS_BLOCK", "sale_price": 145000},
            {"item_id": "ENCHANTED_LAPIS_BLOCK", "sale_price": 155000},  
            {"item_id": "ENCHANTED_LAPIS_BLOCK", "sale_price": 150000},
            {"item_id": "ENCHANTED_LAPIS_BLOCK", "sale_price": 148000}
        ]
        
        for item in auction_data:
            storage.append_record("auctions_ended", item)
            print(f"   Added: {item['item_id']} sold for {item['sale_price']}")
        
        print("\n3. Reading data back...")
        
        # Test bazaar prices
        lapis_prices = storage.get_latest_bazaar_prices("ENCHANTED_LAPIS_LAZULI")
        print(f"   Latest ENCHANTED_LAPIS_LAZULI: buy={lapis_prices['buy_price']}, sell={lapis_prices['sell_price']}")
        
        # Test auction stats
        block_stats = storage.get_auction_price_stats("ENCHANTED_LAPIS_BLOCK")
        print(f"   ENCHANTED_LAPIS_BLOCK stats: median={block_stats['median_price']}, count={block_stats['sale_count']}")
        
        print(f"\n4. File structure:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")


def demo_configuration():
    """Demonstrate configuration options."""
    print("\n=== Configuration Demo ===")
    
    print("\n1. Current configuration (no-database mode disabled):")
    with open("config/config.yaml", "r") as f:
        lines = f.readlines()
    
    # Find and print no-database mode section
    in_section = False
    for line in lines:
        if "no_database_mode:" in line:
            in_section = True
        if in_section:
            print(f"   {line.rstrip()}")
            if line.strip() and not line.startswith(' ') and "no_database_mode:" not in line:
                break
    
    print("\n2. To enable no-database mode, change config to:")
    print("   no_database_mode:")
    print("     enabled: true              # Enable file-based storage")
    print("     data_directory: \"data\"     # Directory to store NDJSON files") 
    print("     max_file_size_mb: 100      # Max size before rotation")
    print("     retention_hours: 168       # Keep data for 7 days")
    
    print("\n3. Or use environment variable:")
    print("   export NO_DATABASE_MODE_ENABLED=true")


def demo_api_differences():
    """Show API behavior differences between modes.""" 
    print("\n=== API Behavior Demo ===")
    
    print("\n1. In database mode:")
    print("   GET /healthz → {\"status\": \"ok\", \"mode\": \"database\"}")
    print("   GET /forecast/{product_id} → Full forecasting available")
    print("   GET /prices/ah/{product_id} → Database queries")
    print("   GET /profit/craft/{product_id} → Database-backed calculations")
    
    print("\n2. In file-based mode:")
    print("   GET /healthz → {\"status\": \"ok\", \"mode\": \"file-based\"}")
    print("   GET /forecast/{product_id} → 501 Not Implemented")
    print("   GET /prices/ah/{product_id} → File-based aggregation") 
    print("   GET /profit/craft/{product_id} → File-backed calculations")
    
    print("\n3. Data collection in both modes:")
    print("   python -m ingestion.bazaar_collector")
    print("   python -m ingestion.auction_collector")
    print("   → Automatically detects mode and writes accordingly")


def demo_use_cases():
    """Show typical use cases for no-database mode."""
    print("\n=== Use Cases Demo ===")
    
    print("\n1. 🔬 Development & Testing:")
    print("   • No PostgreSQL setup required")
    print("   • Fast iteration and debugging")
    print("   • Portable test data")
    
    print("\n2. 📊 Lightweight Monitoring:")
    print("   • Basic price tracking")
    print("   • Simple profitability analysis") 
    print("   • Resource-constrained environments")
    
    print("\n3. 🚀 Quick Deployments:")
    print("   • Docker containers without DB")
    print("   • Edge deployments")
    print("   • Prototype demonstrations")
    
    print("\n4. 💾 Data Archival:")
    print("   • Human-readable backup format")
    print("   • Easy data transfer/sync")
    print("   • Historical analysis")


def main():
    """Run the demo."""
    print("SkyBlock Economy Sim - No-Database Mode Demo")
    print("=" * 50)
    
    demo_file_storage()
    demo_configuration() 
    demo_api_differences()
    demo_use_cases()
    
    print("\n" + "=" * 50)
    print("Demo complete! 🎉")
    print("\nNext steps:")
    print("1. Enable no-database mode in config/config.yaml")
    print("2. Run data collectors: python -m ingestion.bazaar_collector")
    print("3. Start API server: uvicorn services.api.app:app --reload")
    print("4. Test endpoints: curl http://localhost:8000/healthz")
    print("\nSee docs/NO_DATABASE_MODE.md for detailed documentation.")


if __name__ == "__main__":
    main()