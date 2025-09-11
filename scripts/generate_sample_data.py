#!/usr/bin/env python3
"""
Generate Sample Data for Testing

Creates sample feature summaries and bazaar data to test the market pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

def create_sample_feature_summaries():
    """Create sample feature summaries for recent hours."""
    print("📊 Creating sample feature summaries...")
    
    base_path = Path("data/feature_summaries")
    current_time = datetime.now(timezone.utc)
    
    # Common items with realistic price ranges
    items_data = {
        "WHEAT": {"base_price": 3, "volatility": 0.1},
        "ENCHANTED_FLINT": {"base_price": 15000, "volatility": 0.05},
        "COBBLESTONE": {"base_price": 2, "volatility": 0.15},
        "EMERALD": {"base_price": 8000, "volatility": 0.08},
        "DIAMOND": {"base_price": 9000, "volatility": 0.06},
        "ENCHANTED_BREAD": {"base_price": 50000, "volatility": 0.12},
        "CARROT_ITEM": {"base_price": 4, "volatility": 0.2}
    }
    
    # Generate summaries for last 6 hours
    for hour_offset in range(6):
        hour_time = current_time - timedelta(hours=hour_offset)
        
        # Create directory structure
        partition_path = base_path / (
            f"year={hour_time.year}/"
            f"month={hour_time.month:02d}/"
            f"day={hour_time.day:02d}/"
            f"hour={hour_time.hour:02d}"
        )
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Generate summary records for this hour
        records = []
        for item_name, item_config in items_data.items():
            base_price = item_config["base_price"]
            volatility = item_config["volatility"]
            
            # Generate price ladder with some noise
            num_levels = np.random.randint(3, 8)
            floor_price = base_price * (1 + np.random.normal(0, volatility))
            
            prices = []
            counts = []
            total_count = 0
            
            for i in range(num_levels):
                price = floor_price * (1 + i * 0.02 + np.random.normal(0, 0.01))
                count = max(1, np.random.poisson(10) + (5 if i == 0 else 0))  # More items at floor
                
                prices.append(int(price))
                counts.append(count)
                total_count += count
            
            record = {
                "hour_start": hour_time,
                "item_name": item_name,
                "prices": prices,
                "counts": counts,
                "total_count": total_count,
                "floor_price": prices[0],
                "second_lowest_price": prices[1] if len(prices) > 1 else prices[0],
                "auction_count": total_count
            }
            records.append(record)
        
        # Write to parquet
        summary_file = partition_path / "summary.parquet"
        df = pd.DataFrame(records)
        df.to_parquet(summary_file, index=False)
        
        print(f"  ✅ Created {summary_file} with {len(records)} items")
    
    print(f"  📈 Generated feature summaries for {len(items_data)} items over 6 hours")

def create_sample_bazaar_data():
    """Create sample bazaar data."""
    print("🏪 Creating sample bazaar data...")
    
    # Create bazaar_history directory
    bazaar_path = Path("data/bazaar_history")
    bazaar_path.mkdir(parents=True, exist_ok=True)
    
    current_time = datetime.now(timezone.utc)
    
    # Items with market data
    items_data = {
        "WHEAT": {"buy_price": 2.8, "sell_price": 3.2, "volume": 50000},
        "ENCHANTED_FLINT": {"buy_price": 14500, "sell_price": 15500, "volume": 1200},
        "COBBLESTONE": {"buy_price": 1.9, "sell_price": 2.1, "volume": 80000},
        "EMERALD": {"buy_price": 7800, "sell_price": 8200, "volume": 3000},
        "DIAMOND": {"buy_price": 8700, "sell_price": 9300, "volume": 2500}
    }
    
    # Generate data points over last 3 hours with 5-minute intervals
    records = []
    
    for minutes_ago in range(0, 180, 5):  # Every 5 minutes for 3 hours
        timestamp = current_time - timedelta(minutes=minutes_ago)
        
        for item_name, item_config in items_data.items():
            # Add some market movement
            time_factor = np.sin(minutes_ago / 30) * 0.02  # Slow oscillation
            noise = np.random.normal(0, 0.01)  # Random noise
            
            buy_price = item_config["buy_price"] * (1 + time_factor + noise)
            sell_price = item_config["sell_price"] * (1 + time_factor + noise * 0.8)
            
            # Ensure sell > buy
            if sell_price <= buy_price:
                sell_price = buy_price * 1.1
            
            volume_noise = np.random.normal(1, 0.1)
            buy_volume = int(item_config["volume"] * volume_noise * 0.6)
            sell_volume = int(item_config["volume"] * volume_noise * 0.4)
            
            record = {
                "timestamp": timestamp,
                "product_id": item_name,
                "buy_price": round(buy_price, 1),
                "sell_price": round(sell_price, 1),
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_orders": np.random.randint(10, 50),
                "sell_orders": np.random.randint(15, 60)
            }
            records.append(record)
    
    # Save to parquet
    df = pd.DataFrame(records)
    bazaar_file = bazaar_path / "sample_bazaar_data.parquet"
    df.to_parquet(bazaar_file, index=False)
    
    print(f"  ✅ Created {bazaar_file} with {len(records)} records")
    print(f"  📈 Generated bazaar data for {len(items_data)} items over 3 hours")

def create_sample_ndjson_bazaar():
    """Create sample NDJSON bazaar data as fallback."""
    print("📄 Creating sample NDJSON bazaar data...")
    
    ndjson_path = Path("data/bazaar_snapshots.ndjson")
    
    current_time = datetime.now(timezone.utc)
    
    # Generate a few snapshots
    snapshots = []
    
    for minutes_ago in [0, 30, 60, 90, 120]:  # Every 30 minutes
        timestamp = int((current_time - timedelta(minutes=minutes_ago)).timestamp() * 1000)
        
        products = {}
        
        items_data = {
            "WHEAT": {"buy_price": 3.0, "sell_price": 3.5},
            "COBBLESTONE": {"buy_price": 2.0, "sell_price": 2.3},
            "EMERALD": {"buy_price": 8000, "sell_price": 8500}
        }
        
        for item_name, item_config in items_data.items():
            noise = np.random.normal(1, 0.05)
            
            products[item_name] = {
                "buy_summary": [{"pricePerUnit": item_config["buy_price"] * noise}],
                "sell_summary": [{"pricePerUnit": item_config["sell_price"] * noise}],
                "quick_status": {
                    "buyMovingWeek": np.random.randint(10000, 50000),
                    "sellMovingWeek": np.random.randint(5000, 30000),
                    "buyOrders": np.random.randint(20, 80),
                    "sellOrders": np.random.randint(30, 100)
                }
            }
        
        snapshot = {
            "timestamp": timestamp,
            "products": products
        }
        snapshots.append(snapshot)
    
    # Write NDJSON
    with open(ndjson_path, 'w') as f:
        for snapshot in snapshots:
            f.write(json.dumps(snapshot) + '\n')
    
    print(f"  ✅ Created {ndjson_path} with {len(snapshots)} snapshots")

def main():
    """Generate all sample data."""
    print("🎯 Generating Sample Data for Testing")
    print("=" * 50)
    
    try:
        create_sample_feature_summaries()
        create_sample_bazaar_data()
        create_sample_ndjson_bazaar()
        
        print("\n" + "=" * 50)
        print("✅ Sample data generation complete!")
        print("\nYou can now test:")
        print("- Feature summaries: data/feature_summaries/")
        print("- Bazaar data: data/bazaar_history/")
        print("- NDJSON fallback: data/bazaar_snapshots.ndjson")
        print("\nRun scripts/self_test.py to verify the system is working.")
        
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())