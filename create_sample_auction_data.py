#!/usr/bin/env python3
"""
Create sample auction data for testing the auction feature pipeline.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path


def create_sample_auction_data():
    """Create sample auction data for HYPERION and NECRON_CHESTPLATE."""
    
    # Ensure data directory exists
    data_dir = Path("data/auctions_ended")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample auction items with realistic data
    items_data = {
        "HYPERION": {
            "base_price": 800000000,  # 800M coins
            "price_variance": 100000000,  # 100M variance
            "item_name_variants": [
                "Hyperion",
                "Ancient Hyperion", 
                "Fabled Hyperion",
                "Legendary Hyperion",
                "Hyperion (Clean)",
                "Sharpness VII Hyperion",
                "Hyperion (Ultimate Wise V)"
            ]
        },
        "NECRON_CHESTPLATE": {
            "base_price": 50000000,  # 50M coins  
            "price_variance": 10000000,  # 10M variance
            "item_name_variants": [
                "Necron's Chestplate",
                "Ancient Necron's Chestplate",
                "Renowned Necron's Chestplate", 
                "Necron's Chestplate (Clean)",
                "Protection VII Necron's Chestplate",
                "Growth VII Necron's Chestplate"
            ]
        }
    }
    
    records = []
    base_time = datetime.now(timezone.utc) - timedelta(days=7)  # 7 days of data
    
    # Generate auction records for the past week
    for hour_offset in range(0, 168, 2):  # Every 2 hours for 7 days
        timestamp = base_time + timedelta(hours=hour_offset)
        
        for item_id, item_config in items_data.items():
            # Create 1-3 auctions per item per time period
            num_auctions = 1 + (hash(f"{item_id}_{hour_offset}") % 3)
            
            for auction_num in range(num_auctions):
                # Price variation
                price_factor = 0.8 + (hash(f"{item_id}_{hour_offset}_{auction_num}") % 100) / 250  # 0.8 to 1.2
                sale_price = int(item_config["base_price"] * price_factor)
                
                # Random variance
                variance = (hash(f"var_{item_id}_{hour_offset}_{auction_num}") % 1000 - 500) * item_config["price_variance"] // 1000
                sale_price += variance
                sale_price = max(sale_price, item_config["base_price"] // 4)  # Minimum price
                
                # Pick a random item name variant
                name_idx = hash(f"name_{item_id}_{auction_num}") % len(item_config["item_name_variants"])
                item_name = item_config["item_name_variants"][name_idx]
                
                # Auction duration (1-48 hours)
                duration_hours = 1 + (hash(f"dur_{item_id}_{hour_offset}_{auction_num}") % 48)
                start_time = timestamp - timedelta(hours=duration_hours)
                end_time = timestamp
                
                # BIN vs auction
                is_bin = hash(f"bin_{item_id}_{hour_offset}_{auction_num}") % 3 == 0  # 33% BIN
                
                record = {
                    "uuid": f"auction_{item_id}_{hour_offset}_{auction_num}",
                    "item_id": item_id,
                    "item_name": item_name,
                    "tier": "LEGENDARY" if item_id == "HYPERION" else "LEGENDARY",
                    "bin": is_bin,
                    "starting_bid": sale_price if is_bin else int(sale_price * 0.7),
                    "highest_bid": None if is_bin else sale_price,
                    "sale_price": sale_price,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "timestamp": timestamp.isoformat(),
                    "seller": f"seller_{hash(f'sell_{item_id}_{auction_num}') % 1000}",
                    "buyer": f"buyer_{hash(f'buy_{item_id}_{auction_num}') % 1000}",
                    "bids_count": 0 if is_bin else (hash(f"bids_{item_id}_{auction_num}") % 20),
                    "category": "WEAPON" if item_id == "HYPERION" else "ARMOR"
                }
                
                records.append(record)
    
    # Write to NDJSON file
    output_file = data_dir / f"auctions_ended_{datetime.now().strftime('%Y%m%d_%H')}.ndjson"
    
    with open(output_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Created {len(records)} sample auction records in {output_file}")
    return len(records)


if __name__ == "__main__":
    count = create_sample_auction_data()
    print(f"Sample data creation complete: {count} records")