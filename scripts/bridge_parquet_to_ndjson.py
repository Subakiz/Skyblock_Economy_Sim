#!/usr/bin/env python3
"""
Bridge Script: Convert Parquet Feature Summaries to NDJSON Format

This script bridges the gap between the new Parquet-based feature summaries 
and legacy code that expects auction_features.ndjson.

This addresses the core issue where the code shows:
"Auction features file not found: data/auction_features.ndjson"
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

def load_config() -> Dict[str, Any]:
    """Load configuration."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def convert_parquet_summaries_to_ndjson():
    """
    Convert existing Parquet feature summaries to auction_features.ndjson format.
    
    This function reads from data/feature_summaries/ and creates the expected
    data/auction_features.ndjson file that legacy code requires.
    """
    cfg = load_config()
    data_directory = cfg.get("no_database_mode", {}).get("data_directory", "data")
    
    # Paths
    feature_summaries_path = Path(data_directory) / "feature_summaries"
    output_path = Path(data_directory) / "auction_features.ndjson"
    
    if not feature_summaries_path.exists():
        print(f"❌ No feature summaries found at {feature_summaries_path}")
        return False
    
    print(f"🔄 Converting Parquet feature summaries to NDJSON format...")
    print(f"   Source: {feature_summaries_path}")
    print(f"   Target: {output_path}")
    
    # Collect all parquet files
    parquet_files = list(feature_summaries_path.rglob("*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in {feature_summaries_path}")
        return False
    
    print(f"📊 Found {len(parquet_files)} feature summary files")
    
    # Load and combine all summaries
    all_summaries = []
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            all_summaries.append(df)
            print(f"   ✅ Loaded {len(df)} records from {parquet_file.relative_to(feature_summaries_path)}")
        except Exception as e:
            print(f"   ⚠️ Failed to load {parquet_file}: {e}")
    
    if not all_summaries:
        print("❌ No valid summary data found")
        return False
    
    # Combine all dataframes
    combined_df = pd.concat(all_summaries, ignore_index=True)
    print(f"📈 Combined {len(combined_df)} total records for {combined_df['item_name'].nunique()} unique items")
    
    # Convert to auction features format expected by legacy code
    auction_features = []
    
    for _, row in combined_df.iterrows():
        try:
            # Create auction-like feature record from summary data
            feature_record = {
                "item_name": row["item_name"],
                "timestamp": row["hour_start"] if "hour_start" in row else datetime.now(timezone.utc).isoformat(),
                "floor_price": int(row["floor_price"]) if pd.notna(row["floor_price"]) else None,
                "second_lowest_price": int(row["second_lowest_price"]) if pd.notna(row["second_lowest_price"]) else None,
                "auction_count": int(row["auction_count"]) if pd.notna(row["auction_count"]) else 0,
                "total_count": int(row["total_count"]) if pd.notna(row["total_count"]) else 0,
                
                # Add price ladder data if available
                "prices": row.get("prices", []),
                "counts": row.get("counts", []),
                
                # Derived features for ML compatibility
                "is_clean": True,  # Assume clean for items in watchlist
                "has_reforge": False,  # Conservative assumption
                "has_enchants": False,  # Conservative assumption
                "rarity": "COMMON",  # Default rarity
                
                # Market metadata
                "data_source": "feature_summary_bridge",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            auction_features.append(feature_record)
            
        except Exception as e:
            print(f"   ⚠️ Failed to convert record for {row.get('item_name', 'unknown')}: {e}")
    
    if not auction_features:
        print("❌ No feature records could be converted")
        return False
    
    # Write to NDJSON file
    print(f"💾 Writing {len(auction_features)} auction feature records to {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write with proper JSON serialization
    with open(output_path, 'w') as f:
        for record in auction_features:
            # Convert numpy types to native Python for JSON serialization
            clean_record = {}
            for key, value in record.items():
                if value is None:
                    clean_record[key] = None
                elif isinstance(value, (list, np.ndarray)):
                    # Handle arrays/lists
                    if isinstance(value, np.ndarray):
                        clean_record[key] = value.tolist()
                    else:
                        clean_record[key] = value
                elif isinstance(value, (np.int64, np.int32)):
                    clean_record[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    clean_record[key] = float(value)
                elif isinstance(value, np.bool_):
                    clean_record[key] = bool(value)
                elif isinstance(value, pd.Timestamp):
                    clean_record[key] = value.isoformat()
                elif pd.isna(value):
                    clean_record[key] = None
                else:
                    clean_record[key] = value
            
            f.write(json.dumps(clean_record) + '\n')
    
    print(f"✅ Successfully created auction_features.ndjson with {len(auction_features)} records")
    print(f"   Items covered: {len(set(r['item_name'] for r in auction_features))}")
    
    return True

def main():
    """Main function to run the bridge conversion."""
    print("🌉 Parquet to NDJSON Bridge Converter")
    print("=" * 50)
    
    success = convert_parquet_summaries_to_ndjson()
    
    if success:
        print("\n✅ Bridge conversion completed successfully!")
        print("   Legacy code expecting auction_features.ndjson should now work.")
    else:
        print("\n❌ Bridge conversion failed!")
        print("   Check that feature summaries exist in data/feature_summaries/")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)