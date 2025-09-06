"""
Auction Feature Pipeline for No-Database Mode

Processes raw auction data from NDJSON files and creates meaningful features
for auction items like HYPERION, NECRON_CHESTPLATE, etc.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import re

from storage.ndjson_storage import get_storage_instance


def load_config() -> Dict[str, Any]:
    """Load configuration."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_auction_data_from_files(data_directory: str) -> Optional[pd.DataFrame]:
    """Load auction data from NDJSON files."""
    data_path = Path(data_directory)
    
    # Look for auction files
    potential_files = [
        data_path / "auctions.ndjson",
        data_path / "auctions" / "auctions.ndjson"
    ]
    
    # Also look for timestamped auction files
    auctions_dir = data_path / "auctions"
    if auctions_dir.exists():
        import glob
        pattern_files = glob.glob(str(auctions_dir / "auctions_*.ndjson"))
        potential_files.extend(pattern_files)
    
    # Also look for ended auctions
    auctions_ended_dir = data_path / "auctions_ended"
    if auctions_ended_dir.exists():
        import glob
        ended_files = glob.glob(str(auctions_ended_dir / "auctions_ended_*.ndjson"))
        potential_files.extend(ended_files)
    
    records = []
    files_found = 0
    
    for file_path in potential_files:
        if Path(file_path).exists():
            files_found += 1
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            records.append(record)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
    
    if files_found == 0:
        print(f"ERROR: No auction data files found in {data_directory}")
        print("Expected files: auctions.ndjson, auctions/auctions_*.ndjson, or auctions_ended/auctions_ended_*.ndjson")
        return None
    
    if not records:
        print("ERROR: No valid records found in auction data files")
        return None
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(records)} auction records from {files_found} file(s)")
    return df


def extract_item_name_features(item_name: str) -> Dict[str, Any]:
    """Extract features from item name to determine if item is 'clean' etc."""
    features = {
        'is_clean': True,  # Assume clean unless proven otherwise
        'enchantment_count': 0,
        'has_reforge': False,
        'rarity_tier': 'common'
    }
    
    if not item_name:
        return features
    
    item_name_lower = item_name.lower()
    
    # Check for common enchantments that make items "not clean"
    enchant_indicators = [
        'sharpness', 'critical', 'ender', 'giant_killer', 'cubism',
        'smite', 'bane_of_arthropods', 'cleave', 'execute', 'first_strike',
        'lethality', 'life_steal', 'looting', 'luck', 'scavenger',
        'vampirism', 'venomous', 'protection', 'growth', 'thorns'
    ]
    
    for enchant in enchant_indicators:
        if enchant in item_name_lower:
            features['is_clean'] = False
            features['enchantment_count'] += 1
    
    # Check for reforge indicators
    reforge_indicators = [
        'ancient', 'fabled', 'legendary', 'epic', 'rare', 'uncommon',
        'gentle', 'odd', 'fast', 'fair', 'epic', 'sharp', 'heroic',
        'spicy', 'renowned', 'beloved', 'pure', 'smart', 'wise'
    ]
    
    for reforge in reforge_indicators:
        if item_name_lower.startswith(reforge + ' '):
            features['has_reforge'] = True
            break
    
    # Determine rarity (this is rough estimation)
    if 'hyperion' in item_name_lower or 'necron' in item_name_lower:
        features['rarity_tier'] = 'legendary'
    elif any(rare_item in item_name_lower for rare_item in ['dragon', 'divan', 'shadow']):
        features['rarity_tier'] = 'epic'
    
    return features


def calculate_auction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate meaningful features for auction items."""
    
    # Ensure required columns exist
    required_cols = ['item_id', 'item_name', 'bin', 'starting_bid', 'highest_bid', 'start_time', 'end_time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns in auction data: {missing_cols}")
        for col in missing_cols:
            if col in ['bin']:
                df[col] = False
            elif col in ['starting_bid', 'highest_bid']:
                df[col] = 0
            else:
                df[col] = None
    
    # Convert timestamps
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    if 'timestamp' in df.columns:  # For ended auctions
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    features_list = []
    
    for _, row in df.iterrows():
        feature_row = row.to_dict().copy()
        
        # Calculate price_per_hour
        if pd.notna(row.get('start_time')) and pd.notna(row.get('end_time')):
            duration = row['end_time'] - row['start_time']
            duration_hours = max(0.1, duration.total_seconds() / 3600)  # At least 0.1 hour
            
            # Use appropriate price
            if row.get('bin', False) and row.get('starting_bid', 0) > 0:
                price = row['starting_bid']  # BIN price
            elif row.get('sale_price', 0) > 0:  # Ended auction
                price = row['sale_price']
            elif row.get('highest_bid', 0) > 0:
                price = row['highest_bid']
            else:
                price = row.get('starting_bid', 0)
            
            feature_row['price_per_hour'] = price / duration_hours if duration_hours > 0 else 0
        else:
            feature_row['price_per_hour'] = 0
        
        # Extract item name features
        item_name = row.get('item_name', '')
        name_features = extract_item_name_features(item_name)
        feature_row.update(name_features)
        
        # Add final sale price (use sale_price if available, otherwise highest_bid or starting_bid)
        if 'sale_price' in row and pd.notna(row['sale_price']) and row['sale_price'] > 0:
            feature_row['final_price'] = row['sale_price']
        elif row.get('highest_bid', 0) > 0:
            feature_row['final_price'] = row['highest_bid']
        else:
            feature_row['final_price'] = row.get('starting_bid', 0)
        
        # Add timestamp for sorting (use end_time, timestamp, or start_time)
        if pd.notna(row.get('end_time')):
            feature_row['ts'] = row['end_time']
        elif pd.notna(row.get('timestamp')):
            feature_row['ts'] = row['timestamp']
        elif pd.notna(row.get('start_time')):
            feature_row['ts'] = row['start_time']
        else:
            feature_row['ts'] = datetime.now(timezone.utc)
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    
    # Calculate rolling average prices for each item
    features_df = features_df.sort_values('ts')
    
    # Group by item_id and calculate rolling averages
    def calculate_rolling_avg(group):
        group = group.sort_values('ts')
        # Simple rolling averages with a window size instead of time-based
        group['rolling_avg_price_24h'] = group['final_price'].rolling(
            window=24, min_periods=1
        ).mean()
        # 7-day rolling average  
        group['rolling_avg_price_7d'] = group['final_price'].rolling(
            window=168, min_periods=1
        ).mean()
        return group
    
    if 'item_id' in features_df.columns and not features_df.empty:
        features_df = features_df.groupby('item_id').apply(calculate_rolling_avg).reset_index(drop=True)
    else:
        # Fallback if no item_id grouping possible
        features_df['rolling_avg_price_24h'] = features_df['final_price']
        features_df['rolling_avg_price_7d'] = features_df['final_price']
    
    return features_df


def save_auction_features_to_file(features_df: pd.DataFrame, output_path: str):
    """Save auction features to NDJSON file."""
    
    # Convert DataFrame to NDJSON format
    records = features_df.to_dict('records')
    
    # Convert timestamps to ISO strings and handle NaN/None values
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, datetime):
                record[key] = value.isoformat()
            elif isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif isinstance(value, (np.int64, np.int32)):
                record[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                record[key] = float(value)
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved {len(records)} auction feature records to {output_path}")


def build_auction_features_from_files():
    """Build auction features from NDJSON files in no-database mode."""
    cfg = load_config()
    data_directory = cfg.get("no_database_mode", {}).get("data_directory", "data")
    
    # Load raw auction data
    print("Loading auction data from files...")
    df = load_auction_data_from_files(data_directory)
    
    if df is None:
        print("ERROR: Could not load auction data from files")
        return
    
    # Calculate auction-specific features
    print("Calculating auction features...")
    features_df = calculate_auction_features(df)
    
    # Save to output file
    output_path = Path(data_directory) / "auction_features.ndjson"
    save_auction_features_to_file(features_df, str(output_path))
    
    print("Auction features built from files.")


def load_auction_features_from_file(data_directory: str, item_ids: List[str] = None) -> Optional[pd.DataFrame]:
    """Load auction features from file, optionally filtered by item IDs."""
    
    features_path = Path(data_directory) / "auction_features.ndjson"
    
    if not features_path.exists():
        print(f"Auction features file not found: {features_path}")
        print("Run auction feature pipeline first to generate features.")
        return None
    
    records = []
    try:
        with open(features_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    # Filter by item_ids if provided
                    if item_ids is None or record.get('item_id') in item_ids:
                        records.append(record)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading auction features: {e}")
        return None
    
    if not records:
        if item_ids:
            print(f"No auction features found for items: {item_ids}")
        else:
            print("No auction features found in file")
        return None
    
    df = pd.DataFrame(records)
    
    # Convert timestamp columns back to datetime
    time_cols = ['ts', 'start_time', 'end_time', 'timestamp']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print(f"Loaded {len(df)} auction feature records")
    if item_ids:
        print(f"Filtered for items: {item_ids}")
        
    return df


if __name__ == "__main__":
    import os
    # Build auction features when run directly
    build_auction_features_from_files()