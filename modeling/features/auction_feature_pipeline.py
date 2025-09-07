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
    
    # Create a copy of the dataframe to work with
    features_df = df.copy()
    
    # Vectorized calculation of duration and price_per_hour
    valid_duration_mask = pd.notna(features_df.get('start_time')) & pd.notna(features_df.get('end_time'))
    if valid_duration_mask.any():
        duration = features_df.loc[valid_duration_mask, 'end_time'] - features_df.loc[valid_duration_mask, 'start_time']
        duration_hours = np.maximum(0.1, duration.dt.total_seconds() / 3600)
        
        # Vectorized price selection logic
        # Priority: BIN price -> sale_price -> highest_bid -> starting_bid
        price = np.select([
            (features_df.loc[valid_duration_mask, 'bin']) & (features_df.loc[valid_duration_mask, 'starting_bid'] > 0),
            features_df.loc[valid_duration_mask, 'sale_price'].fillna(0) > 0,
            features_df.loc[valid_duration_mask, 'highest_bid'].fillna(0) > 0
        ], [
            features_df.loc[valid_duration_mask, 'starting_bid'],
            features_df.loc[valid_duration_mask, 'sale_price'],
            features_df.loc[valid_duration_mask, 'highest_bid']
        ], default=features_df.loc[valid_duration_mask, 'starting_bid'].fillna(0))
        
        features_df.loc[valid_duration_mask, 'price_per_hour'] = price / duration_hours
    
    # Set price_per_hour to 0 for rows without valid duration
    if 'price_per_hour' not in features_df.columns:
        features_df['price_per_hour'] = 0.0
    features_df.loc[~valid_duration_mask, 'price_per_hour'] = 0.0
    
    # Vectorized final price calculation
    # Priority: sale_price -> highest_bid -> starting_bid
    features_df['final_price'] = np.select([
        features_df['sale_price'].fillna(0) > 0,
        features_df['highest_bid'].fillna(0) > 0
    ], [
        features_df['sale_price'],
        features_df['highest_bid']
    ], default=features_df['starting_bid'].fillna(0))
    
    # Vectorized timestamp selection (use end_time, timestamp, or start_time)
    now_timestamp = datetime.now(timezone.utc)
    features_df['ts'] = np.select([
        pd.notna(features_df.get('end_time')),
        pd.notna(features_df.get('timestamp')),
        pd.notna(features_df.get('start_time'))
    ], [
        features_df.get('end_time'),
        features_df.get('timestamp'), 
        features_df.get('start_time')
    ], default=now_timestamp)
    
    # Vectorized item name feature extraction (simplified for performance)
    if 'item_name' in features_df.columns:
        item_names = features_df['item_name'].fillna('')
        item_names_lower = item_names.str.lower()
        
        # Vectorized enchantment detection
        enchant_patterns = '|'.join(['sharpness', 'critical', 'ender', 'giant_killer', 'cubism',
                                   'smite', 'bane_of_arthropods', 'cleave', 'execute', 'first_strike',
                                   'lethality', 'life_steal', 'looting', 'luck', 'scavenger',
                                   'vampirism', 'venomous', 'protection', 'growth', 'thorns'])
        features_df['is_clean'] = ~item_names_lower.str.contains(enchant_patterns, na=False)
        features_df['enchantment_count'] = item_names_lower.str.count(enchant_patterns)
        
        # Vectorized reforge detection
        reforge_patterns = '|'.join(['^ancient ', '^fabled ', '^legendary ', '^epic ', '^rare ', '^uncommon ',
                                   '^gentle ', '^odd ', '^fast ', '^fair ', '^sharp ', '^heroic ',
                                   '^spicy ', '^renowned ', '^beloved ', '^pure ', '^smart ', '^wise '])
        features_df['has_reforge'] = item_names_lower.str.contains(reforge_patterns, na=False)
        
        # Vectorized rarity tier determination
        features_df['rarity_tier'] = np.select([
            item_names_lower.str.contains('hyperion|necron', na=False),
            item_names_lower.str.contains('dragon|divan|shadow', na=False)
        ], ['legendary', 'epic'], default='common')
    else:
        # Default values if no item_name column
        features_df['is_clean'] = True
        features_df['enchantment_count'] = 0
        features_df['has_reforge'] = False
        features_df['rarity_tier'] = 'common'
    
    # Calculate rolling average prices for each item using time-based windows
    features_df = features_df.sort_values('ts')
    
    # Group by item_name and calculate time-based rolling averages
    def calculate_rolling_avg(group):
        if group.empty:
            return group
        group = group.sort_values('ts')
        
        # Set timestamp as index for time-based rolling windows
        group = group.set_index('ts')
        
        # Time-based rolling averages (24h and 7D)
        group['rolling_avg_price_24h'] = group['final_price'].rolling(
            window='24h', min_periods=1
        ).mean()
        group['rolling_avg_price_7d'] = group['final_price'].rolling(
            window='7D', min_periods=1
        ).mean()
        
        # Reset index to restore 'ts' as a column
        return group.reset_index()
    
    if 'item_name' in features_df.columns and not features_df.empty:
        # Use a more compatible approach for groupby.apply
        grouped = features_df.groupby('item_name', group_keys=False)
        features_df = pd.concat([calculate_rolling_avg(group) for name, group in grouped], ignore_index=True)
    else:
        # Fallback if no item_name grouping possible
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