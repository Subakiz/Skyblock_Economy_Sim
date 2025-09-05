"""
NDJSON File Storage Layer

Provides file-based storage for bazaar and auction data as an alternative to database storage.
Stores data as NDJSON (Newline Delimited JSON) files in the data/ directory.
"""

import os
import json
import gzip
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path
import glob


class NDJSONStorage:
    """File-based storage using NDJSON format."""
    
    def __init__(self, data_directory: str = "data", max_file_size_mb: int = 100, retention_hours: int = 168):
        self.data_dir = Path(data_directory)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.retention_hours = retention_hours
        
        # Create directory structure
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "bazaar").mkdir(exist_ok=True)
        (self.data_dir / "auctions").mkdir(exist_ok=True)
        (self.data_dir / "auctions_ended").mkdir(exist_ok=True)
    
    def _get_current_file_path(self, data_type: str) -> Path:
        """Get the current active file path for a data type."""
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        hour_str = datetime.now(timezone.utc).strftime('%H')
        
        if data_type == "bazaar":
            return self.data_dir / "bazaar" / f"bazaar_{date_str}_{hour_str}.ndjson"
        elif data_type == "auctions":
            return self.data_dir / "auctions" / f"auctions_{date_str}_{hour_str}.ndjson"
        elif data_type == "auctions_ended":
            return self.data_dir / "auctions_ended" / f"auctions_ended_{date_str}_{hour_str}.ndjson"
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def append_record(self, data_type: str, record: Dict[str, Any]):
        """Append a record to the appropriate NDJSON file."""
        file_path = self._get_current_file_path(data_type)
        
        # Add timestamp if not present
        if 'ts' not in record and 'timestamp' not in record:
            record['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Write record as JSON line
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        
        # Check if file rotation is needed
        if file_path.stat().st_size > self.max_file_size_bytes:
            self._rotate_file(file_path)
    
    def _rotate_file(self, file_path: Path):
        """Rotate a file by compressing it with a timestamp suffix."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        compressed_path = file_path.with_suffix(f'.{timestamp}.ndjson.gz')
        
        # Compress and move the current file
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Remove the original file
        file_path.unlink()
    
    def read_records(self, data_type: str, hours_back: int = 24) -> Iterator[Dict[str, Any]]:
        """Read records from NDJSON files for the specified time period."""
        pattern_dir = self.data_dir / data_type
        
        # Get current and historical files
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        # Find all relevant files (both current and compressed)
        files = []
        for pattern in [f"{data_type}_*.ndjson", f"{data_type}_*.ndjson.gz"]:
            files.extend(glob.glob(str(pattern_dir / pattern)))
        
        # Sort files by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        for file_path in files:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)
            if file_mod_time < cutoff_time:
                continue
                
            # Read file (handle both compressed and uncompressed)
            try:
                if file_path.endswith('.gz'):
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                # Check if record is within time window
                                record_time = self._extract_timestamp(record)
                                if record_time and record_time >= cutoff_time:
                                    yield record
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                record_time = self._extract_timestamp(record)
                                if record_time and record_time >= cutoff_time:
                                    yield record
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read file {file_path}: {e}")
                continue
    
    def _extract_timestamp(self, record: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from a record."""
        for key in ['ts', 'timestamp', 'end_time', 'start_time']:
            if key in record:
                try:
                    if isinstance(record[key], str):
                        return datetime.fromisoformat(record[key].replace('Z', '+00:00'))
                    elif isinstance(record[key], datetime):
                        return record[key]
                except (ValueError, TypeError):
                    continue
        return None
    
    def get_latest_bazaar_prices(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest bazaar prices for a product."""
        latest_record = None
        latest_timestamp = None
        
        for record in self.read_records("bazaar", hours_back=2):
            if record.get("product_id") == product_id:
                record_time = self._extract_timestamp(record)
                if record_time and (latest_timestamp is None or record_time > latest_timestamp):
                    latest_record = record
                    latest_timestamp = record_time
                    
        return latest_record
    
    def get_auction_price_stats(self, product_id: str, hours_back: int = 1) -> Dict[str, Any]:
        """Calculate auction price statistics for a product."""
        prices = []
        
        for record in self.read_records("auctions_ended", hours_back=hours_back):
            if record.get("item_id") == product_id and "sale_price" in record:
                try:
                    price = float(record["sale_price"])
                    if price > 0:
                        prices.append(price)
                except (ValueError, TypeError):
                    continue
        
        if not prices:
            return {}
        
        prices.sort()
        n = len(prices)
        
        return {
            "median_price": prices[n // 2] if n > 0 else None,
            "p25_price": prices[n // 4] if n > 0 else None,
            "p75_price": prices[3 * n // 4] if n > 0 else None,
            "avg_price": sum(prices) / n if n > 0 else None,
            "sale_count": n,
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None
        }
    
    def cleanup_old_files(self):
        """Remove files older than the retention period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        
        for subdir in ["bazaar", "auctions", "auctions_ended"]:
            dir_path = self.data_dir / subdir
            if dir_path.exists():
                for file_path in dir_path.glob("*.ndjson*"):
                    file_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                    if file_mod_time < cutoff_time:
                        file_path.unlink()
                        print(f"Cleaned up old file: {file_path}")


def get_storage_instance(config: Dict[str, Any]) -> Optional[NDJSONStorage]:
    """Get NDJSONStorage instance if no-database mode is enabled."""
    no_db_config = config.get("no_database_mode", {})
    if not no_db_config.get("enabled", False):
        return None
        
    return NDJSONStorage(
        data_directory=no_db_config.get("data_directory", "data"),
        max_file_size_mb=no_db_config.get("max_file_size_mb", 100),
        retention_hours=no_db_config.get("retention_hours", 168)
    )