#!/usr/bin/env python3
"""
Feature-First Ingestion Pipeline

Maintains in-memory price ladders per item for the current hour.
Writes compact hourly Parquet feature summaries instead of raw data.
Optionally spools raw NDJSON for retry safety during the hour.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import heapq

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from dotenv import load_dotenv

from ingestion.common.hypixel_client import HypixelClient
from ingestion.item_processing import create_canonical_name


logger = logging.getLogger(__name__)


class InMemoryPriceLadder:
    """Maintains the lowest-N prices for an item during the current hour."""
    
    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self.prices = []  # Min-heap of (price, count)
        self.price_counts = defaultdict(int)  # Map price -> count
        self.total_count = 0
    
    def add_price(self, price: float):
        """Add a price observation to the ladder."""
        self.price_counts[price] += 1
        self.total_count += 1
        
        # Rebuild ladder if needed
        self._rebuild_ladder()
    
    def _rebuild_ladder(self):
        """Rebuild the ladder from price_counts, keeping only lowest-N."""
        # Sort prices and take lowest N
        sorted_prices = sorted(self.price_counts.keys())
        
        # Keep only the lowest max_size unique prices
        self.prices = []
        kept_prices = set()
        
        for price in sorted_prices[:self.max_size]:
            count = self.price_counts[price]
            heapq.heappush(self.prices, (price, count))
            kept_prices.add(price)
        
        # Remove excess prices from counts
        for price in list(self.price_counts.keys()):
            if price not in kept_prices:
                del self.price_counts[price]
    
    def get_ladder_data(self) -> Tuple[List[int], List[int], int]:
        """Get ladder as (prices, counts, total_count)."""
        if not self.prices:
            return [], [], 0
        
        # Sort by price for consistent output
        sorted_pairs = sorted(self.prices)
        prices = [int(price) for price, _ in sorted_pairs]
        counts = [count for _, count in sorted_pairs]
        
        return prices, counts, self.total_count


class FeatureIngestor:
    """
    Feature-first ingestion service.
    
    Maintains in-memory price ladders for the current hour.
    Writes hourly feature summaries to Parquet.
    Optionally spools raw data for retry safety.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureIngestor")
        
        # Load environment variables from .env for defense-in-depth
        load_dotenv()
        
        # Configuration
        market_config = config.get("market", {})
        self.ladder_size = market_config.get("lowest_ladder_size", 16)
        self.intel_interval = market_config.get("intel_interval_seconds", 90)
        self.batch_size = market_config.get("batch_size", 20000)
        self.rows_soft_cap = market_config.get("rows_soft_cap", 300000)
        
        guards_config = config.get("guards", {})
        self.soft_rss_mb = guards_config.get("soft_rss_mb", 1300)
        
        # Data paths
        self.feature_summaries_path = Path("data/feature_summaries")
        self.raw_spool_path = Path("data/raw_spool")
        self.feature_summaries_path.mkdir(parents=True, exist_ok=True)
        self.raw_spool_path.mkdir(parents=True, exist_ok=True)
        
        # Hypixel client
        hypixel_config = config.get("hypixel", {})
        api_key = os.getenv("HYPIXEL_API_KEY")
        if not api_key:
            raise ValueError("HYPIXEL_API_KEY environment variable is required")
        
        self.hypixel_client = HypixelClient(
            base_url=hypixel_config.get("base_url", "https://api.hypixel.net"),
            api_key=api_key,
            max_requests_per_minute=hypixel_config.get("max_requests_per_minute", 120),
            timeout_seconds=hypixel_config.get("timeout_seconds", 10)
        )
        
        # Current hour state
        self.current_hour_start: Optional[datetime] = None
        self.item_ladders: Dict[str, InMemoryPriceLadder] = {}
        self.raw_spool_file: Optional[Path] = None
        self.raw_spool_handle = None
        
        self.logger.info(f"FeatureIngestor initialized with ladder_size={self.ladder_size}")
    
    def _check_memory_guard(self) -> bool:
        """Check if we should skip processing due to memory pressure."""
        try:
            process = psutil.Process()
            rss_mb = process.memory_info().rss / (1024 * 1024)
            
            if rss_mb > self.soft_rss_mb:
                self.logger.warning(f"Memory guard triggered: RSS {rss_mb:.1f}MB > {self.soft_rss_mb}MB, skipping cycle")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Memory guard check failed: {e}")
            return True  # Continue on error
    
    async def _start_new_hour(self, hour_start: datetime):
        """Start a new hour of data collection."""
        if self.current_hour_start == hour_start:
            return  # Already in this hour
        
        # Commit previous hour if exists
        if self.current_hour_start is not None:
            await self._commit_hour_summary()
        
        # Initialize new hour
        self.current_hour_start = hour_start
        self.item_ladders.clear()
        
        # Start raw spool file for retry safety
        spool_filename = f"auctions_{hour_start.strftime('%Y%m%d_%H')}.ndjson"
        self.raw_spool_file = self.raw_spool_path / spool_filename
        
        try:
            self.raw_spool_handle = open(self.raw_spool_file, 'w')
            self.logger.info(f"Started new hour collection: {hour_start}, spool: {spool_filename}")
        except Exception as e:
            self.logger.error(f"Failed to open spool file: {e}")
            self.raw_spool_handle = None
    
    def _build_summary_records(self) -> List[Dict[str, Any]]:
        """Build summary records from current item ladders."""
        summary_records = []
        for item_name, ladder in self.item_ladders.items():
            prices, counts, total_count = ladder.get_ladder_data()
            
            if total_count > 0:  # Only include items with data
                # Include additional fields as required by consumer
                floor_price = prices[0] if prices else 0.0
                second_lowest_price = prices[1] if len(prices) > 1 else floor_price
                auction_count = total_count
                
                summary_records.append({
                    "hour_start": self.current_hour_start,
                    "item_name": item_name,
                    "prices": prices,
                    "counts": counts,
                    "total_count": total_count,
                    "floor_price": floor_price,
                    "second_lowest_price": second_lowest_price,
                    "auction_count": auction_count
                })
        return summary_records

    def _write_summary_for_hour(self, hour_start: datetime, summary_records: List[Dict[str, Any]], 
                               dst_dir: Optional[Path] = None, overwrite: bool = True):
        """Write summary records to Parquet with atomic write."""
        if not summary_records:
            return
            
        try:
            # Use provided directory or default
            if dst_dir is None:
                dst_dir = self.feature_summaries_path
                
            # Create partitioned path
            partition_path = dst_dir / (
                f"year={hour_start.year}/"
                f"month={hour_start.month:02d}/"
                f"day={hour_start.day:02d}/"
                f"hour={hour_start.hour:02d}"
            )
            partition_path.mkdir(parents=True, exist_ok=True)
            
            # Write with atomic operation: summary.parquet.tmp -> summary.parquet
            parquet_file = partition_path / "summary.parquet"
            temp_file = partition_path / "summary.parquet.tmp"
            
            # Write to temporary file first
            df = pd.DataFrame(summary_records)
            df.to_parquet(temp_file, index=False)
            
            # Atomic rename
            import os
            os.replace(str(temp_file), str(parquet_file))
            
            self.logger.info(f"Wrote summary for {hour_start} with {len(summary_records)} items to {partition_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write summary for {hour_start}: {e}")
            # Clean up temp file if it exists
            temp_file = dst_dir / f"year={hour_start.year}/month={hour_start.month:02d}/day={hour_start.day:02d}/hour={hour_start.hour:02d}/summary.parquet.tmp"
            if temp_file.exists():
                temp_file.unlink()

    def flush_current_hour_summary(self):
        """Write an interim summary for the current hour using atomic write."""
        if not self.current_hour_start:
            self.logger.warning("No current hour data to flush")
            return
            
        try:
            summary_records = self._build_summary_records()
            if summary_records:
                self._write_summary_for_hour(self.current_hour_start, summary_records, overwrite=True)
                self.logger.info(f"Wrote interim summary for {self.current_hour_start} with {len(summary_records)} items")
            else:
                self.logger.debug("No data to flush for current hour")
                
        except Exception as e:
            self.logger.error(f"Failed to flush current hour summary: {e}")

    async def _commit_hour_summary(self):
        """Commit the current hour's data as a feature summary."""
        if not self.current_hour_start:
            return
        
        try:
            # Close raw spool
            if self.raw_spool_handle:
                self.raw_spool_handle.close()
                self.raw_spool_handle = None
            
            # Build and write summary records
            summary_records = self._build_summary_records()
            self._write_summary_for_hour(self.current_hour_start, summary_records)
            
            # Auto-update the NDJSON bridge after summary is written
            self._update_ndjson_bridge()
            
            # Clean up raw spool file (ephemeral)
            if self.raw_spool_file and self.raw_spool_file.exists():
                self.raw_spool_file.unlink()
                self.logger.debug(f"Deleted raw spool: {self.raw_spool_file.name}")
        
        except Exception as e:
            self.logger.error(f"Failed to commit hour summary: {e}")
    
    def _update_ndjson_bridge(self):
        """Update the NDJSON bridge file for legacy code compatibility."""
        try:
            self.logger.debug("Updating NDJSON bridge after feature summary commit...")
            
            # Call the bridge conversion function directly
            from scripts.bridge_parquet_to_ndjson import convert_parquet_summaries_to_ndjson
            success = convert_parquet_summaries_to_ndjson()
            
            if success:
                self.logger.debug("NDJSON bridge updated successfully")
            else:
                self.logger.warning("NDJSON bridge update failed")
                
        except Exception as e:
            self.logger.error(f"Failed to update NDJSON bridge: {e}")
    
    def _process_auction_record(self, auction: Dict[str, Any]):
        """Process a single auction record into the current hour's ladders."""
        try:
            # Extract key fields
            item_name = auction.get("item_name", "").strip()
            if not item_name:
                return
            
            # Only process BIN auctions for pricing
            if not auction.get("bin", False):
                return
            
            price = auction.get("starting_bid", 0)
            if not isinstance(price, (int, float)) or price <= 0:
                return
            
            # Get or create ladder for this item
            if item_name not in self.item_ladders:
                self.item_ladders[item_name] = InMemoryPriceLadder(self.ladder_size)
            
            # Add price to ladder
            self.item_ladders[item_name].add_price(float(price))
            
            # Spool raw record for retry safety (if enabled)
            if self.raw_spool_handle:
                json.dump(auction, self.raw_spool_handle)
                self.raw_spool_handle.write('\n')
        
        except Exception as e:
            self.logger.error(f"Failed to process auction record: {e}")
    
    async def run_ingestion_cycle(self):
        """Run a single ingestion cycle."""
        if not self._check_memory_guard():
            return
        
        try:
            cycle_start = time.time()
            self.logger.info("Starting ingestion cycle...")
            
            # Determine current hour
            now = datetime.now(timezone.utc)
            hour_start = now.replace(minute=0, second=0, microsecond=0)
            
            # Start new hour if needed
            await self._start_new_hour(hour_start)
            
            # Fetch auction data with pagination
            total_processed = 0
            page = 0
            
            while total_processed < self.rows_soft_cap:
                try:
                    response = await asyncio.to_thread(
                        self.hypixel_client.get_json,
                        "skyblock/auctions",
                        {"page": page}
                    )
                    
                    auctions = response.get("auctions", [])
                    if not auctions:
                        break  # No more pages
                    
                    # Process auctions
                    for auction in auctions:
                        self._process_auction_record(auction)
                        total_processed += 1
                        
                        if total_processed >= self.rows_soft_cap:
                            self.logger.warning(f"Hit soft cap at {self.rows_soft_cap} records, page {page}")
                            break
                    
                    page += 1
                    
                    # Check if this is the last page
                    total_pages = response.get("totalPages", 1)
                    if page >= total_pages:
                        break
                
                except Exception as e:
                    self.logger.error(f"Error fetching page {page}: {e}")
                    break
            
            processing_time = time.time() - cycle_start
            self.logger.info(f"Ingestion cycle completed: {total_processed} records, "
                           f"{len(self.item_ladders)} unique items, {processing_time:.1f}s")
        
        except Exception as e:
            self.logger.error(f"Ingestion cycle failed: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old raw spool files and feature summaries based on retention policy."""
        try:
            storage_config = self.config.get("storage", {})
            raw_retention_hours = storage_config.get("raw_retention_hours", 2)
            
            # Clean old raw spools
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=raw_retention_hours)
            
            for spool_file in self.raw_spool_path.glob("*.ndjson"):
                try:
                    # Parse timestamp from filename
                    if spool_file.stat().st_mtime < cutoff_time.timestamp():
                        spool_file.unlink()
                        self.logger.debug(f"Cleaned old spool: {spool_file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to clean spool {spool_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


if __name__ == "__main__":
    import yaml
    import os
    
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create and run ingestor
    ingestor = FeatureIngestor(config)
    asyncio.run(ingestor.run_ingestion_cycle())