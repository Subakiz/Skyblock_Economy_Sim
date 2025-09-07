#!/usr/bin/env python3
"""
Standalone Data Ingestion Service for Skyblock Economy Simulator

This service continuously fetches data from the Hypixel API, cleans it using 
the canonical item list method, and writes it to partitioned Parquet datasets.

Key features:
- Canonical item loader from /resources/skyblock/items endpoint
- Intelligent base_item_id cleaner function
- Auction ingestion loop (90 seconds)
- Bazaar ingestion loop (60 seconds)
- Partitioned Parquet output to data/auction_history and data/bazaar_history
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, List, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ingestion.common.hypixel_client import HypixelClient
from ingestion.item_processing import create_canonical_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StandaloneIngestionService:
    """Standalone data ingestion service for auctions and bazaar data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the ingestion service."""
        self.config = self._load_config(config_path)
        self.hypixel_client = self._create_hypixel_client()
        self.canonical_items: Set[str] = set()
        self.running = False
        
        # Data directories
        self.auction_data_path = Path("data/auction_history")
        self.bazaar_data_path = Path("data/bazaar_history")
        
        # Create directories if they don't exist
        self.auction_data_path.mkdir(parents=True, exist_ok=True)
        self.bazaar_data_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _create_hypixel_client(self) -> HypixelClient:
        """Create and configure Hypixel API client."""
        hypixel_config = self.config.get("hypixel", {})
        api_key = os.getenv("HYPIXEL_API_KEY")
        
        if not api_key:
            logger.warning("HYPIXEL_API_KEY not set. Using placeholder for testing.")
            api_key = "test_key"
        
        return HypixelClient(
            base_url=hypixel_config.get("base_url", "https://api.hypixel.net"),
            api_key=api_key,
            max_requests_per_minute=hypixel_config.get("max_requests_per_minute", 120),
            timeout_seconds=hypixel_config.get("timeout_seconds", 10)
        )
    
    async def load_canonical_items(self) -> None:
        """Load canonical items from Hypixel /resources/skyblock/items endpoint."""
        try:
            logger.info("Loading canonical items from Hypixel API...")
            
            # Fetch items from resources endpoint
            try:
                data = self.hypixel_client.get_json("resources/skyblock/items")
                items = data.get("items", [])
            except Exception as e:
                logger.warning(f"Failed to fetch canonical items from API: {e}")
                # Try to load from cache if API fails
                cache_file = Path("data/canonical_items_cache.txt")
                if cache_file.exists():
                    logger.info("Loading canonical items from cache due to API failure")
                    with open(cache_file, 'r') as f:
                        cached_items = [line.strip() for line in f if line.strip()]
                    self.canonical_items = set(cached_items)
                    logger.info(f"Loaded {len(self.canonical_items)} items from cache")
                    return
                else:
                    logger.info("Using fallback canonical item list")
                    items = []
            
            # Extract item names and create canonical set
            canonical_items = set()
            for item in items:
                item_name = item.get("name", "")
                if item_name:
                    # Store in uppercase for consistent matching
                    canonical_items.add(item_name.upper().strip())
                    
                # Also add ID if available for more matching options
                item_id = item.get("id", "")
                if item_id:
                    canonical_items.add(item_id.upper().strip())
            
            self.canonical_items = canonical_items
            logger.info(f"Loaded {len(self.canonical_items)} canonical items")
            
            # Cache to disk for offline use
            cache_file = Path("data/canonical_items_cache.txt")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w') as f:
                for item in sorted(self.canonical_items):
                    f.write(f"{item}\n")
            
            logger.info(f"Cached canonical items to {cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to load canonical items: {e}")
            # Try to load from cache if available
            cache_file = Path("data/canonical_items_cache.txt")
            if cache_file.exists():
                logger.info("Loading canonical items from cache...")
                with open(cache_file, 'r') as f:
                    self.canonical_items = {line.strip() for line in f if line.strip()}
                logger.info(f"Loaded {len(self.canonical_items)} items from cache")
            else:
                logger.warning("No canonical items available - using empty set")
                self.canonical_items = set()
    
    @staticmethod
    def _get_base_item_id(item_name: str, canonical_items: Set[str]) -> str:
        """
        Find the longest matching substring from canonical list to identify base item.
        
        Args:
            item_name: The raw item name from auction
            canonical_items: Set of all valid canonical item names (uppercase)
        
        Returns:
            The base item ID (longest match) or original name if no match found
        """
        if not item_name or not canonical_items:
            return item_name.upper().strip() if item_name else ""
        
        # Normalize input
        normalized_name = item_name.upper().strip()
        
        # First try exact match
        if normalized_name in canonical_items:
            return normalized_name
        
        # Find longest matching substring
        best_match = ""
        best_length = 0
        
        for canonical_item in canonical_items:
            # Check if canonical item is a substring of the auction name
            if canonical_item in normalized_name:
                if len(canonical_item) > best_length:
                    best_match = canonical_item
                    best_length = len(canonical_item)
            
            # Also check if auction name is a substring of canonical item
            elif normalized_name in canonical_item:
                if len(normalized_name) > best_length:
                    best_match = canonical_item
                    best_length = len(normalized_name)
        
        # Return best match or original name if no match found
        return best_match if best_match else normalized_name
    
    async def auction_ingestion_loop(self) -> None:
        """Auction data ingestion loop - runs every 90 seconds."""
        logger.info("Starting auction ingestion loop (90s interval)")
        
        while self.running:
            try:
                start_time = time.time()
                logger.info("Starting auction data collection...")
                
                # Fetch all pages from auctions endpoint
                all_auctions = []
                page = 0
                max_pages = self.config.get("auction_house", {}).get("max_pages_per_cycle", 100)
                
                while page < max_pages:
                    try:
                        response = self.hypixel_client.get_json("skyblock/auctions", {"page": page})
                        auctions = response.get("auctions", [])
                        total_pages = response.get("totalPages", 1)
                        
                        if not auctions:
                            break
                        
                        all_auctions.extend(auctions)
                        page += 1
                        
                        if page >= total_pages:
                            break
                        
                        logger.debug(f"Collected page {page}/{total_pages}")
                        
                    except Exception as e:
                        logger.error(f"Error fetching auction page {page}: {e}")
                        break
                
                logger.info(f"Collected {len(all_auctions)} auctions from {page} pages")
                
                # Process and clean auction data
                auction_records = []
                scan_timestamp = datetime.now(timezone.utc)
                
                for auction in all_auctions:
                    try:
                        # Extract essential fields
                        item_name = auction.get('item_name', '').strip()
                        item_lore = auction.get('item_lore', '')
                        
                        # Create canonical name with attributes
                        canonical_name = create_canonical_name(item_name, item_lore)
                        
                        # Use intelligent base_item_id cleaner on canonical name
                        base_item_id = self._get_base_item_id(canonical_name, self.canonical_items)
                        
                        # Extract price (BIN vs regular auction)
                        if auction.get('bin'):
                            price = float(auction.get('starting_bid', 0))
                        else:
                            price = float(auction.get('highest_bid_amount', 0))
                        
                        record = {
                            'uuid': auction.get('uuid', ''),
                            'item_name': canonical_name,  # Use canonical name
                            'original_name': item_name,   # Keep original for reference
                            'base_item_id': base_item_id,
                            'price': price,
                            'tier': auction.get('tier', ''),
                            'bin': auction.get('bin', False),
                            'seller': auction.get('auctioneer', ''),
                            'bids': len(auction.get('bids', [])),
                            'start_timestamp': datetime.fromtimestamp(
                                auction.get('start', 0) / 1000, tz=timezone.utc
                            ),
                            'end_timestamp': datetime.fromtimestamp(
                                auction.get('end', 0) / 1000, tz=timezone.utc
                            ),
                            'scan_timestamp': scan_timestamp
                        }
                        
                        # Only include records with valid data
                        if record['item_name'] and record['price'] > 0:
                            auction_records.append(record)
                        
                    except Exception as e:
                        logger.debug(f"Error processing auction {auction.get('uuid', 'unknown')}: {e}")
                        continue
                
                # Write to partitioned Parquet dataset
                if auction_records:
                    df = pd.DataFrame(auction_records)
                    
                    # Add partition columns
                    df['year'] = df['scan_timestamp'].dt.year
                    df['month'] = df['scan_timestamp'].dt.month  
                    df['day'] = df['scan_timestamp'].dt.day
                    
                    # Convert to Arrow table and write
                    table = pa.Table.from_pandas(df)
                    pq.write_to_dataset(
                        table,
                        root_path=str(self.auction_data_path),
                        partition_cols=['year', 'month', 'day'],
                        existing_data_behavior='overwrite_or_ignore'
                    )
                    
                    processing_time = time.time() - start_time
                    logger.info(f"Processed and saved {len(auction_records)} auction records "
                               f"in {processing_time:.1f}s")
                else:
                    logger.warning("No valid auction records to save")
                
            except Exception as e:
                logger.error(f"Error in auction ingestion loop: {e}")
            
            # Wait for next cycle (90 seconds)
            await asyncio.sleep(90)
    
    async def bazaar_ingestion_loop(self) -> None:
        """Bazaar data ingestion loop - runs every 60 seconds."""
        logger.info("Starting bazaar ingestion loop (60s interval)")
        
        while self.running:
            try:
                start_time = time.time()
                logger.info("Starting bazaar data collection...")
                
                # Fetch bazaar data
                response = self.hypixel_client.get_json("skyblock/bazaar")
                products = response.get("products", {})
                
                # Flatten product data into records
                bazaar_records = []
                scan_timestamp = datetime.now(timezone.utc)
                
                for product_id, product_data in products.items():
                    try:
                        # Get buy/sell summary data
                        buy_summary = product_data.get("buy_summary", [])
                        sell_summary = product_data.get("sell_summary", [])
                        quick_status = product_data.get("quick_status", {})
                        
                        # Create record for this product
                        record = {
                            'product_id': product_id,
                            'buy_price': quick_status.get('buyPrice', 0.0),
                            'sell_price': quick_status.get('sellPrice', 0.0),
                            'buy_volume': quick_status.get('buyVolume', 0),
                            'sell_volume': quick_status.get('sellVolume', 0),
                            'buy_orders': len(buy_summary),
                            'sell_orders': len(sell_summary),
                            'scan_timestamp': scan_timestamp
                        }
                        
                        # Add top buy/sell orders if available
                        if buy_summary:
                            record['top_buy_price'] = buy_summary[0].get('pricePerUnit', 0.0)
                            record['top_buy_amount'] = buy_summary[0].get('amount', 0)
                        else:
                            record['top_buy_price'] = 0.0
                            record['top_buy_amount'] = 0
                        
                        if sell_summary:
                            record['top_sell_price'] = sell_summary[0].get('pricePerUnit', 0.0)
                            record['top_sell_amount'] = sell_summary[0].get('amount', 0)
                        else:
                            record['top_sell_price'] = 0.0
                            record['top_sell_amount'] = 0
                        
                        bazaar_records.append(record)
                        
                    except Exception as e:
                        logger.debug(f"Error processing bazaar product {product_id}: {e}")
                        continue
                
                # Write to partitioned Parquet dataset
                if bazaar_records:
                    df = pd.DataFrame(bazaar_records)
                    
                    # Add partition columns
                    df['year'] = df['scan_timestamp'].dt.year
                    df['month'] = df['scan_timestamp'].dt.month  
                    df['day'] = df['scan_timestamp'].dt.day
                    
                    # Convert to Arrow table and write
                    table = pa.Table.from_pandas(df)
                    pq.write_to_dataset(
                        table,
                        root_path=str(self.bazaar_data_path),
                        partition_cols=['year', 'month', 'day'],
                        existing_data_behavior='overwrite_or_ignore'
                    )
                    
                    processing_time = time.time() - start_time
                    logger.info(f"Processed and saved {len(bazaar_records)} bazaar records "
                               f"in {processing_time:.1f}s")
                else:
                    logger.warning("No bazaar records to save")
                
            except Exception as e:
                logger.error(f"Error in bazaar ingestion loop: {e}")
            
            # Wait for next cycle (60 seconds)
            await asyncio.sleep(60)
    
    async def start(self) -> None:
        """Start the ingestion service with both loops running concurrently."""
        logger.info("Starting Standalone Data Ingestion Service...")
        
        # Load canonical items first
        await self.load_canonical_items()
        
        # Set running flag
        self.running = True
        
        try:
            # Run both ingestion loops concurrently
            await asyncio.gather(
                self.auction_ingestion_loop(),
                self.bazaar_ingestion_loop()
            )
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in ingestion service: {e}")
        finally:
            self.running = False
            logger.info("Ingestion service stopped")
    
    def stop(self) -> None:
        """Stop the ingestion service."""
        logger.info("Stopping ingestion service...")
        self.running = False


async def main():
    """Main entry point for the standalone ingestion service."""
    service = StandaloneIngestionService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        service.stop()


if __name__ == "__main__":
    asyncio.run(main())