#!/usr/bin/env python3
"""
Hypixel Auction Sniper Cog
High-performance auction sniping with two-speed architecture.
"""

import asyncio
import json
import logging
import time
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Set, List, Optional, Any, Tuple
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import discord
from discord import app_commands
from discord.ext import commands, tasks
import aiofiles
import psutil

# Import project modules
from ingestion.common.hypixel_client import HypixelClient
from ingestion.item_processing import create_canonical_name
from ingestion.feature_consumer import FeatureConsumer


def convert_np(obj):
    """
    Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Any object that might contain NumPy types
        
    Returns:
        The object converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class AuctionSniper(commands.Cog):
    """
    Intelligent auction sniper for Hypixel Skyblock.
    
    Uses a two-speed architecture:
    - Hunter: High-frequency scanning (2s) for quick snipes
    - Analyst: Low-frequency analysis (90s) for market intelligence
    """
    
    # Parquet data directory
    PARQUET_DATA_PATH = Path("data/auction_history")
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.AuctionSniper")
        
        # Configuration
        self.config = self._load_config()
        sniper_config = self.config.get("auction_sniper", {})
        
        # Hypixel API client
        try:
            self.hypixel_client = HypixelClient(
                base_url=self.config["hypixel"]["base_url"],
                api_key=os.getenv("HYPIXEL_API_KEY"),
                max_requests_per_minute=self.config["hypixel"]["max_requests_per_minute"],
                timeout_seconds=self.config["hypixel"]["timeout_seconds"]
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Hypixel client: {e}")
            self.hypixel_client = None
        
        # Sniper state
        self.auction_watchlist: Set[str] = set()  # Items worth sniping
        self.fmv_data: Dict[str, Dict[str, float]] = {}  # Fair Market Value cache
        self.recent_alerts: Dict[str, datetime] = {}  # Cooldown tracking for alerts
        self.recent_auction_uuids: Set[str] = set()  # UUID-based deduplication
        self.uuid_cooldown_minutes = 30  # How long to remember auction UUIDs
        self.alert_channel_id: Optional[int] = None
        self.profit_threshold: float = sniper_config.get("profit_threshold", 100000)
        self.min_auction_count: int = sniper_config.get("min_auction_count", 50)
        self.alert_cooldown_minutes: int = sniper_config.get("alert_cooldown_minutes", 10)  # Default 10 minute cooldown
        
        # Feature consumer for market intelligence
        self.feature_consumer = FeatureConsumer(self.config)
        
        # Memory and performance guards
        guards_config = self.config.get("guards", {})
        self.soft_rss_mb = guards_config.get("soft_rss_mb", 1300)
        self.hard_rss_mb = guards_config.get("hard_rss_mb", 1500)
        
        # Market intelligence configuration
        market_config = self.config.get("market", {})
        self.window_hours = market_config.get("window_hours", 12)
        self.intel_interval = market_config.get("intel_interval_seconds", 90)
        self.rows_soft_cap = market_config.get("rows_soft_cap", 300000)
        
        # Data directories
        self.data_dir = Path("data/sniper")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Parquet data directory
        self.PARQUET_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Load persisted data
        self._load_persisted_data()
        self._load_saved_config()
        
        # Hardcoded fallback for alert channel if not configured
        # This ensures notifications work immediately without requiring manual configuration
        if not self.alert_channel_id:
            self.alert_channel_id = 1414187136609030154  # Default alert channel
            self.logger.info(f"Using hardcoded fallback alert channel: {self.alert_channel_id}")
        
        # Task management
        self.hunter_task_active = False
        self.analyst_task_active = False
        
        self.logger.info(f"Enhanced AuctionSniper initialized with memory guards: soft={self.soft_rss_mb}MB")
        
        # Set up task intervals dynamically
        self.update_market_intelligence.change_interval(seconds=self.intel_interval)
    
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
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from bot's config."""
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_persisted_data(self):
        """Load watchlist and FMV data from disk."""
        try:
            # Load watchlist
            watchlist_file = self.data_dir / "auction_watchlist.json"
            if watchlist_file.exists():
                with open(watchlist_file, "r") as f:
                    data = json.load(f)
                    self.auction_watchlist = set(data.get("items", []))
                    self.logger.info(f"Loaded {len(self.auction_watchlist)} items in watchlist")
            
            # Load FMV data
            fmv_file = self.data_dir / "fmv_cache.json"
            if fmv_file.exists():
                with open(fmv_file, "r") as f:
                    self.fmv_data = json.load(f)
                    self.logger.info(f"Loaded FMV data for {len(self.fmv_data)} items")
        
        except Exception as e:
            self.logger.error(f"Failed to load persisted data: {e}")
    
    def _load_saved_config(self):
        """Load saved sniper configuration from disk."""
        try:
            config_file = self.data_dir / "sniper_config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                    self.alert_channel_id = config_data.get("alert_channel_id")
                    # Don't override config.yaml values unless they were manually set
                    self.logger.info("Loaded saved sniper configuration")
        
        except Exception as e:
            self.logger.error(f"Failed to load saved config: {e}")
    
    async def _save_persisted_data(self):
        """Save watchlist and FMV data to disk asynchronously."""
        try:
            # Save watchlist
            watchlist_file = self.data_dir / "auction_watchlist.json"
            watchlist_data = {
                "items": list(self.auction_watchlist),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "count": len(self.auction_watchlist)
            }
            async with aiofiles.open(watchlist_file, "w") as f:
                await f.write(json.dumps(watchlist_data, indent=2, default=convert_np))
            
            # Save FMV data
            fmv_file = self.data_dir / "fmv_cache.json"
            async with aiofiles.open(fmv_file, "w") as f:
                await f.write(json.dumps(self.fmv_data, indent=2, default=convert_np))
            
            self.logger.debug("Saved persisted data to disk")
        
        except Exception as e:
            self.logger.error(f"Failed to save persisted data: {e}")
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Start sniper tasks when bot is ready."""
        if self.hypixel_client and not self.hunter_task_active:
            self.logger.info("Starting auction sniper tasks...")
            
            # Update task intervals from config
            sniper_config = self.config.get("auction_sniper", {})
            hunter_interval = sniper_config.get("hunter_interval_seconds", 2)
            # Intelligence interval is now fixed at 60 seconds as per requirements
            intelligence_interval = 60
            
            # Change task intervals
            self.high_frequency_snipe_scan.change_interval(seconds=hunter_interval)
            self.update_market_intelligence.change_interval(seconds=intelligence_interval)
            
            # Start tasks
            self.high_frequency_snipe_scan.start()
            self.update_market_intelligence.start()
            self.hunter_task_active = True
            self.analyst_task_active = True
            
            self.logger.info(f"Started sniper tasks: Hunter ({hunter_interval}s), Intelligence ({intelligence_interval}s)")
    
    async def cog_unload(self):
        """Clean shutdown of tasks."""
        self.logger.info("Shutting down auction sniper tasks...")
        if self.high_frequency_snipe_scan.is_running():
            self.high_frequency_snipe_scan.cancel()
        if self.update_market_intelligence.is_running():
            self.update_market_intelligence.cancel()
        await self._save_persisted_data()
    
    @tasks.loop(seconds=2)  # Default, will be updated from config
    async def high_frequency_snipe_scan(self):
        """
        Hunter Task: High-frequency scanning of page 0 for quick snipes.
        Runs every 2 seconds for maximum speed and real-time snipe detection.
        Maintained from original two-speed architecture as specified in requirements.
        """
        # Enhanced debugging for sniper issues
        if not self.hypixel_client:
            self.logger.debug("Sniper scan skipped: No Hypixel client available (API key required)")
            return
            
        if not self.auction_watchlist:
            self.logger.debug(f"Sniper scan skipped: Empty watchlist (need market intelligence update)")
            return
            
        # Log sniper activity periodically for debugging
        scan_count = getattr(self, '_scan_count', 0) + 1
        self._scan_count = scan_count
        if scan_count % 30 == 1:  # Log every 30 scans (1 minute)
            self.logger.info(f"Sniper active: {len(self.auction_watchlist)} watchlist items, "
                           f"FMV data for {len(self.fmv_data)} items")
        
        try:
            start_time = time.time()
            
            # For testing without API key, use test mode
            if not hasattr(self, '_test_mode_enabled'):
                # Enable test mode if we have watchlist and FMV but no API key
                self._test_mode_enabled = bool(self.auction_watchlist and self.fmv_data and not os.getenv("HYPIXEL_API_KEY"))
                if self._test_mode_enabled:
                    self.logger.info("🧪 Test mode enabled: Creating mock auctions for sniper testing")
            
            if self._test_mode_enabled:
                # Create mock auction data for testing
                auctions = self._generate_test_auctions()
                self.logger.debug(f"Generated {len(auctions)} test auctions for evaluation")
            else:
                # Fetch only page 0 for speed (real mode)
                response = self.hypixel_client.get_json("skyblock/auctions", {"page": 0})
                auctions = response.get("auctions", [])
            
            snipes_found = 0
            
            for auction in auctions:
                try:
                    # Quick filters first (fail-fast)
                    if not auction.get("bin", False):  # Must be Buy It Now
                        continue
                    
                    item_name = auction.get("item_name", "").strip()
                    item_lore = auction.get("item_lore", "")
                    
                    # Generate canonical name for consistency with ingested data
                    canonical_name = create_canonical_name(item_name, item_lore)
                    
                    if not canonical_name or canonical_name not in self.auction_watchlist:
                        continue
                    
                    # Update auction with canonical name for verification
                    auction["item_name"] = canonical_name
                    
                    # Passed initial filters, do intensive verification
                    if await self._verify_snipe(auction):
                        await self._alert_snipe(auction)
                        snipes_found += 1
                
                except Exception as e:
                    self.logger.error(f"Error processing auction {auction.get('uuid', 'unknown')}: {e}")
            
            scan_time = time.time() - start_time
            if snipes_found > 0:
                self.logger.info(f"Hunter scan: {len(auctions)} auctions, {snipes_found} snipes found in {scan_time:.2f}s")
        
        except Exception as e:
            self.logger.error(f"Hunter task error: {e}")
    
    @tasks.loop()  # Dynamic interval set in setup
    async def update_market_intelligence(self):
        """
        Market Intelligence Task: Generate market intelligence from feature summaries.
        Uses feature consumer to read compact summaries instead of raw data.
        Memory-efficient with guards and dynamic window scaling.
        """
        if not self._check_memory_guard():
            return
        
        try:
            start_time = time.time()
            self.logger.info("Starting feature-based market intelligence update...")
            
            # Run market intelligence generation in background thread
            intelligence = await asyncio.to_thread(
                self.feature_consumer.generate_market_intelligence,
                self.window_hours
            )
            
            if not intelligence:
                self.logger.warning("No market intelligence generated")
                return
            
            # Update internal state
            self.auction_watchlist = intelligence.get("watchlist", set())
            self.fmv_data = intelligence.get("fmv_data", {})
            
            # Save updated data
            await self._save_persisted_data()
            
            # Log results
            metadata = intelligence.get("metadata", {})
            processing_time = time.time() - start_time
            
            self.logger.info(f"Market intelligence updated: {len(self.auction_watchlist)} watchlist items, "
                           f"FMV data for {len(self.fmv_data)} items "
                           f"(analyzed {metadata.get('items_analyzed', 0)} items from {metadata.get('hours_analyzed', 0)}h window) "
                           f"in {processing_time:.1f}s")
        
        except Exception as e:
            self.logger.error(f"Market intelligence task error: {e}")
    
    async def _update_watchlist_from_parquet(self, auction_records: List[Dict[str, Any]]):
        """Update the auction watchlist based on item volume from Parquet data."""
        try:
            # Count occurrences of each item
            item_counter = Counter()
            
            for record in auction_records:
                item_name = record.get("item_name", "").strip()
                if item_name:
                    item_counter[item_name] += 1
            
            # Update watchlist with items above threshold
            old_watchlist_size = len(self.auction_watchlist)
            self.auction_watchlist = {
                item_name for item_name, count in item_counter.items()
                if count >= self.min_auction_count
            }
            
            new_items = len(self.auction_watchlist) - old_watchlist_size
            self.logger.info(f"Watchlist updated from Parquet: {len(self.auction_watchlist)} items "
                           f"({'+'+ str(new_items) if new_items > 0 else str(new_items)} from previous)")
        
        except Exception as e:
            self.logger.error(f"Failed to update watchlist from Parquet: {e}")
    
    async def _update_watchlist(self, auctions: List[Dict[str, Any]]):
        """Update the auction watchlist based on item volume."""
        try:
            # Count occurrences of each item
            item_counter = Counter()
            
            for auction in auctions:
                item_name = auction.get("item_name", "").strip()
                if item_name:
                    item_counter[item_name] += 1
            
            # Update watchlist with items above threshold
            old_watchlist_size = len(self.auction_watchlist)
            self.auction_watchlist = {
                item_name for item_name, count in item_counter.items()
                if count >= self.min_auction_count
            }
            
            new_items = len(self.auction_watchlist) - old_watchlist_size
            self.logger.info(f"Watchlist updated: {len(self.auction_watchlist)} items "
                           f"({'+'+ str(new_items) if new_items > 0 else str(new_items)} from previous)")
        
        except Exception as e:
            self.logger.error(f"Failed to update watchlist: {e}")
    
    async def _update_fmv_data(self, auctions: List[Dict[str, Any]]):
        """
        Update Fair Market Value data for watchlist items.
        Uses market depth-aware FMV calculation to avoid "price wall" problem.
        """
        try:
            # Group auctions by item (BIN only)
            item_prices = defaultdict(list)
            
            for auction in auctions:
                item_name = auction.get("item_name", "").strip()
                if item_name in self.auction_watchlist and auction.get("bin"):
                    try:
                        price = float(auction.get("starting_bid", 0))
                        if price > 0:
                            item_prices[item_name].append(price)
                    except (ValueError, TypeError):
                        continue
            
            # Calculate market depth-aware FMV
            for item_name, prices in item_prices.items():
                if len(prices) >= 2:
                    prices.sort()
                    
                    # Filter out extreme outliers (prices > 10x median for sanity)
                    median_price = prices[len(prices) // 2]
                    filtered_prices = [p for p in prices if p <= median_price * 10]
                    
                    if len(filtered_prices) >= 2:
                        # Market depth analysis: group by price levels
                        price_levels = self._analyze_market_depth(filtered_prices)
                        fmv, method = self._calculate_depth_aware_fmv(price_levels)
                        
                        self.fmv_data[item_name] = {
                            "fmv": float(fmv),
                            "median": float(filtered_prices[len(filtered_prices) // 2]),
                            "samples": len(prices),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "method": method,
                            "price_levels": len(price_levels)
                        }
                    elif len(filtered_prices) == 1:
                        # Only one price available, use it but with discount
                        fmv = filtered_prices[0] * 0.95  # 5% discount for single sample
                        self.fmv_data[item_name] = {
                            "fmv": float(fmv),
                            "median": float(filtered_prices[0]),
                            "samples": 1,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "method": "single_bin_discounted",
                            "price_levels": 1
                        }
            
            self.logger.debug(f"Updated FMV data using market depth-aware calculation for {len(item_prices)} items")
        
        except Exception as e:
            self.logger.error(f"Failed to update FMV data: {e}")
    
    def _analyze_market_depth(self, sorted_prices: List[float]) -> List[Tuple[float, int]]:
        """
        Analyze market depth by grouping prices into levels and counting quantity at each level.
        Returns list of (price, count) tuples sorted by price.
        """
        from collections import Counter
        
        # Group prices with small tolerance for floating-point precision
        # Round to nearest 100 coins to group similar prices
        rounded_prices = [round(p / 100) * 100 for p in sorted_prices]
        price_counts = Counter(rounded_prices)
        
        # Sort by price (ascending)
        price_levels = sorted(price_counts.items())
        return price_levels
    
    def _calculate_depth_aware_fmv(self, price_levels: List[Tuple[float, int]]) -> Tuple[float, str]:
        """
        Calculate FMV based on market depth analysis.
        
        Logic:
        - If the lowest price level has >2 items (thick wall), FMV = lowest price (no profit opportunity)
        - If the lowest price level has ≤2 items (thin wall), FMV = next price level
        """
        if not price_levels:
            return 0.0, "no_data"
        
        if len(price_levels) == 1:
            # Only one price level, use it with discount
            return price_levels[0][0] * 0.95, "single_level_discounted"
        
        lowest_price, lowest_count = price_levels[0]
        
        # Configurable threshold for "thick wall" (default 3 items)
        sniper_config = self.config.get("auction_sniper", {})
        thick_wall_threshold = sniper_config.get("thick_wall_threshold", 3)
        
        if lowest_count >= thick_wall_threshold:
            # Thick wall: Use lowest price as FMV (represents reality that you can't profit)
            return lowest_price, f"thick_wall_floor_{lowest_count}_items"
        else:
            # Thin wall: Use next price level as FMV (true opportunity)
            next_price, next_count = price_levels[1]
            return next_price, f"thin_wall_next_level_{lowest_count}_at_floor"
    
    def update_market_values_from_parquet_windowed(self, df: pd.DataFrame):
        """
        Memory-efficient update of market values cache from windowed Parquet data.
        Uses market depth-aware FMV calculation to avoid "price wall" problem.
        Works with pre-filtered DataFrame to avoid memory issues.
        """
        try:
            if df.empty:
                self.logger.debug("Empty windowed dataset for FMV calculation")
                return
            
            # Filter for BIN auctions only (for accurate market pricing)
            bin_auctions = df[df['bin'] == True].copy()
            
            if bin_auctions.empty:
                self.logger.debug("No BIN auctions found in windowed data")
                return
            
            # Calculate market depth-aware FMV for watchlist items
            watchlist_items = bin_auctions[bin_auctions['item_name'].isin(self.auction_watchlist)]
            
            if watchlist_items.empty:
                self.logger.debug("No watchlist items found in windowed data")
                return
            
            # Group by item_name and calculate FMV using market depth analysis
            fmv_data = {}
            for item_name in watchlist_items['item_name'].unique():
                item_data = watchlist_items[watchlist_items['item_name'] == item_name]
                
                # Filter out extreme outliers (prices > 10x median for sanity)
                median_price = item_data['price'].median()
                if median_price > 0:
                    # Remove prices that are clearly outliers (more than 10x median)
                    filtered_data = item_data[item_data['price'] <= median_price * 10]
                    if len(filtered_data) >= 2:
                        item_data = filtered_data
                
                # Sort prices for market depth analysis
                sorted_prices = item_data['price'].sort_values().tolist()
                
                if len(sorted_prices) >= 2:
                    # Market depth analysis: group by price levels
                    price_levels = self._analyze_market_depth(sorted_prices)
                    fmv, method = self._calculate_depth_aware_fmv(price_levels)
                    
                    fmv_data[item_name] = {
                        "fmv": float(fmv),
                        "median": float(item_data['price'].median()),
                        "samples": len(sorted_prices),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "method": method,
                        "price_levels": len(price_levels)
                    }
                elif len(sorted_prices) == 1:
                    # Only one price available, use it but with low confidence
                    fmv = sorted_prices[0] * 0.95  # Apply 5% discount for single sample
                    fmv_data[item_name] = {
                        "fmv": float(fmv),
                        "median": float(sorted_prices[0]),
                        "samples": 1,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "method": "single_bin_discounted",
                        "price_levels": 1
                    }
            
            # Update the cache with windowed data
            for item_name, data in fmv_data.items():
                self.fmv_data[item_name] = data
            
            self.logger.debug(f"Updated FMV data using windowed analysis for {len(fmv_data)} items")
            
        except Exception as e:
            self.logger.error(f"Failed to update market values from windowed data: {e}")

    def update_market_values_from_parquet(self):
        """
        Legacy method: Update market values cache from Parquet dataset.
        This method is kept for backward compatibility but should use windowed loading.
        Uses market depth-aware FMV calculation to avoid "price wall" problem.
        """
        try:
            # Check if Parquet data exists
            if not self.PARQUET_DATA_PATH.exists() or not any(self.PARQUET_DATA_PATH.iterdir()):
                self.logger.debug("No Parquet data found for market analysis")
                return
            
            # Use memory-efficient windowed loading instead of full dataset
            two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
            import pyarrow.dataset as ds
            dataset = ds.dataset(self.PARQUET_DATA_PATH, format="parquet")
            
            # Create filter for recent data only
            filter_expr = ds.field('scan_timestamp') > two_hours_ago
            table = dataset.to_table(filter=filter_expr)
            df = table.to_pandas()
            
            if df.empty:
                self.logger.debug("Empty windowed Parquet dataset")
                return
            
            # Use the windowed method for actual processing
            self.update_market_values_from_parquet_windowed(df)
            
            self.logger.debug("Market values updated using legacy method with windowed optimization")
            
        except Exception as e:
            self.logger.error(f"Failed to update market values from Parquet: {e}")
        """
        Update market values cache from Parquet dataset.
        Uses market depth-aware FMV calculation to avoid "price wall" problem.
        """
        try:
            # Check if Parquet data exists
            if not self.PARQUET_DATA_PATH.exists() or not any(self.PARQUET_DATA_PATH.iterdir()):
                self.logger.debug("No Parquet data found for market analysis")
                return
            
            # Load recent Parquet data (last few partitions for current market conditions)
            dataset = pq.ParquetDataset(self.PARQUET_DATA_PATH)
            
            # Read all data using modern PyArrow API
            df = dataset.read().to_pandas()
            
            if df.empty:
                self.logger.debug("Empty Parquet dataset")
                return
            
            # Filter for BIN auctions only (for accurate market pricing)
            bin_auctions = df[df['bin'] == True].copy()
            
            if bin_auctions.empty:
                self.logger.debug("No BIN auctions found in Parquet data")
                return
            
            # Calculate market depth-aware FMV for watchlist items
            watchlist_items = bin_auctions[bin_auctions['item_name'].isin(self.auction_watchlist)]
            
            if watchlist_items.empty:
                self.logger.debug("No watchlist items found in Parquet data")
                return
            
            # Group by item_name and calculate FMV using market depth analysis
            fmv_data = {}
            for item_name in watchlist_items['item_name'].unique():
                item_data = watchlist_items[watchlist_items['item_name'] == item_name]
                
                # Filter out extreme outliers (prices > 10x median for sanity)
                median_price = item_data['price'].median()
                if median_price > 0:
                    # Remove prices that are clearly outliers (more than 10x median)
                    filtered_data = item_data[item_data['price'] <= median_price * 10]
                    if len(filtered_data) >= 2:
                        item_data = filtered_data
                
                # Sort prices for market depth analysis
                sorted_prices = item_data['price'].sort_values().tolist()
                
                if len(sorted_prices) >= 2:
                    # Market depth analysis: group by price levels
                    price_levels = self._analyze_market_depth(sorted_prices)
                    fmv, method = self._calculate_depth_aware_fmv(price_levels)
                    
                    fmv_data[item_name] = {
                        "fmv": float(fmv),
                        "median": float(item_data['price'].median()),
                        "samples": len(sorted_prices),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "method": method,
                        "price_levels": len(price_levels)
                    }
                elif len(sorted_prices) == 1:
                    # Only one price available, use it but with low confidence
                    fmv = sorted_prices[0] * 0.95  # Apply 5% discount for single sample
                    fmv_data[item_name] = {
                        "fmv": float(fmv),
                        "median": float(sorted_prices[0]),
                        "samples": 1,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "method": "single_bin_discounted",
                        "price_levels": 1
                    }
            
            # Update the cache
            for item_name, data in fmv_data.items():
                self.fmv_data[item_name] = data
            
            self.logger.debug(f"Updated FMV data using market depth-aware calculation for {len(fmv_data)} items")
            
        except Exception as e:
            self.logger.error(f"Failed to update market values from Parquet: {e}")
    
    async def _verify_snipe(self, auction: Dict[str, Any]) -> bool:
        """
        Verify if an auction is a valid snipe opportunity.
        Includes manipulation, attribute, profitability, and liquidity checks.
        """
        try:
            item_name = auction.get("item_name", "").strip()
            price = float(auction.get("starting_bid", 0))
            uuid = auction.get("uuid", "")[:8]  # Short UUID for logging
            
            # Enhanced logging for debugging
            self.logger.debug(f"🔍 Evaluating auction: {item_name} @ {price:,.0f} coins (UUID: {uuid}...)")
            
            if price <= 0:
                self.logger.debug(f"❌ {item_name}: Invalid price ({price})")
                return False
            
            # Safety check: Skip extremely expensive items (> 1B coins) to avoid false positives
            if price > 1_000_000_000:
                self.logger.debug(f"❌ {item_name}: Too expensive ({price:,.0f} > 1B)")
                return False
            
            # Manipulation Check: Compare against FMV
            fmv_info = self.fmv_data.get(item_name)
            if not fmv_info:
                self.logger.debug(f"❌ {item_name}: No FMV data available")
                return False  # No FMV data available
            
            fmv = fmv_info.get("fmv", 0)
            samples = fmv_info.get("samples", 0)
            fmv_method = fmv_info.get("method", "unknown")
            
            self.logger.debug(f"📊 {item_name}: FMV={fmv:,.0f}, samples={samples}, method={fmv_method}")
            
            # Enhanced Liquidity Check: Ensure sufficient market activity
            min_samples_config = self.config.get("auction_sniper", {}).get("min_liquidity_samples", 10)
            
            # Require more samples for high-value items (reduce risk)
            if price > 10_000_000:  # Over 10M coins
                min_samples_required = max(min_samples_config, 15)
            elif price > 1_000_000:  # Over 1M coins  
                min_samples_required = max(min_samples_config, 8)
            else:
                min_samples_required = max(min_samples_config, 5)
            
            if samples < min_samples_required:
                self.logger.debug(f"Skipping {item_name}: insufficient liquidity ({samples} < {min_samples_required} samples)")
                return False
            
            # Special handling for single-sample discounted items (very low liquidity)
            if fmv_method == "single_bin_discounted":
                self.logger.debug(f"Skipping {item_name}: too low liquidity (single sample)")
                return False
            
            # Use configurable multiplier from config
            sniper_config = self.config.get("auction_sniper", {})
            max_multiplier = sniper_config.get("max_fmv_multiplier", 1.1)
            
            if price > fmv * max_multiplier:
                self.logger.debug(f"❌ {item_name}: Price too high ({price:,.0f} > {fmv * max_multiplier:,.0f}, {max_multiplier}x FMV)")
                return False
            
            # Attribute Check: Parse NBT for critical attributes
            if not self._check_item_attributes(auction):
                self.logger.debug(f"❌ {item_name}: Failed attribute check")
                return False
            
            # Profitability Check
            auction_house_fee = price * 0.01  # 1% AH fee
            estimated_profit = fmv - price - auction_house_fee
            
            if estimated_profit < self.profit_threshold:
                self.logger.debug(f"❌ {item_name}: Insufficient profit ({estimated_profit:,.0f} < {self.profit_threshold:,.0f})")
                return False
            
            # Additional safety: Ensure profit margin is reasonable (at least 5%)
            profit_margin = estimated_profit / price
            if profit_margin < 0.05:
                self.logger.debug(f"❌ {item_name}: Low profit margin ({profit_margin:.1%} < 5%)")
                return False
            
            # Final liquidity check: Ensure profit justifies risk for low-liquidity items
            if samples < 20 and estimated_profit < self.profit_threshold * 2:
                self.logger.debug(f"❌ {item_name}: Low liquidity requires higher profit ({estimated_profit:,.0f} < {self.profit_threshold * 2:,.0f})")
                return False
            
            self.logger.info(f"✅ Valid snipe found: {item_name} at {price:,.0f} coins "
                           f"(FMV: {fmv:,.0f}, Profit: {estimated_profit:,.0f}, Margin: {profit_margin:.1%}, "
                           f"Samples: {samples}, Method: {fmv_method})")
            return True
        
        except Exception as e:
            self.logger.error(f"Error verifying snipe for {auction.get('item_name', 'unknown')}: {e}")
            return False
    
    def _check_item_attributes(self, auction: Dict[str, Any]) -> bool:
        """
        Check item attributes for weapons/armor.
        Parse NBT data to verify critical attributes.
        """
        try:
            item_name = auction.get("item_name", "")
            item_lore = auction.get("item_lore", "")
            
            # Convert to uppercase for consistent matching
            item_name_upper = item_name.upper()
            
            # Weapon attribute checks
            weapon_keywords = ["SWORD", "BOW", "HYPERION", "VALKYRIE", "SCYLLA", "NECRON_BLADE", 
                             "GIANTS_SWORD", "LIVID_DAGGER", "SHADOW_FURY", "ASPECT_OF_THE"]
            if any(weapon in item_name_upper for weapon in weapon_keywords):
                # Check for ultimate enchantments
                ultimate_enchants = ["Ultimate Enchantment", "Ultimate Wise", "Ultimate Fatal Tempo", 
                                   "Ultimate Combo", "Ultimate Rend", "Ultimate Soul Eater"]
                if any(ult in item_lore for ult in ultimate_enchants):
                    return True
                
                # Check for good reforges
                good_reforges = ["Sharp", "Heroic", "Legendary", "Fabled", "Withered", "Dirty", 
                               "Fast", "Gentle", "Odd", "Smart", "Silky"]
                if any(reforge in item_lore for reforge in good_reforges):
                    return True
                
                # For high-value weapons, require good attributes
                expensive_weapons = ["HYPERION", "VALKYRIE", "SCYLLA", "NECRON_BLADE", "GIANTS_SWORD"]
                if any(weapon in item_name_upper for weapon in expensive_weapons):
                    return False  # Expensive weapon without good attributes
                
                # For other weapons, be less strict
                return True
            
            # Armor attribute checks
            armor_keywords = ["CHESTPLATE", "LEGGINGS", "HELMET", "BOOTS", "_HELMET", "_CHESTPLATE", 
                            "_LEGGINGS", "_BOOTS", "NECRON_", "STORM_", "GOLDOR_", "MAXOR_"]
            if any(armor in item_name_upper for armor in armor_keywords):
                # Check for stars (dungeon upgrades)
                if "⭐" in item_lore or "✪" in item_lore:
                    return True
                
                # Check for good reforges
                good_armor_reforges = ["Ancient", "Renowned", "Spiked", "Reinforced", "Loving", 
                                     "Ridiculous", "Giant", "Smart", "Wise", "Perfect"]
                if any(reforge in item_lore for reforge in good_armor_reforges):
                    return True
                
                # Check for high-level enchantments
                high_enchants = ["Protection VII", "Growth VII", "Sugar Rush III", "True Protection"]
                if any(enchant in item_lore for enchant in high_enchants):
                    return True
                
                # For expensive armor sets, be more strict
                expensive_armor = ["NECRON_", "STORM_", "GOLDOR_", "MAXOR_", "SUPERIOR_DRAGON"]
                if any(armor in item_name_upper for armor in expensive_armor):
                    return False  # Expensive armor without good attributes
            
            # Special item checks
            special_items = ["PET", "RUNE", "UPGRADE_STONE", "REFORGE_STONE"]
            if any(special in item_name_upper for special in special_items):
                # For pets, check level and tier
                if "PET" in item_name_upper:
                    legendary_pet_indicators = ["[Lvl", "LEGENDARY"]
                    return any(indicator in item_lore for indicator in legendary_pet_indicators)
                
                # For runes and stones, generally accept
                return True
            
            # For other items (materials, etc.), accept by default
            return True
        
        except Exception as e:
            self.logger.error(f"Error checking item attributes: {e}")
            return True  # Default to accepting if check fails
    
    async def _alert_snipe(self, auction: Dict[str, Any]):
        """Send snipe alert to configured Discord channel with seller-aware and UUID-aware cooldown."""
        if not self.alert_channel_id:
            return
        
        try:
            item_name = auction.get("item_name", "Unknown Item")
            auctioneer = auction.get("auctioneer", "")
            uuid = auction.get("uuid", "")
            price = float(auction.get("starting_bid", 0))
            current_time = datetime.now(timezone.utc)
            
            # Check auction UUID deduplication (avoid alerting same auction twice)
            if uuid and uuid in self.recent_auction_uuids:
                self.logger.debug(f"Suppressing duplicate alert: same auction UUID {uuid[:8]}...")
                return
            
            # Create seller-aware composite key for cooldown tracking
            alert_key = f"{item_name}_{auctioneer}"
            
            # Check seller cooldown - prevent duplicate alerts from same seller for same item
            price_improved = False
            if alert_key in self.recent_alerts:
                last_alert_time = self.recent_alerts[alert_key]
                time_since_last = (current_time - last_alert_time).total_seconds() / 60  # minutes
                
                if time_since_last < self.alert_cooldown_minutes:
                    # Check if price has improved significantly (allow re-alert for better deals)
                    last_price = getattr(self.recent_alerts.get(alert_key), 'price', None)
                    if last_price and price < last_price * 0.9:  # 10% price improvement
                        price_improved = True
                        self.logger.info(f"Price improved alert: {item_name} from {auctioneer[:8]}... "
                                       f"price dropped {(1 - price/last_price)*100:.1f}%")
                    else:
                        self.logger.debug(f"Suppressing duplicate alert: {item_name} from seller {auctioneer[:8]}... "
                                        f"(cooldown: {time_since_last:.1f}/{self.alert_cooldown_minutes} min)")
                        return
            
            # Clean up expired cooldowns and UUIDs
            cutoff_time = current_time - timedelta(minutes=self.alert_cooldown_minutes * 2)
            expired_keys = [key for key, timestamp in self.recent_alerts.items() if timestamp < cutoff_time]
            for key in expired_keys:
                del self.recent_alerts[key]
            
            # Clean up old UUIDs (TTL-based)
            uuid_cutoff = current_time - timedelta(minutes=self.uuid_cooldown_minutes)
            self.recent_auction_uuids = {
                u for u in self.recent_auction_uuids 
                if (current_time - getattr(u, 'timestamp', current_time)).total_seconds() < self.uuid_cooldown_minutes * 60
            }
            
            # Send the alert
            channel = self.bot.get_channel(self.alert_channel_id)
            if not channel:
                return
            
            fmv_info = self.fmv_data.get(item_name, {})
            fmv = fmv_info.get("fmv", 0)
            floor_price = fmv_info.get("floor_price", 0)
            method = fmv_info.get("method", "unknown")
            estimated_profit = fmv - price - (price * 0.01)  # Account for 1% auction house fee
            
            embed = discord.Embed(
                title="🎯 Auction Snipe Detected!" + (" 📈 Price Improved!" if price_improved else ""),
                description=f"**{item_name}**",
                color=0x00ff00 if not price_improved else 0xffd700,
                timestamp=current_time
            )
            
            embed.add_field(name="💰 Price", value=f"{price:,.0f} coins", inline=True)
            embed.add_field(name="📊 FMV", value=f"{fmv:,.0f} coins", inline=True)
            embed.add_field(name="💸 Est. Profit", value=f"{estimated_profit:,.0f} coins", inline=True)
            
            embed.add_field(name="🔗 View Auction", value=f"`/viewauction {uuid}`", inline=False)
            
            # Add market depth information if available
            if fmv_info:
                floor_count = fmv_info.get("floor_count", 0)
                depth_info = f"Floor: {floor_count} items @ {floor_price:,.0f} | Method: {method}"
                embed.add_field(name="📊 Market Depth", value=depth_info, inline=False)
            
            embed.set_footer(text="Hypixel Auction Sniper", 
                           icon_url="https://hypixel.net/favicon.ico")
            
            await channel.send(embed=embed)
            
            # Record the alert in cooldown cache with price for improvement tracking
            alert_data = type('AlertData', (), {})()
            alert_data.timestamp = current_time
            alert_data.price = price
            self.recent_alerts[alert_key] = alert_data
            
            # Add UUID to recent set
            if uuid:
                self.recent_auction_uuids.add(uuid)
            
            self.logger.info(f"Sent snipe alert for {item_name} from seller {auctioneer[:8]}... to channel {channel.name} "
                           f"(price: {price:,.0f}, profit: {estimated_profit:,.0f})")
        
        except Exception as e:
            self.logger.error(f"Failed to send snipe alert: {e}")
    
    def _generate_test_auctions(self) -> List[Dict[str, Any]]:
        """Generate mock auction data for testing sniper logic without API key."""
        import uuid
        import random
        
        test_auctions = []
        
        # Create test auctions for items in our watchlist
        for item_name in list(self.auction_watchlist)[:5]:  # Test with first 5 items
            fmv_info = self.fmv_data.get(item_name, {})
            fmv = fmv_info.get("fmv", 50000)
            
            # Create auctions at different price points
            test_prices = [
                int(fmv * 0.7),   # Good snipe (30% below FMV)
                int(fmv * 0.95),  # Marginal snipe (5% below FMV)
                int(fmv * 1.2),   # Too expensive (20% above FMV)
            ]
            
            for i, price in enumerate(test_prices):
                auction = {
                    "uuid": str(uuid.uuid4()),
                    "auctioneer": f"test_seller_{random.randint(1000, 9999)}",
                    "profile_id": str(uuid.uuid4()),
                    "item_name": item_name,
                    "item_lore": "",
                    "extra": "",
                    "category": "misc",
                    "tier": "COMMON",
                    "starting_bid": price,
                    "item_bytes": "",
                    "claimed": False,
                    "claimed_bidders": [],
                    "highest_bid_amount": price,
                    "bids": [],
                    "bin": True,  # Buy It Now
                    "start": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "end": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)
                }
                test_auctions.append(auction)
        
        return test_auctions
    
    # Discord Commands
    
    @app_commands.command(name="sniper_channel", description="Set the channel for auction snipe alerts")
    @app_commands.describe(channel="Channel to send snipe alerts to")
    async def set_sniper_channel(self, interaction: discord.Interaction, channel: discord.TextChannel):
        """Set the alert channel for auction snipes."""
        # Immediately defer the response to avoid timeout
        await interaction.response.defer()
        
        if not interaction.user.guild_permissions.administrator:
            await interaction.followup.send("❌ You need Administrator permissions to configure the sniper.", ephemeral=True)
            return
        
        self.alert_channel_id = channel.id
        
        # Save to config (async file I/O)
        await self._save_sniper_config()
        
        await interaction.followup.send(f"✅ Sniper alerts will be sent to {channel.mention}")
        self.logger.info(f"Sniper alert channel set to #{channel.name} ({channel.id})")
    
    @app_commands.command(name="sniper_config", description="Configure sniper settings")
    @app_commands.describe(
        profit_threshold="Minimum profit threshold in coins",
        min_auction_count="Minimum auction count for watchlist items"
    )
    async def configure_sniper(self, interaction: discord.Interaction, 
                              profit_threshold: Optional[int] = None, 
                              min_auction_count: Optional[int] = None):
        """Configure sniper parameters."""
        # Immediately defer the response to avoid timeout
        await interaction.response.defer()
        
        if not interaction.user.guild_permissions.administrator:
            await interaction.followup.send("❌ You need Administrator permissions to configure the sniper.", ephemeral=True)
            return
        
        if profit_threshold is not None:
            self.profit_threshold = max(1000, profit_threshold)  # Minimum 1k
        
        if min_auction_count is not None:
            self.min_auction_count = max(10, min_auction_count)  # Minimum 10
        
        # Save to config
        await self._save_sniper_config()
        
        embed = discord.Embed(title="🔧 Sniper Configuration", color=0x0099ff)
        embed.add_field(name="💰 Profit Threshold", value=f"{self.profit_threshold:,} coins", inline=True)
        embed.add_field(name="📊 Min Auction Count", value=str(self.min_auction_count), inline=True)
        embed.add_field(name="📝 Watchlist Size", value=str(len(self.auction_watchlist)), inline=True)
        
        await interaction.followup.send(embed=embed)
        self.logger.info(f"Sniper configuration updated: profit_threshold={self.profit_threshold}, min_auction_count={self.min_auction_count}")
    
    @app_commands.command(name="sniper_status", description="Check sniper status and performance metrics")
    async def sniper_status(self, interaction: discord.Interaction):
        """Show current sniper status with performance metrics."""
        try:
            await interaction.response.defer()
            
            embed = discord.Embed(title="🎯 Auction Sniper Status", color=0x0099ff, 
                                timestamp=datetime.now(timezone.utc))
            
            # Task status
            hunter_status = "🟢 Running" if self.high_frequency_snipe_scan.is_running() else "🔴 Stopped"
            intelligence_status = "🟢 Running" if self.update_market_intelligence.is_running() else "🔴 Stopped"
            
            embed.add_field(name="🏃 Hunter Task (2s)", value=hunter_status, inline=True)
            embed.add_field(name="🧠 Intelligence Task (90s)", value=intelligence_status, inline=True)
            embed.add_field(name="📢 Alert Channel", 
                           value=f"<#{self.alert_channel_id}>" if self.alert_channel_id else "Not set", 
                           inline=True)
            
            # Configuration
            embed.add_field(name="💰 Profit Threshold", value=f"{self.profit_threshold:,} coins", inline=True)
            embed.add_field(name="📊 Min Auction Count", value=str(self.min_auction_count), inline=True)
            embed.add_field(name="📝 Watchlist Size", value=str(len(self.auction_watchlist)), inline=True)
            
            # FMV data status
            embed.add_field(name="💾 FMV Cache", value=f"{len(self.fmv_data)} items", inline=True)
            
            # Memory usage
            try:
                process = psutil.Process()
                rss_mb = process.memory_info().rss / (1024 * 1024)
                memory_status = f"{rss_mb:.1f}MB"
                if rss_mb > self.soft_rss_mb:
                    memory_status += " ⚠️"
                embed.add_field(name="💾 Memory Usage", value=memory_status, inline=True)
            except Exception:
                embed.add_field(name="💾 Memory Usage", value="Unknown", inline=True)
            
            # API status
            api_status = "🟢 Connected" if self.hypixel_client else "🔴 No API Key"
            embed.add_field(name="🔗 Hypixel API", value=api_status, inline=True)
            
            # Performance metrics
            embed.add_field(name="⏱️ Window Hours", value=str(self.window_hours), inline=True)
            embed.add_field(name="🔒 Recent Alerts", value=str(len(self.recent_alerts)), inline=True)
            embed.add_field(name="🆔 Recent UUIDs", value=str(len(self.recent_auction_uuids)), inline=True)
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Sniper status command error: {e}")
            await interaction.followup.send("❌ Error retrieving sniper status.", ephemeral=True)
    
    @app_commands.command(name="sniper_test", description="Test sniper logic with mock auctions")
    async def sniper_test(self, interaction: discord.Interaction):
        """Test sniper logic with mock auction data."""
        try:
            await interaction.response.defer()
            
            if not self.auction_watchlist:
                await interaction.followup.send("❌ No watchlist items available. Run market intelligence update first.", ephemeral=True)
                return
            
            if not self.fmv_data:
                await interaction.followup.send("❌ No FMV data available. Run market intelligence update first.", ephemeral=True)
                return
            
            # Force enable test mode
            self._test_mode_enabled = True
            
            # Generate test auctions
            test_auctions = self._generate_test_auctions()
            
            embed = discord.Embed(title="🧪 Sniper Test Results", color=0x00ff00, 
                                timestamp=datetime.now(timezone.utc))
            
            snipes_found = 0
            total_evaluated = 0
            results = []
            
            # Test each auction
            for auction in test_auctions:
                total_evaluated += 1
                item_name = auction.get("item_name", "")
                price = auction.get("starting_bid", 0)
                
                if await self._verify_snipe(auction):
                    snipes_found += 1
                    fmv_info = self.fmv_data.get(item_name, {})
                    fmv = fmv_info.get("fmv", 0)
                    profit = fmv - price - (price * 0.01)
                    results.append(f"✅ **{item_name}**: {price:,.0f} → {fmv:,.0f} (Profit: {profit:,.0f})")
                else:
                    results.append(f"❌ **{item_name}**: {price:,.0f} (Failed verification)")
            
            embed.add_field(name="📊 Summary", 
                           value=f"Evaluated: {total_evaluated}\nValid Snipes: {snipes_found}", 
                           inline=False)
            
            if results:
                # Show first 10 results
                results_text = "\n".join(results[:10])
                if len(results) > 10:
                    results_text += f"\n... and {len(results) - 10} more"
                embed.add_field(name="🔍 Test Results", value=results_text, inline=False)
            
            embed.add_field(name="ℹ️ Note", 
                           value="This test uses mock auction data. Check logs for detailed evaluation info.", 
                           inline=False)
            
            await interaction.followup.send(embed=embed)
            self.logger.info(f"Sniper test completed: {snipes_found}/{total_evaluated} auctions passed verification")
        
        except Exception as e:
            self.logger.error(f"Sniper test command error: {e}")
            await interaction.followup.send("❌ Error running sniper test.", ephemeral=True)
    
    async def _save_sniper_config(self):
        """Save sniper configuration to disk asynchronously."""
        try:
            config_file = self.data_dir / "sniper_config.json"
            config_data = {
                "alert_channel_id": self.alert_channel_id,
                "profit_threshold": self.profit_threshold,
                "min_auction_count": self.min_auction_count,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            async with aiofiles.open(config_file, "w") as f:
                await f.write(json.dumps(config_data, indent=2, default=convert_np))
            
            self.logger.debug("Saved sniper configuration")
        
        except Exception as e:
            self.logger.error(f"Failed to save sniper config: {e}")


async def setup(bot):
    """Required setup function for discord.py cog."""
    await bot.add_cog(AuctionSniper(bot))