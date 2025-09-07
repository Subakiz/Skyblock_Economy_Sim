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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, List, Optional, Any, Tuple
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import discord
from discord import app_commands
from discord.ext import commands, tasks
import aiofiles

# Import project modules
from ingestion.common.hypixel_client import HypixelClient
from ingestion.item_processing import create_canonical_name


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
        self.alert_channel_id: Optional[int] = None
        self.profit_threshold: float = sniper_config.get("profit_threshold", 100000)
        self.min_auction_count: int = sniper_config.get("min_auction_count", 50)
        
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
        
        self.logger.info("Auction Sniper initialized")
    
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
                await f.write(json.dumps(watchlist_data, indent=2))
            
            # Save FMV data
            fmv_file = self.data_dir / "fmv_cache.json"
            async with aiofiles.open(fmv_file, "w") as f:
                await f.write(json.dumps(self.fmv_data, indent=2))
            
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
        if not self.hypixel_client or not self.auction_watchlist:
            return
        
        try:
            start_time = time.time()
            
            # Fetch only page 0 for speed
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
    
    @tasks.loop(seconds=60)  # Updated to 60 seconds as requested
    async def update_market_intelligence(self):
        """
        Market Intelligence Task: Read-only market analysis from Parquet data.
        Updates watchlist and FMV data every 60 seconds by reading from pre-processed data.
        No longer fetches data directly - relies on standalone ingestion service.
        """
        try:
            start_time = time.time()
            self.logger.info("Starting market intelligence update from Parquet data...")
            
            # Check if Parquet data exists
            if not self.PARQUET_DATA_PATH.exists() or not any(self.PARQUET_DATA_PATH.iterdir()):
                self.logger.debug("No Parquet auction data found - waiting for ingestion service")
                return
            
            # Load recent auction data from Parquet dataset
            try:
                dataset = pq.ParquetDataset(self.PARQUET_DATA_PATH)
                df = dataset.read().to_pandas()
                
                if df.empty:
                    self.logger.debug("Empty Parquet dataset - waiting for data")
                    return
                
                self.logger.info(f"Loaded {len(df)} auction records from Parquet dataset")
                
                # Convert to list of dictionaries to maintain compatibility with existing methods
                auction_records = df.to_dict('records')
                
                # Update watchlist based on item volume in Parquet data
                await self._update_watchlist_from_parquet(auction_records)
                
                # Update FMV data from Parquet 
                self.update_market_values_from_parquet()
                
                # Persist updated data
                await self._save_persisted_data()
                
                processing_time = time.time() - start_time
                self.logger.info(f"Market intelligence updated: {len(self.auction_watchlist)} watchlist items, "
                               f"FMV data for {len(self.fmv_data)} items in {processing_time:.1f}s")
                
            except Exception as e:
                self.logger.error(f"Error reading Parquet data: {e}")
        
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
        Uses second-lowest BIN price methodology for accurate FMV calculation.
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
            
            # Calculate FMV statistics using second-lowest BIN price
            for item_name, prices in item_prices.items():
                if len(prices) >= 2:
                    prices.sort()
                    
                    # Filter out extreme outliers (prices > 10x median for sanity)
                    median_price = prices[len(prices) // 2]
                    filtered_prices = [p for p in prices if p <= median_price * 10]
                    
                    if len(filtered_prices) >= 2:
                        # Use second-lowest BIN price as FMV
                        fmv = filtered_prices[1]  # Second lowest
                        median = filtered_prices[len(filtered_prices) // 2]
                        
                        self.fmv_data[item_name] = {
                            "fmv": fmv,
                            "median": median,
                            "samples": len(prices),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "method": "second_lowest_bin"
                        }
                    elif len(filtered_prices) == 1:
                        # Only one price available, use it but with discount
                        fmv = filtered_prices[0] * 0.95  # 5% discount for single sample
                        self.fmv_data[item_name] = {
                            "fmv": fmv,
                            "median": filtered_prices[0],
                            "samples": 1,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "method": "single_bin_discounted"
                        }
            
            self.logger.debug(f"Updated FMV data using second-lowest BIN price for {len(item_prices)} items")
        
        except Exception as e:
            self.logger.error(f"Failed to update FMV data: {e}")
    
    def update_market_values_from_parquet(self):
        """
        Update market values cache from Parquet dataset.
        Uses second-lowest BIN price methodology for accurate FMV calculation.
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
            
            # Calculate second-lowest BIN price for watchlist items
            watchlist_items = bin_auctions[bin_auctions['item_name'].isin(self.auction_watchlist)]
            
            if watchlist_items.empty:
                self.logger.debug("No watchlist items found in Parquet data")
                return
            
            # Group by item_name and calculate FMV using second-lowest BIN price
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
                
                # Sort prices and get second-lowest
                sorted_prices = item_data['price'].sort_values().tolist()
                
                if len(sorted_prices) >= 2:
                    # Use second-lowest BIN price as FMV (represents real resale market)
                    fmv = sorted_prices[1]  # Second lowest
                    median = item_data['price'].median()
                    
                    fmv_data[item_name] = {
                        "fmv": float(fmv),
                        "median": float(median),
                        "samples": len(sorted_prices),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "method": "second_lowest_bin"
                    }
                elif len(sorted_prices) == 1:
                    # Only one price available, use it but with low confidence
                    fmv = sorted_prices[0]
                    fmv_data[item_name] = {
                        "fmv": float(fmv) * 0.95,  # Apply 5% discount for single sample
                        "median": float(fmv),
                        "samples": 1,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "method": "single_bin_discounted"
                    }
            
            # Update the cache
            for item_name, data in fmv_data.items():
                self.fmv_data[item_name] = data
            
            self.logger.debug(f"Updated FMV data using second-lowest BIN price for {len(fmv_data)} items")
            
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
            
            if price <= 0:
                return False
            
            # Safety check: Skip extremely expensive items (> 1B coins) to avoid false positives
            if price > 1_000_000_000:
                return False
            
            # Manipulation Check: Compare against FMV
            fmv_info = self.fmv_data.get(item_name)
            if not fmv_info:
                return False  # No FMV data available
            
            fmv = fmv_info.get("fmv", 0)
            samples = fmv_info.get("samples", 0)
            fmv_method = fmv_info.get("method", "unknown")
            
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
                return False
            
            # Attribute Check: Parse NBT for critical attributes
            if not self._check_item_attributes(auction):
                return False
            
            # Profitability Check
            auction_house_fee = price * 0.01  # 1% AH fee
            estimated_profit = fmv - price - auction_house_fee
            
            if estimated_profit < self.profit_threshold:
                return False
            
            # Additional safety: Ensure profit margin is reasonable (at least 5%)
            profit_margin = estimated_profit / price
            if profit_margin < 0.05:
                return False
            
            # Final liquidity check: Ensure profit justifies risk for low-liquidity items
            if samples < 20 and estimated_profit < self.profit_threshold * 2:
                self.logger.debug(f"Skipping {item_name}: low liquidity requires higher profit ({estimated_profit:,.0f} < {self.profit_threshold * 2:,.0f})")
                return False
            
            self.logger.info(f"Valid snipe found: {item_name} at {price:,.0f} coins "
                           f"(FMV: {fmv:,.0f}, Profit: {estimated_profit:,.0f}, Margin: {profit_margin:.1%}, "
                           f"Samples: {samples}, Method: {fmv_method})")
            return True
        
        except Exception as e:
            self.logger.error(f"Error verifying snipe: {e}")
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
        """Send snipe alert to configured Discord channel."""
        if not self.alert_channel_id:
            return
        
        try:
            channel = self.bot.get_channel(self.alert_channel_id)
            if not channel:
                return
            
            item_name = auction.get("item_name", "Unknown Item")
            price = float(auction.get("starting_bid", 0))
            uuid = auction.get("uuid", "")
            fmv_info = self.fmv_data.get(item_name, {})
            fmv = fmv_info.get("fmv", 0)
            estimated_profit = fmv - price - (price * 0.01)
            
            embed = discord.Embed(
                title="🎯 Auction Snipe Detected!",
                description=f"**{item_name}**",
                color=0x00ff00,
                timestamp=datetime.now(timezone.utc)
            )
            
            embed.add_field(name="💰 Price", value=f"{price:,.0f} coins", inline=True)
            embed.add_field(name="📊 FMV", value=f"{fmv:,.0f} coins", inline=True)
            embed.add_field(name="💸 Est. Profit", value=f"{estimated_profit:,.0f} coins", inline=True)
            
            embed.add_field(name="🔗 View Auction", value=f"`/viewauction {uuid}`", inline=False)
            
            embed.set_footer(text="Hypixel Auction Sniper", 
                           icon_url="https://hypixel.net/favicon.ico")
            
            await channel.send(embed=embed)
            
            self.logger.info(f"Sent snipe alert for {item_name} to channel {channel.name}")
        
        except Exception as e:
            self.logger.error(f"Failed to send snipe alert: {e}")
    
    # Discord Commands
    
    @app_commands.command(name="sniper_channel", description="Set the channel for auction snipe alerts")
    @app_commands.describe(channel="Channel to send snipe alerts to")
    async def set_sniper_channel(self, interaction: discord.Interaction, channel: discord.TextChannel):
        """Set the alert channel for auction snipes."""
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("❌ You need Administrator permissions to configure the sniper.", ephemeral=True)
            return
        
        self.alert_channel_id = channel.id
        
        # Save to config
        await self._save_sniper_config()
        
        await interaction.response.send_message(f"✅ Sniper alerts will be sent to {channel.mention}")
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
    
    @app_commands.command(name="sniper_status", description="Check sniper status and statistics")
    async def sniper_status(self, interaction: discord.Interaction):
        """Show current sniper status."""
        embed = discord.Embed(title="🎯 Auction Sniper Status", color=0x0099ff, 
                            timestamp=datetime.now(timezone.utc))
        
        # Task status
        hunter_status = "🟢 Running" if self.high_frequency_snipe_scan.is_running() else "🔴 Stopped"
        intelligence_status = "🟢 Running" if self.update_market_intelligence.is_running() else "🔴 Stopped"
        
        embed.add_field(name="🏃 Hunter Task (2s)", value=hunter_status, inline=True)
        embed.add_field(name="🧠 Intelligence Task (60s)", value=intelligence_status, inline=True)
        embed.add_field(name="📢 Alert Channel", 
                       value=f"<#{self.alert_channel_id}>" if self.alert_channel_id else "Not set", 
                       inline=True)
        
        # Configuration
        embed.add_field(name="💰 Profit Threshold", value=f"{self.profit_threshold:,} coins", inline=True)
        embed.add_field(name="📊 Min Auction Count", value=str(self.min_auction_count), inline=True)
        embed.add_field(name="📝 Watchlist Size", value=str(len(self.auction_watchlist)), inline=True)
        
        # FMV data status
        embed.add_field(name="💾 FMV Cache", value=f"{len(self.fmv_data)} items", inline=True)
        
        # API status
        api_status = "🟢 Connected" if self.hypixel_client else "🔴 No API Key"
        embed.add_field(name="🔗 Hypixel API", value=api_status, inline=True)
        
        await interaction.response.send_message(embed=embed)
    
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
                await f.write(json.dumps(config_data, indent=2))
            
            self.logger.debug("Saved sniper configuration")
        
        except Exception as e:
            self.logger.error(f"Failed to save sniper config: {e}")


async def setup(bot):
    """Required setup function for discord.py cog."""
    await bot.add_cog(AuctionSniper(bot))