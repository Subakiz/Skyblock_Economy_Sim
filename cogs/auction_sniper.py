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

import discord
from discord import app_commands
from discord.ext import commands, tasks

# Import project modules
from ingestion.common.hypixel_client import HypixelClient


class AuctionSniper(commands.Cog):
    """
    Intelligent auction sniper for Hypixel Skyblock.
    
    Uses a two-speed architecture:
    - Hunter: High-frequency scanning (2s) for quick snipes
    - Analyst: Low-frequency analysis (90s) for market intelligence
    """
    
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
        
        # Load persisted data
        self._load_persisted_data()
        self._load_saved_config()
        
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
    
    def _save_persisted_data(self):
        """Save watchlist and FMV data to disk."""
        try:
            # Save watchlist
            watchlist_file = self.data_dir / "auction_watchlist.json"
            with open(watchlist_file, "w") as f:
                json.dump({
                    "items": list(self.auction_watchlist),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(self.auction_watchlist)
                }, f, indent=2)
            
            # Save FMV data
            fmv_file = self.data_dir / "fmv_cache.json"
            with open(fmv_file, "w") as f:
                json.dump(self.fmv_data, f, indent=2)
            
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
            analyst_interval = sniper_config.get("analyst_interval_seconds", 90)
            
            # Change task intervals
            self.high_frequency_snipe_scan.change_interval(seconds=hunter_interval)
            self.full_market_analysis.change_interval(seconds=analyst_interval)
            
            # Start tasks
            self.high_frequency_snipe_scan.start()
            self.full_market_analysis.start()
            self.hunter_task_active = True
            self.analyst_task_active = True
            
            self.logger.info(f"Started sniper tasks: Hunter ({hunter_interval}s), Analyst ({analyst_interval}s)")
    
    def cog_unload(self):
        """Clean shutdown of tasks."""
        self.logger.info("Shutting down auction sniper tasks...")
        if self.high_frequency_snipe_scan.is_running():
            self.high_frequency_snipe_scan.cancel()
        if self.full_market_analysis.is_running():
            self.full_market_analysis.cancel()
        self._save_persisted_data()
    
    @tasks.loop(seconds=2)  # Default, will be updated from config
    async def high_frequency_snipe_scan(self):
        """
        Hunter Task: High-frequency scanning of page 0 for quick snipes.
        Runs every 2 seconds for maximum speed.
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
                    if not item_name or item_name not in self.auction_watchlist:
                        continue
                    
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
    
    @tasks.loop(seconds=90)  # Default, will be updated from config
    async def full_market_analysis(self):
        """
        Analyst Task: Full market analysis with pagination.
        Updates watchlist and FMV data every 90 seconds.
        """
        if not self.hypixel_client:
            return
        
        try:
            start_time = time.time()
            self.logger.info("Starting full market analysis...")
            
            # Collect all auctions with pagination
            all_auctions = []
            page = 0
            
            while True:
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
                        
                except Exception as e:
                    self.logger.error(f"Error fetching page {page}: {e}")
                    break
            
            self.logger.info(f"Collected {len(all_auctions)} total auctions from {page} pages")
            
            # Analyze market data
            await self._update_watchlist(all_auctions)
            await self._update_fmv_data(all_auctions)
            
            # Persist data
            self._save_persisted_data()
            
            analysis_time = time.time() - start_time
            self.logger.info(f"Market analysis complete: {len(self.auction_watchlist)} watchlist items, "
                           f"FMV data for {len(self.fmv_data)} items in {analysis_time:.1f}s")
        
        except Exception as e:
            self.logger.error(f"Analyst task error: {e}")
    
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
        """Update Fair Market Value data for watchlist items."""
        try:
            # Group auctions by item
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
            
            # Calculate FMV statistics
            for item_name, prices in item_prices.items():
                if len(prices) >= 3:  # Need at least 3 data points
                    prices.sort()
                    # Remove outliers (bottom and top 10%)
                    trim_count = max(1, len(prices) // 10)
                    trimmed_prices = prices[trim_count:-trim_count] if len(prices) > 2 * trim_count else prices
                    
                    if trimmed_prices:
                        fmv = sum(trimmed_prices) / len(trimmed_prices)
                        median = trimmed_prices[len(trimmed_prices) // 2]
                        
                        self.fmv_data[item_name] = {
                            "fmv": fmv,
                            "median": median,
                            "samples": len(prices),
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
            
            self.logger.debug(f"Updated FMV data for {len(item_prices)} items")
        
        except Exception as e:
            self.logger.error(f"Failed to update FMV data: {e}")
    
    async def _verify_snipe(self, auction: Dict[str, Any]) -> bool:
        """
        Verify if an auction is a valid snipe opportunity.
        Includes manipulation, attribute, and profitability checks.
        """
        try:
            item_name = auction.get("item_name", "").strip()
            price = float(auction.get("starting_bid", 0))
            
            if price <= 0:
                return False
            
            # Manipulation Check: Compare against FMV
            fmv_info = self.fmv_data.get(item_name)
            if not fmv_info:
                return False  # No FMV data available
            
            fmv = fmv_info.get("fmv", 0)
            if price > fmv * 1.1:  # Price is significantly above FMV
                return False
            
            # Attribute Check: Parse NBT for critical attributes
            if not self._check_item_attributes(auction):
                return False
            
            # Profitability Check
            auction_house_fee = price * 0.01  # 1% AH fee
            estimated_profit = fmv - price - auction_house_fee
            
            if estimated_profit < self.profit_threshold:
                return False
            
            self.logger.info(f"Valid snipe found: {item_name} at {price:,.0f} coins "
                           f"(FMV: {fmv:,.0f}, Profit: {estimated_profit:,.0f})")
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
            # For now, implement basic checks
            # This could be expanded to parse actual NBT data
            item_name = auction.get("item_name", "")
            item_lore = auction.get("item_lore", "")
            
            # Basic attribute requirements
            if any(weapon_type in item_name.upper() for weapon_type in 
                   ["SWORD", "BOW", "HYPERION", "VALKYRIE", "SCYLLA"]):
                # Check for reforge or ultimate enchants in lore
                if "Ultimate" in item_lore or any(reforge in item_lore for reforge in 
                    ["Sharp", "Heroic", "Legendary", "Fabled", "Withered"]):
                    return True
                return False  # Weapon without good attributes
            
            if any(armor_type in item_name.upper() for armor_type in 
                   ["CHESTPLATE", "LEGGINGS", "HELMET", "BOOTS"]):
                # Check for stars or reforge
                if "⭐" in item_lore or any(reforge in item_lore for reforge in
                    ["Ancient", "Renowned", "Spiked", "Reinforced"]):
                    return True
                return False  # Armor without good attributes
            
            # For other items, accept by default
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
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("❌ You need Administrator permissions to configure the sniper.", ephemeral=True)
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
        
        await interaction.response.send_message(embed=embed)
        self.logger.info(f"Sniper configuration updated: profit_threshold={self.profit_threshold}, min_auction_count={self.min_auction_count}")
    
    @app_commands.command(name="sniper_status", description="Check sniper status and statistics")
    async def sniper_status(self, interaction: discord.Interaction):
        """Show current sniper status."""
        embed = discord.Embed(title="🎯 Auction Sniper Status", color=0x0099ff, 
                            timestamp=datetime.now(timezone.utc))
        
        # Task status
        hunter_status = "🟢 Running" if self.high_frequency_snipe_scan.is_running() else "🔴 Stopped"
        analyst_status = "🟢 Running" if self.full_market_analysis.is_running() else "🔴 Stopped"
        
        embed.add_field(name="🏃 Hunter Task (2s)", value=hunter_status, inline=True)
        embed.add_field(name="🔍 Analyst Task (90s)", value=analyst_status, inline=True)
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
        """Save sniper configuration to disk."""
        try:
            config_file = self.data_dir / "sniper_config.json"
            config_data = {
                "alert_channel_id": self.alert_channel_id,
                "profit_threshold": self.profit_threshold,
                "min_auction_count": self.min_auction_count,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.debug("Saved sniper configuration")
        
        except Exception as e:
            self.logger.error(f"Failed to save sniper config: {e}")


async def setup(bot):
    """Required setup function for discord.py cog."""
    await bot.add_cog(AuctionSniper(bot))