#!/usr/bin/env python3
"""
Market Pulse Cog for Discord Bot

Provides /market_pulse command that generates market signals from:
- Most recent feature summaries
- Recent bazaar snapshot data
- Top spread items, rising demand, supply drops, floor acceleration
"""

import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Tuple
import json

import discord
from discord.ext import commands
from discord import app_commands
import pandas as pd
import numpy as np
import yaml

from ingestion.feature_consumer import FeatureConsumer

logger = logging.getLogger(__name__)


class MarketPulseCog(commands.Cog):
    """Market pulse analysis for identifying trading opportunities."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.MarketPulseCog")
        
        # Load config
        try:
            with open("config/config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        # Data paths for bazaar fallback
        self.bazaar_paths = [
            Path("data/bazaar_history"),
            Path("data/bazaar"),
            Path("data/bazaar_snapshots.ndjson")
        ]
        
        self.logger.info("Market Pulse cog initialized")
    
    def _load_recent_bazaar_data(self, hours: int = 3) -> Optional[pd.DataFrame]:
        """Load recent bazaar data as fallback when features unavailable."""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            # Try parquet sources first
            for data_path in self.bazaar_paths[:2]:  # Skip NDJSON for now
                if not data_path.exists():
                    continue
                    
                try:
                    if data_path.is_dir():
                        dfs = []
                        for file_path in data_path.glob("*.parquet"):
                            df = pd.read_parquet(file_path)
                            dfs.append(df)
                        
                        if dfs:
                            df = pd.concat(dfs, ignore_index=True)
                            
                            # Convert timestamp column
                            for ts_col in ['ts', 'timestamp', 'time', 'created_at']:
                                if ts_col in df.columns:
                                    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                                    df = df[df[ts_col] >= start_time]
                                    break
                            
                            if len(df) > 0:
                                self.logger.info(f"Loaded {len(df)} bazaar records from {data_path}")
                                return df
                                
                except Exception as e:
                    self.logger.error(f"Failed to load from {data_path}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load bazaar data: {e}")
            return None
    
    def _analyze_feature_signals(self, intelligence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze signals from feature summaries."""
        signals = []
        
        try:
            fmv_data = intelligence.get("fmv_data", {})
            
            # Signal 1: High spread items (potential arbitrage)
            spread_items = []
            for item, data in fmv_data.items():
                floor_price = data.get("floor_price", 0)
                second_price = data.get("second_price", 0)
                
                if floor_price > 0 and second_price > floor_price:
                    spread_bps = ((second_price - floor_price) / floor_price) * 10000
                    if spread_bps > 500:  # >5% spread
                        spread_items.append({
                            "item": item,
                            "spread_bps": round(spread_bps),
                            "floor_price": floor_price,
                            "floor_count": data.get("floor_count", 0)
                        })
            
            # Sort by spread and take top 5
            spread_items.sort(key=lambda x: x["spread_bps"], reverse=True)
            for item_data in spread_items[:5]:
                signals.append({
                    "type": "High Spread",
                    "item": item_data["item"],
                    "value": f"{item_data['spread_bps']} bps",
                    "detail": f"Floor: {item_data['floor_price']:,.0f} ({item_data['floor_count']} items)",
                    "score": item_data["spread_bps"]
                })
            
            # Signal 2: Thin floor opportunities
            thin_floors = []
            for item, data in fmv_data.items():
                floor_count = data.get("floor_count", 0)
                floor_price = data.get("floor_price", 0)
                
                if floor_count <= 2 and floor_price >= 10000:  # Thin floor on valuable items
                    thin_floors.append({
                        "item": item,
                        "floor_count": floor_count,
                        "floor_price": floor_price,
                        "method": data.get("method", "unknown")
                    })
            
            # Sort by price and take highest value thin floors
            thin_floors.sort(key=lambda x: x["floor_price"], reverse=True)
            for item_data in thin_floors[:3]:
                signals.append({
                    "type": "Thin Floor",
                    "item": item_data["item"],
                    "value": f"{item_data['floor_count']} items",
                    "detail": f"Floor: {item_data['floor_price']:,.0f}",
                    "score": item_data["floor_price"] / 1000  # Convert to score
                })
            
        except Exception as e:
            self.logger.error(f"Failed to analyze feature signals: {e}")
        
        return signals
    
    def _analyze_bazaar_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze signals from bazaar data when features unavailable."""
        signals = []
        
        try:
            if df.empty:
                return signals
            
            # Normalize column names
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'product' in col_lower or col_lower in ['item', 'name']:
                    column_mapping[col] = 'product_id'
                elif 'buy' in col_lower and 'price' in col_lower:
                    column_mapping[col] = 'buy_price'
                elif 'sell' in col_lower and 'price' in col_lower:
                    column_mapping[col] = 'sell_price'
                elif 'buy' in col_lower and ('volume' in col_lower or 'moving' in col_lower):
                    column_mapping[col] = 'buy_volume'
                elif 'sell' in col_lower and ('volume' in col_lower or 'moving' in col_lower):
                    column_mapping[col] = 'sell_volume'
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            if 'product_id' not in df.columns:
                return signals
            
            # Get latest data per product
            latest_data = df.groupby('product_id').last().reset_index()
            
            # Signal 1: High spread items
            if 'buy_price' in latest_data.columns and 'sell_price' in latest_data.columns:
                latest_data['spread_bps'] = (
                    (latest_data['sell_price'] - latest_data['buy_price']) / 
                    latest_data['buy_price'] * 10000
                ).fillna(0)
                
                high_spread = latest_data[latest_data['spread_bps'] > 500].nlargest(5, 'spread_bps')
                
                for _, row in high_spread.iterrows():
                    signals.append({
                        "type": "High Spread",
                        "item": row['product_id'],
                        "value": f"{row['spread_bps']:.0f} bps",
                        "detail": f"Buy: {row['buy_price']:,.0f} → Sell: {row['sell_price']:,.0f}",
                        "score": row['spread_bps']
                    })
            
            # Signal 2: Volume imbalance
            if 'buy_volume' in latest_data.columns and 'sell_volume' in latest_data.columns:
                latest_data['volume_ratio'] = (
                    latest_data['buy_volume'] / (latest_data['sell_volume'] + 1)
                ).fillna(0)
                
                # High buy pressure (buy_volume >> sell_volume)
                high_demand = latest_data[latest_data['volume_ratio'] > 3].nlargest(3, 'volume_ratio')
                
                for _, row in high_demand.iterrows():
                    signals.append({
                        "type": "High Demand",
                        "item": row['product_id'],
                        "value": f"{row['volume_ratio']:.1f}x ratio",
                        "detail": f"Buy: {row['buy_volume']:,.0f} vs Sell: {row['sell_volume']:,.0f}",
                        "score": row['volume_ratio'] * 10
                    })
        
        except Exception as e:
            self.logger.error(f"Failed to analyze bazaar signals: {e}")
        
        return signals
    
    def _create_pulse_embed(self, signals: List[Dict[str, Any]], source: str, time_window: str) -> discord.Embed:
        """Create Discord embed for market pulse."""
        embed = discord.Embed(
            title="📈 Market Pulse",
            description=f"Top trading signals from {source}",
            color=0x00ff00 if signals else 0xff0000
        )
        
        embed.add_field(
            name="📊 Analysis Window",
            value=time_window,
            inline=True
        )
        
        embed.add_field(
            name="📡 Data Source",
            value=source,
            inline=True
        )
        
        if not signals:
            embed.add_field(
                name="⚠️ No Signals",
                value="No actionable market signals found. Check data availability.",
                inline=False
            )
        else:
            # Sort by score and take top 10
            signals.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            signal_text = ""
            for i, signal in enumerate(signals[:10], 1):
                signal_text += f"**{i}. {signal['type']}** - {signal['item']}\n"
                signal_text += f"   {signal['value']} • {signal['detail']}\n\n"
            
            embed.add_field(
                name="🎯 Top Signals",
                value=signal_text[:1024],  # Discord limit
                inline=False
            )
        
        embed.set_footer(text=f"Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return embed
    
    @app_commands.command(name="market_pulse", description="Get current market pulse with top trading signals")
    @app_commands.describe(
        hours="Time window for analysis (1-6 hours, default: 2)"
    )
    async def market_pulse(self, interaction: discord.Interaction, hours: int = 2):
        """Generate market pulse analysis with actionable signals."""
        await interaction.response.defer()
        
        try:
            # Validate input
            if not 1 <= hours <= 6:
                await interaction.followup.send("⚠️ Hours must be between 1 and 6.", ephemeral=True)
                return
            
            self.logger.info(f"Generating market pulse for {hours} hour window")
            
            # Try to get signals from feature summaries first
            signals = []
            source = "Unknown"
            
            try:
                consumer = FeatureConsumer(self.config)
                intelligence = consumer.generate_market_intelligence(window_hours=hours)
                
                if intelligence.get("fmv_data"):
                    signals = self._analyze_feature_signals(intelligence)
                    source = f"Feature Summaries ({len(intelligence['fmv_data'])} items)"
                    self.logger.info(f"Generated {len(signals)} signals from feature summaries")
                else:
                    self.logger.warning("No feature data available, falling back to bazaar")
                    
            except Exception as e:
                self.logger.error(f"Failed to load feature data: {e}")
            
            # Fallback to bazaar data if features unavailable
            if not signals:
                bazaar_df = self._load_recent_bazaar_data(hours)
                if bazaar_df is not None and not bazaar_df.empty:
                    signals = self._analyze_bazaar_signals(bazaar_df)
                    source = f"Bazaar Data ({len(bazaar_df)} records)"
                    self.logger.info(f"Generated {len(signals)} signals from bazaar data")
                else:
                    source = "No Data Available"
            
            # Create embed
            time_window = f"Last {hours} hour{'s' if hours != 1 else ''}"
            embed = self._create_pulse_embed(signals, source, time_window)
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in market_pulse command: {e}")
            await interaction.followup.send(
                f"❌ Error generating market pulse: {str(e)}",
                ephemeral=True
            )


async def setup(bot: commands.Bot):
    """Setup function for the cog."""
    await bot.add_cog(MarketPulseCog(bot))
    logging.getLogger(__name__).info("Market Pulse cog loaded")