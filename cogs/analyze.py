#!/usr/bin/env python3
"""
Enhanced Analyze Cog for Discord Bot

Provides /analyze command that gives comprehensive item analysis:
- Current mid, spread, spread_bps
- Z-score analysis of price/spread
- Volume imbalance analysis
- Floor price from features if available
- Opportunity/risk assessment
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


class AnalyzeCog(commands.Cog):
    """Enhanced item analysis with statistical insights."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.AnalyzeCog")
        
        # Load config
        try:
            with open("config/config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        # Data paths
        self.bazaar_paths = [
            Path("data/bazaar_history"),
            Path("data/bazaar"),
            Path("data/bazaar_snapshots.ndjson")
        ]
        
        self.logger.info("Enhanced Analyze cog initialized")
    
    def _normalize_item_name(self, item: str) -> str:
        """Normalize item name for matching."""
        return item.strip().replace(' ', '_').replace('-', '_').upper()
    
    def _load_item_bazaar_data(self, item: str, hours: int = 6) -> Optional[pd.DataFrame]:
        """Load bazaar data for specific item."""
        try:
            normalized_item = self._normalize_item_name(item)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            # Try parquet sources
            for data_path in self.bazaar_paths[:2]:
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
                            
                            # Normalize column names and filter by item
                            column_mapping = {}
                            for col in df.columns:
                                col_lower = col.lower()
                                if 'product' in col_lower or col_lower in ['item', 'name']:
                                    column_mapping[col] = 'product_id'
                                elif 'timestamp' in col_lower or col_lower in ['ts', 'time', 'created_at']:
                                    column_mapping[col] = 'timestamp'
                                elif 'buy' in col_lower and 'price' in col_lower:
                                    column_mapping[col] = 'buy_price'
                                elif 'sell' in col_lower and 'price' in col_lower:
                                    column_mapping[col] = 'sell_price'
                                elif 'buy' in col_lower and ('volume' in col_lower or 'moving' in col_lower):
                                    column_mapping[col] = 'buy_volume'
                                elif 'sell' in col_lower and ('volume' in col_lower or 'moving' in col_lower):
                                    column_mapping[col] = 'sell_volume'
                                elif 'buy' in col_lower and 'order' in col_lower:
                                    column_mapping[col] = 'buy_orders'
                                elif 'sell' in col_lower and 'order' in col_lower:
                                    column_mapping[col] = 'sell_orders'
                            
                            df = df.rename(columns=column_mapping)
                            
                            if 'product_id' in df.columns and 'timestamp' in df.columns:
                                # Filter by item and time
                                df['product_id_norm'] = df['product_id'].astype(str).str.upper()
                                item_df = df[df['product_id_norm'] == normalized_item].copy()
                                
                                if not item_df.empty:
                                    item_df['timestamp'] = pd.to_datetime(item_df['timestamp'], utc=True)
                                    item_df = item_df[item_df['timestamp'] >= start_time]
                                    
                                    if not item_df.empty:
                                        self.logger.info(f"Found {len(item_df)} records for {item} from {data_path}")
                                        return item_df.sort_values('timestamp')
                                        
                except Exception as e:
                    self.logger.error(f"Failed to load from {data_path}: {e}")
                    continue
            
            # Try NDJSON fallback
            ndjson_path = self.bazaar_paths[2]
            if ndjson_path.exists():
                try:
                    records = []
                    with open(ndjson_path, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                # Check if this record contains our item
                                if 'products' in data:
                                    for product_id, product_data in data['products'].items():
                                        if self._normalize_item_name(product_id) == normalized_item:
                                            record = {
                                                'timestamp': data.get('timestamp', data.get('lastUpdated')),
                                                'product_id': product_id,
                                                'buy_price': product_data.get('buy_summary', [{}])[0].get('pricePerUnit', 0) if product_data.get('buy_summary') else 0,
                                                'sell_price': product_data.get('sell_summary', [{}])[0].get('pricePerUnit', 0) if product_data.get('sell_summary') else 0,
                                                'buy_volume': product_data.get('quick_status', {}).get('buyMovingWeek', 0),
                                                'sell_volume': product_data.get('quick_status', {}).get('sellMovingWeek', 0),
                                                'buy_orders': product_data.get('quick_status', {}).get('buyOrders', 0),
                                                'sell_orders': product_data.get('quick_status', {}).get('sellOrders', 0)
                                            }
                                            records.append(record)
                            except json.JSONDecodeError:
                                continue
                    
                    if records:
                        df = pd.DataFrame(records)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                        df = df[df['timestamp'] >= start_time]
                        
                        if not df.empty:
                            self.logger.info(f"Found {len(df)} records for {item} from NDJSON")
                            return df.sort_values('timestamp')
                            
                except Exception as e:
                    self.logger.error(f"Failed to load from NDJSON: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load bazaar data for {item}: {e}")
            return None
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical analysis for the item."""
        try:
            if df.empty:
                return {}
            
            # Calculate mid prices and spreads
            df = df.copy()
            df['mid_price'] = (df['buy_price'] + df['sell_price']) / 2
            df['spread'] = df['sell_price'] - df['buy_price']
            df['spread_bps'] = (df['spread'] / df['buy_price'] * 10000).fillna(0)
            
            # Current values (latest)
            latest = df.iloc[-1]
            current_mid = latest.get('mid_price', 0)
            current_spread = latest.get('spread', 0)
            current_spread_bps = latest.get('spread_bps', 0)
            current_buy_volume = latest.get('buy_volume', 0)
            current_sell_volume = latest.get('sell_volume', 0)
            
            # Statistical analysis (recent window for z-scores)
            recent_window = min(len(df), 100)  # Last 100 data points
            recent_df = df.tail(recent_window)
            
            # Z-scores
            mid_price_mean = recent_df['mid_price'].mean()
            mid_price_std = recent_df['mid_price'].std()
            mid_price_zscore = (current_mid - mid_price_mean) / mid_price_std if mid_price_std > 0 else 0
            
            spread_mean = recent_df['spread_bps'].mean()
            spread_std = recent_df['spread_bps'].std()
            spread_zscore = (current_spread_bps - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Volume imbalance
            total_volume = current_buy_volume + current_sell_volume
            buy_volume_pct = (current_buy_volume / total_volume * 100) if total_volume > 0 else 50
            volume_imbalance = buy_volume_pct - 50  # Positive = more buying pressure
            
            # Data quality metrics
            coverage_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            data_points = len(df)
            
            return {
                'current_mid': current_mid,
                'current_spread': current_spread,
                'current_spread_bps': current_spread_bps,
                'mid_price_zscore': mid_price_zscore,
                'spread_zscore': spread_zscore,
                'buy_volume': current_buy_volume,
                'sell_volume': current_sell_volume,
                'volume_imbalance': volume_imbalance,
                'buy_orders': latest.get('buy_orders', 0),
                'sell_orders': latest.get('sell_orders', 0),
                'coverage_hours': coverage_hours,
                'data_points': data_points,
                'price_volatility': mid_price_std / mid_price_mean if mid_price_mean > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate statistics: {e}")
            return {}
    
    def _get_feature_data(self, item: str) -> Optional[Dict[str, Any]]:
        """Get feature data for the item if available."""
        try:
            normalized_item = self._normalize_item_name(item)
            consumer = FeatureConsumer(self.config)
            intelligence = consumer.generate_market_intelligence(window_hours=12)
            
            fmv_data = intelligence.get("fmv_data", {})
            
            # Try exact match first
            if normalized_item in fmv_data:
                return fmv_data[normalized_item]
            
            # Try fuzzy matching
            for fmv_item in fmv_data.keys():
                if self._normalize_item_name(fmv_item) == normalized_item:
                    return fmv_data[fmv_item]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get feature data: {e}")
            return None
    
    def _generate_insights(self, stats: Dict[str, Any], features: Optional[Dict[str, Any]]) -> List[str]:
        """Generate trading insights and risk assessment."""
        insights = []
        
        try:
            # Price analysis
            mid_zscore = stats.get('mid_price_zscore', 0)
            if mid_zscore > 2:
                insights.append("🔴 Price significantly above recent average (+2σ)")
            elif mid_zscore > 1:
                insights.append("🟡 Price above recent average (+1σ)")
            elif mid_zscore < -2:
                insights.append("🟢 Price significantly below recent average (-2σ)")
            elif mid_zscore < -1:
                insights.append("🟡 Price below recent average (-1σ)")
            else:
                insights.append("⚪ Price near recent average")
            
            # Spread analysis
            spread_bps = stats.get('current_spread_bps', 0)
            spread_zscore = stats.get('spread_zscore', 0)
            
            if spread_bps > 1000:
                insights.append("💰 Very high spread - potential arbitrage opportunity")
            elif spread_bps > 500:
                insights.append("💵 High spread - good trading opportunity")
            elif spread_bps < 100:
                insights.append("⚠️ Very tight spread - limited profit margin")
            
            if spread_zscore > 1:
                insights.append("📈 Spread wider than usual")
            elif spread_zscore < -1:
                insights.append("📉 Spread tighter than usual")
            
            # Volume analysis
            volume_imbalance = stats.get('volume_imbalance', 0)
            if volume_imbalance > 20:
                insights.append("🔥 Strong buying pressure (high demand)")
            elif volume_imbalance > 10:
                insights.append("📈 Moderate buying pressure")
            elif volume_imbalance < -20:
                insights.append("❄️ Strong selling pressure (high supply)")
            elif volume_imbalance < -10:
                insights.append("📉 Moderate selling pressure")
            
            # Feature-based insights
            if features:
                floor_count = features.get('floor_count', 0)
                method = features.get('method', '')
                
                if floor_count <= 2:
                    insights.append("⚡ Thin floor detected - potential buy opportunity")
                elif floor_count >= 10:
                    insights.append("🏗️ Thick floor - stable support level")
                
                if 'thin_wall' in method:
                    insights.append("🎯 Market depth analysis suggests price instability")
            
            # Data quality warning
            coverage = stats.get('coverage_hours', 0)
            if coverage < 2:
                insights.append("⚠️ Limited data coverage - analysis may be incomplete")
            
        except Exception as e:
            self.logger.error(f"Failed to generate insights: {e}")
            insights.append("❌ Error generating insights")
        
        return insights
    
    def _create_analysis_embed(self, item: str, stats: Dict[str, Any], features: Optional[Dict[str, Any]], insights: List[str]) -> discord.Embed:
        """Create Discord embed for item analysis."""
        embed = discord.Embed(
            title=f"📊 Market Analysis: {item}",
            color=0x00ff00 if stats else 0xff0000
        )
        
        if not stats:
            embed.add_field(
                name="❌ No Data",
                value="No bazaar data found for this item. Check item name or data availability.",
                inline=False
            )
            return embed
        
        # Current market data
        current_mid = stats.get('current_mid', 0)
        current_spread = stats.get('current_spread', 0)
        current_spread_bps = stats.get('current_spread_bps', 0)
        
        market_text = f"**Mid Price:** {current_mid:,.0f} coins\n"
        market_text += f"**Spread:** {current_spread:,.0f} coins ({current_spread_bps:.0f} bps)\n"
        
        if features:
            floor_price = features.get('floor_price', 0)
            floor_count = features.get('floor_count', 0)
            market_text += f"**Floor Price:** {floor_price:,.0f} coins ({floor_count} items)\n"
        
        embed.add_field(
            name="💰 Current Market",
            value=market_text,
            inline=True
        )
        
        # Statistical analysis
        mid_zscore = stats.get('mid_price_zscore', 0)
        spread_zscore = stats.get('spread_zscore', 0)
        volatility = stats.get('price_volatility', 0)
        
        stats_text = f"**Price Z-Score:** {mid_zscore:+.2f}σ\n"
        stats_text += f"**Spread Z-Score:** {spread_zscore:+.2f}σ\n"
        stats_text += f"**Volatility:** {volatility:.1%}\n"
        
        embed.add_field(
            name="📈 Statistics",
            value=stats_text,
            inline=True
        )
        
        # Volume analysis
        buy_volume = stats.get('buy_volume', 0)
        sell_volume = stats.get('sell_volume', 0)
        volume_imbalance = stats.get('volume_imbalance', 0)
        buy_orders = stats.get('buy_orders', 0)
        sell_orders = stats.get('sell_orders', 0)
        
        volume_text = f"**Buy Volume:** {buy_volume:,.0f}\n"
        volume_text += f"**Sell Volume:** {sell_volume:,.0f}\n"
        volume_text += f"**Imbalance:** {volume_imbalance:+.1f}%\n"
        volume_text += f"**Orders:** {buy_orders}/{sell_orders}\n"
        
        embed.add_field(
            name="📦 Volume & Orders",
            value=volume_text,
            inline=True
        )
        
        # Insights
        if insights:
            insights_text = "\n".join(insights[:8])  # Limit for Discord
            embed.add_field(
                name="🎯 Trading Insights",
                value=insights_text,
                inline=False
            )
        
        # Data coverage
        coverage = stats.get('coverage_hours', 0)
        data_points = stats.get('data_points', 0)
        
        coverage_text = f"**Time Coverage:** {coverage:.1f} hours\n"
        coverage_text += f"**Data Points:** {data_points}\n"
        
        if features:
            last_updated = features.get('updated_at', 'Unknown')
            coverage_text += f"**Features:** {last_updated[:16] if last_updated != 'Unknown' else 'Unknown'}\n"
        
        embed.add_field(
            name="📡 Data Coverage",
            value=coverage_text,
            inline=True
        )
        
        embed.set_footer(text=f"Analysis generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return embed
    
    @app_commands.command(name="analyze", description="Comprehensive market analysis for a specific item")
    @app_commands.describe(
        item="Item name to analyze (e.g., WHEAT, ENCHANTED_FLINT)",
        hours="Data window in hours (1-12, default: 6)"
    )
    async def analyze(self, interaction: discord.Interaction, item: str, hours: int = 6):
        """Perform comprehensive market analysis for an item."""
        await interaction.response.defer()
        
        try:
            # Validate input
            if not 1 <= hours <= 12:
                await interaction.followup.send("⚠️ Hours must be between 1 and 12.", ephemeral=True)
                return
            
            if not item.strip():
                await interaction.followup.send("⚠️ Please provide an item name.", ephemeral=True)
                return
            
            item = item.strip()
            self.logger.info(f"Analyzing {item} over {hours} hour window")
            
            # Load bazaar data
            bazaar_df = self._load_item_bazaar_data(item, hours)
            
            # Calculate statistics
            stats = self._calculate_statistics(bazaar_df) if bazaar_df is not None else {}
            
            # Get feature data
            features = self._get_feature_data(item)
            
            # Generate insights
            insights = self._generate_insights(stats, features) if stats else []
            
            # Create embed
            embed = self._create_analysis_embed(item, stats, features, insights)
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in analyze command: {e}")
            await interaction.followup.send(
                f"❌ Error analyzing {item}: {str(e)}",
                ephemeral=True
            )


async def setup(bot: commands.Bot):
    """Setup function for the cog."""
    await bot.add_cog(AnalyzeCog(bot))
    logging.getLogger(__name__).info("Enhanced Analyze cog loaded")