#!/usr/bin/env python3
"""
Enhanced Plot Cog for Discord Bot

Provides an enhanced /plot command that:
- Loads bazaar snapshots from data/bazaar or data/bazaar_snapshots.ndjson
- Generates two-panel charts: prices+spread (top), volumes+orders (bottom)
- Uses 1-minute median resampling and percentile clipping to avoid spikes
- Computes mid, spread, spread_bps, demand/supply volumes, and order counts
"""

import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List
from io import BytesIO

import discord
from discord.ext import commands
from discord import app_commands
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class PlotCog(commands.Cog):
    """Enhanced plotting functionality for market data visualization."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.PlotCog")
        
        # Data paths to check in priority order
        self.data_paths = [
            Path("data/bazaar_history"),  # Parquet under bazaar_history (highest priority)
            Path("data/bazaar"),         # Parquet under bazaar/
            Path("data/bazaar_snapshots.ndjson")  # NDJSON file (fallback)
        ]
        
        self.logger.info("Enhanced Plot cog initialized")
    
    def _normalize_item_name(self, item: str) -> str:
        """Normalize item name (spaces to underscores, uppercase)."""
        return item.strip().replace(' ', '_').replace('-', '_').upper()
    
    def _get_item_suggestions(self, item: str, available_items: List[str], max_suggestions: int = 5) -> List[str]:
        """Get fuzzy suggestions for item names."""
        normalized_item = self._normalize_item_name(item)
        normalized_available = [self._normalize_item_name(i) for i in available_items]
        
        # Simple alias mapping for common items
        aliases = {
            'WHEAT': ['WHEAT', 'SEEDS'],
            'ENCHANTED_FLINT': ['ENCHANTED_FLINT', 'E_FLINT', 'EFLINT'],
            'COBBLESTONE': ['COBBLESTONE', 'COBBLE', 'STONE'],
            'EMERALD': ['EMERALD', 'EMERALDS'],
            'DIAMOND': ['DIAMOND', 'DIAMONDS']
        }
        
        suggestions = []
        
        # Check exact matches first (after normalization)
        if normalized_item in normalized_available:
            idx = normalized_available.index(normalized_item)
            suggestions.append(available_items[idx])
        
        # Check alias matches
        for canonical, alias_list in aliases.items():
            if normalized_item in [self._normalize_item_name(a) for a in alias_list]:
                if canonical in normalized_available:
                    idx = normalized_available.index(canonical)
                    if available_items[idx] not in suggestions:
                        suggestions.append(available_items[idx])
        
        # Simple substring matching
        for i, norm_item in enumerate(normalized_available):
            if normalized_item in norm_item or norm_item in normalized_item:
                if available_items[i] not in suggestions:
                    suggestions.append(available_items[i])
        
        return suggestions[:max_suggestions]
    
    def _detect_data_source(self) -> Optional[Path]:
        """Detect available bazaar data source."""
        for path in self.data_paths:
            if path.exists():
                self.logger.debug(f"Found data source: {path}")
                return path
        return None
    
    def _load_bazaar_data(self, item: str, hours: int = 3) -> Optional[pd.DataFrame]:
        """Load bazaar data for the specified item and time window."""
        try:
            data_source = self._detect_data_source()
            if not data_source:
                self.logger.warning("No bazaar data source found")
                return None
            
            # Calculate time window
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            if data_source.is_dir():
                # Directory-based storage
                dfs = []
                for file_path in data_source.glob("*.parquet"):
                    try:
                        df = pd.read_parquet(file_path)
                        dfs.append(df)
                    except Exception as e:
                        self.logger.error(f"Failed to read {file_path}: {e}")
                
                if not dfs:
                    return None
                    
                df = pd.concat(dfs, ignore_index=True)
                
            else:
                # NDJSON file
                try:
                    df = pd.read_json(data_source, lines=True)
                except Exception as e:
                    self.logger.error(f"Failed to read NDJSON {data_source}: {e}")
                    return None
            
            # Detect timestamp column robustly
            timestamp_cols = ['timestamp', 'time', 'datetime', 'created_at', 'updated_at', 'ts']
            timestamp_col = None
            
            for col in timestamp_cols:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            # Fallback: look for any column with 'time' in the name
            if timestamp_col is None:
                time_cols = [col for col in df.columns if 'time' in col.lower()]
                if time_cols:
                    timestamp_col = time_cols[0]
            
            if timestamp_col is None:
                self.logger.error(f"No timestamp column found in columns: {list(df.columns)}")
                return None
            
            # Convert timestamp and filter by time window
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df[(df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)]
            
            # Filter by item (flexible product ID detection)
            product_cols = ['product_id', 'productId', 'product', 'item_name', 'item', 'name']
            product_col = None
            
            for col in product_cols:
                if col in df.columns:
                    product_col = col
                    break
            
            if product_col is None:
                self.logger.error(f"No product column found in columns: {list(df.columns)}")
                return None
            
            # Store available items for suggestions
            available_items = df[product_col].unique().tolist()
            
            # Normalize input item name and try to match
            normalized_item = self._normalize_item_name(item)
            
            # Try exact normalized match first
            item_data = df[df[product_col].str.upper().str.replace(' ', '_').str.replace('-', '_') == normalized_item]
            
            if item_data.empty:
                # Store available items for later suggestion use
                self._last_available_items = available_items
                return None
            
            # Ensure we have required price columns
            buy_col = None
            sell_col = None
            
            # Check for various column naming patterns
            for col in ['buy_price', 'buyPrice', 'buy_price_coins', 'instant_buy']:
                if col in item_data.columns:
                    buy_col = col
                    break
            
            for col in ['sell_price', 'sellPrice', 'sell_price_coins', 'instant_sell']:
                if col in item_data.columns:
                    sell_col = col
                    break
            
            if buy_col is None or sell_col is None:
                self.logger.error(f"Required price columns not found. Available: {list(item_data.columns)}")
                return None
            
            # Rename columns to standard names
            item_data = item_data.rename(columns={
                timestamp_col: 'timestamp',
                buy_col: 'buy_price',
                sell_col: 'sell_price'
            })
            
            # Add volume and order columns if available
            for col in ['buy_volume', 'buyVolume', 'buy_volume_coins']:
                if col in item_data.columns:
                    item_data = item_data.rename(columns={col: 'buy_volume'})
                    break
            
            for col in ['sell_volume', 'sellVolume', 'sell_volume_coins']:
                if col in item_data.columns:
                    item_data = item_data.rename(columns={col: 'sell_volume'})
                    break
            
            for col in ['buy_orders', 'buyOrders']:
                if col in item_data.columns:
                    item_data = item_data.rename(columns={col: 'buy_orders'})
                    break
                    
            for col in ['sell_orders', 'sellOrders']:
                if col in item_data.columns:
                    item_data = item_data.rename(columns={col: 'sell_orders'})
                    break
            
            # Fill missing volume/order columns with defaults
            if 'buy_volume' not in item_data.columns:
                item_data['buy_volume'] = 0
            if 'sell_volume' not in item_data.columns:
                item_data['sell_volume'] = 0
            if 'buy_orders' not in item_data.columns:
                item_data['buy_orders'] = 0
            if 'sell_orders' not in item_data.columns:
                item_data['sell_orders'] = 0
            
            self.logger.info(f"Loaded {len(item_data)} records for {item} from {data_source}")
            return item_data.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"Failed to load bazaar data for {item}: {e}")
            return None
    
    def _resample_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to 1-minute median and clip outliers."""
        try:
            # Set timestamp as index for resampling
            df = df.set_index('timestamp')
            
            # Resample to 1-minute using median for numeric columns only
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            resampled = df[numeric_columns].resample('1min').median()
            
            # Remove rows where all price data is NaN
            resampled = resampled.dropna(subset=['buy_price', 'sell_price'], how='all')
            
            if resampled.empty:
                return pd.DataFrame()
            
            # Clip prices to 1-99 percentiles to avoid spikes
            for price_col in ['buy_price', 'sell_price']:
                if price_col in resampled.columns:
                    p1 = resampled[price_col].quantile(0.01)
                    p99 = resampled[price_col].quantile(0.99)
                    resampled[price_col] = resampled[price_col].clip(p1, p99)
            
            # Reset index to have timestamp as column
            resampled = resampled.reset_index()
            
            self.logger.debug(f"Resampled data to {len(resampled)} 1-minute intervals")
            return resampled
            
        except Exception as e:
            self.logger.error(f"Failed to resample and clean data: {e}")
            # Return original data if resampling fails
            return df
    
    def _compute_enhanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute enhanced metrics: mid, spread, spread_bps."""
        try:
            # Compute mid price
            df['mid'] = (df['buy_price'] + df['sell_price']) / 2
            
            # Compute spread
            df['spread'] = df['sell_price'] - df['buy_price']
            
            # Compute spread in basis points (bps)
            df['spread_bps'] = (df['spread'] / df['mid']) * 10000
            
            # Ensure non-negative spreads
            df['spread'] = df['spread'].clip(lower=0)
            df['spread_bps'] = df['spread_bps'].clip(lower=0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to compute enhanced metrics: {e}")
            return df
    
    def _create_enhanced_plot(self, df: pd.DataFrame, item: str) -> BytesIO:
        """Create two-panel enhanced plot."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'{item} — Market Overview', fontsize=16, fontweight='bold')
            
            # Top panel: Prices + Spread
            ax1.plot(df['timestamp'], df['buy_price'], 
                    label='Buy Price (Insta-Buy)', color='#2E8B57', linewidth=2)
            ax1.plot(df['timestamp'], df['sell_price'], 
                    label='Sell Price (Insta-Sell)', color='#DC143C', linewidth=2)
            ax1.plot(df['timestamp'], df['mid'], 
                    label='Mid Price', color='#1f77b4', linewidth=1, linestyle='--')
            
            # Add spread as shaded area
            ax1.fill_between(df['timestamp'], df['buy_price'], df['sell_price'], 
                           alpha=0.3, color='#FFD700', label='Spread')
            
            ax1.set_ylabel('Price (coins)', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Prices and Spread', fontsize=14)
            
            # Format y-axis for prices (use k, M notation for large numbers)
            def format_price(x, p):
                if x >= 1_000_000:
                    return f'{x/1_000_000:.1f}M'
                elif x >= 1_000:
                    return f'{x/1_000:.1f}k'
                else:
                    return f'{x:.0f}'
            
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_price))
            
            # Bottom panel: Volumes + Order Counts
            # Bar plots for volumes
            bar_width = 0.8 * (df['timestamp'].iloc[1] - df['timestamp'].iloc[0])
            
            ax2.bar(df['timestamp'], df['buy_volume'], 
                   width=bar_width, alpha=0.7, color='#2E8B57', 
                   label='Buy Volume (Demand)')
            ax2.bar(df['timestamp'], -df['sell_volume'], 
                   width=bar_width, alpha=0.7, color='#DC143C', 
                   label='Sell Volume (Supply)')
            
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Volumes and Order Counts', fontsize=14)
            
            # Twin axis for order counts (dashed lines)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df['timestamp'], df['buy_orders'], 
                         color='#228B22', linestyle='--', linewidth=1,
                         label='Buy Orders', alpha=0.8)
            ax2_twin.plot(df['timestamp'], df['sell_orders'], 
                         color='#B22222', linestyle='--', linewidth=1,
                         label='Sell Orders', alpha=0.8)
            ax2_twin.set_ylabel('Order Count', fontsize=12)
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Format x-axis for both panels
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.tick_params(axis='x', rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            self.logger.error(f"Failed to create enhanced plot: {e}")
            plt.close()
            return None
    
    @app_commands.command(name="plot", description="Generate enhanced market overview chart for an item")
    @app_commands.describe(
        item="Name of the item to plot (e.g., WHEAT, ENCHANTED_FLINT)",
        hours="Number of hours of data to include (default: 3)"
    )
    async def plot_command(self, interaction: discord.Interaction, item: str, hours: int = 3):
        """Generate an enhanced market overview chart for an item."""
        
        # Validate hours parameter
        if hours < 1 or hours > 72:
            await interaction.response.send_message(
                "❌ Hours must be between 1 and 72.", ephemeral=True
            )
            return
        
        await interaction.response.defer()
        
        try:
            self.logger.info(f"Plot command requested for {item} ({hours}h)")
            
            # Load bazaar data
            df = self._load_bazaar_data(item, hours)
            
            if df is None or df.empty:
                # Try to get suggestions if available
                suggestions = []
                if hasattr(self, '_last_available_items'):
                    suggestions = self._get_item_suggestions(item, self._last_available_items)
                
                error_msg = f"❌ No bazaar data found for **{item}** in the last {hours} hours."
                
                if suggestions:
                    error_msg += f"\n\n💡 **Did you mean:**\n" + "\n".join([f"• `{suggestion}`" for suggestion in suggestions])
                else:
                    error_msg += f"\n\n💡 **Examples:** `WHEAT`, `ENCHANTED_FLINT`, `COBBLESTONE`"
                    
                paths_searched = [str(p) for p in self.data_paths if p.exists()]
                if paths_searched:
                    error_msg += f"\n\n📁 **Searched:** {', '.join(paths_searched)}"
                
                await interaction.followup.send(error_msg)
                return
            
            # Resample and clean data
            df_clean = self._resample_and_clean_data(df)
            
            if df_clean.empty:
                await interaction.followup.send(
                    f"❌ No valid data points found for **{item}** after cleaning."
                )
                return
            
            # Compute enhanced metrics
            df_enhanced = self._compute_enhanced_metrics(df_clean)
            
            # Create enhanced plot
            plot_buffer = self._create_enhanced_plot(df_enhanced, item)
            
            if plot_buffer is None:
                await interaction.followup.send(
                    f"❌ Failed to generate plot for **{item}**. Please try again later."
                )
                return
            
            # Create Discord file and embed
            file = discord.File(plot_buffer, filename=f"{item.lower()}_market_overview.png")
            
            embed = discord.Embed(
                title=f"📊 {item} — Market Overview",
                description=f"Enhanced market analysis ({len(df_enhanced)} data points, {hours}h window)",
                color=discord.Color.blue(),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add market statistics
            if not df_enhanced.empty:
                avg_mid = df_enhanced['mid'].mean()
                avg_spread = df_enhanced['spread'].mean()
                avg_spread_bps = df_enhanced['spread_bps'].mean()
                total_buy_vol = df_enhanced['buy_volume'].sum()
                total_sell_vol = df_enhanced['sell_volume'].sum()
                
                embed.add_field(
                    name="💰 Price Summary",
                    value=f"Avg Mid: {avg_mid:,.0f} coins\nAvg Spread: {avg_spread:,.0f} coins\nSpread: {avg_spread_bps:.0f} bps",
                    inline=True
                )
                
                embed.add_field(
                    name="📈 Volume Summary",
                    value=f"Buy Volume: {total_buy_vol:,.0f}\nSell Volume: {total_sell_vol:,.0f}\nImbalance: {((total_buy_vol - total_sell_vol) / (total_buy_vol + total_sell_vol) * 100):+.1f}%",
                    inline=True
                )
            
            embed.set_image(url=f"attachment://{file.filename}")
            embed.set_footer(text="Enhanced Plot • 1-min median resampling • Spike filtering applied")
            
            await interaction.followup.send(embed=embed, file=file)
            self.logger.info(f"Successfully sent enhanced plot for {item}")
            
        except Exception as e:
            self.logger.error(f"Error in plot command for {item}: {e}")
            await interaction.followup.send(
                f"❌ An error occurred while generating the plot for **{item}**. "
                f"Please check the logs for details."
            )


async def setup(bot: commands.Bot):
    """Setup function for the cog."""
    await bot.add_cog(PlotCog(bot))
    logging.getLogger(__name__).info("Enhanced Plot cog loaded")