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
import json

import discord
from discord.ext import commands
from discord import app_commands
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import aiohttp

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
    
    async def _load_bazaar_data(self, item: str, hours: int = 3) -> Optional[pd.DataFrame]:
        """Load bazaar data for the specified item and time window with multiple fallbacks."""
        try:
            search_results = []  # Track what we searched and why it failed
            
            # Calculate time window
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            # Try each data source in priority order
            for i, data_path in enumerate(self.data_paths, 1):
                source_name = data_path.name if data_path.is_file() else data_path.name + "/"
                self.logger.debug(f"Checking data source {i}: {source_name}")
                
                if not data_path.exists():
                    search_results.append(f"❌ {source_name}: Path does not exist")
                    continue
                
                try:
                    df = None
                    files_found = 0
                    
                    if data_path.is_dir():
                        # Directory-based storage (parquet files)
                        parquet_files = list(data_path.glob("*.parquet"))
                        files_found = len(parquet_files)
                        
                        if not parquet_files:
                            search_results.append(f"❌ {source_name}: No parquet files found")
                            continue
                        
                        dfs = []
                        for file_path in parquet_files:
                            try:
                                file_df = pd.read_parquet(file_path)
                                dfs.append(file_df)
                            except Exception as e:
                                self.logger.error(f"Failed to read {file_path}: {e}")
                        
                        if not dfs:
                            search_results.append(f"❌ {source_name}: {files_found} files found but all failed to read")
                            continue
                            
                        df = pd.concat(dfs, ignore_index=True)
                        
                    else:
                        # NDJSON file
                        try:
                            # For NDJSON, we need to parse it more carefully for bazaar data
                            records = []
                            line_count = 0
                            
                            with open(data_path, 'r') as f:
                                for line in f:
                                    try:
                                        data = json.loads(line.strip())
                                        line_count += 1
                                        
                                        # Parse bazaar snapshot format
                                        if 'products' in data:
                                            timestamp = data.get('timestamp', data.get('lastUpdated'))
                                            for product_id, product_data in data['products'].items():
                                                record = {
                                                    'timestamp': timestamp,
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
                                files_found = line_count
                            else:
                                search_results.append(f"❌ {source_name}: {line_count} lines read but no valid bazaar data found")
                                continue
                                
                        except Exception as e:
                            search_results.append(f"❌ {source_name}: Failed to read NDJSON - {e}")
                            continue
                    
                    if df is None or df.empty:
                        search_results.append(f"❌ {source_name}: No data loaded from {files_found} files")
                        continue
                    
                    # Detect and normalize column names
                    timestamp_col = None
                    product_col = None
                    
                    # Find timestamp column
                    for col in ['timestamp', 'time', 'datetime', 'created_at', 'updated_at', 'ts']:
                        if col in df.columns:
                            timestamp_col = col
                            break
                    
                    if timestamp_col is None:
                        time_cols = [col for col in df.columns if 'time' in col.lower()]
                        if time_cols:
                            timestamp_col = time_cols[0]
                    
                    # Find product column
                    for col in ['product_id', 'productId', 'product', 'item_name', 'item', 'name']:
                        if col in df.columns:
                            product_col = col
                            break
                    
                    if timestamp_col is None:
                        search_results.append(f"❌ {source_name}: No timestamp column found in {list(df.columns)}")
                        continue
                    
                    if product_col is None:
                        search_results.append(f"❌ {source_name}: No product column found in {list(df.columns)}")
                        continue
                    
                    # Convert timestamp and filter by time
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
                    time_filtered_df = df[(df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)]
                    
                    if time_filtered_df.empty:
                        oldest_time = df[timestamp_col].min()
                        newest_time = df[timestamp_col].max()
                        search_results.append(f"❌ {source_name}: {len(df)} rows found but none in time window. Available: {oldest_time} to {newest_time}")
                        continue
                    
                    # Filter by item
                    normalized_item = self._normalize_item_name(item)
                    available_items = time_filtered_df[product_col].unique().tolist()
                    
                    # Try exact normalized match
                    item_mask = time_filtered_df[product_col].str.upper().str.replace(' ', '_').str.replace('-', '_') == normalized_item
                    item_data = time_filtered_df[item_mask]
                    
                    if item_data.empty:
                        # Try fuzzy matching for suggestions
                        suggestions = self._get_item_suggestions(item, available_items, max_suggestions=5)
                        search_results.append(f"❌ {source_name}: {len(time_filtered_df)} rows in time window but item '{item}' not found. Available items: {len(available_items)}. Suggestions: {', '.join(suggestions) if suggestions else 'None'}")
                        continue
                    
                    # Normalize column names to standard format
                    column_mapping = {timestamp_col: 'timestamp', product_col: 'product_id'}
                    
                    # Map price columns
                    for col in ['buy_price', 'buyPrice', 'buy_price_coins', 'instant_buy']:
                        if col in item_data.columns:
                            column_mapping[col] = 'buy_price'
                            break
                    
                    for col in ['sell_price', 'sellPrice', 'sell_price_coins', 'instant_sell']:
                        if col in item_data.columns:
                            column_mapping[col] = 'sell_price'
                            break
                    
                    if 'buy_price' not in column_mapping.values() or 'sell_price' not in column_mapping.values():
                        search_results.append(f"❌ {source_name}: Item found but missing required price columns. Available: {list(item_data.columns)}")
                        continue
                    
                    # Map volume and order columns (optional)
                    for col in ['buy_volume', 'buyVolume', 'buy_volume_coins']:
                        if col in item_data.columns:
                            column_mapping[col] = 'buy_volume'
                            break
                    
                    for col in ['sell_volume', 'sellVolume', 'sell_volume_coins']:
                        if col in item_data.columns:
                            column_mapping[col] = 'sell_volume'
                            break
                    
                    for col in ['buy_orders', 'buyOrders']:
                        if col in item_data.columns:
                            column_mapping[col] = 'buy_orders'
                            break
                    
                    for col in ['sell_orders', 'sellOrders']:
                        if col in item_data.columns:
                            column_mapping[col] = 'sell_orders'
                            break
                    
                    # Apply column mapping
                    item_data = item_data.rename(columns=column_mapping)
                    
                    # Fill missing volume/order columns with defaults
                    for col in ['buy_volume', 'sell_volume', 'buy_orders', 'sell_orders']:
                        if col not in item_data.columns:
                            item_data[col] = 0
                    
                    # Success!
                    coverage_hours = (item_data['timestamp'].max() - item_data['timestamp'].min()).total_seconds() / 3600
                    search_results.append(f"✅ {source_name}: Found {len(item_data)} records for {item}, coverage: {coverage_hours:.1f}h")
                    
                    self.logger.info(f"Successfully loaded {len(item_data)} records for {item} from {source_name}")
                    self._last_search_results = search_results  # Store for error reporting
                    return item_data.sort_values('timestamp')
                    
                except Exception as e:
                    search_results.append(f"❌ {source_name}: Exception - {str(e)}")
                    self.logger.error(f"Error loading from {data_path}: {e}")
                    continue
            
            # All local sources failed, try live API fallback
            try:
                live_data = await self._fetch_live_bazaar_data(item)
                if live_data is not None and not live_data.empty:
                    search_results.append(f"✅ Live API: Retrieved current bazaar data for {item}")
                    self._last_search_results = search_results
                    return live_data
                else:
                    search_results.append(f"❌ Live API: No data returned for {item}")
            except Exception as e:
                search_results.append(f"❌ Live API: {str(e)}")
            
            # Store search results for error reporting
            self._last_search_results = search_results
            self.logger.warning(f"Failed to load data for {item} from any source")
            return None
            
        except Exception as e:
            self.logger.error(f"Critical error in _load_bazaar_data: {e}")
            return None
    
    async def _fetch_live_bazaar_data(self, item: str) -> Optional[pd.DataFrame]:
        """Fetch current bazaar data from live Hypixel API as fallback."""
        try:
            # Get API key from environment
            import os
            api_key = os.getenv('HYPIXEL_API_KEY')
            if not api_key:
                self.logger.warning("No HYPIXEL_API_KEY available for live API fallback")
                return None
            
            # Fetch live bazaar data
            url = f"https://api.hypixel.net/skyblock/bazaar"
            params = {'key': api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        self.logger.error(f"Live API request failed: {response.status}")
                        return None
                    
                    data = await response.json()
                    
                    if not data.get('success', False):
                        self.logger.error(f"Live API returned error: {data}")
                        return None
                    
                    products = data.get('products', {})
                    normalized_item = self._normalize_item_name(item)
                    
                    # Find the item in the response
                    product_data = None
                    for product_id, prod_data in products.items():
                        if self._normalize_item_name(product_id) == normalized_item:
                            product_data = prod_data
                            break
                    
                    if product_data is None:
                        self.logger.warning(f"Item {item} not found in live bazaar data")
                        return None
                    
                    # Create a single-row DataFrame with current data
                    current_time = datetime.now(timezone.utc)
                    
                    record = {
                        'timestamp': current_time,
                        'product_id': item,
                        'buy_price': product_data.get('buy_summary', [{}])[0].get('pricePerUnit', 0) if product_data.get('buy_summary') else 0,
                        'sell_price': product_data.get('sell_summary', [{}])[0].get('pricePerUnit', 0) if product_data.get('sell_summary') else 0,
                        'buy_volume': product_data.get('quick_status', {}).get('buyMovingWeek', 0),
                        'sell_volume': product_data.get('quick_status', {}).get('sellMovingWeek', 0),
                        'buy_orders': product_data.get('quick_status', {}).get('buyOrders', 0),
                        'sell_orders': product_data.get('quick_status', {}).get('sellOrders', 0)
                    }
                    
                    # Create DataFrame with just this single point
                    df = pd.DataFrame([record])
                    
                    self.logger.info(f"Retrieved live bazaar data for {item}")
                    return df
        
        except Exception as e:
            self.logger.error(f"Failed to fetch live bazaar data: {e}")
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
            # Check if DataFrame is empty or has insufficient data
            if df.empty:
                self.logger.warning(f"Cannot create plot for {item}: DataFrame is empty")
                return None
            
            if len(df) < 2:
                self.logger.warning(f"Cannot create plot for {item}: Only {len(df)} data point(s), need at least 2")
                return None
            
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
            df = await self._load_bazaar_data(item, hours)
            
            if df is None or df.empty:
                # Get detailed search results for error reporting
                search_results = getattr(self, '_last_search_results', [])
                
                embed = discord.Embed(
                    title=f"❌ No Data Found: {item}",
                    color=0xff0000,
                    description=f"No bazaar data found for **{item}** in the last {hours} hours."
                )
                
                if search_results:
                    # Show what was searched and why it failed
                    search_text = "\n".join(search_results[-10:])  # Last 10 results
                    embed.add_field(
                        name="🔍 Search Results",
                        value=f"```\n{search_text}\n```",
                        inline=False
                    )
                
                # Add suggestions if available
                if hasattr(self, '_last_available_items'):
                    suggestions = self._get_item_suggestions(item, self._last_available_items)
                    if suggestions:
                        embed.add_field(
                            name="💡 Did you mean?",
                            value="\n".join([f"• `{suggestion}`" for suggestion in suggestions]),
                            inline=False
                        )
                
                # Add common examples
                embed.add_field(
                    name="📝 Examples",
                    value="`WHEAT`, `ENCHANTED_FLINT`, `COBBLESTONE`, `EMERALD`",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed)
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
                # Check if this was due to insufficient data
                if len(df_enhanced) < 2:
                    await interaction.followup.send(
                        f"❌ Insufficient data for **{item}**. Found {len(df_enhanced)} data point(s), but need at least 2 to create a chart. Try a longer time window."
                    )
                else:
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