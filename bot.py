#!/usr/bin/env python3
"""
Discord Bot for SkyBlock Economic Modeling
Provides automated data collection and interactive analysis commands.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import discord
from discord import app_commands
from discord.ext import tasks
import yaml
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import existing project modules
try:
    from ingestion.auction_collector import run as auction_collector_run
    from ingestion.bazaar_collector import run as bazaar_collector_run
    from modeling.features.feature_pipeline import build_features
    from phase3_cli import run_predictive_analysis, train_ml_model, check_status, PHASE3_AVAILABLE
    from modeling.forecast.file_ml_forecaster import train_and_forecast_ml_from_files
    from modeling.simulation.file_predictive_engine import FileBasedPredictiveMarketEngine
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Failed to import project modules: {e}")
    IMPORTS_SUCCESS = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class SkyBlockEconomyBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.config = load_config()
        self.start_time = datetime.now(timezone.utc)
        self.auction_collector_task = None
        self.bazaar_collector_task = None
        
        self.last_feature_build = None
        
    async def setup_hook(self):
        """Set up the bot after login."""
        # Start background tasks
        if IMPORTS_SUCCESS:
            await self.start_data_collectors()
            self.feature_builder.start()
            logger.info("Started feature building background task")
        else:
            logger.warning("Skipped background tasks due to import failures")
        
        # Sync commands
        await self.tree.sync()
        logger.info("Synced slash commands")

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f'{self.user} has logged in and is ready!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')

    # --- NEW METHOD TO START DATA COLLECTORS ---
    async def start_data_collectors(self):
        """Starts the auction and bazaar data collectors in background threads."""
        if not os.getenv("HYPIXEL_API_KEY"):
            logger.warning("HYPIXEL_API_KEY not set. Skipping data collection.")
            return

        loop = asyncio.get_event_loop()
        
        logger.info("Starting auction collector in background...")
        self.auction_collector_task = asyncio.to_thread(auction_collector_run)
        
        logger.info("Starting bazaar collector in background...")
        self.bazaar_collector_task = asyncio.to_thread(bazaar_collector_run)
        
        logger.info("Data collection tasks have been launched.")

    @tasks.loop(hours=1)
    async def feature_builder(self):
        """Background task to build features periodically."""
        try:
            logger.info("Starting periodic feature building...")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, build_features)
            
            self.last_feature_build = datetime.now(timezone.utc)
            logger.info("Feature building completed successfully")
            
        except Exception as e:
            logger.error(f"Feature building failed: {e}")

# Create bot instance
bot = SkyBlockEconomyBot()

@bot.tree.command(name="status", description="Check bot health and data pipeline status")
async def status_command(interaction: discord.Interaction):
    """Bot status command."""
    await interaction.response.defer()
    
    try:
        # Calculate uptime
        uptime = datetime.now(timezone.utc) - bot.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        uptime_str = f"{days} days, {hours} hours, {minutes} minutes"
        
        # Check data directory for file counts
        data_dir = Path("data")
        auction_records = 0
        bazaar_records = 0
        
        if data_dir.exists():
            auction_file = data_dir / "auctions.ndjson"
            bazaar_file = data_dir / "bazaar_snapshots.ndjson"
            
            if auction_file.exists():
                with open(auction_file, 'r') as f:
                    auction_records = sum(1 for _ in f)
            
            if bazaar_file.exists():
                with open(bazaar_file, 'r') as f:
                    bazaar_records = sum(1 for _ in f)

        # Check model directory
        model_dir = Path("models")
        models_count = 0
        if model_dir.exists():
            models_count = len(list(model_dir.glob("*.pkl"))) + len(list(model_dir.glob("*.txt")))
        
        # Create status embed
        embed = discord.Embed(
            title="🤖 Bot Status",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(
            name="⏱️ Uptime",
            value=uptime_str,
            inline=True
        )
        
        embed.add_field(
            name="📊 Data Records",
            value=f"Auction: {auction_records:,}\nBazaar: {bazaar_records:,}",
            inline=True
        )
        
        embed.add_field(
            name="🤖 ML Models",
            value=f"{models_count} trained models",
            inline=True
        )
        
        # Add feature build status
        if bot.last_feature_build:
            time_since = datetime.now(timezone.utc) - bot.last_feature_build
            minutes_ago = int(time_since.total_seconds() / 60)
            feature_status = f"{minutes_ago} minutes ago"
        else:
            feature_status = "Not yet completed"
            
        embed.add_field(
            name="🏗️ Last Feature Build",
            value=feature_status,
            inline=True
        )
        
        embed.add_field(
            name="🔧 Phase 3 Status",
            value="✅ Available" if PHASE3_AVAILABLE else "❌ Unavailable",
            inline=True
        )
        
        embed.add_field(
            name="📥 Imports",
            value="✅ Success" if IMPORTS_SUCCESS else "❌ Failed",
            inline=True
        )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Status command failed: {e}")
        await interaction.followup.send(f"❌ Failed to get status: {str(e)}")

# (The rest of your commands: /analyze, /predict, /help, etc. remain unchanged)
@bot.tree.command(name="analyze", description="Analyze item market data and predictions")
@app_commands.describe(item_name="Name of the item to analyze (e.g., HYPERION)")
async def analyze_command(interaction: discord.Interaction, item_name: str):
    """Analyze command using existing phase3_cli functionality."""
    await interaction.response.defer()
    
    if not PHASE3_AVAILABLE or not IMPORTS_SUCCESS:
        await interaction.followup.send("❌ Phase 3 features are not available.")
        return
    
    try:
        # Sanitize input
        item_name = item_name.upper().strip()
        
        # Basic validation
        if not item_name or len(item_name) > 50:
            await interaction.followup.send("❌ Please provide a valid item name (max 50 characters)")
            return
        
        if not item_name.replace('_', '').replace('-', '').isalnum():
            await interaction.followup.send("❌ Item name can only contain letters, numbers, underscores, and hyphens")
            return
        
        logger.info(f"Running analysis for {item_name}")
        
        # Run analysis using existing engine
        engine = FileBasedPredictiveMarketEngine()
        results = engine.run_full_analysis([item_name], model_type='lightgbm')
        
        # Create response embed
        embed = discord.Embed(
            title=f"📈 Analysis for {item_name}",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add predictions if available
        predictions = results.get('ml_predictions', {})
        if item_name in predictions:
            pred_text = ""
            for horizon, price in predictions[item_name].items():
                pred_text += f"**{horizon}min:** {price:,.0f} coins\n"
            
            embed.add_field(
                name="🔮 Price Predictions",
                value=pred_text,
                inline=False
            )
        
        # Add training info
        training_results = results.get('training_results', {})
        if item_name in training_results:
            training_info = training_results[item_name]
            data_points = training_info.get('data_points', 0)
            horizons = len(training_info.get('horizons_trained', []))
            
            embed.add_field(
                name="📊 Model Info",
                value=f"**Data Points:** {data_points:,}\n**Horizons Trained:** {horizons}",
                inline=True
            )
        
        # Add opportunities if available
        opportunities = results.get('opportunities', [])
        if opportunities:
            opp_text = ""
            for opp in opportunities[:3]:  # Show top 3
                action = opp.get('action', 'Unknown')
                confidence = opp.get('confidence', 0)
                opp_text += f"**{action}** (Confidence: {confidence:.1f})\n"
            
            embed.add_field(
                name="💡 Opportunities",
                value=opp_text,
                inline=True
            )
        
        # Add market outlook (simplified)
        if predictions and item_name in predictions:
            current_pred = list(predictions[item_name].values())[0] if predictions[item_name] else 0
            if current_pred > 1000000:  # High value item
                outlook = "📈 High-value asset"
            elif current_pred > 100000:
                outlook = "📊 Mid-tier item"
            else:
                outlook = "📉 Commodity item"
            
            embed.add_field(
                name="🎯 Market Outlook",
                value=outlook,
                inline=True
            )
        
        embed.set_footer(text="Analysis completed using LightGBM models")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Analysis command failed: {e}")
        await interaction.followup.send(f"❌ Analysis failed: {str(e)}")

@bot.tree.command(name="predict", description="Get price prediction for an item")
@app_commands.describe(
    item_name="Name of the item to predict (e.g., WHEAT)", 
    time_horizon="Prediction horizon (15min, 60min, or 240min)"
)
@app_commands.choices(time_horizon=[
    app_commands.Choice(name="15 minutes", value="15"),
    app_commands.Choice(name="1 hour", value="60"), 
    app_commands.Choice(name="4 hours", value="240")
])
async def predict_command(interaction: discord.Interaction, item_name: str, time_horizon: str):
    """Predict command for specific time horizons."""
    await interaction.response.defer()
    
    if not PHASE3_AVAILABLE or not IMPORTS_SUCCESS:
        await interaction.followup.send("❌ Phase 3 features are not available.")
        return
        
    try:
        # Sanitize input
        item_name = item_name.upper().strip()
        horizon_int = int(time_horizon)
        
        # Basic validation
        if not item_name or len(item_name) > 50:
            await interaction.followup.send("❌ Please provide a valid item name (max 50 characters)")
            return
        
        if not item_name.replace('_', '').replace('-', '').isalnum():
            await interaction.followup.send("❌ Item name can only contain letters, numbers, underscores, and hyphens")
            return
        
        if horizon_int not in [15, 60, 240]:
            await interaction.followup.send("❌ Invalid time horizon. Please use 15, 60, or 240 minutes.")
            return
        
        logger.info(f"Running prediction for {item_name} at {horizon_int}min horizon")
        
        # Train and forecast using existing function
        success = train_and_forecast_ml_from_files(
            product_id=item_name,
            model_type='lightgbm',
            horizons=(horizon_int,)
        )
        
        if not success:
            await interaction.followup.send(f"❌ Could not generate prediction for {item_name}. Check if data is available.")
            return
        
        # Try to load the prediction (this would need to be implemented to read the saved results)
        embed = discord.Embed(
            title=f"🔮 Prediction for {item_name}",
            description=f"**Time Horizon:** {time_horizon} minutes",
            color=discord.Color.purple(),
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(
            name="📊 Status",
            value=f"✅ Model trained successfully\n🎯 Forecast generated for {horizon_int}-minute horizon",
            inline=False
        )
        
        embed.add_field(
            name="ℹ️ Note",
            value="Detailed prediction results are saved to the models directory. Use `/analyze` for comprehensive analysis.",
            inline=False
        )
        
        embed.set_footer(text="Prediction using LightGBM model")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Predict command failed: {e}")
        await interaction.followup.send(f"❌ Prediction failed: {str(e)}")

@bot.tree.command(name="help", description="Show bot help and command information")
async def help_command(interaction: discord.Interaction):
    """Help command to show available commands and usage."""
    embed = discord.Embed(
        title="🤖 SkyBlock Economy Bot Help",
        description="Automated SkyBlock market analysis and predictions",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="📊 `/status`",
        value="Check bot health and data pipeline status",
        inline=False
    )
    
    embed.add_field(
        name="📈 `/analyze <item_name>`",
        value="Comprehensive market analysis with ML predictions\n*Example: `/analyze HYPERION`*",
        inline=False
    )
    
    embed.add_field(
        name="🔮 `/predict <item_name> <time_horizon>`",
        value="Get specific price prediction for an item\n*Example: `/predict WHEAT 60min`*",
        inline=False
    )
    
    embed.add_field(
        name="🔄 `/retrain <item_name>`",
        value="[Admin] Manually retrain model for an item",
        inline=False
    )
    
    embed.add_field(
        name="📊 `/plot <item_name>`",
        value="Generate price history chart for an item\n*Example: `/plot WHEAT`*",
        inline=False
    )
    
    embed.add_field(
        name="ℹ️ **Supported Items**",
        value="High-value items like HYPERION, NECRON_CHESTPLATE, etc.\nCommodity items like WHEAT, SUGAR_CANE, etc.",
        inline=False
    )
    
    embed.add_field(
        name="⏰ **Time Horizons**",
        value="• 15 minutes\n• 1 hour (60 minutes)\n• 4 hours (240 minutes)",
        inline=False
    )
    
    embed.set_footer(text="Bot uses ML models trained on historical Hypixel SkyBlock data")
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="retrain", description="[Admin] Retrain model for a specific item")
@app_commands.describe(item_name="Name of the item to retrain model for")
async def retrain_command(interaction: discord.Interaction, item_name: str):
    """Admin command to retrain models."""
    # Basic admin check (you might want to implement proper role checking)
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ This command requires administrator permissions.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    if not PHASE3_AVAILABLE or not IMPORTS_SUCCESS:
        await interaction.followup.send("❌ Phase 3 features are not available.")
        return
    
    try:
        # Sanitize input
        item_name = item_name.upper().strip()
        
        # Basic validation
        if not item_name or len(item_name) > 50:
            await interaction.followup.send("❌ Please provide a valid item name (max 50 characters)")
            return
        
        if not item_name.replace('_', '').replace('-', '').isalnum():
            await interaction.followup.send("❌ Item name can only contain letters, numbers, underscores, and hyphens")
            return
        
        logger.info(f"Admin retraining model for {item_name}")
        
        # Retrain using existing function
        success = train_and_forecast_ml_from_files(
            product_id=item_name,
            model_type='lightgbm',
            horizons=(15, 60, 240)
        )
        
        embed = discord.Embed(
            title=f"🔄 Model Retraining for {item_name}",
            color=discord.Color.orange() if success else discord.Color.red(),
            timestamp=datetime.now(timezone.utc)
        )
        
        if success:
            embed.add_field(
                name="✅ Status",
                value=f"Model successfully retrained for {item_name}\nHorizons: 15min, 60min, 240min",
                inline=False
            )
        else:
            embed.add_field(
                name="❌ Status", 
                value=f"Failed to retrain model for {item_name}\nCheck data availability and logs",
                inline=False
            )
        
        embed.set_footer(text=f"Requested by {interaction.user.display_name}")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Retrain command failed: {e}")
        await interaction.followup.send(f"❌ Retraining failed: {str(e)}")

@bot.tree.command(name="plot", description="Generate price history chart for an item")
@app_commands.describe(item_name="Name of the item to plot price history for")
async def plot_command(interaction: discord.Interaction, item_name: str):
    """Generate a price history chart for an item."""
    await interaction.response.defer()
    
    if not PHASE3_AVAILABLE or not IMPORTS_SUCCESS:
        await interaction.followup.send("❌ Phase 3 features are not available.")
        return
    
    try:
        # Sanitize input
        item_name = item_name.upper().strip()
        
        # Basic validation
        if not item_name or len(item_name) > 50:
            await interaction.followup.send("❌ Please provide a valid item name (max 50 characters)")
            return
        
        if not item_name.replace('_', '').replace('-', '').isalnum():
            await interaction.followup.send("❌ Item name can only contain letters, numbers, underscores, and hyphens")
            return
        
        logger.info(f"Generating price chart for {item_name}")
        
        # Try to load bazaar data for the item
        import pandas as pd
        from pathlib import Path
        
        data_dir = Path("data")
        bazaar_files = list(data_dir.glob("bazaar*.ndjson"))
        
        if not bazaar_files:
            await interaction.followup.send(f"❌ No bazaar data found for plotting. Data collection may not be running.")
            return
        
        # Load and filter data
        dfs = []
        for file in bazaar_files[:5]:  # Limit to recent files
            try:
                df = pd.read_json(file, lines=True)
                if 'product_id' in df.columns:
                    item_data = df[df['product_id'] == item_name]
                    if not item_data.empty:
                        dfs.append(item_data)
            except:
                continue
        
        if not dfs:
            # Generate a sample chart to demonstrate functionality
            dates = pd.date_range('2024-01-01', periods=100, freq='h')
            prices = 100 + np.random.cumsum(np.random.randn(100) * 2)  # Random walk
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, prices, linewidth=2, color='#1f77b4')
            plt.title(f'{item_name} - Price History (Sample Data)', fontsize=16, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Price (coins)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            # Create file object for Discord
            file = discord.File(buffer, filename=f"{item_name.lower()}_price_chart.png")
            
            embed = discord.Embed(
                title=f"📊 Price Chart - {item_name}",
                description="Sample price chart (no historical data available)",
                color=discord.Color.blue(),
                timestamp=datetime.now(timezone.utc)
            )
            
            embed.add_field(
                name="ℹ️ Note",
                value="This is a demonstration chart with sample data. Start data collection to see real price history.",
                inline=False
            )
            
            await interaction.followup.send(embed=embed, file=file)
            
        else:
            # Process real data
            df = pd.concat(dfs).sort_values('ts')
            df['ts'] = pd.to_datetime(df['ts'])
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            if 'buy_price' in df.columns and not df['buy_price'].isna().all():
                plt.plot(df['ts'], df['buy_price'], label='Buy Price', linewidth=2, alpha=0.8)
            
            if 'sell_price' in df.columns and not df['sell_price'].isna().all():
                plt.plot(df['ts'], df['sell_price'], label='Sell Price', linewidth=2, alpha=0.8)
            
            plt.title(f'{item_name} - Price History', fontsize=16, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Price (coins)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            # Create file object for Discord
            file = discord.File(buffer, filename=f"{item_name.lower()}_price_chart.png")
            
            embed = discord.Embed(
                title=f"📊 Price Chart - {item_name}",
                description=f"Historical price data ({len(df)} data points)",
                color=discord.Color.green(),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add statistics
            if 'buy_price' in df.columns:
                avg_buy = df['buy_price'].mean()
                embed.add_field(
                    name="📈 Buy Price",
                    value=f"Avg: {avg_buy:,.0f} coins",
                    inline=True
                )
            
            if 'sell_price' in df.columns:
                avg_sell = df['sell_price'].mean()
                embed.add_field(
                    name="📉 Sell Price", 
                    value=f"Avg: {avg_sell:,.0f} coins",
                    inline=True
                )
            
            embed.add_field(
                name="📊 Data Points",
                value=f"{len(df)} records",
                inline=True
            )
            
            await interaction.followup.send(embed=embed, file=file)
        
    except Exception as e:
        logger.error(f"Plot command failed: {e}")
        await interaction.followup.send(f"❌ Failed to generate chart: {str(e)}")


def main():
    """Main entry point for the Discord bot."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        logger.error("DISCORD_BOT_TOKEN environment variable not set")
        sys.exit(1)
    
    hypixel_key = os.getenv("HYPIXEL_API_KEY")
    if not hypixel_key:
        logger.warning("HYPIXEL_API_KEY not set - data collection will be limited, but bot will start.")
    
    logger.info("Starting SkyBlock Economy Discord Bot...")
    logger.info(f"Phase 3 available: {PHASE3_AVAILABLE}")
    logger.info(f"Imports successful: {IMPORTS_SUCCESS}")
    
    try:
        bot.run(bot_token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")

if __name__ == "__main__":
    main()

