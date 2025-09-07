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
from typing import Dict, Any, Optional, List

import discord
from discord import app_commands
from discord.ext import commands, tasks
import yaml
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import numpy as np
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import existing project modules
try:
    # Data collection is now handled by standalone ingestion service
    # from ingestion.auction_collector import run as auction_collector_run
    # from ingestion.bazaar_collector import run as bazaar_collector_run
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

async def event_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Autocomplete function for event names."""
    try:
        events_path = Path("data/events.json")
        if not events_path.exists():
            return []
        
        with open(events_path, "r") as f:
            events = json.load(f)
        
        # Filter events based on current input
        choices = []
        for event_id, event_data in events.items():
            event_name = event_data.get('name', event_id)
            if current.lower() in event_id.lower() or current.lower() in event_name.lower():
                choices.append(app_commands.Choice(name=event_name, value=event_id))
        
        return choices[:25]  # Discord limit
    except Exception:
        return []

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class SkyBlockEconomyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        self.config = load_config()
        self.start_time = datetime.now(timezone.utc)
        
        # Data collection is now handled by standalone ingestion service
        # self.auction_collector_task = None
        # self.bazaar_collector_task = None
        
        self.last_feature_build = None
        
    async def setup_hook(self):
        """Set up the bot after login."""
        # Load auction sniper cog
        try:
            await self.load_extension("cogs.auction_sniper")
            logger.info("Loaded auction sniper cog")
        except Exception as e:
            logger.error(f"Failed to load auction sniper cog: {e}")
        
        # Start background tasks
        if IMPORTS_SUCCESS:
            # Data collection is now handled by standalone ingestion service
            # await self.start_data_collectors()
            self.feature_builder.start()
            logger.info("Started feature building background task")
            logger.info("Note: Data collection is handled by separate ingestion service")
        else:
            logger.warning("Skipped background tasks due to import failures")
        
        # Sync commands
        await self.tree.sync()
        logger.info("Synced slash commands")

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f'{self.user} has logged in and is ready!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')

    # Data collection is now handled by standalone ingestion service
    # async def start_data_collectors(self):
    #     """Starts the auction and bazaar data collectors in background threads."""
    #     if not os.getenv("HYPIXEL_API_KEY"):
    #         logger.warning("HYPIXEL_API_KEY not set. Skipping data collection.")
    #         return
    # 
    #     try:
    #         # Start auction collector in background
    #         logger.info("Starting auction collector in background...")
    #         self.auction_collector_task = asyncio.create_task(
    #             asyncio.to_thread(auction_collector_run)
    #         )
    #         
    #         # Start bazaar collector in background  
    #         logger.info("Starting bazaar collector in background...")
    #         self.bazaar_collector_task = asyncio.create_task(
    #             asyncio.to_thread(bazaar_collector_run)
    #         )
    #         
    #         logger.info("Data collection tasks have been launched.")
    #         
    #     except Exception as e:
    #         logger.error(f"Failed to start data collectors: {e}")

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
            # Continue running despite the error
    
    # Data collection shutdown is now handled by standalone ingestion service
    # async def shutdown_data_collectors(self):
    #     """Gracefully shut down data collection tasks."""
    #     logger.info("Shutting down data collection tasks...")
    #     
    #     if self.auction_collector_task and not self.auction_collector_task.done():
    #         self.auction_collector_task.cancel()
    #         try:
    #             await self.auction_collector_task
    #         except asyncio.CancelledError:
    #             logger.info("Auction collector task cancelled")
    #     
    #     if self.bazaar_collector_task and not self.bazaar_collector_task.done():
    #         self.bazaar_collector_task.cancel()
    #         try:
    #             await self.bazaar_collector_task
    #         except asyncio.CancelledError:
    #             logger.info("Bazaar collector task cancelled")
    #     
    #     # Stop feature builder task
    #     if self.feature_builder.is_running():
    #         self.feature_builder.cancel()
    #         logger.info("Feature builder task cancelled")
    #     
    #     logger.info("Data collection shutdown complete")

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
            # Check for Parquet datasets (new architecture)
            auction_parquet_dir = data_dir / "auction_history"
            bazaar_parquet_dir = data_dir / "bazaar_history"
            
            # Count auction records from Parquet
            if auction_parquet_dir.exists() and any(auction_parquet_dir.rglob("*.parquet")):
                try:
                    import pyarrow.parquet as pq
                    parquet_files = list(auction_parquet_dir.rglob("*.parquet"))
                    auction_records = sum(pq.read_table(f).num_rows for f in parquet_files[:10])  # Sample recent files
                except Exception as e:
                    logger.debug(f"Error reading auction Parquet data: {e}")
                    auction_records = len(list(auction_parquet_dir.rglob("*.parquet"))) * 1000  # Estimate
            
            # Count bazaar records from Parquet
            if bazaar_parquet_dir.exists() and any(bazaar_parquet_dir.rglob("*.parquet")):
                try:
                    import pyarrow.parquet as pq
                    parquet_files = list(bazaar_parquet_dir.rglob("*.parquet"))
                    bazaar_records = sum(pq.read_table(f).num_rows for f in parquet_files[:10])  # Sample recent files
                except Exception as e:
                    logger.debug(f"Error reading bazaar Parquet data: {e}")
                    bazaar_records = len(list(bazaar_parquet_dir.rglob("*.parquet"))) * 100  # Estimate
            
            # Fallback to old NDJSON files if Parquet doesn't exist
            if auction_records == 0:
                auction_file = data_dir / "auctions.ndjson"
                if auction_file.exists():
                    with open(auction_file, 'r') as f:
                        auction_records = sum(1 for _ in f)
            
            if bazaar_records == 0:
                bazaar_file = data_dir / "bazaar_snapshots.ndjson"
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
        
        # Run analysis using existing engine (in separate thread to avoid blocking)
        engine = FileBasedPredictiveMarketEngine()
        results = await asyncio.to_thread(engine.run_full_analysis, [item_name], model_type='lightgbm')
        
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

@bot.tree.command(name="compare", description="Compare two items for arbitrage opportunities")
@app_commands.describe(
    item_a="First item to compare (e.g., HYPERION)",
    item_b="Second item to compare (e.g., NECRON_CHESTPLATE)"
)
async def compare_command(interaction: discord.Interaction, item_a: str, item_b: str):
    """Compare two items for cross-item arbitrage analysis."""
    await interaction.response.defer()
    
    if not IMPORTS_SUCCESS:
        await interaction.followup.send("❌ Analysis features are not available.")
        return
    
    try:
        # Sanitize inputs
        item_a = item_a.upper().strip()
        item_b = item_b.upper().strip()
        
        # Basic validation
        for item in [item_a, item_b]:
            if not item or len(item) > 50:
                await interaction.followup.send("❌ Please provide valid item names (max 50 characters each)")
                return
            if not item.replace('_', '').replace('-', '').isalnum():
                await interaction.followup.send("❌ Item names can only contain letters, numbers, underscores, and hyphens")
                return
        
        if item_a == item_b:
            await interaction.followup.send("❌ Please provide two different items to compare")
            return
        
        logger.info(f"Running comparison analysis: {item_a} vs {item_b}")
        
        # Run cross-item analysis
        engine = FileBasedPredictiveMarketEngine()
        results = await asyncio.to_thread(engine.run_cross_item_analysis, item_a, item_b)
        
        # Create response embed
        embed = discord.Embed(
            title=f"⚖️ Comparison: {item_a} vs {item_b}",
            color=discord.Color.gold(),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add price predictions
        predictions = results.get('predictions', {})
        if predictions:
            pred_text = ""
            if item_a in predictions:
                pred_a = predictions[item_a]
                if pred_a:
                    best_pred_a = list(pred_a.values())[0]
                    pred_text += f"**{item_a}:** {best_pred_a:,.0f} coins\n"
            
            if item_b in predictions:
                pred_b = predictions[item_b]
                if pred_b:
                    best_pred_b = list(pred_b.values())[0]
                    pred_text += f"**{item_b}:** {best_pred_b:,.0f} coins\n"
            
            if pred_text:
                embed.add_field(
                    name="📈 Price Predictions",
                    value=pred_text,
                    inline=False
                )
        
        # Add comparison analysis
        comparison = results.get('comparison', {})
        if comparison:
            comp_text = f"**Analysis:** {comparison.get('price_comparison', 'Unknown')}\n"
            comp_text += f"**Relative Value:** {comparison.get('relative_value', 'Unknown')}\n"
            comp_text += f"**Recommendation:** {comparison.get('recommendation', 'Unknown')}"
            
            embed.add_field(
                name="🔍 Comparison Analysis",
                value=comp_text,
                inline=False
            )
        
        # Add arbitrage opportunities
        opportunities = results.get('arbitrage_opportunities', [])
        if opportunities:
            opp_text = ""
            for opp in opportunities[:3]:  # Show top 3
                opp_text += f"• {opp.get('description', 'Unknown opportunity')}\n"
            
            embed.add_field(
                name="💰 Arbitrage Opportunities",
                value=opp_text,
                inline=False
            )
        
        # Add correlation info
        correlation = results.get('correlation_analysis', {})
        if correlation.get('data_points', 0) > 0:
            corr_coef = correlation.get('correlation_coefficient', 0)
            strength = correlation.get('correlation_strength', 'unknown')
            data_points = correlation.get('data_points', 0)
            
            embed.add_field(
                name="📊 Price Correlation",
                value=f"**Coefficient:** {corr_coef:.3f}\n**Strength:** {strength.title()}\n**Data Points:** {data_points:,}",
                inline=True
            )
        
        embed.set_footer(text="Cross-item analysis completed")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Compare command failed: {e}")
        await interaction.followup.send(f"❌ Comparison analysis failed: {str(e)}")

@bot.tree.command(name="event_impact", description="Analyze impact of a historical SkyBlock event")
@app_commands.describe(event_name="Name of the event to analyze")
@app_commands.autocomplete(event_name=event_autocomplete)
async def event_impact_command(interaction: discord.Interaction, event_name: str):
    """Analyze the impact of a specific historical event."""
    await interaction.response.defer()
    
    if not IMPORTS_SUCCESS:
        await interaction.followup.send("❌ Analysis features are not available.")
        return
    
    try:
        # Load events to validate input
        events_path = Path("data/events.json")
        if not events_path.exists():
            await interaction.followup.send("❌ Events database not found.")
            return
        
        with open(events_path, "r") as f:
            events = json.load(f)
        
        # Find matching event (case-insensitive)
        matching_event = None
        for event_id in events.keys():
            if event_id.lower() == event_name.lower():
                matching_event = event_id
                break
            elif event_name.upper() in event_id.upper():
                matching_event = event_id
                break
        
        if not matching_event:
            available_events = ", ".join(list(events.keys())[:5])  # Show first 5
            await interaction.followup.send(f"❌ Event '{event_name}' not found. Available events: {available_events}...")
            return
        
        logger.info(f"Running event impact analysis for: {matching_event}")
        
        # Run event impact analysis
        engine = FileBasedPredictiveMarketEngine()
        results = await asyncio.to_thread(engine.run_event_impact_analysis, matching_event)
        
        event_data = results.get('event_data', {})
        impact_analysis = results.get('impact_analysis', {})
        
        # Create response embed
        embed = discord.Embed(
            title=f"📅 Event Impact: {event_data.get('name', matching_event)}",
            description=event_data.get('description', 'No description available'),
            color=discord.Color.purple(),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add event details
        event_type = event_data.get('type', 'Unknown')
        start_date = event_data.get('start_date', 'Unknown')
        end_date = event_data.get('end_date', 'Unknown')
        
        embed.add_field(
            name="📋 Event Details",
            value=f"**Type:** {event_type}\n**Start:** {start_date[:10]}\n**End:** {end_date[:10]}",
            inline=True
        )
        
        # Add impact effects
        effects = event_data.get('effects', {})
        if effects:
            effects_text = ""
            for category, impact in effects.items():
                sign = "+" if impact > 0 else ""
                effects_text += f"**{category.replace('_', ' ').title()}:** {sign}{impact*100:.1f}%\n"
            
            embed.add_field(
                name="📊 Market Effects",
                value=effects_text,
                inline=True
            )
        
        # Add affected items
        affected_items = event_data.get('affected_items', [])
        if affected_items:
            items_text = ", ".join(affected_items[:5])  # Show first 5
            if len(affected_items) > 5:
                items_text += f" +{len(affected_items) - 5} more"
            
            embed.add_field(
                name="🎯 Affected Items",
                value=items_text,
                inline=False
            )
        
        # Add recommendations
        recommendations = impact_analysis.get('recommendations', [])
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations[:3]])
            embed.add_field(
                name="💡 Recommendations",
                value=rec_text,
                inline=False
            )
        
        embed.set_footer(text="Event impact analysis completed")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Event impact command failed: {e}")
        await interaction.followup.send(f"❌ Event impact analysis failed: {str(e)}")

@bot.tree.command(name="market_pulse", description="Get holistic SkyBlock market overview and sentiment")
async def market_pulse_command(interaction: discord.Interaction):
    """Generate holistic market pulse analysis."""
    await interaction.response.defer()
    
    if not IMPORTS_SUCCESS:
        await interaction.followup.send("❌ Analysis features are not available.")
        return
    
    try:
        logger.info("Running market pulse analysis")
        
        # Run market pulse analysis
        engine = FileBasedPredictiveMarketEngine()
        results = await asyncio.to_thread(engine.run_market_pulse_analysis)
        
        market_sentiment = results.get('market_sentiment', {})
        basket_performance = results.get('basket_performance', {})
        market_insights = results.get('market_insights', [])
        
        # Create response embed
        sentiment_emoji = market_sentiment.get('emoji', '⚖️')
        sentiment_name = market_sentiment.get('sentiment', 'Neutral')
        
        embed = discord.Embed(
            title=f"🌡️ SkyBlock Market Pulse",
            description=f"Overall Market Sentiment: **{sentiment_name}** {sentiment_emoji}",
            color=discord.Color.green() if sentiment_name == 'Bullish' else 
                  discord.Color.red() if sentiment_name == 'Bearish' else discord.Color.light_grey(),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add sector performance
        if basket_performance:
            sector_text = ""
            sorted_baskets = sorted(basket_performance.items(), 
                                  key=lambda x: x[1].get('change_24h', 0), reverse=True)
            
            for basket_name, performance in sorted_baskets[:6]:  # Show top 6
                name = performance.get('name', basket_name.replace('_', ' ').title())
                change = performance.get('change_24h', 0)
                trend_emoji = "▲" if change > 0 else "▼" if change < 0 else "➡️"
                
                sector_text += f"• **{name}:** {trend_emoji} {change:+.1f}%\n"
            
            embed.add_field(
                name="📈 Sector Performance (24h)",
                value=sector_text or "No data available",
                inline=False
            )
        
        # Add market insights
        if market_insights:
            insights_text = "\n".join([f"• {insight}" for insight in market_insights[:4]])
            embed.add_field(
                name="🔍 Market Insights",
                value=insights_text,
                inline=False
            )
        
        # Add summary stats
        avg_change = market_sentiment.get('average_change', 0)
        positive_sectors = market_sentiment.get('positive_sectors', 0)
        total_sectors = market_sentiment.get('total_sectors', 0)
        
        embed.add_field(
            name="📊 Market Summary",
            value=f"**Average Change:** {avg_change:+.1f}%\n**Positive Sectors:** {positive_sectors}/{total_sectors}",
            inline=True
        )
        
        embed.set_footer(text="Market pulse analysis • Data may be limited without active trading data")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        logger.error(f"Market pulse command failed: {e}")
        await interaction.followup.send(f"❌ Market pulse analysis failed: {str(e)}")

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
        name="⚖️ `/compare <item_a> <item_b>`",
        value="Compare two items for arbitrage opportunities\n*Example: `/compare HYPERION NECRON_CHESTPLATE`*",
        inline=False
    )
    
    embed.add_field(
        name="📅 `/event_impact <event_name>`",
        value="Analyze impact of historical SkyBlock events\n*Example: `/event_impact MAYOR_DERPY`*",
        inline=False
    )
    
    embed.add_field(
        name="🌡️ `/market_pulse`",
        value="Get holistic market overview and sentiment analysis",
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
        bazaar_parquet_dir = data_dir / "bazaar_history"
        
        # Check for Parquet data first (new architecture)
        if bazaar_parquet_dir.exists() and any(bazaar_parquet_dir.rglob("*.parquet")):
            try:
                import pyarrow.parquet as pq
                
                # Read recent Parquet files
                parquet_files = list(bazaar_parquet_dir.rglob("*.parquet"))
                parquet_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
                
                dfs = []
                for file in parquet_files[:5]:  # Limit to recent files
                    try:
                        df = pq.read_table(file).to_pandas()
                        if 'product_id' in df.columns:
                            item_data = df[df['product_id'] == item_name]
                            if not item_data.empty:
                                dfs.append(item_data)
                    except Exception as e:
                        logger.debug(f"Error reading Parquet file {file}: {e}")
                        continue
                        
                if not dfs:
                    # Try partial matches if exact match fails
                    for file in parquet_files[:5]:
                        try:
                            df = pq.read_table(file).to_pandas()
                            if 'product_id' in df.columns:
                                # Case-insensitive partial match
                                item_data = df[df['product_id'].str.contains(item_name, case=False, na=False)]
                                if not item_data.empty:
                                    dfs.append(item_data)
                                    break  # Found data, stop searching
                        except Exception as e:
                            logger.debug(f"Error reading Parquet file {file}: {e}")
                            continue
                            
            except ImportError:
                await interaction.followup.send("❌ PyArrow is required for reading Parquet data but is not available.")
                return
        
        # Fallback to old NDJSON files if Parquet data not available
        if not dfs:
            bazaar_files = list(data_dir.glob("bazaar*.ndjson"))
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
            prices = 100 + np.cumsum(np.random.randn(100) * 2)  # Random walk
            
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
        logger.error(f"Failed to generate plot: {e}", exc_info=True)
        await interaction.followup.send("❌ An internal error occurred while generating the chart.")


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

