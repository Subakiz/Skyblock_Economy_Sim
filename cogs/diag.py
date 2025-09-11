#!/usr/bin/env python3
"""
Diagnostics Cog for Discord Bot

Provides /diag command group for troubleshooting:
- /diag features: feature summaries status
- /diag bazaar: bazaar data status  
- /diag env: environment variables and versions
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
import json

import discord
from discord.ext import commands
from discord import app_commands
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DiagnosticsCog(commands.Cog):
    """Administrative diagnostics for troubleshooting the market pipeline."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.DiagnosticsCog")
        
        # Load config
        try:
            with open("config/config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        self.logger.info("Diagnostics cog initialized")
    
    def _scan_feature_summaries(self) -> Dict[str, Any]:
        """Scan feature summaries directory and return status."""
        try:
            market_config = self.config.get("market", {})
            summaries_path = Path(market_config.get("feature_summaries_path", "data/feature_summaries"))
            
            if not summaries_path.exists():
                return {
                    "status": "missing",
                    "path": str(summaries_path),
                    "message": "Feature summaries directory does not exist"
                }
            
            # Scan for parquet files
            parquet_files = list(summaries_path.rglob("*.parquet"))
            
            if not parquet_files:
                return {
                    "status": "empty",
                    "path": str(summaries_path),
                    "total_files": 0,
                    "message": "No parquet files found in feature summaries directory"
                }
            
            # Analyze files
            earliest_time = None
            latest_time = None
            recent_files = []
            total_items = 0
            
            two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
            
            for file_path in parquet_files:
                try:
                    # Parse time from path structure year=/month=/day=/hour=
                    parts = file_path.parts
                    year = month = day = hour = None
                    
                    for part in parts:
                        if part.startswith("year="):
                            year = int(part.split("=")[1])
                        elif part.startswith("month="):
                            month = int(part.split("=")[1])
                        elif part.startswith("day="):
                            day = int(part.split("=")[1])
                        elif part.startswith("hour="):
                            hour = int(part.split("=")[1])
                    
                    if all(x is not None for x in [year, month, day, hour]):
                        file_time = datetime(year, month, day, hour, tzinfo=timezone.utc)
                        
                        if earliest_time is None or file_time < earliest_time:
                            earliest_time = file_time
                        if latest_time is None or file_time > latest_time:
                            latest_time = file_time
                        
                        # Check if file is recent
                        if file_time >= two_hours_ago:
                            try:
                                df = pd.read_parquet(file_path)
                                item_count = len(df)
                                total_items += item_count
                                recent_files.append({
                                    "time": file_time,
                                    "path": str(file_path.relative_to(summaries_path)),
                                    "items": item_count
                                })
                            except Exception as e:
                                recent_files.append({
                                    "time": file_time,
                                    "path": str(file_path.relative_to(summaries_path)),
                                    "error": str(e)
                                })
                
                except Exception as e:
                    self.logger.error(f"Failed to parse file {file_path}: {e}")
                    continue
            
            return {
                "status": "ok",
                "path": str(summaries_path),
                "total_files": len(parquet_files),
                "earliest_time": earliest_time,
                "latest_time": latest_time,
                "recent_files": recent_files,
                "recent_items": total_items
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error scanning feature summaries: {e}"
            }
    
    def _scan_bazaar_data(self) -> Dict[str, Any]:
        """Scan bazaar data sources and return status."""
        try:
            sources = []
            three_hours_ago = datetime.now(timezone.utc) - timedelta(hours=3)
            
            # Check bazaar_history
            bazaar_history_path = Path("data/bazaar_history")
            if bazaar_history_path.exists():
                try:
                    files = list(bazaar_history_path.glob("*.parquet"))
                    total_rows = 0
                    recent_rows = 0
                    
                    for file_path in files:
                        try:
                            df = pd.read_parquet(file_path)
                            total_rows += len(df)
                            
                            # Check for recent data
                            for ts_col in ['ts', 'timestamp', 'time', 'created_at']:
                                if ts_col in df.columns:
                                    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                                    recent_data = df[df[ts_col] >= three_hours_ago]
                                    recent_rows += len(recent_data)
                                    break
                        except Exception as e:
                            self.logger.error(f"Failed to read {file_path}: {e}")
                    
                    sources.append({
                        "name": "bazaar_history",
                        "path": str(bazaar_history_path),
                        "exists": True,
                        "files": len(files),
                        "total_rows": total_rows,
                        "recent_rows": recent_rows
                    })
                except Exception as e:
                    sources.append({
                        "name": "bazaar_history",
                        "path": str(bazaar_history_path),
                        "exists": True,
                        "error": str(e)
                    })
            else:
                sources.append({
                    "name": "bazaar_history",
                    "path": str(bazaar_history_path),
                    "exists": False
                })
            
            # Check bazaar
            bazaar_path = Path("data/bazaar")
            if bazaar_path.exists():
                try:
                    files = list(bazaar_path.glob("*.parquet"))
                    total_rows = 0
                    
                    for file_path in files[:5]:  # Sample first 5 files
                        try:
                            df = pd.read_parquet(file_path)
                            total_rows += len(df)
                        except Exception:
                            pass
                    
                    sources.append({
                        "name": "bazaar",
                        "path": str(bazaar_path),
                        "exists": True,
                        "files": len(files),
                        "sample_rows": total_rows
                    })
                except Exception as e:
                    sources.append({
                        "name": "bazaar",
                        "path": str(bazaar_path),
                        "exists": True,
                        "error": str(e)
                    })
            else:
                sources.append({
                    "name": "bazaar",
                    "path": str(bazaar_path),
                    "exists": False
                })
            
            # Check NDJSON
            ndjson_path = Path("data/bazaar_snapshots.ndjson")
            if ndjson_path.exists():
                try:
                    file_size = ndjson_path.stat().st_size
                    modified_time = datetime.fromtimestamp(ndjson_path.stat().st_mtime, tz=timezone.utc)
                    
                    # Sample first few lines to get products
                    sample_products = set()
                    line_count = 0
                    
                    with open(ndjson_path, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if 'products' in data:
                                    sample_products.update(data['products'].keys())
                                line_count += 1
                                if line_count >= 10:  # Sample first 10 lines
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    sources.append({
                        "name": "bazaar_snapshots.ndjson",
                        "path": str(ndjson_path),
                        "exists": True,
                        "size_mb": file_size / (1024 * 1024),
                        "modified": modified_time,
                        "sample_products": len(sample_products),
                        "sample_lines": line_count
                    })
                except Exception as e:
                    sources.append({
                        "name": "bazaar_snapshots.ndjson",
                        "path": str(ndjson_path),
                        "exists": True,
                        "error": str(e)
                    })
            else:
                sources.append({
                    "name": "bazaar_snapshots.ndjson",
                    "path": str(ndjson_path),
                    "exists": False
                })
            
            return {
                "status": "ok",
                "sources": sources
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error scanning bazaar data: {e}"
            }
    
    def _check_environment(self) -> Dict[str, Any]:
        """Check environment variables and system info."""
        try:
            env_info = {
                "python_version": sys.version,
                "working_directory": str(Path.cwd())
            }
            
            # Check environment variables (mask sensitive parts)
            env_vars = {}
            
            discord_token = os.getenv("DISCORD_BOT_TOKEN")
            if discord_token:
                env_vars["DISCORD_BOT_TOKEN"] = f"{discord_token[:10]}...{discord_token[-4:]}" if len(discord_token) > 14 else "***"
            else:
                env_vars["DISCORD_BOT_TOKEN"] = "Not set"
            
            hypixel_key = os.getenv("HYPIXEL_API_KEY")
            if hypixel_key:
                env_vars["HYPIXEL_API_KEY"] = f"{hypixel_key[:8]}...{hypixel_key[-4:]}" if len(hypixel_key) > 12 else "***"
            else:
                env_vars["HYPIXEL_API_KEY"] = "Not set"
            
            # Check package versions
            packages = {}
            try:
                import pandas
                packages["pandas"] = pandas.__version__
            except ImportError:
                packages["pandas"] = "Not installed"
            
            try:
                import pyarrow
                packages["pyarrow"] = pyarrow.__version__
            except ImportError:
                packages["pyarrow"] = "Not installed"
            
            try:
                import matplotlib
                packages["matplotlib"] = matplotlib.__version__
            except ImportError:
                packages["matplotlib"] = "Not installed"
            
            try:
                import discord
                packages["discord.py"] = discord.__version__
            except ImportError:
                packages["discord.py"] = "Not installed"
            
            return {
                "status": "ok",
                "environment": env_vars,
                "system": env_info,
                "packages": packages
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking environment: {e}"
            }
    
    @app_commands.command(name="diag_features", description="Diagnostics for feature summaries")
    async def diag_features(self, interaction: discord.Interaction):
        """Diagnose feature summaries status."""
        await interaction.response.defer()
        
        try:
            self.logger.info("Running feature summaries diagnostics")
            result = self._scan_feature_summaries()
            
            embed = discord.Embed(
                title="🔍 Feature Summaries Diagnostics",
                color=0x00ff00 if result["status"] == "ok" else 0xff0000
            )
            
            embed.add_field(
                name="📁 Path",
                value=f"`{result.get('path', 'Unknown')}`",
                inline=False
            )
            
            if result["status"] == "ok":
                embed.add_field(
                    name="📊 Summary",
                    value=f"**Total Files:** {result['total_files']}\n"
                          f"**Recent Files:** {len(result['recent_files'])}\n"
                          f"**Recent Items:** {result['recent_items']}",
                    inline=True
                )
                
                if result.get('earliest_time') and result.get('latest_time'):
                    embed.add_field(
                        name="⏰ Time Range",
                        value=f"**Earliest:** {result['earliest_time'].strftime('%Y-%m-%d %H:%M')}\n"
                              f"**Latest:** {result['latest_time'].strftime('%Y-%m-%d %H:%M')}\n"
                              f"**Age:** {(datetime.now(timezone.utc) - result['latest_time']).total_seconds() / 3600:.1f}h",
                        inline=True
                    )
                
                # Show recent files
                if result['recent_files']:
                    recent_text = ""
                    for file_info in result['recent_files'][-5:]:  # Last 5 files
                        if 'error' in file_info:
                            recent_text += f"❌ {file_info['path']}: {file_info['error']}\n"
                        else:
                            recent_text += f"✅ {file_info['path']}: {file_info['items']} items\n"
                    
                    embed.add_field(
                        name="📋 Recent Files (Last 2h)",
                        value=recent_text[:1024],
                        inline=False
                    )
                else:
                    embed.add_field(
                        name="⚠️ Recent Files",
                        value="No files found in the last 2 hours",
                        inline=False
                    )
            else:
                embed.add_field(
                    name="❌ Status",
                    value=result.get('message', 'Unknown error'),
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in diag_features: {e}")
            await interaction.followup.send(f"❌ Error running diagnostics: {str(e)}", ephemeral=True)
    
    @app_commands.command(name="diag_bazaar", description="Diagnostics for bazaar data sources")
    async def diag_bazaar(self, interaction: discord.Interaction):
        """Diagnose bazaar data sources status."""
        await interaction.response.defer()
        
        try:
            self.logger.info("Running bazaar data diagnostics")
            result = self._scan_bazaar_data()
            
            embed = discord.Embed(
                title="🏪 Bazaar Data Diagnostics",
                color=0x00ff00 if result["status"] == "ok" else 0xff0000
            )
            
            if result["status"] == "ok":
                for source in result["sources"]:
                    name = source["name"]
                    
                    if source["exists"]:
                        if "error" in source:
                            value = f"❌ Error: {source['error']}"
                        elif name == "bazaar_history":
                            value = f"✅ **Files:** {source.get('files', 0)}\n"
                            value += f"**Total Rows:** {source.get('total_rows', 0):,}\n"
                            value += f"**Recent Rows (3h):** {source.get('recent_rows', 0):,}"
                        elif name == "bazaar":
                            value = f"✅ **Files:** {source.get('files', 0)}\n"
                            value += f"**Sample Rows:** {source.get('sample_rows', 0):,}"
                        elif name == "bazaar_snapshots.ndjson":
                            modified = source.get('modified')
                            age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600 if modified else 0
                            value = f"✅ **Size:** {source.get('size_mb', 0):.1f} MB\n"
                            value += f"**Modified:** {age_hours:.1f}h ago\n"
                            value += f"**Sample Products:** {source.get('sample_products', 0)}"
                        else:
                            value = "✅ Available"
                    else:
                        value = "❌ Not found"
                    
                    embed.add_field(
                        name=f"📁 {name}",
                        value=value,
                        inline=True
                    )
            else:
                embed.add_field(
                    name="❌ Error",
                    value=result.get('message', 'Unknown error'),
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in diag_bazaar: {e}")
            await interaction.followup.send(f"❌ Error running diagnostics: {str(e)}", ephemeral=True)
    
    @app_commands.command(name="diag_env", description="Diagnostics for environment and system")
    async def diag_env(self, interaction: discord.Interaction):
        """Diagnose environment variables and system status."""
        await interaction.response.defer()
        
        try:
            self.logger.info("Running environment diagnostics")
            result = self._check_environment()
            
            embed = discord.Embed(
                title="🔧 Environment Diagnostics",
                color=0x00ff00 if result["status"] == "ok" else 0xff0000
            )
            
            if result["status"] == "ok":
                # Environment variables
                env_text = ""
                for var, value in result["environment"].items():
                    status = "✅" if value != "Not set" else "❌"
                    env_text += f"{status} **{var}:** {value}\n"
                
                embed.add_field(
                    name="🔐 Environment Variables",
                    value=env_text,
                    inline=False
                )
                
                # System info
                system_info = result["system"]
                system_text = f"**Python:** {system_info['python_version'].split()[0]}\n"
                system_text += f"**Directory:** `{system_info['working_directory']}`"
                
                embed.add_field(
                    name="🖥️ System",
                    value=system_text,
                    inline=True
                )
                
                # Package versions
                packages_text = ""
                for pkg, version in result["packages"].items():
                    status = "✅" if version != "Not installed" else "❌"
                    packages_text += f"{status} **{pkg}:** {version}\n"
                
                embed.add_field(
                    name="📦 Packages",
                    value=packages_text,
                    inline=True
                )
            else:
                embed.add_field(
                    name="❌ Error",
                    value=result.get('message', 'Unknown error'),
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in diag_env: {e}")
            await interaction.followup.send(f"❌ Error running diagnostics: {str(e)}", ephemeral=True)


async def setup(bot: commands.Bot):
    """Setup function for the cog."""
    await bot.add_cog(DiagnosticsCog(bot))
    logging.getLogger(__name__).info("Diagnostics cog loaded")