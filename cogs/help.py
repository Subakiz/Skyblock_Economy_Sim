#!/usr/bin/env python3
"""
Dynamic Help Command Cog

Provides a comprehensive /help command that dynamically lists all slash commands
from the bot's command tree, including sniper commands and other functionality.
"""

import logging
from typing import Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands


class HelpCog(commands.Cog):
    """Dynamic help command that lists all available slash commands."""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.HelpCog")
        
        self.logger.info("Help cog initialized")
    
    @app_commands.command(name="help", description="Show all available commands and their descriptions")
    async def help_command(self, interaction: discord.Interaction):
        """Show comprehensive help for all commands."""
        try:
            await interaction.response.defer()
            
            # Get all registered slash commands
            commands_by_category = self._categorize_commands()
            
            # Create main help embed
            embed = discord.Embed(
                title="🤖 Skyblock Economy Bot - Commands Help",
                description="Here are all available commands organized by category:",
                color=0x0099ff
            )
            
            # Add commands by category
            for category, commands_list in commands_by_category.items():
                if commands_list:
                    command_text = "\n".join([
                        f"`/{cmd.name}` - {cmd.description or 'No description'}"
                        for cmd in commands_list
                    ])
                    embed.add_field(
                        name=f"📁 {category}",
                        value=command_text,
                        inline=False
                    )
            
            # Add usage examples
            embed.add_field(
                name="💡 Usage Examples",
                value=(
                    "`/sniper_channel #auction-alerts` - Set snipe alert channel\n"
                    "`/sniper_config profit_threshold:100000` - Set minimum profit\n"
                    "`/sniper_status` - Check sniper status and stats\n"
                    "`/plot ENCHANTED_BREAD` - Generate price history chart"
                ),
                inline=False
            )
            
            # Add bot info
            embed.set_footer(
                text=f"Bot uptime: {self._get_uptime()} | Total commands: {len([cmd for cmds in commands_by_category.values() for cmd in cmds])}",
                icon_url=self.bot.user.avatar.url if self.bot.user.avatar else None
            )
            
            await interaction.followup.send(embed=embed)
            self.logger.info(f"Help command used by {interaction.user.name}")
            
        except Exception as e:
            self.logger.error(f"Help command error: {e}")
            await interaction.followup.send("❌ Error generating help. Please try again later.", ephemeral=True)
    
    def _categorize_commands(self) -> Dict[str, List[app_commands.Command]]:
        """Categorize commands by functionality."""
        categories = {
            "Auction Sniper": [],
            "Market Analysis": [],
            "Data Visualization": [],
            "Bot Management": [],
            "Other": []
        }
        
        try:
            # Get all global commands from the command tree
            for command in self.bot.tree.walk_commands():
                if isinstance(command, app_commands.Command):
                    category = self._determine_command_category(command)
                    categories[category].append(command)
            
            # Remove empty categories
            return {k: v for k, v in categories.items() if v}
            
        except Exception as e:
            self.logger.error(f"Error categorizing commands: {e}")
            return {}
    
    def _determine_command_category(self, command: app_commands.Command) -> str:
        """Determine which category a command belongs to."""
        command_name = command.name.lower()
        
        # Auction Sniper commands
        if any(keyword in command_name for keyword in ["sniper", "snipe", "auction"]):
            return "Auction Sniper"
        
        # Market Analysis commands  
        if any(keyword in command_name for keyword in ["analyze", "predict", "forecast", "market"]):
            return "Market Analysis"
        
        # Data Visualization commands
        if any(keyword in command_name for keyword in ["plot", "chart", "graph", "visualize"]):
            return "Data Visualization"
        
        # Bot Management commands
        if any(keyword in command_name for keyword in ["help", "status", "config", "setup"]):
            return "Bot Management"
        
        # Everything else
        return "Other"
    
    def _get_uptime(self) -> str:
        """Get bot uptime as a human-readable string."""
        try:
            if hasattr(self.bot, 'start_time'):
                from datetime import datetime, timezone
                uptime = datetime.now(timezone.utc) - self.bot.start_time
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                
                if days > 0:
                    return f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    return f"{hours}h {minutes}m"
                else:
                    return f"{minutes}m"
            else:
                return "Unknown"
        except Exception as e:
            self.logger.error(f"Error calculating uptime: {e}")
            return "Error"


async def setup(bot):
    """Load the help cog."""
    await bot.add_cog(HelpCog(bot))