#!/usr/bin/env python3
"""
Storage Janitor Cog

Manages disk storage usage to stay within configured limits.
Prunes oldest data partitions when storage exceeds capacity thresholds.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

from discord.ext import commands, tasks


class StorageJanitor(commands.Cog):
    """
    Automated storage management to maintain disk usage within limits.
    
    Monitors data directories and prunes oldest partitions when storage
    exceeds configured thresholds.
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(f"{__name__}.StorageJanitor")
        
        # Load configuration
        self.config = self._load_config()
        storage_config = self.config.get("storage", {})
        
        self.cap_gb = storage_config.get("cap_gb", 70)
        self.headroom_gb = storage_config.get("headroom_gb", 5)
        self.raw_retention_hours = storage_config.get("raw_retention_hours", 2)
        self.summary_retention_days = storage_config.get("summary_retention_days", 30)
        
        # Data directories to monitor
        self.data_root = Path("data")
        self.monitored_paths = [
            self.data_root / "feature_summaries",
            self.data_root / "auction_history", 
            self.data_root / "bazaar_history",
            self.data_root / "raw_spool"
        ]
        
        self.logger.info(f"StorageJanitor initialized: {self.cap_gb}GB cap with {self.headroom_gb}GB headroom")
        
        # Start monitoring task
        self.storage_monitoring_task.start()
    
    def _load_config(self) -> Dict:
        """Load configuration from bot's config file."""
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def disk_usage_gb(self, path: Path) -> float:
        """Calculate disk usage for a path in GB."""
        try:
            if not path.exists():
                return 0.0
            
            if path.is_file():
                return path.stat().st_size / (1024**3)
            
            # Directory - calculate total size
            total_size = 0
            for item in path.rglob("*"):
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass  # Skip files that can't be accessed
            
            return total_size / (1024**3)
        
        except Exception as e:
            self.logger.error(f"Error calculating disk usage for {path}: {e}")
            return 0.0
    
    def get_total_usage_gb(self) -> float:
        """Get total usage across all monitored data directories."""
        total_usage = 0.0
        for path in self.monitored_paths:
            usage = self.disk_usage_gb(path)
            total_usage += usage
        return total_usage
    
    def get_partition_paths(self, root_path: Path) -> List[Tuple[Path, datetime]]:
        """
        Get all time-partitioned directories with their timestamps.
        
        Returns list of (path, datetime) tuples sorted by age (oldest first).
        """
        partitions = []
        
        try:
            if not root_path.exists():
                return partitions
            
            # Look for partitioned structure: year=YYYY/month=MM/day=DD/hour=HH
            for year_dir in root_path.glob("year=*"):
                try:
                    year = int(year_dir.name.split("=")[1])
                    
                    for month_dir in year_dir.glob("month=*"):
                        try:
                            month = int(month_dir.name.split("=")[1])
                            
                            for day_dir in month_dir.glob("day=*"):
                                try:
                                    day = int(day_dir.name.split("=")[1])
                                    
                                    # Check if there are hour directories
                                    hour_dirs = list(day_dir.glob("hour=*"))
                                    if hour_dirs:
                                        for hour_dir in hour_dirs:
                                            try:
                                                hour = int(hour_dir.name.split("=")[1])
                                                partition_time = datetime(year, month, day, hour, tzinfo=timezone.utc)
                                                partitions.append((hour_dir, partition_time))
                                            except (ValueError, IndexError) as e:
                                                self.logger.warning(f"Invalid hour partition {hour_dir}: {e}")
                                    else:
                                        # No hour subdirs, treat day as partition
                                        partition_time = datetime(year, month, day, tzinfo=timezone.utc)
                                        partitions.append((day_dir, partition_time))
                                        
                                except (ValueError, IndexError) as e:
                                    self.logger.warning(f"Invalid day partition {day_dir}: {e}")
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Invalid month partition {month_dir}: {e}")
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Invalid year partition {year_dir}: {e}")
            
            # Sort by timestamp (oldest first)
            partitions.sort(key=lambda x: x[1])
            
        except Exception as e:
            self.logger.error(f"Error scanning partitions in {root_path}: {e}")
        
        return partitions
    
    def prune_to_cap(self, cap_gb: Optional[float] = None, headroom_gb: Optional[float] = None) -> Dict[str, float]:
        """
        Prune oldest partitions until disk usage is under cap - headroom.
        
        Returns dict with pruning statistics.
        """
        if cap_gb is None:
            cap_gb = self.cap_gb
        if headroom_gb is None:
            headroom_gb = self.headroom_gb
        
        target_gb = cap_gb - headroom_gb
        
        stats = {
            "initial_usage_gb": 0.0,
            "final_usage_gb": 0.0,
            "pruned_gb": 0.0,
            "partitions_deleted": 0,
            "directories_deleted": []
        }
        
        try:
            # Check current usage
            stats["initial_usage_gb"] = self.get_total_usage_gb()
            
            if stats["initial_usage_gb"] <= target_gb:
                self.logger.info(f"Storage usage {stats['initial_usage_gb']:.2f}GB is under target {target_gb:.2f}GB, no pruning needed")
                stats["final_usage_gb"] = stats["initial_usage_gb"]
                return stats
            
            self.logger.warning(f"Storage usage {stats['initial_usage_gb']:.2f}GB exceeds target {target_gb:.2f}GB, starting pruning...")
            
            # Collect all partitions from all monitored paths
            all_partitions = []
            for path in self.monitored_paths:
                partitions = self.get_partition_paths(path)
                all_partitions.extend(partitions)
            
            # Sort by age (oldest first)
            all_partitions.sort(key=lambda x: x[1])
            
            # Delete partitions until we're under target
            for partition_path, partition_time in all_partitions:
                current_usage = self.get_total_usage_gb()
                
                if current_usage <= target_gb:
                    break  # Target reached
                
                # Calculate partition size before deletion
                partition_size_gb = self.disk_usage_gb(partition_path)
                
                try:
                    # Delete the partition
                    if partition_path.exists():
                        if partition_path.is_dir():
                            shutil.rmtree(partition_path)
                        else:
                            partition_path.unlink()
                        
                        stats["partitions_deleted"] += 1
                        stats["pruned_gb"] += partition_size_gb
                        stats["directories_deleted"].append(str(partition_path))
                        
                        self.logger.info(f"Pruned partition: {partition_path} ({partition_size_gb:.3f}GB, {partition_time})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to delete partition {partition_path}: {e}")
            
            # Clean up empty parent directories
            self._cleanup_empty_directories()
            
            # Final usage check
            stats["final_usage_gb"] = self.get_total_usage_gb()
            
            self.logger.info(f"Pruning completed: {stats['initial_usage_gb']:.2f}GB → {stats['final_usage_gb']:.2f}GB "
                           f"({stats['partitions_deleted']} partitions, {stats['pruned_gb']:.2f}GB freed)")
            
        except Exception as e:
            self.logger.error(f"Error during pruning: {e}")
        
        return stats
    
    def _cleanup_empty_directories(self):
        """Remove empty parent directories after pruning."""
        try:
            for root_path in self.monitored_paths:
                if not root_path.exists():
                    continue
                
                # Walk up the directory tree and remove empty dirs
                for year_dir in list(root_path.glob("year=*")):
                    for month_dir in list(year_dir.glob("month=*")):
                        for day_dir in list(month_dir.glob("day=*")):
                            # Remove empty day directories
                            if day_dir.is_dir() and not any(day_dir.iterdir()):
                                try:
                                    day_dir.rmdir()
                                    self.logger.debug(f"Removed empty day directory: {day_dir}")
                                except OSError:
                                    pass
                        
                        # Remove empty month directories
                        if month_dir.is_dir() and not any(month_dir.iterdir()):
                            try:
                                month_dir.rmdir()
                                self.logger.debug(f"Removed empty month directory: {month_dir}")
                            except OSError:
                                pass
                    
                    # Remove empty year directories
                    if year_dir.is_dir() and not any(year_dir.iterdir()):
                        try:
                            year_dir.rmdir()
                            self.logger.debug(f"Removed empty year directory: {year_dir}")
                        except OSError:
                            pass
        
        except Exception as e:
            self.logger.error(f"Error during empty directory cleanup: {e}")
    
    def prune_by_retention_policy(self):
        """Prune data based on retention policies regardless of disk usage."""
        try:
            current_time = datetime.now(timezone.utc)
            pruned_count = 0
            
            # Prune raw spool files (short retention)
            raw_spool_path = self.data_root / "raw_spool"
            if raw_spool_path.exists():
                cutoff_time = current_time - timedelta(hours=self.raw_retention_hours)
                
                for spool_file in raw_spool_path.glob("*.ndjson"):
                    try:
                        if spool_file.stat().st_mtime < cutoff_time.timestamp():
                            spool_file.unlink()
                            pruned_count += 1
                            self.logger.debug(f"Pruned old raw spool: {spool_file.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to prune raw spool {spool_file}: {e}")
            
            # Prune old feature summaries (longer retention)
            summaries_path = self.data_root / "feature_summaries"
            if summaries_path.exists():
                cutoff_time = current_time - timedelta(days=self.summary_retention_days)
                
                partitions = self.get_partition_paths(summaries_path)
                for partition_path, partition_time in partitions:
                    if partition_time < cutoff_time:
                        try:
                            if partition_path.is_dir():
                                shutil.rmtree(partition_path)
                            else:
                                partition_path.unlink()
                            
                            pruned_count += 1
                            self.logger.debug(f"Pruned old summary partition: {partition_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to prune summary partition {partition_path}: {e}")
            
            if pruned_count > 0:
                self.logger.info(f"Retention policy pruning: removed {pruned_count} items")
                self._cleanup_empty_directories()
        
        except Exception as e:
            self.logger.error(f"Error during retention policy pruning: {e}")
    
    @tasks.loop(hours=4)  # Run every 4 hours
    async def storage_monitoring_task(self):
        """Background task to monitor and manage storage usage."""
        try:
            # Run pruning operations in background thread to avoid blocking
            await asyncio.to_thread(self._run_storage_maintenance)
            
        except Exception as e:
            self.logger.error(f"Storage monitoring task error: {e}")
    
    def _run_storage_maintenance(self):
        """Run storage maintenance operations."""
        start_time = time.time()
        
        # Check current usage
        current_usage = self.get_total_usage_gb()
        target_usage = self.cap_gb - self.headroom_gb
        
        self.logger.info(f"Storage check: {current_usage:.2f}GB used, {target_usage:.2f}GB target")
        
        # Prune by retention policy first
        self.prune_by_retention_policy()
        
        # Then prune by disk usage if needed
        if current_usage > target_usage:
            self.prune_to_cap()
        
        maintenance_time = time.time() - start_time
        self.logger.info(f"Storage maintenance completed in {maintenance_time:.1f}s")
    
    @storage_monitoring_task.before_loop
    async def before_storage_monitoring(self):
        """Wait for bot to be ready before starting storage monitoring."""
        await self.bot.wait_until_ready()
        
        # Run initial storage check
        try:
            await asyncio.to_thread(self._run_storage_maintenance)
        except Exception as e:
            self.logger.error(f"Initial storage check failed: {e}")
    
    def cog_unload(self):
        """Cleanup when cog is unloaded."""
        self.storage_monitoring_task.cancel()


async def setup(bot):
    """Load the storage janitor cog."""
    await bot.add_cog(StorageJanitor(bot))