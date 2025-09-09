#!/usr/bin/env python3
"""
Continuous Feature Ingestor Script

Long-running script that uses FeatureIngestor to continuously:
- Collect auction data from Hypixel API
- Build in-memory price ladders
- Write hourly feature summaries to partitioned Parquet files
- Perform interim flushes every 120-180 seconds for fresh data

Requires HYPIXEL_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
import signal
import psutil
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.feature_ingestor import FeatureIngestor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContinuousFeatureIngestor:
    """Wrapper for FeatureIngestor that runs continuously with interim flushes."""
    
    def __init__(self, config: dict):
        self.config = config
        self.ingestor: Optional[FeatureIngestor] = None
        self.running = False
        self.interim_flush_interval = 150  # 2.5 minutes
        
        # Create logs directory if it doesn't exist
        logs_path = Path('logs')
        logs_path.mkdir(exist_ok=True)
        
        # Setup file logging
        file_handler = logging.FileHandler(logs_path / 'feature_ingestor.log', mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logger.info("Continuous Feature Ingestor initialized")
    
    async def run(self):
        """Main continuous ingestion loop."""
        # Validate environment
        api_key = os.environ.get('HYPIXEL_API_KEY')
        if not api_key:
            logger.error("HYPIXEL_API_KEY environment variable is required")
            return
        
        self.running = True
        logger.info("Starting continuous feature ingestion...")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Create FeatureIngestor instance
            self.ingestor = FeatureIngestor(self.config)
            
            last_interim_flush = datetime.now(timezone.utc)
            
            while self.running:
                start_time = datetime.now(timezone.utc)
                
                try:
                    # Run one ingestion cycle
                    await self.ingestor.run_ingestion_cycle()
                    
                    # Check if it's time for interim flush
                    if (datetime.now(timezone.utc) - last_interim_flush).total_seconds() >= self.interim_flush_interval:
                        await self._interim_flush()
                        last_interim_flush = datetime.now(timezone.utc)
                    
                    # Memory monitoring
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.debug(f"Memory usage: {memory_mb:.1f} MB")
                    
                    # Check memory guards
                    guards_config = self.config.get("guards", {})
                    soft_rss_mb = guards_config.get("soft_rss_mb", 1300)
                    
                    if memory_mb > soft_rss_mb:
                        logger.warning(f"Memory usage ({memory_mb:.1f} MB) exceeds soft limit ({soft_rss_mb} MB)")
                        # Could implement memory cleanup here if needed
                    
                except Exception as e:
                    logger.error(f"Error in ingestion cycle: {e}")
                    # Continue running after errors
                
                # Calculate sleep time to maintain regular intervals
                cycle_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                market_config = self.config.get("market", {})
                intel_interval = market_config.get("intel_interval_seconds", 90)
                
                sleep_time = max(0, intel_interval - cycle_duration)
                if sleep_time > 0:
                    logger.debug(f"Cycle completed in {cycle_duration:.1f}s, sleeping for {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Fatal error in continuous ingestion: {e}")
        finally:
            await self._shutdown()
    
    async def _interim_flush(self):
        """Perform interim flush to ensure fresh data availability."""
        if not self.ingestor:
            return
        
        try:
            logger.info("Performing interim flush...")
            
            # Use the new flush method from FeatureIngestor
            self.ingestor.flush_current_hour_summary()
            
        except Exception as e:
            logger.error(f"Error during interim flush: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down continuous feature ingestor...")
        
        if self.ingestor:
            try:
                # Commit any pending data
                if hasattr(self.ingestor, '_commit_hour_summary'):
                    await self.ingestor._commit_hour_summary()
                
                # Cleanup
                if hasattr(self.ingestor, 'cleanup_old_data'):
                    await self.ingestor.cleanup_old_data()
                    
            except Exception as e:
                logger.error(f"Error during shutdown cleanup: {e}")
        
        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return
    
    # Validate required packages
    try:
        import pyarrow
        import psutil
    except ImportError as e:
        logger.error(f"Required package not available: {e}")
        logger.error("Please ensure pyarrow and psutil are installed")
        return
    
    # Create and run continuous ingestor
    continuous_ingestor = ContinuousFeatureIngestor(config)
    await continuous_ingestor.run()


if __name__ == "__main__":
    asyncio.run(main())