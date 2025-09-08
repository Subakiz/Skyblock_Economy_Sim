#!/usr/bin/env python3
"""
Feature Consumer for Market Intelligence

Reads hourly feature summaries and produces:
- Watchlist of active items (by volume)
- Fair Market Value (FMV) data with market depth awareness
- Compact market state for the auction sniper
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)


class FeatureConsumer:
    """
    Consumes hourly feature summaries to produce market intelligence.
    
    Merges price ladders across recent hours to compute:
    - Floor and second-lowest prices with counts
    - Market depth-aware FMV calculations
    - Watchlist based on activity volume
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureConsumer")
        
        # Configuration
        market_config = config.get("market", {})
        self.window_hours = market_config.get("window_hours", 12)
        self.thin_wall_threshold = market_config.get("thin_wall_threshold", 2)
        self.min_auction_count = market_config.get("min_auction_count", 25)
        self.max_watchlist_items = market_config.get("max_watchlist_items", 2000)
        
        # Data paths
        self.feature_summaries_path = Path("data/feature_summaries")
        
        self.logger.info(f"FeatureConsumer initialized with window={self.window_hours}h")
    
    def _load_recent_summaries(self, window_hours: Optional[int] = None) -> pd.DataFrame:
        """Load feature summaries from the last N hours."""
        try:
            if window_hours is None:
                window_hours = self.window_hours
            
            if not self.feature_summaries_path.exists():
                self.logger.debug("Feature summaries directory does not exist")
                return pd.DataFrame()
            
            # Calculate time window
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=window_hours)
            
            # Use PyArrow dataset for efficient reading
            try:
                dataset = ds.dataset(self.feature_summaries_path, format="parquet")
                
                # Filter by hour_start timestamp
                filter_expr = (
                    (ds.field("hour_start") >= start_time) &
                    (ds.field("hour_start") <= end_time)
                )
                
                table = dataset.to_table(filter=filter_expr)
                df = table.to_pandas()
                
                self.logger.debug(f"Loaded {len(df)} summary records from last {window_hours}h")
                return df
                
            except Exception as e:
                self.logger.error(f"Failed to load summaries with PyArrow: {e}")
                # Fallback: manual directory scanning
                return self._load_summaries_fallback(start_time, end_time)
        
        except Exception as e:
            self.logger.error(f"Failed to load recent summaries: {e}")
            return pd.DataFrame()
    
    def _load_summaries_fallback(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fallback method to load summaries by scanning directories."""
        try:
            summary_files = []
            
            # Scan partition directories
            for year_dir in self.feature_summaries_path.glob("year=*"):
                for month_dir in year_dir.glob("month=*"):
                    for day_dir in month_dir.glob("day=*"):
                        for hour_dir in day_dir.glob("hour=*"):
                            summary_file = hour_dir / "summary.parquet"
                            if summary_file.exists():
                                # Parse time from path
                                try:
                                    year = int(year_dir.name.split("=")[1])
                                    month = int(month_dir.name.split("=")[1])
                                    day = int(day_dir.name.split("=")[1])
                                    hour = int(hour_dir.name.split("=")[1])
                                    
                                    file_time = datetime(year, month, day, hour, tzinfo=timezone.utc)
                                    
                                    if start_time <= file_time <= end_time:
                                        summary_files.append(summary_file)
                                except Exception as e:
                                    self.logger.error(f"Failed to parse time from {hour_dir}: {e}")
            
            # Load and concatenate files
            if not summary_files:
                return pd.DataFrame()
            
            dfs = []
            for file_path in summary_files:
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Failed to read {file_path}: {e}")
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                self.logger.debug(f"Fallback loaded {len(result)} records from {len(dfs)} files")
                return result
            else:
                return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Fallback loading failed: {e}")
            return pd.DataFrame()
    
    def _merge_price_ladders(self, item_summaries: List[Dict[str, Any]]) -> Tuple[List[Tuple[float, int]], int]:
        """
        Merge price ladders from multiple hours for an item.
        
        Returns:
            - List of (price, count) tuples sorted by price
            - Total count across all hours
        """
        try:
            # Aggregate price counts across all hours
            price_counts = defaultdict(int)
            total_count = 0
            
            for summary in item_summaries:
                prices = summary.get("prices", [])
                counts = summary.get("counts", [])
                hour_total = summary.get("total_count", 0)
                
                # Add counts for each price level
                for price, count in zip(prices, counts):
                    if price > 0 and count > 0:  # Sanity check
                        price_counts[float(price)] += count
                
                total_count += hour_total
            
            # Sort by price and convert to list of tuples
            if not price_counts:
                return [], 0
            
            sorted_prices = sorted(price_counts.items())  # (price, count) tuples
            
            return sorted_prices, total_count
        
        except Exception as e:
            self.logger.error(f"Failed to merge price ladders: {e}")
            return [], 0
    
    def _calculate_market_depth_fmv(self, price_ladder: List[Tuple[float, int]]) -> Tuple[float, str]:
        """
        Calculate Fair Market Value using market depth analysis.
        
        Logic:
        - If floor level has <= thin_wall_threshold items: FMV = second price
        - If floor level has > thin_wall_threshold items: FMV = floor price
        - Apply discount factor for single level scenarios
        """
        if not price_ladder:
            return 0.0, "no_data"
        
        if len(price_ladder) == 1:
            # Only one price level - apply discount
            floor_price, floor_count = price_ladder[0]
            return floor_price * 0.95, "single_level_discounted"
        
        # Get floor and next levels
        floor_price, floor_count = price_ladder[0]
        second_price, second_count = price_ladder[1]
        
        # Market depth analysis
        if floor_count <= self.thin_wall_threshold:
            # Thin wall at floor - use second price as FMV
            return second_price, "thin_wall_second_price"
        else:
            # Thick wall at floor - use floor price as FMV
            return floor_price, "thick_wall_floor_price"
    
    def generate_market_intelligence(self, window_hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate market intelligence from recent feature summaries.
        
        Returns:
            {
                "watchlist": Set[str] - items with sufficient activity,
                "fmv_data": Dict[str, Dict] - FMV calculations per item,
                "metadata": Dict - summary statistics
            }
        """
        try:
            self.logger.info("Generating market intelligence from feature summaries...")
            
            # Load recent summaries
            df = self._load_recent_summaries(window_hours)
            
            if df.empty:
                self.logger.warning("No recent summaries found")
                return {
                    "watchlist": set(),
                    "fmv_data": {},
                    "metadata": {
                        "hours_analyzed": window_hours or self.window_hours,
                        "items_analyzed": 0,
                        "watchlist_size": 0
                    }
                }
            
            # Group by item_name
            item_groups = df.groupby("item_name")
            
            # Calculate market intelligence per item
            watchlist = set()
            fmv_data = {}
            
            for item_name, item_df in item_groups:
                try:
                    # Convert to list of summary dicts
                    item_summaries = item_df.to_dict('records')
                    
                    # Merge price ladders across hours
                    price_ladder, total_count = self._merge_price_ladders(item_summaries)
                    
                    # Check if item qualifies for watchlist
                    if total_count >= self.min_auction_count and price_ladder:
                        watchlist.add(item_name)
                        
                        # Calculate market depth-aware FMV
                        fmv, method = self._calculate_market_depth_fmv(price_ladder)
                        
                        # Store FMV data
                        fmv_data[item_name] = {
                            "fmv": fmv,
                            "floor_price": price_ladder[0][0],
                            "floor_count": price_ladder[0][1],
                            "second_price": price_ladder[1][0] if len(price_ladder) > 1 else None,
                            "second_count": price_ladder[1][1] if len(price_ladder) > 1 else None,
                            "total_count": total_count,
                            "price_levels": len(price_ladder),
                            "method": method,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                
                except Exception as e:
                    self.logger.error(f"Failed to process item {item_name}: {e}")
            
            # Limit watchlist size (keep highest volume items)
            if len(watchlist) > self.max_watchlist_items:
                # Sort by total_count and keep top items
                item_volumes = [(item, fmv_data[item]["total_count"]) for item in watchlist]
                item_volumes.sort(key=lambda x: x[1], reverse=True)
                
                top_items = {item for item, _ in item_volumes[:self.max_watchlist_items]}
                
                # Remove excess items from both watchlist and fmv_data
                excess_items = watchlist - top_items
                watchlist = top_items
                
                for item in excess_items:
                    fmv_data.pop(item, None)
                
                self.logger.info(f"Limited watchlist to top {self.max_watchlist_items} items by volume")
            
            metadata = {
                "hours_analyzed": window_hours or self.window_hours,
                "items_analyzed": len(item_groups),
                "watchlist_size": len(watchlist),
                "fmv_items": len(fmv_data),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"Market intelligence generated: {len(watchlist)} watchlist items, "
                           f"FMV data for {len(fmv_data)} items")
            
            return {
                "watchlist": watchlist,
                "fmv_data": fmv_data,
                "metadata": metadata
            }
        
        except Exception as e:
            self.logger.error(f"Failed to generate market intelligence: {e}")
            return {
                "watchlist": set(),
                "fmv_data": {},
                "metadata": {
                    "error": str(e),
                    "hours_analyzed": window_hours or self.window_hours
                }
            }


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create and run consumer
    consumer = FeatureConsumer(config)
    intelligence = consumer.generate_market_intelligence()
    
    print(f"Generated intelligence:")
    print(f"  Watchlist: {len(intelligence['watchlist'])} items")
    print(f"  FMV Data: {len(intelligence['fmv_data'])} items")
    print(f"  Metadata: {intelligence['metadata']}")