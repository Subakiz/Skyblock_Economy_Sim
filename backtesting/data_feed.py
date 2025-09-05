"""
Data feed for backtesting - streams historical price snapshots.
"""

import psycopg2
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass 
class MarketSnapshot:
    """Single point-in-time market data snapshot."""
    timestamp: datetime
    bazaar_data: Dict[str, Dict[str, float]]  # product_id -> {buy_price, sell_price, ...}
    ah_data: Dict[str, Dict[str, float]]      # product_id -> {median_price, sale_count, ...}

class DataFeed:
    """Historical data feed for backtesting."""
    
    def __init__(self, conn, start_date: datetime, end_date: datetime, 
                 interval_minutes: int = 15):
        self.conn = conn
        self.start_date = start_date
        self.end_date = end_date
        self.interval_minutes = interval_minutes
    
    def stream_snapshots(self) -> Iterator[MarketSnapshot]:
        """Stream market snapshots at regular intervals."""
        current_time = self.start_date
        
        while current_time <= self.end_date:
            # Get bazaar data for this timestamp
            bazaar_data = self._get_bazaar_snapshot(current_time)
            
            # Get AH aggregated data for this timestamp  
            ah_data = self._get_ah_snapshot(current_time)
            
            yield MarketSnapshot(
                timestamp=current_time,
                bazaar_data=bazaar_data,
                ah_data=ah_data
            )
            
            current_time += timedelta(minutes=self.interval_minutes)
    
    def _get_bazaar_snapshot(self, timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """Get bazaar data closest to the given timestamp."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT product_id, buy_price, sell_price, buy_volume, sell_volume
                FROM bazaar_snapshots 
                WHERE ts <= %s
                AND ts >= %s - INTERVAL '1 hour'
                ORDER BY product_id, ts DESC
            """, (timestamp, timestamp))
            
            bazaar_data = {}
            seen_products = set()
            
            for row in cur.fetchall():
                product_id, buy_price, sell_price, buy_vol, sell_vol = row
                
                # Take the most recent data point for each product
                if product_id not in seen_products:
                    bazaar_data[product_id] = {
                        "buy_price": float(buy_price) if buy_price else 0.0,
                        "sell_price": float(sell_price) if sell_price else 0.0,
                        "buy_volume": float(buy_vol) if buy_vol else 0.0,
                        "sell_volume": float(sell_vol) if sell_vol else 0.0,
                    }
                    seen_products.add(product_id)
            
            return bazaar_data
    
    def _get_ah_snapshot(self, timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """Get AH aggregated data closest to the given timestamp."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT item_id, median_price, p25_price, p75_price, sale_count
                FROM ah_prices_15m
                WHERE time_bucket <= %s
                AND time_bucket >= %s - INTERVAL '2 hours'
                ORDER BY item_id, time_bucket DESC
            """, (timestamp, timestamp))
            
            ah_data = {}
            seen_items = set()
            
            for row in cur.fetchall():
                item_id, median_price, p25_price, p75_price, sale_count = row
                
                # Take the most recent data point for each item
                if item_id not in seen_items:
                    ah_data[item_id] = {
                        "median_price": float(median_price) if median_price else 0.0,
                        "p25_price": float(p25_price) if p25_price else 0.0,
                        "p75_price": float(p75_price) if p75_price else 0.0,
                        "sale_count": int(sale_count) if sale_count else 0,
                    }
                    seen_items.add(item_id)
            
            return ah_data