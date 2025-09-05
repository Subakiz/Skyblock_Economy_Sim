"""
Hypixel SkyBlock Auction House data collector.

Polls /skyblock/auctions (paginated) and /skyblock/auctions_ended endpoints,
handles pagination, rate limiting, deduplication and persistence.
"""

import os
import time
import json
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import yaml

from ingestion.common.hypixel_client import HypixelClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration with environment variable overrides."""
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Environment variable overrides
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    cfg["storage"]["database_url"] = db_url
    
    return cfg

def upsert_auction(conn, auction_data: Dict[str, Any], table_name: str = "auctions"):
    """Insert or update auction record with deduplication by UUID."""
    with conn.cursor() as cur:
        if table_name == "auctions":
            cur.execute("""
                INSERT INTO auctions (
                    uuid, item_id, item_name, tier, bin, starting_bid, highest_bid,
                    start_time, end_time, seller, bids_count, category, attributes, raw_data
                ) VALUES (
                    %(uuid)s, %(item_id)s, %(item_name)s, %(tier)s, %(bin)s, %(starting_bid)s, %(highest_bid)s,
                    %(start_time)s, %(end_time)s, %(seller)s, %(bids_count)s, %(category)s, %(attributes)s, %(raw_data)s
                ) ON CONFLICT (uuid) DO UPDATE SET
                    highest_bid = EXCLUDED.highest_bid,
                    bids_count = EXCLUDED.bids_count,
                    updated_at = now()
            """, auction_data)
        else:  # auctions_ended
            cur.execute("""
                INSERT INTO auctions_ended (
                    uuid, item_id, item_name, tier, bin, starting_bid, highest_bid, sale_price,
                    start_time, end_time, seller, buyer, bids_count, category, attributes, raw_data
                ) VALUES (
                    %(uuid)s, %(item_id)s, %(item_name)s, %(tier)s, %(bin)s, %(starting_bid)s, %(highest_bid)s, %(sale_price)s,
                    %(start_time)s, %(end_time)s, %(seller)s, %(buyer)s, %(bids_count)s, %(category)s, %(attributes)s, %(raw_data)s
                ) ON CONFLICT (uuid) DO NOTHING
            """, auction_data)

def extract_auction_data(raw_auction: Dict[str, Any], enable_raw_capture: bool = False) -> Dict[str, Any]:
    """Extract and normalize auction data from raw API response."""
    # Convert timestamps from milliseconds to datetime
    start_time = None
    end_time = None
    
    if raw_auction.get("start"):
        start_time = datetime.fromtimestamp(raw_auction["start"] / 1000, tz=timezone.utc)
    if raw_auction.get("end"):
        end_time = datetime.fromtimestamp(raw_auction["end"] / 1000, tz=timezone.utc)
    
    # Extract item attributes (enchants, reforge, etc.)
    attributes = {}
    item_bytes = raw_auction.get("item_bytes")
    if item_bytes:
        # In real implementation, you'd decode NBT data here
        # For now, just store the raw bytes info
        attributes["has_item_data"] = True
    
    extra = raw_auction.get("extra", "")
    if extra:
        attributes["extra"] = extra

    # Determine item_id from item_name if not provided directly
    item_name = raw_auction.get("item_name", "")
    item_id = item_name.upper().replace(" ", "_").replace("'", "")
    
    data = {
        "uuid": raw_auction.get("uuid"),
        "item_id": item_id,
        "item_name": item_name,
        "tier": raw_auction.get("tier"),
        "bin": raw_auction.get("bin", False),
        "starting_bid": raw_auction.get("starting_bid"),
        "highest_bid": raw_auction.get("highest_bid_amount"),
        "start_time": start_time,
        "end_time": end_time,
        "seller": raw_auction.get("auctioneer"),
        "bids_count": len(raw_auction.get("bids", [])),
        "category": raw_auction.get("category"),
        "attributes": json.dumps(attributes) if attributes else None,
        "raw_data": json.dumps(raw_auction) if enable_raw_capture else None
    }
    
    return data

def extract_ended_auction_data(raw_auction: Dict[str, Any], enable_raw_capture: bool = False) -> Dict[str, Any]:
    """Extract and normalize ended auction data from raw API response."""
    base_data = extract_auction_data(raw_auction, enable_raw_capture)
    
    # Add ended-auction specific fields
    base_data.update({
        "sale_price": raw_auction.get("price"),  # final sale price
        "buyer": raw_auction.get("buyer"),
    })
    
    return base_data

def collect_auctions(client: HypixelClient, conn, cfg: Dict[str, Any]) -> int:
    """Collect active auctions with pagination support."""
    endpoint = cfg["hypixel"]["endpoints"]["auctions"]
    max_pages = cfg["auction_house"]["max_pages_per_cycle"]
    enable_raw_capture = cfg["auction_house"]["enable_raw_capture"]
    
    total_auctions = 0
    page = 0
    
    while page < max_pages:
        try:
            logger.info(f"Fetching auctions page {page}")
            data = client.get_json(endpoint, {"page": page})
            
            auctions = data.get("auctions", [])
            if not auctions:
                logger.info(f"No more auctions found at page {page}")
                break
                
            total_pages = data.get("totalPages", 1)
            logger.info(f"Processing page {page + 1}/{total_pages} with {len(auctions)} auctions")
            
            # Process auctions in this page
            for auction in auctions:
                try:
                    auction_data = extract_auction_data(auction, enable_raw_capture)
                    upsert_auction(conn, auction_data, "auctions")
                    total_auctions += 1
                except Exception as e:
                    logger.error(f"Error processing auction {auction.get('uuid', 'unknown')}: {e}")
            
            conn.commit()
            
            # Check if we've reached the last page
            if page >= total_pages - 1:
                logger.info(f"Reached last page ({total_pages})")
                break
                
            page += 1
            
        except Exception as e:
            logger.error(f"Error fetching auctions page {page}: {e}")
            break
    
    logger.info(f"Collected {total_auctions} auctions across {page + 1} pages")
    return total_auctions

def collect_ended_auctions(client: HypixelClient, conn, cfg: Dict[str, Any]) -> int:
    """Collect recently ended auctions."""
    endpoint = cfg["hypixel"]["endpoints"]["auctions_ended"]
    enable_raw_capture = cfg["auction_house"]["enable_raw_capture"]
    
    total_ended = 0
    
    try:
        logger.info("Fetching ended auctions")
        data = client.get_json(endpoint)
        
        ended_auctions = data.get("auctions", [])
        logger.info(f"Processing {len(ended_auctions)} ended auctions")
        
        for auction in ended_auctions:
            try:
                auction_data = extract_ended_auction_data(auction, enable_raw_capture)
                upsert_auction(conn, auction_data, "auctions_ended")
                total_ended += 1
            except Exception as e:
                logger.error(f"Error processing ended auction {auction.get('uuid', 'unknown')}: {e}")
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error fetching ended auctions: {e}")
    
    logger.info(f"Collected {total_ended} ended auctions")
    return total_ended

def run():
    """Main auction collection loop."""
    cfg = load_config()
    
    client = HypixelClient(
        base_url=cfg["hypixel"]["base_url"],
        api_key=os.getenv("HYPIXEL_API_KEY"),
        max_requests_per_minute=cfg["hypixel"]["max_requests_per_minute"],
        timeout_seconds=cfg["hypixel"]["timeout_seconds"],
    )
    
    db_url = cfg["storage"]["database_url"]
    poll_interval = cfg["auction_house"]["poll_interval_seconds"]
    
    logger.info("Connecting to database...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = False  # We want explicit transaction control
    
    logger.info(f"Starting auction collector loop (poll interval: {poll_interval}s)...")
    
    try:
        while True:
            cycle_start = datetime.now(timezone.utc)
            logger.info(f"[{cycle_start.isoformat()}] Starting collection cycle")
            
            # Collect active auctions
            active_count = collect_auctions(client, conn, cfg)
            
            # Collect ended auctions  
            ended_count = collect_ended_auctions(client, conn, cfg)
            
            cycle_end = datetime.now(timezone.utc)
            duration = (cycle_end - cycle_start).total_seconds()
            
            logger.info(f"[{cycle_end.isoformat()}] Cycle complete: {active_count} active, {ended_count} ended auctions in {duration:.1f}s")
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        logger.info("Auction collector stopped by user")
    except Exception as e:
        logger.error(f"Auction collector error: {e}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    run()