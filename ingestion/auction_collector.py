import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import yaml

from ingestion.common.hypixel_client import HypixelClient
from storage.ndjson_storage import get_storage_instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration for no-database mode."""
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # No database configuration needed in no-database mode
    return cfg

# Database functions removed - using file-based storage only

def extract_auction_data(raw_auction: Dict[str, Any], enable_raw_capture: bool = False) -> Dict[str, Any]:
    """Extract and normalize auction data from raw API response."""
    start_time = datetime.fromtimestamp(raw_auction.get("start", 0) / 1000, tz=timezone.utc)
    end_time = datetime.fromtimestamp(raw_auction.get("end", 0) / 1000, tz=timezone.utc)
    
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
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "seller": raw_auction.get("auctioneer"),
        "bids_count": len(raw_auction.get("bids", [])),
        "category": raw_auction.get("category"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    if enable_raw_capture:
        data["raw_data"] = raw_auction
        
    return data

def extract_ended_auction_data(raw_auction: Dict[str, Any], enable_raw_capture: bool = False) -> Dict[str, Any]:
    """Extract and normalize ended auction data from raw API response."""
    base_data = extract_auction_data(raw_auction, enable_raw_capture)
    
    base_data.update({
        "sale_price": raw_auction.get("price"),
        "buyer": raw_auction.get("buyer"),
    })
    
    return base_data

def collect_auctions(client: HypixelClient, cfg: Dict[str, Any]) -> int:
    """Collect active auctions with pagination support (file-based storage only)."""
    endpoint = cfg["hypixel"]["endpoints"]["auctions"]
    max_pages = cfg["auction_house"]["max_pages_per_cycle"]
    enable_raw_capture = cfg["auction_house"]["enable_raw_capture"]
    
    data_dir = cfg.get("no_database_mode", {}).get("data_directory", "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "auctions.ndjson")
    
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
            
            records_to_write = []
            for auction in auctions:
                try:
                    auction_data = extract_auction_data(auction, enable_raw_capture)
                    records_to_write.append(auction_data)
                    total_auctions += 1
                except Exception as e:
                    logger.error(f"Error processing auction {auction.get('uuid', 'unknown')}: {e}")
            
            # Write all records for this page
            with open(output_path, "a") as f:
                for record in records_to_write:
                    f.write(json.dumps(record) + "\n")

            if page >= total_pages - 1:
                logger.info(f"Reached last page ({total_pages})")
                break
                
            page += 1
            
        except Exception as e:
            logger.error(f"Error fetching auctions page {page}: {e}")
            break
    
    logger.info(f"Collected {total_auctions} auctions across {page + 1} pages")
    return total_auctions

def collect_ended_auctions(client: HypixelClient, cfg: Dict[str, Any]) -> int:
    """Collect recently ended auctions (file-based storage only)."""
    endpoint = cfg["hypixel"]["endpoints"]["auctions_ended"]
    enable_raw_capture = cfg["auction_house"]["enable_raw_capture"]
    
    data_dir = cfg.get("no_database_mode", {}).get("data_directory", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Use separate directory for ended auctions
    ended_dir = os.path.join(data_dir, "auctions_ended")
    os.makedirs(ended_dir, exist_ok=True)
    output_path = os.path.join(ended_dir, "auctions_ended.ndjson")
    
    total_ended = 0
    try:
        logger.info("Fetching ended auctions")
        data = client.get_json(endpoint)
        
        ended_auctions = data.get("auctions", [])
        logger.info(f"Processing {len(ended_auctions)} ended auctions")
        
        records_to_write = []
        for auction in ended_auctions:
            try:
                auction_data = extract_ended_auction_data(auction, enable_raw_capture)
                records_to_write.append(auction_data)
                total_ended += 1
            except Exception as e:
                logger.error(f"Error processing ended auction {auction.get('uuid', 'unknown')}: {e}")
        
        # Write all ended auction records
        with open(output_path, "a") as f:
            for record in records_to_write:
                f.write(json.dumps(record) + "\n")
            
    except Exception as e:
        logger.error(f"Error fetching ended auctions: {e}")
    
    logger.info(f"Collected {total_ended} ended auctions")
    return total_ended

def run():
    """Main auction collection loop (file-based storage only)."""
    cfg = load_config()
    
    client = HypixelClient(
        base_url=cfg["hypixel"]["base_url"],
        api_key=os.getenv("HYPIXEL_API_KEY"),
        max_requests_per_minute=cfg["hypixel"]["max_requests_per_minute"],
        timeout_seconds=cfg["hypixel"]["timeout_seconds"],
    )
    
    poll_interval = cfg["auction_house"]["poll_interval_seconds"]
    logger.info(f"Starting auction collector loop (File-based mode, poll interval: {poll_interval}s)...")
    
    try:
        while True:
            cycle_start = datetime.now(timezone.utc)
            logger.info(f"[{cycle_start.isoformat()}] Starting collection cycle")
            
            try:
                active_count = collect_auctions(client, cfg)
                ended_count = collect_ended_auctions(client, cfg)
                
                cycle_end = datetime.now(timezone.utc)
                duration = (cycle_end - cycle_start).total_seconds()
                
                logger.info(f"[{cycle_end.isoformat()}] Cycle complete: {active_count} active, {ended_count} ended auctions in {duration:.1f}s (stored to NDJSON files)")
                
            except Exception as e:
                logger.error(f"Error in collection cycle: {e}")
                # Continue to next cycle after error
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        logger.info("Auction collector stopped by user")
    except Exception as e:
        logger.error(f"Auction collector error: {e}")
        raise
    finally:
        logger.info("Auction collector stopped")

if __name__ == "__main__":
    run()
