import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any

import yaml

from ingestion.common.hypixel_client import HypixelClient

def load_config() -> Dict[str, Any]:
    """Load configuration for no-database mode."""
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# Database functions removed - using file-based storage only

def run():
    """Main bazaar collection loop (file-based storage only)."""
    cfg = load_config()
    client = HypixelClient(
        base_url=cfg["hypixel"]["base_url"],
        api_key=os.getenv("HYPIXEL_API_KEY"),
        max_requests_per_minute=cfg["hypixel"]["max_requests_per_minute"],
        timeout_seconds=cfg["hypixel"]["timeout_seconds"],
    )
    endpoint = cfg["hypixel"]["endpoints"]["bazaar"]
    poll_interval = cfg["hypixel"]["poll_interval_seconds"]
    
    print("Starting Bazaar collector loop (File-based mode)...")

    try:
        while True:
            ts = datetime.now(timezone.utc)
            try:
                data = client.get_json(endpoint)
                products = data.get("products", {})
                count = 0
                
                # File-based storage
                data_dir = cfg.get("no_database_mode", {}).get("data_directory", "data")
                os.makedirs(data_dir, exist_ok=True)
                output_path = os.path.join(data_dir, "bazaar_snapshots.ndjson")
                
                records_to_write = []
                for pid, pdata in products.items():
                    qs = pdata.get("quick_status", {})
                    record = {
                        "ts": ts.isoformat(),
                        "product_id": pid,
                        "buy_price": qs.get("buyPrice"),
                        "sell_price": qs.get("sellPrice"),
                        "buy_volume": qs.get("buyVolume"),
                        "sell_volume": qs.get("sellVolume"),
                        "buy_orders": qs.get("buyOrders"),
                        "sell_orders": qs.get("sellOrders")
                    }
                    records_to_write.append(record)
                    count += 1

                # Append all records for this cycle in one go
                with open(output_path, "a") as f:
                    for record in records_to_write:
                        f.write(json.dumps(record) + "\n")
                
                print(f"[{ts.isoformat()}] Wrote {count} bazaar snapshots to {output_path}")
                
            except Exception as e:
                print(f"[{ts.isoformat()}] Error in bazaar collection cycle: {e}")
                # Continue to next cycle after error
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("Bazaar collector stopped by user")
    except Exception as e:
        print(f"Bazaar collector error: {e}")
        raise
    finally:
        print("Bazaar collector stopped")

if __name__ == "__main__":
    run()
