import os
import time
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from typing import Dict, Any

import yaml

from ingestion.common.hypixel_client import HypixelClient

def load_config() -> Dict[str, Any]:
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    # env interpolation for DATABASE_URL
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    cfg["storage"]["database_url"] = db_url
    return cfg

def upsert_bazaar_snapshot(conn, ts, product_id, qs):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bazaar_snapshots
                (ts, product_id, buy_price, sell_price, buy_volume, sell_volume, buy_orders, sell_orders)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ts, product_id) DO NOTHING
            """,
            (
                ts,
                product_id,
                qs.get("buyPrice"),
                qs.get("sellPrice"),
                qs.get("buyVolume"),
                qs.get("sellVolume"),
                qs.get("buyOrders"),
                qs.get("sellOrders"),
            ),
        )

def run():
    cfg = load_config()
    client = HypixelClient(
        base_url=cfg["hypixel"]["base_url"],
        api_key=os.getenv("HYPIXEL_API_KEY"),
        max_requests_per_minute=cfg["hypixel"]["max_requests_per_minute"],
        timeout_seconds=cfg["hypixel"]["timeout_seconds"],
    )
    endpoint = cfg["hypixel"]["endpoints"]["bazaar"]
    poll_interval = cfg["hypixel"]["poll_interval_seconds"]
    db_url = cfg["storage"]["database_url"]

    print("Connecting to DB...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    print("Starting Bazaar collector loop...")

    try:
        while True:
            ts = datetime.now(timezone.utc)
            data = client.get_json(endpoint)
            products = data.get("products", {})
            count = 0
            with conn:
                with conn.cursor() as cur:
                    for pid, pdata in products.items():
                        qs = pdata.get("quick_status", {})
                        upsert_bazaar_snapshot(conn, ts, pid, qs)
                        count += 1
            print(f"[{ts.isoformat()}] Upserted {count} bazaar snapshots")
            time.sleep(poll_interval)
    finally:
        conn.close()

if __name__ == "__main__":
    run()