#!/usr/bin/env python3
"""
Example usage demonstrations for the SkyBlock Economic Modeling components.
This shows how to use each component individually without requiring full setup.
"""

import os
import time
from typing import Dict, Any

# Example 1: Using the HypixelClient (requires HYPIXEL_API_KEY)
def demo_hypixel_client():
    """Demonstrate HypixelClient usage."""
    print("=== HypixelClient Demo ===")
    
    # Check if API key is available
    api_key = os.getenv("HYPIXEL_API_KEY")
    if not api_key:
        print("⚠️  HYPIXEL_API_KEY not set. Skipping API demo.")
        print("   To test with real data, set: export HYPIXEL_API_KEY='your-key-here'")
        return
    
    from ingestion.common.hypixel_client import HypixelClient
    
    try:
        client = HypixelClient(
            base_url="https://api.hypixel.net",
            api_key=api_key,
            max_requests_per_minute=110
        )
        
        print("Fetching bazaar data...")
        data = client.get_json("/skyblock/bazaar")
        products = data.get("products", {})
        
        print(f"✓ Successfully fetched {len(products)} products")
        
        # Show a few example products
        sample_products = list(products.keys())[:3]
        for product_id in sample_products:
            product_data = products[product_id]
            quick_status = product_data.get("quick_status", {})
            buy_price = quick_status.get("buyPrice", "N/A")
            sell_price = quick_status.get("sellPrice", "N/A")
            print(f"  {product_id}: Buy={buy_price}, Sell={sell_price}")
            
    except Exception as e:
        print(f"✗ API request failed: {e}")

# Example 2: Demonstrate TokenBucket rate limiting
def demo_token_bucket():
    """Demonstrate TokenBucket rate limiter."""
    print("\n=== TokenBucket Demo ===")
    
    from ingestion.common.hypixel_client import TokenBucket
    
    # Create a bucket that allows 2 requests per minute (for demo purposes)
    bucket = TokenBucket(rate_per_minute=2, burst=2)
    
    print("Testing rate limiting (2 requests/minute, burst=2)...")
    
    # First 2 requests should be fast (using burst capacity)
    start_time = time.time()
    print("Request 1...", end=" ")
    bucket.consume(1)
    print(f"✓ ({time.time() - start_time:.2f}s)")
    
    print("Request 2...", end=" ")
    bucket.consume(1)
    print(f"✓ ({time.time() - start_time:.2f}s)")
    
    # Third request should be delayed
    print("Request 3 (should be delayed)...", end=" ")
    bucket.consume(1)
    elapsed = time.time() - start_time
    print(f"✓ ({elapsed:.2f}s)")
    
    if elapsed > 25:  # Should take ~30 seconds for the 3rd request
        print("✓ Rate limiting working correctly")
    else:
        print("⚠️  Rate limiting may not be working as expected")

# Example 3: Load and validate configuration
def demo_config():
    """Demonstrate configuration loading."""
    print("\n=== Configuration Demo ===")
    
    import yaml
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("Configuration sections:")
    for section, values in config.items():
        print(f"  {section}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"    {key}: {value}")
        else:
            print(f"    {values}")
    
    # Show environment variable handling
    database_url = os.getenv("DATABASE_URL", config["storage"]["database_url"])
    print(f"\nResolved DATABASE_URL: {database_url}")

# Example 4: Item ontology usage
def demo_ontology():
    """Demonstrate item ontology usage."""
    print("\n=== Item Ontology Demo ===")
    
    import json
    
    with open("item_ontology.json", "r") as f:
        ontology = json.load(f)
    
    print(f"Loaded {len(ontology)} items from ontology:")
    
    for item_id, item_data in ontology.items():
        print(f"\n{item_id} ({item_data['display_name']}):")
        
        # Show crafting recipe
        if item_data.get("craft", {}).get("inputs"):
            print("  Crafting recipe:")
            for input_item in item_data["craft"]["inputs"]:
                print(f"    - {input_item['qty']}x {input_item['item']}")
        
        # Show sources and sinks
        sources = item_data.get("sources", [])
        sinks = item_data.get("sinks", [])
        print(f"  Sources: {len(sources)}, Sinks: {len(sinks)}")

# Example 5: Database schema info
def demo_schema():
    """Show database schema information."""
    print("\n=== Database Schema Demo ===")
    
    with open("storage/schema.sql", "r") as f:
        schema = f.read()
    
    tables = []
    for line in schema.split('\n'):
        line = line.strip()
        if line.startswith('CREATE TABLE'):
            table_name = line.split()[4]  # Extract table name
            tables.append(table_name.replace('(', ''))
    
    print("Database tables:")
    for table in tables:
        print(f"  ✓ {table}")
    
    print(f"\nTotal schema size: {len(schema)} characters")

def main():
    """Run all demos."""
    print("SkyBlock Economic Modeling - Component Demos")
    print("=" * 50)
    
    # Run demos
    demo_config()
    demo_ontology()
    demo_schema()
    demo_token_bucket()
    demo_hypixel_client()
    
    print("\n" + "=" * 50)
    print("Demo complete! 🎉")
    print("\nNext steps:")
    print("1. Set up your database and run: psql \"$DATABASE_URL\" -f storage/schema.sql")
    print("2. Set HYPIXEL_API_KEY environment variable")
    print("3. Start data collection: python -m ingestion.bazaar_collector")

if __name__ == "__main__":
    main()