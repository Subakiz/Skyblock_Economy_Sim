"""
File-based data access layer for crafting profitability analysis.
Provides alternatives to database queries using NDJSON files.
"""

import os
import yaml
from typing import Dict, Any, Optional, Tuple
from storage.ndjson_storage import get_storage_instance


def get_bazaar_price_from_files(storage, product_id: str, price_type: str = "buy") -> Optional[float]:
    """Get latest Bazaar price (buy/sell) for a product from NDJSON files."""
    record = storage.get_latest_bazaar_prices(product_id)
    
    if not record:
        return None
    
    if price_type == "buy":
        return record.get("buy_price")
    else:  # sell
        return record.get("sell_price")


def get_ah_price_stats_from_files(storage, product_id: str, horizon: str = "1h", pricing: str = "median") -> Tuple[Optional[float], int]:
    """Get Auction House price statistics for a product from NDJSON files."""
    # Map horizon to hours
    hours_back = 1 if horizon in ["1h"] else 0.25  # 15 minutes
    
    stats = storage.get_auction_price_stats(product_id, hours_back=int(hours_back * 2))  # Double for safety
    
    if not stats:
        return None, 0
    
    # Map pricing preference to stats
    price_map = {
        "median": "median_price",
        "p25": "p25_price",
        "p50": "median_price",
        "p75": "p75_price",
        "avg": "avg_price"
    }
    
    price_key = price_map.get(pricing, "median_price")
    price = stats.get(price_key)
    count = stats.get("sale_count", 0)
    
    return price, count


def compute_craft_cost_from_files(
    storage, 
    ontology: Dict[str, Any], 
    product_id: str
) -> Tuple[float, str]:
    """
    Compute the cost to craft a product using NDJSON files instead of database.
    Returns (total_cost, path_description).
    """
    if product_id not in ontology:
        raise ValueError(f"Product {product_id} not found in ontology")
    
    item_data = ontology[product_id]
    craft_info = item_data.get("craft")
    
    if not craft_info:
        raise ValueError(f"Product {product_id} has no crafting recipe")
    
    total_cost = 0.0
    path_components = []
    
    # Get crafting inputs
    inputs = craft_info.get("inputs", [])
    fees = craft_info.get("fees", {})
    
    for input_item in inputs:
        input_id = input_item["item"]
        qty_needed = input_item["qty"]
        
        # Try to get price from Bazaar first (buy orders)
        input_price = get_bazaar_price_from_files(storage, input_id, "buy")
        
        if input_price is None:
            # Fall back to AH if not on Bazaar
            ah_price, _ = get_ah_price_stats_from_files(storage, input_id, "1h", "median")
            input_price = ah_price
        
        if input_price is None:
            # If still no price, try to recursively craft this input
            try:
                recursive_cost, recursive_path = compute_craft_cost_from_files(storage, ontology, input_id)
                input_price = recursive_cost
                path_components.append(f"craft({input_id}:{recursive_path})")
            except (ValueError, RecursionError):
                raise ValueError(f"No price available for input {input_id}")
        else:
            path_components.append(f"buy({input_id})")
        
        total_cost += input_price * qty_needed
    
    # Add fees
    npc_fee = fees.get("npc", 0)
    tax_bps = fees.get("tax_bps", 0)
    
    total_cost += npc_fee
    total_cost += total_cost * (tax_bps / 10000)  # basis points to percentage
    
    path_desc = " + ".join(path_components)
    return total_cost, path_desc


def load_config_for_file_mode() -> Dict[str, Any]:
    """Load configuration for file mode only (no database config needed)."""
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # No database configuration needed in no-database mode
    return cfg


def get_file_storage_if_enabled() -> Optional[Any]:
    """Get file storage instance if no-database mode is enabled."""
    cfg = load_config_for_file_mode()
    return get_storage_instance(cfg)