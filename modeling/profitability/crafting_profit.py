"""
Crafting Profitability Engine

Combines item ontology with live Bazaar and Auction House prices to compute:
- Craft input costs  
- Expected sale prices
- Margins (gross/net after fees)
- ROI and turnover-adjusted profitability

Supports both database and file-based (NDJSON) data sources.
"""

import os
import sys
import json
import yaml
import argparse
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from modeling.profitability.file_data_access import (
    get_file_storage_if_enabled, 
    get_bazaar_price_from_files,
    get_ah_price_stats_from_files,
    compute_craft_cost_from_files
)

@dataclass
class CraftProfitability:
    """Results from craft profitability analysis."""
    product_id: str
    craft_cost: float
    expected_sale_price: float
    gross_margin: float
    net_margin: float  # after fees
    roi_percent: float
    turnover_adj_profit: float
    best_path: str
    sell_volume: int
    data_age_minutes: int

def load_config() -> Dict[str, Any]:
    """Load configuration with environment variable overrides."""
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    cfg["storage"]["database_url"] = db_url
    
    return cfg

def load_item_ontology() -> Dict[str, Any]:
    """Load item ontology JSON file."""
    with open("item_ontology.json", "r") as f:
        return json.load(f)

def get_bazaar_price(conn, product_id: str, price_type: str = "buy") -> Optional[float]:
    """Get latest Bazaar price (buy/sell) for a product."""
    with conn.cursor() as cur:
        if price_type == "buy":
            cur.execute("""
                SELECT buy_price FROM bazaar_snapshots 
                WHERE product_id = %s AND buy_price IS NOT NULL
                ORDER BY ts DESC LIMIT 1
            """, (product_id,))
        else:  # sell
            cur.execute("""
                SELECT sell_price FROM bazaar_snapshots 
                WHERE product_id = %s AND sell_price IS NOT NULL
                ORDER BY ts DESC LIMIT 1
            """, (product_id,))
        
        row = cur.fetchone()
        return float(row[0]) if row and row[0] else None

def get_ah_price_stats(conn, product_id: str, horizon: str = "1h", pricing: str = "median") -> Tuple[Optional[float], int]:
    """Get Auction House price statistics for a product."""
    
    # Map horizon to view table
    view_name = "ah_prices_1h" if horizon in ["1h", "4h"] else "ah_prices_15m"
    
    # Map pricing preference to column
    price_columns = {
        "median": "median_price",
        "p25": "p25_price", 
        "p50": "median_price",
        "p75": "p75_price",
        "avg": "avg_price"
    }
    price_col = price_columns.get(pricing, "median_price")
    
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT {price_col}, sale_count
            FROM {view_name}
            WHERE item_id = %s AND sale_count > 0
            ORDER BY time_bucket DESC
            LIMIT 1
        """, (product_id,))
        
        row = cur.fetchone()
        if row and row[0]:
            return float(row[0]), int(row[1])
        return None, 0

def compute_craft_cost(ontology: Dict[str, Any], conn, product_id: str) -> Tuple[float, str]:
    """
    Compute the cost to craft a product using the item ontology.
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
        input_price = get_bazaar_price(conn, input_id, "buy")
        
        if input_price is None:
            # Fall back to AH if not on Bazaar
            ah_price, _ = get_ah_price_stats(conn, input_id, "1h", "median")
            input_price = ah_price
        
        if input_price is None:
            # If still no price, try to recursively craft this input
            try:
                recursive_cost, recursive_path = compute_craft_cost(ontology, conn, input_id)
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

def compute_profitability(
    conn, 
    ontology: Dict[str, Any], 
    product_id: str, 
    horizon: str = "1h",
    pricing: str = "median",
    ah_fee_bps: int = 100
) -> CraftProfitability:
    """
    Compute full profitability analysis for a craftable product.
    
    Automatically detects whether to use database or file-based data source.
    """
    # Check if file-based mode is enabled
    storage = get_file_storage_if_enabled()
    
    if storage is not None:
        return compute_profitability_from_files(
            storage, ontology, product_id, horizon, pricing, ah_fee_bps
        )
    else:
        return compute_profitability_from_db(
            conn, ontology, product_id, horizon, pricing, ah_fee_bps
        )


def compute_profitability_from_files(
    storage,
    ontology: Dict[str, Any], 
    product_id: str, 
    horizon: str = "1h",
    pricing: str = "median",
    ah_fee_bps: int = 100
) -> CraftProfitability:
    """
    Compute profitability analysis using file-based data source.
    """
    # Compute craft cost
    try:
        craft_cost, best_path = compute_craft_cost_from_files(storage, ontology, product_id)
    except ValueError as e:
        raise ValueError(f"Cannot compute craft cost: {e}")
    
    # Get expected sale price (prefer AH for crafted items)
    sale_price, sale_volume = get_ah_price_stats_from_files(storage, product_id, horizon, pricing)
    
    if sale_price is None:
        # Fall back to Bazaar sell price
        sale_price = get_bazaar_price_from_files(storage, product_id, "sell")
        sale_volume = 0  # No volume data for Bazaar
    
    if sale_price is None:
        raise ValueError(f"No sale price available for {product_id}")
    
    # Calculate margins
    gross_margin = sale_price - craft_cost
    ah_fee = sale_price * (ah_fee_bps / 10000)
    net_margin = gross_margin - ah_fee
    
    # Calculate ROI
    roi_percent = (net_margin / craft_cost * 100) if craft_cost > 0 else 0
    
    # Estimate turnover-adjusted profit (simple proxy using volume)
    turnover_multiplier = min(sale_volume / 50.0, 2.0) if sale_volume > 0 else 0.5
    turnover_adj_profit = net_margin * turnover_multiplier
    
    # Estimate data age (simplified - assume recent)
    data_age_minutes = 15  # Placeholder
    
    return CraftProfitability(
        product_id=product_id,
        craft_cost=craft_cost,
        expected_sale_price=sale_price,
        gross_margin=gross_margin,
        net_margin=net_margin,
        roi_percent=roi_percent,
        turnover_adj_profit=turnover_adj_profit,
        best_path=best_path,
        sell_volume=sale_volume,
        data_age_minutes=data_age_minutes
    )


def compute_profitability_from_db(
    conn, 
    ontology: Dict[str, Any], 
    product_id: str, 
    horizon: str = "1h",
    pricing: str = "median",
    ah_fee_bps: int = 100
) -> CraftProfitability:
    """
    Compute profitability analysis using database.
    """
    # Compute craft cost
    try:
        craft_cost, best_path = compute_craft_cost(ontology, conn, product_id)
    except ValueError as e:
        raise ValueError(f"Cannot compute craft cost: {e}")
    
    # Get expected sale price (prefer AH for crafted items)
    sale_price, sale_volume = get_ah_price_stats(conn, product_id, horizon, pricing)
    
    if sale_price is None:
        # Fall back to Bazaar sell price
        sale_price = get_bazaar_price(conn, product_id, "sell")
        sale_volume = 0  # No volume data for Bazaar
    
    if sale_price is None:
        raise ValueError(f"No sale price available for {product_id}")
    
    # Calculate margins
    gross_margin = sale_price - craft_cost
    ah_fee = sale_price * (ah_fee_bps / 10000)
    net_margin = gross_margin - ah_fee
    
    # Calculate ROI
    roi_percent = (net_margin / craft_cost * 100) if craft_cost > 0 else 0
    
    # Estimate turnover-adjusted profit (simple proxy using volume)
    turnover_multiplier = min(sale_volume / 50.0, 2.0) if sale_volume > 0 else 0.5
    turnover_adj_profit = net_margin * turnover_multiplier
    
    # Estimate data age (simplified - assume recent)
    data_age_minutes = 15  # Placeholder
    
    return CraftProfitability(
        product_id=product_id,
        craft_cost=craft_cost,
        expected_sale_price=sale_price,
        gross_margin=gross_margin,
        net_margin=net_margin,
        roi_percent=roi_percent,
        turnover_adj_profit=turnover_adj_profit,
        best_path=best_path,
        sell_volume=sale_volume,
        data_age_minutes=data_age_minutes
    )

def print_profitability_report(result: CraftProfitability):
    """Print a formatted profitability report."""
    print(f"\n{'='*60}")
    print(f"CRAFTING PROFITABILITY ANALYSIS: {result.product_id}")
    print(f"{'='*60}")
    print(f"Craft Cost:        {result.craft_cost:,.2f} coins")
    print(f"Expected Sale:     {result.expected_sale_price:,.2f} coins")
    print(f"Gross Margin:      {result.gross_margin:,.2f} coins")
    print(f"Net Margin:        {result.net_margin:,.2f} coins")
    print(f"ROI:               {result.roi_percent:.2f}%")
    print(f"Turnover Adj.:     {result.turnover_adj_profit:,.2f} coins")
    print(f"Sale Volume:       {result.sell_volume} recent sales")
    print(f"Data Age:          {result.data_age_minutes} minutes")
    print(f"\nBest Craft Path:")
    print(f"  {result.best_path}")
    print(f"{'='*60}")

def main():
    """CLI entry point for craft profitability analysis."""
    parser = argparse.ArgumentParser(description="Compute crafting profitability")
    parser.add_argument("product_id", help="Product ID to analyze")
    parser.add_argument("--horizon", default="1h", choices=["15m", "1h"], 
                       help="Price data time horizon")
    parser.add_argument("--pricing", default="median", 
                       choices=["median", "p25", "p50", "p75", "avg"],
                       help="Pricing method for sale price")
    
    args = parser.parse_args()
    
    # Load configuration and ontology
    cfg = load_config()
    ontology = load_item_ontology()
    ah_fee_bps = cfg["auction_house"]["fee_bps"]
    
    # Check if file-based mode is enabled
    storage = get_file_storage_if_enabled()
    
    conn = None
    if storage is None:
        # Database mode
        db_url = cfg["storage"]["database_url"]
        conn = psycopg2.connect(db_url)
        print("Using database mode")
    else:
        print("Using file-based mode")
    
    try:
        # Compute profitability (automatically detects mode)
        result = compute_profitability(
            conn, ontology, args.product_id, 
            args.horizon, args.pricing, ah_fee_bps
        )
        
        # Print report
        print_profitability_report(result)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()