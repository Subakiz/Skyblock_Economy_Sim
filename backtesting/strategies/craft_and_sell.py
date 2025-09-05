"""
Craft and sell strategy: craft items via ontology then sell on AH.
"""

import json
from typing import List, Dict, Any, Optional
from backtesting.execution import Order, OrderSide, OrderType
from backtesting.data_feed import MarketSnapshot

def load_ontology() -> Dict[str, Any]:
    """Load item ontology for crafting recipes."""
    with open("item_ontology.json", "r") as f:
        return json.load(f)

class CraftAndSellStrategy:
    """Craft profitable items and sell them."""
    
    def __init__(self, 
                 target_items: List[str],
                 min_profit_margin: float = 100000,    # minimum profit per craft
                 max_craft_batches: int = 5):          # max crafting operations
        self.target_items = target_items
        self.min_profit_margin = min_profit_margin
        self.max_craft_batches = max_craft_batches
        self.ontology = load_ontology()
    
    def can_craft_item(self, item_id: str, inventory: Dict[str, int]) -> bool:
        """Check if we have enough materials to craft an item."""
        if item_id not in self.ontology:
            return False
        
        craft_info = self.ontology[item_id].get("craft")
        if not craft_info:
            return False
        
        inputs = craft_info.get("inputs", [])
        for input_item in inputs:
            required_id = input_item["item"]
            required_qty = input_item["qty"]
            
            available_qty = inventory.get(required_id, 0)
            if available_qty < required_qty:
                return False
        
        return True
    
    def calculate_craft_profit(self, 
                              item_id: str, 
                              snapshot: MarketSnapshot) -> Optional[float]:
        """Calculate profit from crafting an item."""
        if item_id not in self.ontology:
            return None
        
        craft_info = self.ontology[item_id].get("craft")
        if not craft_info:
            return None
        
        # Calculate input cost
        total_input_cost = 0
        inputs = craft_info.get("inputs", [])
        
        for input_item in inputs:
            input_id = input_item["item"]
            qty_needed = input_item["qty"]
            
            # Try bazaar sell price first (what we pay)
            bazaar_data = snapshot.bazaar_data.get(input_id, {})
            input_price = bazaar_data.get("sell_price", 0)
            
            if not input_price:
                # Fall back to AH median
                ah_data = snapshot.ah_data.get(input_id, {})
                input_price = ah_data.get("median_price", 0)
            
            if not input_price:
                return None  # Cannot price input
            
            total_input_cost += input_price * qty_needed
        
        # Add crafting fees
        fees = craft_info.get("fees", {})
        total_input_cost += fees.get("npc", 0)
        tax_bps = fees.get("tax_bps", 0)
        total_input_cost *= (1 + tax_bps / 10000.0)
        
        # Calculate expected sale price
        ah_data = snapshot.ah_data.get(item_id, {})
        sale_price = ah_data.get("median_price", 0)
        
        if not sale_price:
            return None
        
        # Account for AH fees (assume 1%)
        net_sale_price = sale_price * 0.99
        
        return net_sale_price - total_input_cost
    
    def generate_orders(self, 
                       snapshot: MarketSnapshot, 
                       portfolio: Dict[str, Any]) -> List[Order]:
        """Generate orders for craft-and-sell strategy."""
        orders = []
        inventory = portfolio.get("inventory", {})
        
        # Check each target item for profitability
        for item_id in self.target_items:
            # Skip if we already have inventory of this item
            if inventory.get(item_id, 0) > 0:
                # Sell existing inventory
                orders.append(Order(
                    product_id=item_id,
                    side=OrderSide.SELL,
                    order_type=OrderType.BIN,
                    quantity=inventory[item_id]
                ))
                continue
            
            # Calculate craft profitability
            profit = self.calculate_craft_profit(item_id, snapshot)
            if not profit or profit < self.min_profit_margin:
                continue
            
            # Check if we can craft (have materials)
            if not self.can_craft_item(item_id, inventory):
                # Need to buy materials
                craft_info = self.ontology[item_id].get("craft", {})
                inputs = craft_info.get("inputs", [])
                
                for input_item in inputs:
                    input_id = input_item["item"]
                    qty_needed = input_item["qty"]
                    available = inventory.get(input_id, 0)
                    
                    if available < qty_needed:
                        # Buy the missing materials
                        orders.append(Order(
                            product_id=input_id,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=qty_needed - available
                        ))
        
        return orders[:self.max_craft_batches * 3]  # Limit order count