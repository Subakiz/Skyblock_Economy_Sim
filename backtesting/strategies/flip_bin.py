"""
Simple flip strategy: buy low, sell at median BIN price.
"""

from typing import List, Dict, Any, Optional
from backtesting.execution import Order, OrderSide, OrderType
from backtesting.data_feed import MarketSnapshot

class FlipStrategy:
    """Buy underpriced BINs and relist at market price."""
    
    def __init__(self, 
                 target_items: List[str],
                 min_profit_margin: float = 50000,  # minimum profit in coins
                 max_position_size: int = 10):      # max inventory per item
        self.target_items = target_items
        self.min_profit_margin = min_profit_margin
        self.max_position_size = max_position_size
    
    def generate_orders(self, 
                       snapshot: MarketSnapshot, 
                       portfolio: Dict[str, Any]) -> List[Order]:
        """Generate buy/sell orders based on current market conditions."""
        orders = []
        
        current_inventory = portfolio.get("inventory", {})
        
        for item_id in self.target_items:
            ah_data = snapshot.ah_data.get(item_id, {})
            if not ah_data:
                continue
            
            median_price = ah_data.get("median_price", 0)
            p25_price = ah_data.get("p25_price", 0)
            
            if not median_price or not p25_price:
                continue
            
            current_qty = current_inventory.get(item_id, 0)
            
            # Buy signal: P25 significantly below median (undervalued)
            if p25_price > 0 and current_qty < self.max_position_size:
                potential_profit = median_price - p25_price
                if potential_profit >= self.min_profit_margin:
                    # Buy at P25 level
                    orders.append(Order(
                        product_id=item_id,
                        side=OrderSide.BUY,
                        order_type=OrderType.BIN,
                        quantity=1,
                        target_price=p25_price
                    ))
            
            # Sell signal: have inventory and can sell at profit
            if current_qty > 0 and median_price > 0:
                # Sell at median price
                orders.append(Order(
                    product_id=item_id,
                    side=OrderSide.SELL,
                    order_type=OrderType.BIN,
                    quantity=min(current_qty, 3),  # sell up to 3 at a time
                    target_price=median_price
                ))
        
        return orders