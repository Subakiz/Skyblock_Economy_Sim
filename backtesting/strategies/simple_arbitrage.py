"""
Simple arbitrage strategy: exploit price differences between Bazaar and AH.
"""

from typing import List, Dict, Any
from backtesting.execution import Order, OrderSide, OrderType
from backtesting.data_feed import MarketSnapshot

class SimpleArbitrageStrategy:
    """Exploit price differences between Bazaar and Auction House."""
    
    def __init__(self, 
                 target_items: List[str],
                 min_arbitrage_profit: float = 10000,   # minimum profit per unit
                 max_position_size: int = 100):         # max quantity per trade
        self.target_items = target_items
        self.min_arbitrage_profit = min_arbitrage_profit
        self.max_position_size = max_position_size
    
    def generate_orders(self, 
                       snapshot: MarketSnapshot, 
                       portfolio: Dict[str, Any]) -> List[Order]:
        """Generate arbitrage orders between Bazaar and AH."""
        orders = []
        inventory = portfolio.get("inventory", {})
        
        for item_id in self.target_items:
            bazaar_data = snapshot.bazaar_data.get(item_id, {})
            ah_data = snapshot.ah_data.get(item_id, {})
            
            if not bazaar_data or not ah_data:
                continue
            
            bazaar_buy = bazaar_data.get("buy_price", 0)    # what bazaar pays us
            bazaar_sell = bazaar_data.get("sell_price", 0)  # what we pay bazaar
            ah_median = ah_data.get("median_price", 0)      # AH market price
            ah_p25 = ah_data.get("p25_price", 0)           # cheap AH listings
            
            current_qty = inventory.get(item_id, 0)
            
            # Arbitrage opportunity 1: Buy cheap from Bazaar, sell on AH
            if bazaar_sell > 0 and ah_median > 0:
                profit_per_unit = ah_median * 0.99 - bazaar_sell  # after AH fees
                if profit_per_unit >= self.min_arbitrage_profit and current_qty < self.max_position_size:
                    orders.append(Order(
                        product_id=item_id,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=min(10, self.max_position_size - current_qty)
                    ))
            
            # Arbitrage opportunity 2: Buy cheap on AH, sell to Bazaar
            if ah_p25 > 0 and bazaar_buy > 0:
                profit_per_unit = bazaar_buy * 0.9875 - ah_p25  # after Bazaar fees
                if profit_per_unit >= self.min_arbitrage_profit and current_qty < self.max_position_size:
                    orders.append(Order(
                        product_id=item_id,
                        side=OrderSide.BUY,
                        order_type=OrderType.BIN,
                        quantity=min(5, self.max_position_size - current_qty),
                        target_price=ah_p25
                    ))
            
            # Liquidate inventory when profitable
            if current_qty > 0:
                if bazaar_buy > 0:
                    # Sell to bazaar
                    orders.append(Order(
                        product_id=item_id,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=current_qty
                    ))
                elif ah_median > 0:
                    # Sell on AH
                    orders.append(Order(
                        product_id=item_id,
                        side=OrderSide.SELL,
                        order_type=OrderType.BIN,
                        quantity=current_qty
                    ))
        
        return orders