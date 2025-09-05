"""
Execution engine with realistic fill simulation (latency, slippage, fees).
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    BIN = "bin"  # Buy-it-now on AH

@dataclass
class Order:
    """Order to be executed."""
    product_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    target_price: Optional[float] = None
    
@dataclass  
class Fill:
    """Execution result."""
    product_id: str
    side: OrderSide
    quantity: int
    fill_price: float
    fees: float
    net_amount: float  # amount after fees
    executed_at: float  # timestamp

class ExecutionEngine:
    """Realistic execution simulation with fees, latency, and slippage."""
    
    def __init__(self, 
                 latency_ms: int = 500,
                 slippage_bps: int = 10,
                 ah_fee_bps: int = 100,
                 bazaar_fee_bps: int = 125):
        self.latency_ms = latency_ms
        self.slippage_bps = slippage_bps
        self.ah_fee_bps = ah_fee_bps
        self.bazaar_fee_bps = bazaar_fee_bps
    
    def execute_order(self, order: Order, market_data: Dict[str, Dict[str, float]], 
                     timestamp: float) -> Optional[Fill]:
        """
        Execute an order against market data with realistic constraints.
        Returns None if order cannot be filled.
        """
        product_data = market_data.get(order.product_id)
        if not product_data:
            return None
        
        # Simulate execution latency
        execution_time = timestamp + (self.latency_ms / 1000.0)
        
        # Determine execution venue and base price
        if order.order_type == OrderType.BIN:
            # AH Buy-it-now
            if order.side == OrderSide.BUY:
                base_price = product_data.get("median_price", 0)
                fee_bps = self.ah_fee_bps
            else:  # selling on AH
                base_price = product_data.get("p25_price", 0)  # conservative estimate
                fee_bps = self.ah_fee_bps
        else:
            # Bazaar market order
            if order.side == OrderSide.BUY:
                base_price = product_data.get("sell_price", 0)  # buy from sell offers
                fee_bps = self.bazaar_fee_bps
            else:  # selling to bazaar
                base_price = product_data.get("buy_price", 0)   # sell to buy offers
                fee_bps = self.bazaar_fee_bps
        
        if not base_price or base_price <= 0:
            return None
        
        # Apply slippage (negative for buys, positive for sells to be conservative)
        slippage_multiplier = 1 + (self.slippage_bps / 10000.0)
        if order.side == OrderSide.BUY:
            fill_price = base_price * slippage_multiplier
        else:
            fill_price = base_price / slippage_multiplier
        
        # Calculate fees
        gross_amount = fill_price * order.quantity
        fees = gross_amount * (fee_bps / 10000.0)
        
        if order.side == OrderSide.BUY:
            net_amount = -(gross_amount + fees)  # negative cash flow
        else:
            net_amount = gross_amount - fees     # positive cash flow
        
        return Fill(
            product_id=order.product_id,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            fees=fees,
            net_amount=net_amount,
            executed_at=execution_time
        )
    
    def can_fill_order(self, order: Order, market_data: Dict[str, Dict[str, float]]) -> bool:
        """Check if an order can be filled given current market data."""
        product_data = market_data.get(order.product_id)
        if not product_data:
            return False
        
        if order.order_type == OrderType.BIN:
            return product_data.get("median_price", 0) > 0
        else:
            if order.side == OrderSide.BUY:
                return product_data.get("sell_price", 0) > 0
            else:
                return product_data.get("buy_price", 0) > 0