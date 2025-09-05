"""
Backtesting engine that simulates portfolio performance over time.
"""

import json
import math
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from backtesting.data_feed import DataFeed, MarketSnapshot
from backtesting.execution import ExecutionEngine, Fill

@dataclass
class Portfolio:
    """Portfolio state tracking cash, inventory, and positions."""
    cash: float = 0.0
    inventory: Dict[str, int] = field(default_factory=dict)
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    
    def add_cash(self, amount: float):
        """Add cash to portfolio (can be negative for expenses)."""
        self.cash += amount
    
    def add_inventory(self, product_id: str, quantity: int):
        """Add inventory (can be negative for sales)."""
        current = self.inventory.get(product_id, 0)
        new_quantity = current + quantity
        
        if new_quantity <= 0:
            self.inventory.pop(product_id, None)
        else:
            self.inventory[product_id] = new_quantity

@dataclass
class BacktestMetrics:
    """Summary metrics for a backtest run."""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    final_cash: float = 0.0
    final_inventory_value: float = 0.0

class BacktestingEngine:
    """Main backtesting simulation engine."""
    
    def __init__(self, 
                 initial_capital: float,
                 execution_engine: ExecutionEngine,
                 max_inventory_slots: int = 1000):
        self.initial_capital = initial_capital
        self.execution_engine = execution_engine
        self.max_inventory_slots = max_inventory_slots
        
        # State tracking
        self.portfolio = Portfolio(cash=initial_capital)
        self.trade_history: List[Fill] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
    def can_afford_trade(self, estimated_cost: float) -> bool:
        """Check if portfolio has enough cash for a trade."""
        return self.portfolio.cash >= estimated_cost
    
    def has_inventory_space(self, additional_items: int = 1) -> bool:
        """Check if portfolio has inventory space."""
        current_slots = sum(self.portfolio.inventory.values())
        return current_slots + additional_items <= self.max_inventory_slots
    
    def calculate_portfolio_value(self, market_data: Dict[str, Dict[str, float]]) -> float:
        """Calculate total portfolio value including inventory."""
        total_value = self.portfolio.cash
        
        # Value inventory at current market prices
        for product_id, quantity in self.portfolio.inventory.items():
            # Use conservative pricing for inventory valuation
            ah_data = market_data.get(product_id, {})
            bazaar_data = market_data.get(product_id, {})
            
            # Try AH P25 (conservative) or bazaar buy price
            market_price = ah_data.get("p25_price", 0)
            if not market_price:
                market_price = bazaar_data.get("buy_price", 0)
            
            if market_price:
                total_value += market_price * quantity * 0.95  # 5% haircut for liquidity
        
        return total_value
    
    def execute_fills(self, fills: List[Fill]):
        """Apply executed fills to portfolio."""
        for fill in fills:
            # Update cash
            self.portfolio.add_cash(fill.net_amount)
            
            # Update inventory
            if fill.side.value == "buy":
                self.portfolio.add_inventory(fill.product_id, fill.quantity)
            else:
                self.portfolio.add_inventory(fill.product_id, -fill.quantity)
            
            # Track trades
            self.trade_history.append(fill)
    
    def run_backtest(self, 
                    data_feed: DataFeed, 
                    strategy,
                    progress_callback=None) -> BacktestMetrics:
        """Run full backtest simulation."""
        
        snapshot_count = 0
        peak_value = self.initial_capital
        max_drawdown = 0.0
        
        for snapshot in data_feed.stream_snapshots():
            snapshot_count += 1
            
            # Combine market data (bazaar + AH)
            combined_market_data = {**snapshot.bazaar_data, **snapshot.ah_data}
            
            # Generate strategy orders
            portfolio_dict = {
                "cash": self.portfolio.cash,
                "inventory": self.portfolio.inventory.copy(),
                "total_value": self.portfolio.total_value
            }
            
            orders = strategy.generate_orders(snapshot, portfolio_dict)
            
            # Execute orders
            fills = []
            for order in orders:
                # Check constraints
                if order.side.value == "buy":
                    estimated_cost = (order.target_price or combined_market_data.get(order.product_id, {}).get("sell_price", 0)) * order.quantity
                    if not self.can_afford_trade(estimated_cost):
                        continue
                    if not self.has_inventory_space(order.quantity):
                        continue
                
                # Execute order
                fill = self.execution_engine.execute_order(
                    order, combined_market_data, snapshot.timestamp.timestamp()
                )
                
                if fill:
                    fills.append(fill)
            
            # Apply fills
            self.execute_fills(fills)
            
            # Calculate portfolio value
            current_value = self.calculate_portfolio_value(combined_market_data)
            self.portfolio.total_value = current_value
            
            # Track drawdown
            if current_value > peak_value:
                peak_value = current_value
            
            drawdown = peak_value - current_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            # Record portfolio state
            self.portfolio_history.append({
                "timestamp": snapshot.timestamp,
                "cash": self.portfolio.cash,
                "inventory_value": current_value - self.portfolio.cash,
                "total_value": current_value,
                "drawdown": drawdown,
            })
            
            if progress_callback and snapshot_count % 100 == 0:
                progress_callback(snapshot_count, current_value)
        
        # Calculate final metrics
        return self._calculate_metrics(max_drawdown, peak_value)
    
    def _calculate_metrics(self, max_drawdown: float, peak_value: float) -> BacktestMetrics:
        """Calculate summary metrics from backtest results."""
        final_value = self.portfolio.total_value
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        max_drawdown_pct = (max_drawdown / peak_value) * 100 if peak_value > 0 else 0
        
        # Simple Sharpe calculation (annualized, assume daily returns)
        if len(self.portfolio_history) > 1:
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_val = self.portfolio_history[i-1]["total_value"]
                curr_val = self.portfolio_history[i]["total_value"]
                if prev_val > 0:
                    returns.append((curr_val - prev_val) / prev_val)
            
            if returns:
                avg_return = sum(returns) / len(returns)
                return_std = math.sqrt(sum((r - avg_return)**2 for r in returns) / len(returns))
                sharpe_ratio = (avg_return / return_std) * math.sqrt(365) if return_std > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Trade statistics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for fill in self.trade_history if fill.net_amount > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_trade_pnl = sum(fill.net_amount for fill in self.trade_history) / total_trades if total_trades > 0 else 0
        
        return BacktestMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            final_cash=self.portfolio.cash,
            final_inventory_value=final_value - self.portfolio.cash
        )