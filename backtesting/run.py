"""
CLI runner for backtesting strategies.
"""

import os
import sys
import yaml
import argparse
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, Any

from backtesting.engine import BacktestingEngine, BacktestMetrics
from backtesting.execution import ExecutionEngine
from backtesting.data_feed import DataFeed
from backtesting.strategies.flip_bin import FlipStrategy
from backtesting.strategies.craft_and_sell import CraftAndSellStrategy
from backtesting.strategies.simple_arbitrage import SimpleArbitrageStrategy

def load_config() -> Dict[str, Any]:
    """Load configuration with environment variable overrides."""
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    db_url = os.getenv("DATABASE_URL") or cfg["storage"]["database_url"]
    cfg["storage"]["database_url"] = db_url
    
    return cfg

def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

def create_strategy(strategy_name: str, item_id: str, params: Dict[str, Any]):
    """Create strategy instance based on name and parameters."""
    target_items = [item_id] if item_id else ["ENCHANTED_LAPIS_BLOCK", "ENCHANTED_LAPIS_LAZULI"]
    
    if strategy_name == "flip_bin":
        return FlipStrategy(
            target_items=target_items,
            min_profit_margin=params.get("min_profit", 50000),
            max_position_size=params.get("max_position", 10)
        )
    elif strategy_name == "craft_and_sell":
        return CraftAndSellStrategy(
            target_items=target_items,
            min_profit_margin=params.get("min_profit", 100000),
            max_craft_batches=params.get("max_batches", 5)
        )
    elif strategy_name == "simple_arbitrage":
        return SimpleArbitrageStrategy(
            target_items=target_items,
            min_arbitrage_profit=params.get("min_profit", 10000),
            max_position_size=params.get("max_position", 100)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def print_backtest_results(metrics: BacktestMetrics, duration_days: int):
    """Print formatted backtest results."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Duration:              {duration_days} days")
    print(f"Total Return:          {metrics.total_return:,.2f} coins ({metrics.total_return_pct:.2f}%)")
    print(f"Max Drawdown:          {metrics.max_drawdown:,.2f} coins ({metrics.max_drawdown_pct:.2f}%)")
    print(f"Sharpe Ratio:          {metrics.sharpe_ratio:.3f}")
    print(f"Total Trades:          {metrics.total_trades}")
    print(f"Win Rate:              {metrics.win_rate:.1f}%")
    print(f"Avg Trade P&L:         {metrics.avg_trade_pnl:,.2f} coins")
    print(f"Final Cash:            {metrics.final_cash:,.2f} coins")
    print(f"Final Inventory Value: {metrics.final_inventory_value:,.2f} coins")
    print("="*60)

def main():
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument("--strategy", required=True,
                       choices=["flip_bin", "craft_and_sell", "simple_arbitrage"],
                       help="Strategy to backtest")
    parser.add_argument("--item", help="Target item ID (optional)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10_000_000,
                       help="Starting capital in coins")
    parser.add_argument("--params", help="Strategy parameters YAML file")
    parser.add_argument("--interval", type=int, default=15,
                       help="Data feed interval in minutes")
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if start_date >= end_date:
        print("Error: Start date must be before end date", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    cfg = load_config()
    
    # Load strategy parameters
    strategy_params = {}
    if args.params:
        try:
            with open(args.params, "r") as f:
                strategy_params = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading params file: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Create components
    db_url = cfg["storage"]["database_url"]
    conn = psycopg2.connect(db_url)
    
    execution_engine = ExecutionEngine(
        latency_ms=cfg["backtesting"]["execution_latency_ms"],
        slippage_bps=cfg["backtesting"]["slippage_bps"],
        ah_fee_bps=cfg["auction_house"]["fee_bps"]
    )
    
    backtesting_engine = BacktestingEngine(
        initial_capital=args.capital,
        execution_engine=execution_engine,
        max_inventory_slots=cfg["backtesting"]["max_inventory_slots"]
    )
    
    data_feed = DataFeed(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
        interval_minutes=args.interval
    )
    
    try:
        strategy = create_strategy(args.strategy, args.item, strategy_params)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run backtest
    print(f"Starting backtest: {args.strategy}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Capital: {args.capital:,.0f} coins")
    print(f"Target item: {args.item or 'Multiple'}")
    print("Running...")
    
    def progress_callback(snapshot_count: int, current_value: float):
        print(f"  Processed {snapshot_count} snapshots, Portfolio: {current_value:,.0f} coins")
    
    try:
        metrics = backtesting_engine.run_backtest(
            data_feed, strategy, progress_callback
        )
        
        duration_days = (end_date - start_date).days
        print_backtest_results(metrics, duration_days)
        
    except Exception as e:
        print(f"Backtest failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()