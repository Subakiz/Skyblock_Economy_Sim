"""
Unit tests for Phase 2 components.
"""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from ingestion.auction_collector import extract_auction_data, extract_ended_auction_data
from modeling.profitability.crafting_profit import compute_craft_cost, load_item_ontology
from backtesting.execution import ExecutionEngine, Order, OrderSide, OrderType
from backtesting.strategies.flip_bin import FlipStrategy

class TestAuctionCollector(unittest.TestCase):
    """Test auction data extraction and normalization."""
    
    def test_extract_auction_data(self):
        """Test auction data extraction from raw API response."""
        raw_auction = {
            "uuid": "test-uuid-123",
            "item_name": "Enchanted Lapis Lazuli",
            "tier": "UNCOMMON",
            "bin": True,
            "starting_bid": 50000,
            "highest_bid_amount": 55000,
            "start": 1640995200000,  # 2022-01-01 00:00:00 UTC
            "end": 1641081600000,    # 2022-01-02 00:00:00 UTC
            "auctioneer": "test_seller",
            "category": "MISC",
            "bids": [{"bidder": "bidder1", "amount": 55000}]
        }
        
        extracted = extract_auction_data(raw_auction, enable_raw_capture=True)
        
        self.assertEqual(extracted["uuid"], "test-uuid-123")
        self.assertEqual(extracted["item_id"], "ENCHANTED_LAPIS_LAZULI")
        self.assertEqual(extracted["item_name"], "Enchanted Lapis Lazuli")
        self.assertEqual(extracted["tier"], "UNCOMMON")
        self.assertTrue(extracted["bin"])
        self.assertEqual(extracted["starting_bid"], 50000)
        self.assertEqual(extracted["highest_bid"], 55000)
        self.assertEqual(extracted["seller"], "test_seller")
        self.assertEqual(extracted["category"], "MISC")
        self.assertEqual(extracted["bids_count"], 1)
        self.assertIsNotNone(extracted["raw_data"])
        
        # Test timestamp conversion
        self.assertIsInstance(extracted["start_time"], datetime)
        self.assertIsInstance(extracted["end_time"], datetime)
        self.assertEqual(extracted["start_time"].year, 2022)

    def test_extract_ended_auction_data(self):
        """Test ended auction data extraction."""
        raw_auction = {
            "uuid": "test-uuid-456",
            "item_name": "Enchanted Lapis Block", 
            "price": 150000,  # final sale price
            "buyer": "test_buyer",
            "bin": True,
            "start": 1640995200000,
            "end": 1641081600000,
            "auctioneer": "test_seller"
        }
        
        extracted = extract_ended_auction_data(raw_auction)
        
        self.assertEqual(extracted["uuid"], "test-uuid-456")
        self.assertEqual(extracted["sale_price"], 150000)
        self.assertEqual(extracted["buyer"], "test_buyer")

class TestCraftProfitability(unittest.TestCase):
    """Test craft profitability calculations."""
    
    def setUp(self):
        self.test_ontology = {
            "ENCHANTED_LAPIS_BLOCK": {
                "display_name": "Enchanted Lapis Block",
                "craft": {
                    "inputs": [
                        {"item": "ENCHANTED_LAPIS_LAZULI", "qty": 160}
                    ],
                    "fees": {"npc": 0, "tax_bps": 0}
                }
            },
            "ENCHANTED_LAPIS_LAZULI": {
                "display_name": "Enchanted Lapis Lazuli",
                "craft": {
                    "inputs": [
                        {"item": "LAPIS_LAZULI", "qty": 160}
                    ],
                    "fees": {"npc": 0, "tax_bps": 0}
                }
            }
        }
        
        # Mock database connection
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value.__enter__.return_value = self.mock_cursor

    def test_compute_craft_cost_simple(self):
        """Test simple craft cost calculation."""
        # Mock bazaar price for ENCHANTED_LAPIS_LAZULI
        self.mock_cursor.fetchone.return_value = (100,)  # 100 coins per unit
        
        cost, path = compute_craft_cost(self.test_ontology, self.mock_conn, "ENCHANTED_LAPIS_BLOCK")
        
        expected_cost = 100 * 160  # 160 units at 100 coins each
        self.assertEqual(cost, expected_cost)
        self.assertIn("buy(ENCHANTED_LAPIS_LAZULI)", path)

    def test_compute_craft_cost_recursive(self):
        """Test recursive craft cost calculation."""
        # Setup mock to return None for ENCHANTED_LAPIS_LAZULI (not on bazaar or AH)
        # but 2 coins for LAPIS_LAZULI
        def mock_fetchone_side_effect():
            calls = mock_fetchone_side_effect.call_count
            mock_fetchone_side_effect.call_count += 1
            if calls == 0:
                return None  # ENCHANTED_LAPIS_LAZULI not on bazaar
            elif calls == 1: 
                return None  # ENCHANTED_LAPIS_LAZULI not on AH either
            elif calls == 2:
                return (2,)  # LAPIS_LAZULI costs 2 coins on bazaar
            return None
        
        mock_fetchone_side_effect.call_count = 0
        self.mock_cursor.fetchone.side_effect = mock_fetchone_side_effect
        
        cost, path = compute_craft_cost(self.test_ontology, self.mock_conn, "ENCHANTED_LAPIS_BLOCK")
        
        # Cost should be: 160 * (160 * 2) = 160 * 320 = 51,200
        expected_cost = 160 * (160 * 2)
        self.assertEqual(cost, expected_cost)
        self.assertIn("craft(ENCHANTED_LAPIS_LAZULI:", path)

class TestExecutionEngine(unittest.TestCase):
    """Test execution engine simulation."""
    
    def setUp(self):
        self.engine = ExecutionEngine(
            latency_ms=100,
            slippage_bps=5,
            ah_fee_bps=100
        )
        
        self.market_data = {
            "TEST_ITEM": {
                "median_price": 1000.0,
                "p25_price": 900.0,
                "buy_price": 950.0,
                "sell_price": 1050.0
            }
        }

    def test_execute_buy_order_bin(self):
        """Test BIN buy order execution."""
        order = Order(
            product_id="TEST_ITEM",
            side=OrderSide.BUY,
            order_type=OrderType.BIN,
            quantity=1
        )
        
        fill = self.engine.execute_order(order, self.market_data, 1640995200.0)
        
        self.assertIsNotNone(fill)
        self.assertEqual(fill.product_id, "TEST_ITEM")
        self.assertEqual(fill.side, OrderSide.BUY)
        self.assertEqual(fill.quantity, 1)
        self.assertGreater(fill.fill_price, 1000.0)  # Should include slippage
        self.assertLess(fill.net_amount, 0)  # Negative for buy order

    def test_execute_sell_order_bazaar(self):
        """Test Bazaar sell order execution."""
        order = Order(
            product_id="TEST_ITEM", 
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1
        )
        
        fill = self.engine.execute_order(order, self.market_data, 1640995200.0)
        
        self.assertIsNotNone(fill)
        self.assertEqual(fill.side, OrderSide.SELL)
        self.assertLess(fill.fill_price, 950.0)  # Should include slippage
        self.assertGreater(fill.net_amount, 0)  # Positive for sell order
        self.assertGreater(fill.fees, 0)  # Should have fees

    def test_cannot_fill_missing_data(self):
        """Test order cannot be filled without market data."""
        order = Order(
            product_id="MISSING_ITEM",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1
        )
        
        fill = self.engine.execute_order(order, self.market_data, 1640995200.0)
        self.assertIsNone(fill)

class TestFlipStrategy(unittest.TestCase):
    """Test flip strategy logic."""
    
    def setUp(self):
        self.strategy = FlipStrategy(
            target_items=["TEST_ITEM"],
            min_profit_margin=50000,
            max_position_size=5
        )
        
        from backtesting.data_feed import MarketSnapshot
        self.snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            bazaar_data={},
            ah_data={
                "TEST_ITEM": {
                    "median_price": 200000,
                    "p25_price": 120000,  # Good flip opportunity
                    "sale_count": 10
                }
            }
        )

    def test_generates_buy_order_on_opportunity(self):
        """Test strategy generates buy order when profitable."""
        portfolio = {"inventory": {}, "cash": 1000000}
        
        orders = self.strategy.generate_orders(self.snapshot, portfolio)
        
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(order.product_id, "TEST_ITEM")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.BIN)
        self.assertEqual(order.quantity, 1)

    def test_generates_sell_order_with_inventory(self):
        """Test strategy generates sell order when holding inventory."""
        portfolio = {"inventory": {"TEST_ITEM": 2}, "cash": 1000000}
        
        orders = self.strategy.generate_orders(self.snapshot, portfolio)
        
        # Should generate sell order for existing inventory
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        self.assertEqual(len(sell_orders), 1)
        self.assertEqual(sell_orders[0].quantity, 2)

    def test_respects_position_limits(self):
        """Test strategy respects maximum position size."""
        portfolio = {"inventory": {"TEST_ITEM": 5}, "cash": 1000000}  # At max
        
        orders = self.strategy.generate_orders(self.snapshot, portfolio)
        
        # Should not generate buy orders when at position limit
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        self.assertEqual(len(buy_orders), 0)

if __name__ == "__main__":
    unittest.main()