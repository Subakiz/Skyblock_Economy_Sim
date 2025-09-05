"""
Integration tests for Phase 2 API endpoints.
"""

import unittest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from services.api.app import app

class TestPhase2API(unittest.TestCase):
    """Test new API endpoints for Phase 2."""
    
    def setUp(self):
        self.client = TestClient(app)

    @patch('services.api.app.psycopg2.connect')
    def test_ah_prices_endpoint(self, mock_connect):
        """Test AH prices endpoint."""
        # Mock database response
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (
            150000,  # median_price
            120000,  # p25_price  
            180000,  # p75_price
            25,      # sale_count
            "2025-01-01T12:00:00"  # time_bucket
        )
        mock_connect.return_value = mock_conn
        
        response = self.client.get("/prices/ah/ENCHANTED_LAPIS_BLOCK?window=1h")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["product_id"], "ENCHANTED_LAPIS_BLOCK")
        self.assertEqual(data["window"], "1h")
        self.assertEqual(data["median_price"], 150000)
        self.assertEqual(data["p25_price"], 120000)
        self.assertEqual(data["sale_count"], 25)

    @patch('services.api.app.psycopg2.connect')
    @patch('services.api.app.compute_profitability')
    def test_craft_profitability_endpoint(self, mock_compute, mock_connect):
        """Test craft profitability endpoint."""
        from modeling.profitability.crafting_profit import CraftProfitability
        
        # Mock profitability calculation result
        mock_result = CraftProfitability(
            product_id="ENCHANTED_LAPIS_BLOCK",
            craft_cost=100000,
            expected_sale_price=150000,
            gross_margin=50000,
            net_margin=48500,  # after fees
            roi_percent=48.5,
            turnover_adj_profit=45000,
            best_path="buy(ENCHANTED_LAPIS_LAZULI)",
            sell_volume=15,
            data_age_minutes=30
        )
        mock_compute.return_value = mock_result
        
        response = self.client.get("/profit/craft/ENCHANTED_LAPIS_BLOCK?horizon=1h&pricing=median")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["product_id"], "ENCHANTED_LAPIS_BLOCK")
        self.assertEqual(data["craft_cost"], 100000)
        self.assertEqual(data["expected_sale_price"], 150000)
        self.assertEqual(data["gross_margin"], 50000)
        self.assertEqual(data["roi_percent"], 48.5)
        self.assertEqual(data["best_path"], "buy(ENCHANTED_LAPIS_LAZULI)")

    def test_backtest_endpoint(self):
        """Test backtest endpoint (mock implementation)."""
        backtest_request = {
            "strategy": "flip_bin",
            "params": {"min_profit": 50000},
            "start_date": "2025-01-01",
            "end_date": "2025-01-31", 
            "capital": 1000000,
            "item_id": "ENCHANTED_LAPIS_BLOCK"
        }
        
        response = self.client.post("/backtest/run", json=backtest_request)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_return", data)
        self.assertIn("total_return_pct", data)
        self.assertIn("max_drawdown", data)
        self.assertIn("sharpe_ratio", data)
        self.assertIn("total_trades", data)

    @patch('services.api.app.psycopg2.connect')
    def test_ah_prices_not_found(self, mock_connect):
        """Test AH prices endpoint when data not found."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # No data found
        mock_connect.return_value = mock_conn
        
        response = self.client.get("/prices/ah/NONEXISTENT_ITEM")
        
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

    def test_backtest_invalid_dates(self):
        """Test backtest endpoint with invalid date format."""
        backtest_request = {
            "strategy": "flip_bin",
            "params": {},
            "start_date": "invalid-date",  # Invalid format
            "end_date": "2025-01-31",
            "capital": 1000000
        }
        
        response = self.client.post("/backtest/run", json=backtest_request)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Date parsing error", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()