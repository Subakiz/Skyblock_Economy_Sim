#!/usr/bin/env python3

"""
Test the API endpoints in file-based mode.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
import json

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.ndjson_storage import NDJSONStorage
from modeling.forecast.file_ml_forecaster import train_and_forecast_ml_from_files


def create_api_test_data():
    """Create test data for API testing."""
    
    # Create data directory
    data_dir = "data"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    storage = NDJSONStorage(data_directory=data_dir)
    product_id = "ENCHANTED_LAPIS_BLOCK"
    
    print(f"Creating API test data for {product_id}...")
    
    base_time = datetime.now(timezone.utc) - timedelta(hours=48)
    base_price = 150000
    
    # Create bazaar data
    for i in range(600):
        ts = base_time + timedelta(minutes=i * 5)
        
        trend = 100 * i
        volatility = 5000 * (0.5 - abs(0.5 - (i % 100) / 100))
        noise = (hash(str(i)) % 1000 - 500) * 10
        
        price = base_price + trend + volatility + noise
        
        record = {
            "product_id": product_id,
            "buy_price": price * 1.02,
            "sell_price": price * 0.98,
            "buy_volume": 1000 + (i % 500),
            "sell_volume": 800 + (i % 400),
            "ts": ts.isoformat()
        }
        
        storage.append_record("bazaar", record)
    
    # Create auction data
    for i in range(50):
        ts = base_time + timedelta(hours=i)
        
        price_variation = (hash(str(i)) % 1000 - 500) * 100
        sale_price = base_price + price_variation
        
        record = {
            "item_id": product_id,
            "sale_price": sale_price,
            "timestamp": ts.isoformat()
        }
        
        storage.append_record("auctions_ended", record)
    
    print("✓ Test data created")
    
    # Train ML model
    print("Training ML model for API testing...")
    success = train_and_forecast_ml_from_files(product_id, model_type="lightgbm")
    if success:
        print("✓ ML model trained and forecasts generated")
    else:
        print("✗ ML model training failed")
    
    return success


def test_api_endpoints():
    """Test API endpoints manually."""
    print("\n=== API Endpoint Testing ===")
    print("Start the API server with: uvicorn services.api.app:app --host 0.0.0.0 --port 8000")
    print("Then test these endpoints:")
    print()
    print("1. Health check (should show file-based mode):")
    print("   curl http://localhost:8000/healthz")
    print()
    print("2. Get auction house prices:")
    print("   curl http://localhost:8000/prices/ah/ENCHANTED_LAPIS_BLOCK")
    print()
    print("3. Get craft profitability:")
    print("   curl http://localhost:8000/profit/craft/ENCHANTED_LAPIS_BLOCK")
    print()
    print("4. Get forecast (now supported in file mode!):")
    print("   curl http://localhost:8000/forecast/ENCHANTED_LAPIS_BLOCK?horizon_minutes=60")
    print()
    print("5. Train ML model:")
    print('   curl -X POST http://localhost:8000/ml/train -H "Content-Type: application/json" \\')
    print('        -d \'{"product_id": "ENCHANTED_LAPIS_BLOCK", "model_type": "lightgbm", "horizons": [15, 60, 240]}\'')
    print()
    print("6. Run predictive analysis:")
    print('   curl -X POST http://localhost:8000/analysis/predictive -H "Content-Type: application/json" \\')
    print('        -d \'{"items": ["ENCHANTED_LAPIS_BLOCK"], "model_type": "lightgbm", "include_opportunities": true}\'')
    print()


if __name__ == "__main__":
    print("File-Based API Testing Setup")
    print("=" * 50)
    
    success = create_api_test_data()
    
    if success:
        test_api_endpoints()
        print("\n✓ Test data and models ready for API testing")
    else:
        print("\n✗ Failed to create test data")
        sys.exit(1)