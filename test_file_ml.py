#!/usr/bin/env python3

"""
Test file-based ML forecasting and feature engineering.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.ndjson_storage import NDJSONStorage
from modeling.profitability.file_feature_engineering import create_feature_dataframe_from_files
from modeling.forecast.file_ml_forecaster import (
    train_and_forecast_ml_from_files,
    get_latest_forecast_from_files,
    fetch_multivariate_series_from_files
)


def create_test_data(storage, product_id="ENCHANTED_LAPIS_BLOCK"):
    """Create sample test data for ML forecasting."""
    print(f"Creating test data for {product_id}...")
    
    base_time = datetime.now(timezone.utc) - timedelta(hours=48)  # 2 days ago
    base_price = 150000
    
    # Create realistic bazaar data with trends and volatility
    for i in range(600):  # 600 records over 48 hours (5-minute intervals)
        ts = base_time + timedelta(minutes=i * 5)
        
        # Add some trend and volatility
        trend = 100 * i  # Gradual price increase
        volatility = 5000 * (0.5 - abs(0.5 - (i % 100) / 100))  # Cyclical volatility
        noise = (hash(str(i)) % 1000 - 500) * 10  # Random noise
        
        price = base_price + trend + volatility + noise
        
        record = {
            "product_id": product_id,
            "buy_price": price * 1.02,  # 2% spread
            "sell_price": price * 0.98,
            "buy_volume": 1000 + (i % 500),
            "sell_volume": 800 + (i % 400),
            "ts": ts.isoformat()
        }
        
        storage.append_record("bazaar", record)
    
    print(f"Created {600} bazaar records")
    
    # Create some auction data too
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
    
    print(f"Created {50} auction records")


def test_file_feature_engineering():
    """Test file-based feature engineering."""
    print("\n=== Testing File-Based Feature Engineering ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = NDJSONStorage(data_directory=temp_dir)
        product_id = "ENCHANTED_LAPIS_BLOCK"
        
        # Create test data
        create_test_data(storage, product_id)
        
        # Test feature engineering
        df = create_feature_dataframe_from_files(storage, product_id, hours_back=48)
        
        if df is None:
            print("✗ Feature engineering failed")
            return False
        
        print(f"✓ Created feature DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # Check required features exist
        required_features = [
            'mid_price', 'ma_15', 'ma_60', 'spread_bps', 'vol_window_30',
            'price_lag_1', 'price_lag_5', 'momentum_1', 'momentum_5',
            'ma_crossover', 'ma_ratio', 'vol_price_ratio',
            'hour_of_day', 'day_of_week', 'day_of_month',
            'market_volatility', 'market_momentum'
        ]
        
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            print(f"✗ Missing features: {missing}")
            return False
        
        print("✓ All required features present")
        
        # Check data quality
        if df['mid_price'].isna().sum() > len(df) * 0.1:  # More than 10% NaN
            print(f"✗ Too many NaN values in mid_price: {df['mid_price'].isna().sum()}")
            return False
        
        print("✓ Data quality checks passed")
        return True


def test_file_ml_forecasting():
    """Test file-based ML forecasting."""
    print("\n=== Testing File-Based ML Forecasting ===")
    
    # Update config to enable file mode for testing
    config_content = """
hypixel:
  base_url: "https://api.hypixel.net"
  
storage:
  database_url: "postgresql://user:pass@localhost:5432/skyblock"

no_database_mode:
  enabled: true
  data_directory: "test_data"
  max_file_size_mb: 100
  retention_hours: 168
  
phase3:
  ml_models:
    default_model: "lightgbm"
"""
    
    # Create temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config = f.name
    
    # Create test data directory
    test_data_dir = "test_data"
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    
    try:
        storage = NDJSONStorage(data_directory=test_data_dir)
        product_id = "ENCHANTED_LAPIS_BLOCK"
        
        # Create test data
        create_test_data(storage, product_id)
        
        # Backup original config
        original_config = None
        if os.path.exists("config/config.yaml"):
            with open("config/config.yaml", 'r') as f:
                original_config = f.read()
        
        # Use temporary config
        shutil.copy(temp_config, "config/config.yaml")
        
        # Test ML forecasting
        print("Training ML model with file data...")
        success = train_and_forecast_ml_from_files(product_id, model_type="lightgbm")
        
        if not success:
            print("✗ ML training failed")
            return False
        
        print("✓ ML model trained successfully")
        
        # Test forecast retrieval
        forecast = get_latest_forecast_from_files(storage, product_id, 60)
        
        if not forecast:
            print("✗ No forecast found")
            return False
        
        print(f"✓ Forecast retrieved: {forecast['forecast_price']:.2f}")
        
        # Test multiple horizons
        horizons = [15, 60, 240]
        forecasts_found = 0
        
        for horizon in horizons:
            forecast = get_latest_forecast_from_files(storage, product_id, horizon)
            if forecast:
                forecasts_found += 1
                print(f"  {horizon}min: {forecast['forecast_price']:.2f}")
        
        if forecasts_found == 0:
            print("✗ No forecasts found for any horizon")
            return False
        
        print(f"✓ Found forecasts for {forecasts_found}/{len(horizons)} horizons")
        
        # Restore original config
        if original_config:
            with open("config/config.yaml", 'w') as f:
                f.write(original_config)
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(temp_config):
            os.unlink(temp_config)
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)


def main():
    """Run file-based ML tests."""
    print("File-Based ML Forecasting Tests")
    print("=" * 50)
    
    tests = [
        test_file_feature_engineering,
        test_file_ml_forecasting,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All file-based ML tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())