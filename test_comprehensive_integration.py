#!/usr/bin/env python3

"""
Comprehensive integration test for file-based mode with all features.
Tests the complete workflow from data creation to ML forecasting to API endpoints.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
import requests
import subprocess
import time
import json

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_comprehensive_test():
    """Run a comprehensive test of all file-based features."""
    print("🚀 Comprehensive File-Based Mode Integration Test")
    print("=" * 60)
    
    # Clean up any existing data
    if os.path.exists("data"):
        shutil.rmtree("data")
    if os.path.exists("models"):
        shutil.rmtree("models")
    
    # Step 1: Verify config is set correctly
    print("\n1️⃣ Verifying configuration...")
    
    try:
        from storage.ndjson_storage import get_storage_instance
        from modeling.profitability.file_data_access import load_config_for_file_mode
        
        config = load_config_for_file_mode()
        storage = get_storage_instance(config)
        
        if storage is None:
            print("❌ File mode not enabled in config")
            return False
        
        print("✅ File mode configuration verified")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Step 2: Create comprehensive test data
    print("\n2️⃣ Creating comprehensive test data...")
    
    try:
        success = subprocess.run([
            sys.executable, "setup_api_test.py"
        ], capture_output=True, text=True, timeout=120)
        
        if success.returncode != 0:
            print(f"❌ Test data creation failed: {success.stderr}")
            return False
        
        print("✅ Test data and ML models created successfully")
    except subprocess.TimeoutExpired:
        print("❌ Test data creation timed out")
        return False
    except Exception as e:
        print(f"❌ Test data creation error: {e}")
        return False
    
    # Step 3: Start API server
    print("\n3️⃣ Starting API server...")
    
    api_process = None
    try:
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "services.api.app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8000/healthz", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("mode") == "file-based":
                    print("✅ API server started in file-based mode")
                else:
                    print(f"❌ API server not in file-based mode: {health_data}")
                    return False
            else:
                print(f"❌ API server health check failed: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Could not connect to API server: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False
    
    # Step 4: Test all API endpoints
    print("\n4️⃣ Testing API endpoints...")
    
    test_results = {}
    product_id = "ENCHANTED_LAPIS_BLOCK"
    
    # Test basic endpoints
    endpoints_to_test = [
        ("GET", f"/prices/ah/{product_id}", None, "Auction house prices"),
        ("GET", f"/profit/craft/{product_id}", None, "Craft profitability"),
        ("GET", f"/forecast/{product_id}?horizon_minutes=60", None, "ML forecasting"),
        ("POST", "/ml/train", {
            "product_id": product_id, 
            "model_type": "lightgbm", 
            "horizons": [15, 60]
        }, "ML model training"),
        ("POST", "/analysis/predictive", {
            "items": [product_id], 
            "model_type": "lightgbm", 
            "include_opportunities": True
        }, "Predictive analysis"),
        ("POST", "/backtest/run", {
            "strategy": "simple_buy_hold",
            "params": {},
            "start_date": "2024-01-01",
            "end_date": "2024-02-01", 
            "capital": 1000000
        }, "Backtesting")
    ]
    
    for method, endpoint, data, description in endpoints_to_test:
        try:
            print(f"   Testing {description}...")
            
            if method == "GET":
                response = requests.get(f"http://localhost:8000{endpoint}", timeout=30)
            else:
                response = requests.post(
                    f"http://localhost:8000{endpoint}",
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=45
                )
            
            if response.status_code == 200:
                result_data = response.json()
                test_results[description] = "✅ PASS"
                
                # Additional validation for key endpoints
                if "forecast" in endpoint.lower():
                    if "forecast_price" in result_data:
                        print(f"     Forecast: ${result_data['forecast_price']:.2f}")
                    else:
                        print(f"     Warning: No forecast_price in response")
                
                elif "predictive" in endpoint:
                    opportunities = result_data.get("trading_opportunities", [])
                    print(f"     Found {len(opportunities)} trading opportunities")
                
            else:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                    error_msg += f": {error_detail}"
                except:
                    error_msg += f": {response.text[:100]}"
                
                test_results[description] = f"❌ FAIL - {error_msg}"
                
        except requests.exceptions.Timeout:
            test_results[description] = "❌ FAIL - Timeout"
        except Exception as e:
            test_results[description] = f"❌ FAIL - {str(e)[:50]}"
    
    # Cleanup
    if api_process:
        api_process.terminate()
        api_process.wait()
    
    # Step 5: Report results
    print("\n5️⃣ Test Results Summary:")
    print("-" * 40)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if "✅ PASS" in result)
    
    for description, result in test_results.items():
        print(f"{description:.<30} {result}")
    
    print("-" * 40)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! File-based mode has full feature parity!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)