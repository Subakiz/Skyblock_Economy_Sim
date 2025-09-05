#!/usr/bin/env python3
"""
Quick test to verify Phase 3 API endpoints work correctly.
"""

import requests
import json
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

API_BASE = "http://localhost:8000"

def test_api_status():
    """Test basic API health."""
    try:
        response = requests.get(f"{API_BASE}/healthz", timeout=5)
        print(f"✓ API Health: {response.status_code}")
        return True
    except Exception as e:
        print(f"✗ API Health failed: {e}")
        return False

def test_phase3_endpoints():
    """Test Phase 3 specific endpoints."""
    
    # Test scenarios endpoint
    try:
        response = requests.get(f"{API_BASE}/scenarios/available", timeout=10)
        if response.status_code == 200:
            scenarios = response.json()
            print(f"✓ Available scenarios: {len(scenarios['scenarios'])} scenarios")
        else:
            print(f"✗ Scenarios endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Scenarios endpoint failed: {e}")
    
    # Test models status endpoint
    try:
        response = requests.get(f"{API_BASE}/models/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✓ Models status: Phase 3 available = {status['phase3_available']}")
        else:
            print(f"✗ Models status failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Models status failed: {e}")
    
    # Test market simulation endpoint
    try:
        sim_request = {
            "scenario": "normal_market",
            "n_agents": 10,
            "steps": 5,
            "market_volatility": 0.02
        }
        response = requests.post(f"{API_BASE}/simulation/market", 
                               json=sim_request, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Market simulation: {result['total_trades']} trades, "
                  f"sentiment = {result['market_sentiment']:.3f}")
        else:
            print(f"✗ Market simulation failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:200]}")
    except Exception as e:
        print(f"✗ Market simulation failed: {e}")

if __name__ == "__main__":
    print("Phase 3 API Integration Test")
    print("=" * 40)
    
    print("\nNote: This test requires the API server to be running.")
    print("Start with: uvicorn services.api.app:app --reload")
    print("\nTesting Phase 3 endpoints...")
    
    if not test_api_status():
        print("\n✗ API server is not running or not accessible.")
        print("Please start the server first: uvicorn services.api.app:app --reload")
        sys.exit(1)
    
    test_phase3_endpoints()
    
    print("\n✓ Phase 3 API integration test completed!")