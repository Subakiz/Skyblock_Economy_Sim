#!/usr/bin/env python3
"""
Test script for Part Two expansion features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_events_loading():
    """Test that events.json loads correctly"""
    print("Testing events.json loading...")
    try:
        import json
        with open("data/events.json", "r") as f:
            events = json.load(f)
        print(f"✅ Loaded {len(events)} events")
        
        # Test a few events
        for event_id, event_data in list(events.items())[:3]:
            print(f"   - {event_data.get('name', event_id)}: {event_data.get('type', 'Unknown type')}")
        return True
    except Exception as e:
        print(f"❌ Failed to load events: {e}")
        return False

def test_market_baskets():
    """Test market baskets configuration"""
    print("\nTesting market baskets configuration...")
    try:
        import yaml
        with open("config/market_baskets.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        baskets = config.get('market_baskets', {})
        print(f"✅ Loaded {len(baskets)} market baskets")
        
        for basket_name, basket_data in baskets.items():
            items_count = len(basket_data.get('items', []))
            print(f"   - {basket_data.get('name', basket_name)}: {items_count} items")
        return True
    except Exception as e:
        print(f"❌ Failed to load market baskets: {e}")
        return False

def test_enhanced_auction_pipeline():
    """Test enhanced auction feature pipeline"""
    print("\nTesting enhanced auction feature pipeline...")
    try:
        from modeling.features.auction_feature_pipeline import load_events, calculate_event_features
        
        # Test event loading
        events = load_events()
        print(f"✅ Event loading function works: {len(events)} events")
        
        # Test event features calculation (with dummy data)
        import pandas as pd
        from datetime import datetime, timezone
        
        dummy_df = pd.DataFrame({
            'item_id': ['HYPERION', 'WHEAT'],
            'ts': [datetime.now(timezone.utc), datetime.now(timezone.utc)],
            'final_price': [1000000, 1000]
        })
        
        enhanced_df = calculate_event_features(dummy_df, events)
        print("✅ Event features calculation works")
        print(f"   - Added event columns: {[col for col in enhanced_df.columns if 'event' in col or 'mayor' in col]}")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced auction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictive_engine_methods():
    """Test new predictive engine methods"""
    print("\nTesting enhanced predictive engine...")
    try:
        from modeling.simulation.file_predictive_engine import FileBasedPredictiveMarketEngine
        
        engine = FileBasedPredictiveMarketEngine()
        print("✅ FileBasedPredictiveMarketEngine instantiated")
        
        # Test method existence
        methods = ['run_cross_item_analysis', 'run_event_impact_analysis', 'run_market_pulse_analysis']
        for method in methods:
            if hasattr(engine, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Predictive engine test failed: {e}")
        return False

def test_bot_commands():
    """Test that bot has new commands"""
    print("\nTesting bot commands...")
    try:
        from bot import SkyBlockEconomyBot
        print("✅ Bot imported successfully")
        
        # Note: We can't easily test the actual command registration without running the bot
        # But we can check that the functions exist
        import bot
        
        command_functions = ['compare_command', 'event_impact_command', 'market_pulse_command', 'event_autocomplete']
        for func_name in command_functions:
            if hasattr(bot, func_name):
                print(f"✅ Command function {func_name} exists")
            else:
                print(f"❌ Command function {func_name} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Bot commands test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Part Two Expansion Features")
    print("=" * 50)
    
    tests = [
        test_events_loading,
        test_market_baskets,
        test_enhanced_auction_pipeline,
        test_predictive_engine_methods,
        test_bot_commands
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Part Two expansion features are working!")
        print("\n📋 New Features Available:")
        print("   - /compare <item_a> <item_b> - Cross-item arbitrage analysis")
        print("   - /event_impact <event_name> - Event impact analysis") 
        print("   - /market_pulse - Holistic market overview")
        print("   - Enhanced auction pipeline with event awareness")
        print("   - Market baskets for sector analysis")
    else:
        print("❌ Some tests failed - check the output above")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())