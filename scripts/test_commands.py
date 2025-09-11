#!/usr/bin/env python3
"""
Test New Commands Functionality

Tests the new commands without requiring Discord:
- FeatureConsumer intelligence generation
- Market pulse analysis  
- Analyze functionality
- Diagnostics
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_feature_consumer():
    """Test FeatureConsumer with sample data."""
    print("📊 Testing FeatureConsumer with sample data...")
    
    try:
        import yaml
        from ingestion.feature_consumer import FeatureConsumer
        
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        consumer = FeatureConsumer(config)
        intelligence = consumer.generate_market_intelligence(window_hours=6)
        
        watchlist = intelligence.get('watchlist', set())
        fmv_data = intelligence.get('fmv_data', {})
        metadata = intelligence.get('metadata', {})
        
        print(f"  ✅ Watchlist: {len(watchlist)} items")
        print(f"  ✅ FMV data: {len(fmv_data)} items")
        print(f"  ✅ Hours analyzed: {metadata.get('hours_analyzed', 'Unknown')}")
        
        # Show sample watchlist items
        sample_items = list(watchlist)[:3]
        if sample_items:
            print(f"  📋 Sample items: {', '.join(sample_items)}")
        
        # Show sample FMV data
        if fmv_data:
            item_name = list(fmv_data.keys())[0]
            item_data = fmv_data[item_name]
            print(f"  💰 {item_name}: FMV={item_data.get('fmv', 0):,.0f}, "
                  f"Floor={item_data.get('floor_price', 0):,.0f}, "
                  f"Method={item_data.get('method', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FeatureConsumer test failed: {e}")
        return False

async def test_market_pulse():
    """Test market pulse analysis."""
    print("📈 Testing Market Pulse analysis...")
    
    try:
        import yaml
        from cogs.market_pulse import MarketPulseCog
        
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create a mock bot
        class MockBot:
            pass
        
        pulse_cog = MarketPulseCog(MockBot())
        
        # Test feature signal analysis
        from ingestion.feature_consumer import FeatureConsumer
        consumer = FeatureConsumer(config)
        intelligence = consumer.generate_market_intelligence(window_hours=6)
        
        signals = pulse_cog._analyze_feature_signals(intelligence)
        print(f"  ✅ Generated {len(signals)} feature signals")
        
        # Show sample signals
        for i, signal in enumerate(signals[:3], 1):
            print(f"  📊 Signal {i}: {signal['type']} - {signal['item']} ({signal['value']})")
        
        # Test bazaar signal analysis 
        import pandas as pd
        bazaar_file = Path("data/bazaar_history/sample_bazaar_data.parquet")
        if bazaar_file.exists():
            df = pd.read_parquet(bazaar_file)
            bazaar_signals = pulse_cog._analyze_bazaar_signals(df)
            print(f"  ✅ Generated {len(bazaar_signals)} bazaar signals")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Market Pulse test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_analyze():
    """Test analyze functionality."""
    print("🔍 Testing Analyze functionality...")
    
    try:
        import yaml
        from cogs.analyze import AnalyzeCog
        
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create a mock bot
        class MockBot:
            pass
        
        analyze_cog = AnalyzeCog(MockBot())
        
        # Test item analysis with sample data
        test_item = "WHEAT"
        
        # Load bazaar data for the item
        bazaar_df = analyze_cog._load_item_bazaar_data(test_item, hours=3)
        
        if bazaar_df is not None and not bazaar_df.empty:
            print(f"  ✅ Loaded {len(bazaar_df)} records for {test_item}")
            
            # Calculate statistics
            stats = analyze_cog._calculate_statistics(bazaar_df)
            print(f"  ✅ Calculated statistics: mid={stats.get('current_mid', 0):.2f}, "
                  f"spread_bps={stats.get('current_spread_bps', 0):.0f}")
            
            # Get feature data
            features = analyze_cog._get_feature_data(test_item)
            if features:
                print(f"  ✅ Found feature data: floor={features.get('floor_price', 0):,.0f}")
            else:
                print(f"  ⚠️ No feature data found for {test_item}")
            
            # Generate insights
            insights = analyze_cog._generate_insights(stats, features)
            print(f"  ✅ Generated {len(insights)} insights")
            
            for i, insight in enumerate(insights[:3], 1):
                print(f"  💡 Insight {i}: {insight}")
        else:
            print(f"  ⚠️ No bazaar data found for {test_item}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Analyze test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_diagnostics():
    """Test diagnostics functionality."""
    print("🔍 Testing Diagnostics functionality...")
    
    try:
        import yaml
        from cogs.diag import DiagnosticsCog
        
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create a mock bot
        class MockBot:
            pass
        
        diag_cog = DiagnosticsCog(MockBot())
        
        # Test feature summaries scan
        feature_result = diag_cog._scan_feature_summaries()
        print(f"  ✅ Feature scan: {feature_result.get('status', 'unknown')} - "
              f"{feature_result.get('total_files', 0)} files")
        
        # Test bazaar data scan
        bazaar_result = diag_cog._scan_bazaar_data()
        if bazaar_result.get('status') == 'ok':
            sources = bazaar_result.get('sources', [])
            available_sources = [s['name'] for s in sources if s.get('exists', False)]
            print(f"  ✅ Bazaar scan: {len(available_sources)} sources available")
        
        # Test environment check
        env_result = diag_cog._check_environment()
        if env_result.get('status') == 'ok':
            packages = env_result.get('packages', {})
            available_packages = [pkg for pkg, version in packages.items() if version != "Not installed"]
            print(f"  ✅ Environment check: {len(available_packages)} packages available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Diagnostics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all functionality tests."""
    print("🧪 Testing New Commands Functionality")
    print("=" * 50)
    
    tests = [
        ("FeatureConsumer", test_feature_consumer),
        ("Market Pulse", test_market_pulse),
        ("Analyze", test_analyze),
        ("Diagnostics", test_diagnostics),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All command tests passed! New functionality is working.")
        return 0
    else:
        print("⚠️ Some command tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)