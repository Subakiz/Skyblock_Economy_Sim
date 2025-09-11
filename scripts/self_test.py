#!/usr/bin/env python3
"""
Self-Test Script for SkyBlock Economy System

Verifies end-to-end functionality:
- Environment variables are present
- Feature summaries exist or can be created
- Bazaar data is accessible  
- Core components can be imported and instantiated

Non-zero exit code on failure.
"""

import sys
import os
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_environment():
    """Test environment variables and dependencies."""
    print("🔧 Testing environment...")
    
    # Load .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("  ⚠️ python-dotenv not available, checking system env vars only")
    
    # Check environment variables
    discord_token = os.getenv('DISCORD_BOT_TOKEN')
    hypixel_key = os.getenv('HYPIXEL_API_KEY')
    
    env_success = True
    
    if not discord_token:
        print("  ❌ DISCORD_BOT_TOKEN not set")
        env_success = False
    elif discord_token.startswith('test_'):
        print(f"  ⚠️ DISCORD_BOT_TOKEN is a test token ({len(discord_token)} chars) - OK for testing")
    else:
        print(f"  ✅ DISCORD_BOT_TOKEN present ({len(discord_token)} chars)")
    
    if not hypixel_key:
        print("  ❌ HYPIXEL_API_KEY not set")
        env_success = False
    elif hypixel_key.startswith('test_'):
        print(f"  ⚠️ HYPIXEL_API_KEY is a test key ({len(hypixel_key)} chars) - OK for testing")
    else:
        print(f"  ✅ HYPIXEL_API_KEY present ({len(hypixel_key)} chars)")
    
    # For testing purposes, accept test tokens
    if discord_token and hypixel_key:
        env_success = True
    
    # Check required packages
    required_packages = [
        'discord', 'pandas', 'pyarrow', 'matplotlib', 'yaml', 'aiohttp', 'psutil', 'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} missing")
    
    if missing_packages:
        print(f"  ❌ Missing packages: {missing_packages}")
        env_success = False
    
    return env_success

def test_directories():
    """Test that required directories exist or can be created."""
    print("📁 Testing directories...")
    
    required_dirs = [
        "data",
        "data/feature_summaries", 
        "data/raw_spool",
        "logs"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {dir_path} exists/created")
        except Exception as e:
            print(f"  ❌ Failed to create {dir_path}: {e}")
            return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("⚙️ Testing configuration...")
    
    try:
        import yaml
        
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            print(f"  ❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key sections
        required_sections = ['market', 'guards', 'storage']
        for section in required_sections:
            if section not in config:
                print(f"  ❌ Missing config section: {section}")
                return False
            else:
                print(f"  ✅ Config section {section} present")
        
        # Check key values
        market_config = config.get('market', {})
        intel_interval = market_config.get('intel_interval_seconds')
        if intel_interval is None:
            print(f"  ❌ Missing market.intel_interval_seconds")
            return False
        else:
            print(f"  ✅ intel_interval_seconds: {intel_interval}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False

def test_feature_consumer():
    """Test FeatureConsumer functionality."""
    print("📊 Testing FeatureConsumer...")
    
    try:
        import yaml
        from ingestion.feature_consumer import FeatureConsumer
        
        # Load config
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create consumer
        consumer = FeatureConsumer(config)
        print("  ✅ FeatureConsumer instantiated")
        
        # Test intelligence generation (may be empty if no data)
        intelligence = consumer.generate_market_intelligence(window_hours=1)
        
        watchlist_size = len(intelligence.get('watchlist', set()))
        fmv_count = len(intelligence.get('fmv_data', {}))
        
        print(f"  ✅ Generated intelligence: {watchlist_size} watchlist items, {fmv_count} FMV entries")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FeatureConsumer test failed: {e}")
        traceback.print_exc()
        return False

def test_feature_ingestor():
    """Test FeatureIngestor can be instantiated."""
    print("🔄 Testing FeatureIngestor...")
    
    try:
        import yaml
        from ingestion.feature_ingestor import FeatureIngestor
        
        # Load config
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create ingestor (this will test env loading and hypixel client creation)
        ingestor = FeatureIngestor(config)
        print("  ✅ FeatureIngestor instantiated")
        
        # Test flush method (should not fail even with no data)
        ingestor.flush_current_hour_summary()
        print("  ✅ flush_current_hour_summary() executed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FeatureIngestor test failed: {e}")
        # Don't print full traceback for expected failures (like missing API key in test env)
        if "HYPIXEL_API_KEY" in str(e):
            print("  💡 This is expected if HYPIXEL_API_KEY is not set")
        return False

def test_current_hour_summary():
    """Test if a current hour summary exists or can be created."""
    print("📈 Testing current hour summary...")
    
    try:
        summaries_path = Path("data/feature_summaries")
        current_time = datetime.now(timezone.utc)
        
        # Look for a summary file from the current hour
        current_hour_path = summaries_path / (
            f"year={current_time.year}/"
            f"month={current_time.month:02d}/"
            f"day={current_time.day:02d}/"
            f"hour={current_time.hour:02d}/"
            "summary.parquet"
        )
        
        if current_hour_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(current_hour_path)
                print(f"  ✅ Current hour summary exists: {len(df)} items at {current_hour_path}")
                return True
            except Exception as e:
                print(f"  ⚠️ Current hour summary exists but failed to read: {e}")
        
        # Look for any recent summaries (last 3 hours)
        recent_summaries = []
        for hour_offset in range(3):
            check_time = current_time - timedelta(hours=hour_offset)
            check_path = summaries_path / (
                f"year={check_time.year}/"
                f"month={check_time.month:02d}/"
                f"day={check_time.day:02d}/"
                f"hour={check_time.hour:02d}/"
                "summary.parquet"
            )
            if check_path.exists():
                recent_summaries.append(check_path)
        
        if recent_summaries:
            print(f"  ✅ Found {len(recent_summaries)} recent summaries (last 3h)")
            return True
        else:
            print(f"  ⚠️ No recent feature summaries found")
            print(f"     Expected location: {current_hour_path}")
            print(f"     Run the feature ingestor to generate summaries")
            return False
        
    except Exception as e:
        print(f"  ❌ Summary test failed: {e}")
        return False

def test_bazaar_data():
    """Test bazaar data availability."""
    print("🏪 Testing bazaar data...")
    
    bazaar_paths = [
        Path("data/bazaar_history"),
        Path("data/bazaar"),
        Path("data/bazaar_snapshots.ndjson")
    ]
    
    found_data = False
    
    for path in bazaar_paths:
        if not path.exists():
            print(f"  ❌ {path} does not exist")
            continue
        
        try:
            if path.is_dir():
                files = list(path.glob("*.parquet"))
                if files:
                    print(f"  ✅ {path} contains {len(files)} parquet files")
                    found_data = True
                else:
                    print(f"  ⚠️ {path} exists but contains no parquet files")
            else:
                # NDJSON file
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {path} exists ({size_mb:.1f} MB)")
                found_data = True
                
        except Exception as e:
            print(f"  ❌ Error checking {path}: {e}")
    
    if not found_data:
        print(f"  ⚠️ No bazaar data sources found")
        print(f"     Run data ingestion to populate bazaar data")
    
    return found_data

def main():
    """Run all self-tests."""
    print("🧪 SkyBlock Economy System Self-Test")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Directories", test_directories), 
        ("Configuration", test_config),
        ("FeatureConsumer", test_feature_consumer),
        ("FeatureIngestor", test_feature_ingestor),
        ("Current Hour Summary", test_current_hour_summary),
        ("Bazaar Data", test_bazaar_data),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
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
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)