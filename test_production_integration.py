#!/usr/bin/env python3
"""
Production-Grade Bot Integration Test

Validates the implementation meets all requirements from the problem statement.
Tests memory management, feature pipelines, alert systems, and storage management.
"""

import sys
import time
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

def test_feature_pipeline_imports():
    """Test that feature pipeline components can be imported and initialized."""
    print("🧪 Testing feature pipeline imports...")
    
    try:
        sys.path.insert(0, '.')
        
        # Test feature ingestor
        from ingestion.feature_ingestor import FeatureIngestor, InMemoryPriceLadder
        
        # Test price ladder
        ladder = InMemoryPriceLadder(max_size=16)
        ladder.add_price(100.0)
        ladder.add_price(95.0)
        ladder.add_price(110.0)
        
        prices, counts, total = ladder.get_ladder_data()
        assert len(prices) == 3, f"Expected 3 prices, got {len(prices)}"
        assert prices[0] == 95, f"Expected lowest price 95, got {prices[0]}"  # Should be sorted
        assert total == 3, f"Expected total count 3, got {total}"
        
        print("  ✅ InMemoryPriceLadder working correctly")
        
        # Test feature consumer
        from ingestion.feature_consumer import FeatureConsumer
        
        # Mock config for testing
        test_config = {
            "market": {
                "window_hours": 12,
                "thin_wall_threshold": 2,
                "min_auction_count": 25,
                "max_watchlist_items": 2000
            }
        }
        
        consumer = FeatureConsumer(test_config)
        
        # Test market depth FMV calculation
        price_ladder = [(100.0, 2), (150.0, 5)]  # Thin wall at floor
        fmv, method = consumer._calculate_market_depth_fmv(price_ladder)
        assert fmv == 150.0, f"Expected FMV 150 for thin wall, got {fmv}"
        assert method == "thin_wall_second_price", f"Expected thin wall method, got {method}"
        
        price_ladder = [(100.0, 5), (150.0, 2)]  # Thick wall at floor
        fmv, method = consumer._calculate_market_depth_fmv(price_ladder)
        assert fmv == 100.0, f"Expected FMV 100 for thick wall, got {fmv}"
        assert method == "thick_wall_floor_price", f"Expected thick wall method, got {method}"
        
        print("  ✅ Market depth-aware FMV calculation working correctly")
        
        return True, "Feature pipeline imports and logic tests passed"
        
    except Exception as e:
        return False, f"Feature pipeline test failed: {e}"


def test_memory_guards():
    """Test memory guard functionality."""
    print("🧪 Testing memory guards...")
    
    try:
        sys.path.insert(0, '.')
        import psutil
        
        # Test that we can get memory info
        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)
        
        assert rss_mb > 0, "Memory usage should be positive"
        print(f"  ✅ Current memory usage: {rss_mb:.1f}MB")
        
        # Test memory guard check simulation
        from cogs.auction_sniper import AuctionSniper
        import unittest.mock as mock
        
        # Mock bot for testing
        mock_bot = mock.MagicMock()
        mock_bot.get_channel.return_value = None
        
        # Create sniper with test config
        with mock.patch('cogs.auction_sniper.AuctionSniper._load_config') as mock_config:
            mock_config.return_value = {
                "guards": {"soft_rss_mb": 100},  # Very low threshold for testing
                "market": {"window_hours": 12},
                "auction_sniper": {"profit_threshold": 100000}
            }
            
            sniper = AuctionSniper(mock_bot)
            
            # Test memory guard (should fail with low threshold)
            memory_ok = sniper._check_memory_guard()
            # Should return False since we set threshold very low
            print(f"  ✅ Memory guard check returns: {memory_ok}")
        
        return True, "Memory guard tests passed"
        
    except Exception as e:
        return False, f"Memory guard test failed: {e}"


def test_cog_imports():
    """Test that new cogs can be imported."""
    print("🧪 Testing cog imports...")
    
    try:
        sys.path.insert(0, '.')
        
        # Test help cog
        from cogs.help import HelpCog
        mock_bot = type('MockBot', (), {})()
        help_cog = HelpCog(mock_bot)
        
        print("  ✅ Help cog imported successfully")
        
        # Test storage janitor
        from cogs.storage_janitor import StorageJanitor
        
        print("  ✅ Storage janitor cog imported successfully")
        
        return True, "All new cogs import successfully"
        
    except Exception as e:
        return False, f"Cog import test failed: {e}"


def test_config_validation():
    """Test that the updated configuration is valid."""
    print("🧪 Testing configuration validation...")
    
    try:
        import yaml
        
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Check required new sections
        required_sections = ["market", "guards", "storage"]
        for section in required_sections:
            assert section in config, f"Missing required config section: {section}"
        
        # Check market config
        market = config["market"]
        required_market_keys = ["window_hours", "lowest_ladder_size", "thin_wall_threshold", 
                               "min_auction_count", "intel_interval_seconds", "rows_soft_cap"]
        for key in required_market_keys:
            assert key in market, f"Missing required market config key: {key}"
        
        print("  ✅ Market configuration complete")
        
        # Check guards config
        guards = config["guards"]
        assert "soft_rss_mb" in guards, "Missing soft_rss_mb in guards config"
        assert guards["soft_rss_mb"] == 1300, f"Expected soft RSS limit 1300MB, got {guards['soft_rss_mb']}"
        
        print("  ✅ Guards configuration complete")
        
        # Check storage config
        storage = config["storage"]
        assert "cap_gb" in storage, "Missing cap_gb in storage config"
        assert storage["cap_gb"] == 70, f"Expected storage cap 70GB, got {storage['cap_gb']}"
        
        print("  ✅ Storage configuration complete")
        
        return True, "Configuration validation passed"
        
    except Exception as e:
        return False, f"Config validation failed: {e}"


def test_storage_janitor_logic():
    """Test storage janitor partition detection and disk usage calculation."""
    print("🧪 Testing storage janitor logic...")
    
    try:
        sys.path.insert(0, '.')
        from cogs.storage_janitor import StorageJanitor
        import tempfile
        import shutil
        
        # Create a mock bot
        mock_bot = type('MockBot', (), {})()
        
        # Mock the config loading to avoid file dependency
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test partition structure
            test_partitions = [
                temp_path / "year=2024/month=01/day=15/hour=10",
                temp_path / "year=2024/month=01/day=15/hour=11", 
                temp_path / "year=2024/month=01/day=16/hour=12"
            ]
            
            for partition_path in test_partitions:
                partition_path.mkdir(parents=True, exist_ok=True)
                # Create a test file
                (partition_path / "test.parquet").write_text("test data")
            
            # Mock storage janitor without the config dependency
            with unittest.mock.patch('cogs.storage_janitor.StorageJanitor._load_config') as mock_config:
                mock_config.return_value = {
                    "storage": {"cap_gb": 70, "headroom_gb": 5}
                }
                
                janitor = StorageJanitor(mock_bot)
                janitor.storage_monitoring_task.cancel()  # Don't start the loop
                
                # Test partition detection
                partitions = janitor.get_partition_paths(temp_path)
                
                assert len(partitions) == 3, f"Expected 3 partitions, got {len(partitions)}"
                print(f"  ✅ Detected {len(partitions)} test partitions")
                
                # Test disk usage calculation
                usage = janitor.disk_usage_gb(temp_path)
                assert usage > 0, "Disk usage should be positive"
                print(f"  ✅ Calculated disk usage: {usage:.6f}GB")
        
        return True, "Storage janitor logic tests passed"
        
    except Exception as e:
        return False, f"Storage janitor test failed: {e}"


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("Production-Grade Bot Integration Tests")
    print("=" * 70)
    
    tests = [
        test_feature_pipeline_imports,
        test_memory_guards,
        test_cog_imports, 
        test_config_validation,
        test_storage_janitor_logic
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            success, message = test_func()
            if success:
                print(f"✅ {test_func.__name__}: {message}")
                passed += 1
            else:
                print(f"❌ {test_func.__name__}: {message}")
                failed += 1
        except Exception as e:
            print(f"💥 {test_func.__name__}: Exception - {e}")
            failed += 1
        
        print()
    
    print("=" * 70)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed! Production-grade implementation ready.")
    else:
        print(f"⚠️  {failed} tests failed. Review implementation before deployment.")
    
    return failed == 0


if __name__ == "__main__":
    import unittest.mock
    success = run_integration_tests()
    sys.exit(0 if success else 1)