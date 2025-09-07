#!/usr/bin/env python3
"""
Basic smoke test for the SkyBlock Economic Modeling components.
Tests basic imports and functionality without requiring external dependencies.
"""
import sys
import os
import json
import yaml

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from ingestion.common.hypixel_client import HypixelClient, TokenBucket
        print("✓ HypixelClient imported successfully")
        
        from services.api.app import app
        print("✓ FastAPI app imported successfully")
        
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration file loading."""
    print("\nTesting configuration...")
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Check required sections for no-database mode
        required_sections = ['hypixel', 'no_database_mode', 'features', 'forecast']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        # Verify no-database mode is enabled
        if not config.get('no_database_mode', {}).get('enabled', False):
            print(f"✗ No-database mode not enabled in config")
            return False
        
        print("✓ Configuration file valid")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_ontology():
    """Test item ontology file."""
    print("\nTesting item ontology...")
    try:
        with open("item_ontology.json", "r") as f:
            ontology = json.load(f)
        
        # Check that we have at least one item
        if not ontology:
            print("✗ Empty ontology file")
            return False
            
        # Check structure of first item
        first_item = next(iter(ontology.values()))
        required_keys = ['display_name', 'craft', 'sinks', 'sources']
        for key in required_keys:
            if key not in first_item:
                print(f"✗ Missing ontology key: {key}")
                return False
        
        print(f"✓ Item ontology valid ({len(ontology)} items)")
        return True
    except Exception as e:
        print(f"✗ Ontology test failed: {e}")
        return False

def test_token_bucket():
    """Test the TokenBucket rate limiter."""
    print("\nTesting TokenBucket...")
    try:
        from ingestion.common.hypixel_client import TokenBucket
        import time
        
        # Create a bucket with 60 tokens per minute (1 per second)
        bucket = TokenBucket(rate_per_minute=60, burst=5)
        
        # Should be able to consume 5 tokens immediately
        start_time = time.time()
        for i in range(5):
            bucket.consume(1)
        elapsed = time.time() - start_time
        
        if elapsed > 0.1:  # Should be nearly instantaneous
            print(f"✗ TokenBucket too slow for burst: {elapsed}s")
            return False
        
        print("✓ TokenBucket working correctly")
        return True
    except Exception as e:
        print(f"✗ TokenBucket test failed: {e}")
        return False

def test_sql_schema():
    """Test that SQL schema file exists and is readable."""
    print("\nTesting SQL schema...")
    try:
        with open("storage/schema.sql", "r") as f:
            schema = f.read()
        
        # Check for required tables
        required_tables = ['bazaar_snapshots', 'bazaar_features', 'model_forecasts']
        for table in required_tables:
            if table not in schema:
                print(f"✗ Missing table in schema: {table}")
                return False
        
        print("✓ SQL schema contains all required tables")
        return True
    except Exception as e:
        print(f"✗ SQL schema test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("SkyBlock Economic Modeling - Smoke Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_ontology,
        test_token_bucket,
        test_sql_schema,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The scaffold is ready for deployment.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())