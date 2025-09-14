#!/usr/bin/env python3
"""
Test script to verify the auction sniper fixes work correctly.
Tests both the bridge conversion and sniper logic improvements.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_auction_features_bridge():
    """Test that the bridge conversion worked."""
    logger.info("🧪 Testing auction features bridge...")
    
    try:
        from modeling.features.auction_feature_pipeline import load_auction_features_from_file
        
        # Test loading the generated file
        df = load_auction_features_from_file('data')
        
        if df is not None and not df.empty:
            logger.info(f"✅ Bridge test PASSED: Loaded {len(df)} auction features for {df['item_name'].nunique()} items")
            logger.info(f"   Items: {list(df['item_name'].unique())}")
            return True
        else:
            logger.error("❌ Bridge test FAILED: Could not load auction features")
            return False
            
    except Exception as e:
        logger.error(f"❌ Bridge test FAILED: {e}")
        return False

def test_feature_consumer():
    """Test that FeatureConsumer generates market intelligence."""
    logger.info("🧪 Testing FeatureConsumer market intelligence...")
    
    try:
        import yaml
        from ingestion.feature_consumer import FeatureConsumer
        
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Create feature consumer
        consumer = FeatureConsumer(config)
        
        # Generate intelligence
        intelligence = consumer.generate_market_intelligence()
        
        if intelligence and intelligence.get("watchlist"):
            watchlist = intelligence["watchlist"]
            fmv_data = intelligence["fmv_data"]
            logger.info(f"✅ FeatureConsumer test PASSED: {len(watchlist)} watchlist items, {len(fmv_data)} FMV entries")
            return True, intelligence
        else:
            logger.error("❌ FeatureConsumer test FAILED: No market intelligence generated")
            return False, None
            
    except Exception as e:
        logger.error(f"❌ FeatureConsumer test FAILED: {e}")
        return False, None

async def test_sniper_logic():
    """Test the auction sniper logic improvements."""
    logger.info("🧪 Testing auction sniper logic...")
    
    try:
        import yaml
        from unittest.mock import MagicMock
        from cogs.auction_sniper import AuctionSniper
        
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Create mock bot
        mock_bot = MagicMock()
        
        # Create auction sniper
        sniper = AuctionSniper(mock_bot)
        
        # Test that it loads without the market intelligence running
        logger.info(f"   Watchlist size: {len(sniper.auction_watchlist)}")
        logger.info(f"   FMV data size: {len(sniper.fmv_data)}")
        
        # Run market intelligence update
        await sniper.update_market_intelligence()
        
        logger.info(f"   After intelligence update:")
        logger.info(f"   Watchlist size: {len(sniper.auction_watchlist)}")
        logger.info(f"   FMV data size: {len(sniper.fmv_data)}")
        
        if sniper.auction_watchlist and sniper.fmv_data:
            # Test the test auction generation
            test_auctions = sniper._generate_test_auctions()
            logger.info(f"   Generated {len(test_auctions)} test auctions")
            
            # Test verification logic
            snipes_found = 0
            for auction in test_auctions:
                if await sniper._verify_snipe(auction):
                    snipes_found += 1
                    item_name = auction.get("item_name")
                    price = auction.get("starting_bid")
                    logger.info(f"   ✅ Valid snipe: {item_name} @ {price:,.0f}")
            
            logger.info(f"✅ Sniper logic test PASSED: Found {snipes_found} valid snipes out of {len(test_auctions)} test auctions")
            return True
        else:
            logger.error("❌ Sniper logic test FAILED: No market intelligence available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sniper logic test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    logger.info("🚀 Starting auction sniper fix verification tests...")
    logger.info("=" * 60)
    
    # Test 1: Bridge conversion
    bridge_pass = test_auction_features_bridge()
    
    # Test 2: Feature consumer
    consumer_pass, intelligence = test_feature_consumer()
    
    # Test 3: Sniper logic
    sniper_pass = await test_sniper_logic()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY:")
    logger.info(f"   Bridge Conversion: {'✅ PASS' if bridge_pass else '❌ FAIL'}")
    logger.info(f"   Feature Consumer: {'✅ PASS' if consumer_pass else '❌ FAIL'}")
    logger.info(f"   Sniper Logic: {'✅ PASS' if sniper_pass else '❌ FAIL'}")
    
    all_pass = bridge_pass and consumer_pass and sniper_pass
    
    if all_pass:
        logger.info("🎉 ALL TESTS PASSED! Auction sniper fixes are working correctly.")
        return 0
    else:
        logger.error("💥 SOME TESTS FAILED! Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)