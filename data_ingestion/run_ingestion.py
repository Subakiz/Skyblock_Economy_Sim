#!/usr/bin/env python3
"""
Standalone Data Ingestion Service Entry Point
Required by the production-grade Hypixel Market Analysis & Sniping Application.

This script serves as the canonical entry point for the data ingestion service,
as specified in the mandated two-process architecture.
"""

import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import and run the existing standalone ingestion service
from ingestion.standalone_ingestion_service import main

if __name__ == "__main__":
    print("=" * 70)
    print("Skyblock Economy Simulator - Data Ingestion Service")
    print("=" * 70)
    print("Production-grade market data ingestion process starting...")
    print("This service will continuously fetch and process Hypixel auction data.")
    print("")
    print("Features:")
    print("- Canonical item normalization from /resources/items API")
    print("- Auction ingestion loop (90 second intervals)")
    print("- Bazaar ingestion loop (60 second intervals)")
    print("- Partitioned Parquet dataset output")
    print("")
    print("Press Ctrl+C to stop the service.")
    print("=" * 70)
    
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Data ingestion service stopped by user.")
    except Exception as e:
        print(f"\n❌ Service error: {e}")
        sys.exit(1)